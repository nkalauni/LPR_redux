#!/usr/bin/env python
# coding: utf-8

# # Set up Pycap LPR to operate with PEST++ for optimization using fish response curves

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import pyemu
import geopandas as gp
import matplotlib.pyplot as plt
import os, shutil, platform, zipfile
import dash
from dash import html, dcc, Input, Output
import plotly.express as px
import folium
from shapely.geometry import MultiPoint
import matplotlib
import matplotlib.colors as mcolors

GPM2CFS = 0.002228

#######################################################################################
# function to create all files necessary to run Multiple Objective Optimization (MOU) #
#######################################################################################
def prepare_MOU_files(pump_lbound_fraction=0,
                      pump_ubound_fraction=1.2,
                      objectives='fish_dollars', #'dep_q','fish_q', or 'fish_dollars'
                      depletion_potential_threshold=0.1,
                      scenario_name=None
                    ):
    objectives = objectives.lower()
    if objectives not in ['depletion_q','fish_q','fish_dollars']:
        raise Exception('Objectives must be either: depletion_q,fish_q, or fish_dollars')
    if scenario_name is None:
        scenario_name = objectives 

    #### PyCap Run Name is what all your outputs will have as a name. 
    pycap_run_name = "LPR_Redux"

    #### Base directory for runs
    parent_run_path = Path("../pycap_runs")

    #### depletion potential calculations directory
    base_run_path = parent_run_path / "pycap_base"
    pest_path = parent_run_path / "pycap_pest"
    template_path = pest_path / f"{scenario_name}_template"
    fish_path = Path('../Inputs/fish_curves')
    econ_path = Path('../econ')

    #### set path to save configurations in
    configs_path = Path('./configurations')
    if not configs_path.exists():
        configs_path.mkdir()

    #### finally define the scripts directory
    script_path = Path("../scripts")

    #### Path at which to run MOU
    run_path = pest_path / f"run_{scenario_name}"

    # ### Let's check out the fish curves 
    ref_q = 8.6
    
    ### prepare to save the run options 
    run_options = {
        'depletion_potential_threshold': depletion_potential_threshold,
        'objectives': objectives,
        'pump_lbound_fraction': pump_lbound_fraction,
        'pump_ubound_fraction': pump_ubound_fraction,
        'ref_q': ref_q,
        'run_path': str(run_path),
        'scenario_name': scenario_name
    }


    with open(configs_path / f'{scenario_name}.yml', 'w+') as ofp:
        yaml.dump(run_options, ofp)
    print(f"wrote configuration to: {str(configs_path / scenario_name)}.yml")

    # # Parameterization for PEST++ 
    if not template_path.exists():
        template_path.mkdir(parents=True)


    # #### let's load up the configuration file
    with open(base_run_path / f"{pycap_run_name}.yml", 'r') as ifp:
            indat = yaml.safe_load(ifp)


    # #### and now parameterize inputs to vary in optimization. This will only be pumping rates for now
    # 
    # First we set up template (`TPL`) files that allow PEST++ to update model input values by name. We do this by reading in the input (`YML`) file and replacing numeric values with updated values being changed by the algorithm.

    # well-by-well pumping rates
    well_keys = [i for i in indat.keys() if i.startswith('well_')]

    pars = list()
    parvals = list()

    # then again for pumping rate Q
    for k in well_keys:
        cpar = f'{k}__q'
        pars.append(cpar)
        parvals.append(indat[k]['Q'])
        indat[k]['Q'] = f'~{cpar:^45}~'


    # save out tpl file
    with open(template_path / f"{pycap_run_name}.yml.tpl", 'w') as ofp:
        ofp.write('ptf ~\n')
        documents = yaml.dump(indat, ofp, default_flow_style = False, sort_keys = False)


    # create DataFrame of parameters
    pars_df = pd.DataFrame(index = pars, data= {'parval1':parvals})


    # ### Next we need to be able to read in model ouputs to PEST++
    # Now we write an instruction file (`INS`) that can navigate model output and read it into PEST++

    # make ins file and external forward run file
    # set base case depletion observations
    basedeplobs = [f"{indat[k]['name']}:bdpl" for k in indat.keys() if 'stream' in k]

    # get list of unique stream names used in the run
    unique_rivers = list(set([i.split(':')[0] for i in basedeplobs]))

    # add in the totals/sums of proposed/existing/combined depletions for each stream
    basedeplobs.extend([f'{i}:{j}:bdpl' for i in unique_rivers for j in ['total_proposed','total_existing','total_combined']])

    with open(template_path / 'basedeplobs.dat','w') as ofp:
        [ofp.write(i + '\n') for i in basedeplobs]


    # ### Now read in the base case observation values for depletion data

    base_data = pd.read_csv(base_run_path/"output" / f'{pycap_run_name}.table_report.base_stream_depletion.csv', index_col=0)
    # read in the observation names and make a DataFrame to keep the results in
    bdplobs = pd.read_csv(template_path/'basedeplobs.dat', header=None)
    bdplobs.columns =['obsname']
    bdplobs.index = bdplobs.obsname
    bdplobs['obs_values'] = np.nan

    # now map the actual output values to the DataFrame
    for cob in bdplobs.obsname:
        riv,wel,_ = cob.split(':')
        bdplobs.loc[cob, 'obs_values'] = base_data.loc[wel][riv]
    if 'fish' in objectives:
        bdplobs.loc['fish_prob'] = ['fish_prob',100.0]
    if 'dollar' in objectives:
        bdplobs.loc['ag_receipts'] = ['ag_receipts',1e8]

    # ### We can combine all the outputs into a single dataframe and make the instruction file we'll need to read in the results


    bdplobs['obs_values'].to_csv(template_path / 'allobs.out', sep = ' ', header=None)

    with open(template_path / 'allobs.out.ins', 'w') as ofp:
        ofp.write('pif ~\n')
        [ofp.write(f'l1 w !{i}!\n') for i in bdplobs.index]


    # ### Now we need to make a PEST control file to orchestrate everything. Luckily, `pyemu` makes this straightforward now that we have made the `tpl` and `ins` files 

    cwd = os.getcwd()
    os.chdir(template_path)
    pst = pyemu.Pst.from_io_files(*pyemu.utils.parse_dir_for_io_files('.'))
    os.chdir(cwd)

    pars = pst.parameter_data
    obs = pst.observation_data


    # # let's clean up some of the data and add important values
    # name paramter groups according to the type of parameter
    pars.loc[pars.parnme.str.endswith("q"), "pargp"] = "pumping"
    # set initial values
    pars.loc[pars_df.index,'parval1'] = pars_df.parval1

    # set upper and lower bounds on pumping rates as well - 
    pars.loc[pars.pargp=="pumping","parlbnd"] = pars.loc[pars.pargp=="pumping", "parval1"]*pump_lbound_fraction 
    pars.loc[pars.pargp=="pumping","parubnd"] = pars.loc[pars.pargp=="pumping", "parval1"]*pump_ubound_fraction
    pars.partrans = 'none'

    # ### We need to make some definitions for multi-objective optimization algorithm (MOU)

    # ### first, optionally, we can consider only the wells greater than a specified depletion potential threshold be managed.
    # ### Do this by setting the variable `dp_thresh`

    dp = gp.read_file(base_run_path / 'depletion_potential.json')
    dp.set_index('index', inplace=True)
    dp_thresh = depletion_potential_threshold # user provided - as function arg


    # ### now let's assign only the wells exceeding the depletion potential threshold to be adjustable (e.g. "decision variables" or "decvars")

    pars["wellname"] = [i.split("__")[0] for i in pars.index]

    # we only adjust wells for which we have receipts information if the pumping objective is in dollars
    if 'dollar' in objectives:
        receipts = pd.read_csv(econ_path / 'total_receipts.csv', index_col=0)
        pars_df['wellno'] = [int(i.split('_')[1]) for i in pars_df.index]
        pars_df=pars_df.merge(receipts['total_receipts'], 
                      left_on = 'wellno', 
                      right_index=True, 
                      how='outer').fillna(0)
        pars_df.to_csv(template_path / 'wells_and_receipts.csv')
        pars.loc[((pars.wellname.isin(dp.loc[dp.Depletion_Potential >= dp_thresh].index)) &
             (pars_df.loc[pars.index, 'total_receipts'] > 0)), 'pargp'] = 'decvars'
    else:
        pars.loc[pars.wellname.isin(dp.loc[dp.Depletion_Potential >= dp_thresh].index),'pargp'] = 'decvars'
    # we need to "fix" (e.g. make unadjustable) the pumping rates that are not decision variables
    pars.loc[pars.pargp=="pumping", "partrans"] = "fixed"
    pars.sample(5)


    # ### we need to sample a prior population of pumping rates, centered reasonably closely to the initial pumping rates. To do this, we assume we can temporarily set the upper and lower bounds to, say, +/1 10% of the initial value
    pars.loc[pars.pargp=="decvars", "parlbnd"] = pars.loc[pars.pargp=="decvars", "parval1"]*0.8
    pars.loc[pars.pargp=="decvars", "parubnd"] = pars.loc[pars.pargp=="decvars", "parval1"]*1.2

    # we start with about 2x the number of decision variables as the population size
    num_reals = len(pars.loc[pars.pargp=="decvars"])*2
    # sample decision variables from a uniform distribution
    dvpop = pyemu.ParameterEnsemble.from_uniform_draw(pst,num_reals=num_reals)

    # record to external file for PESTPP-MOU
    dvpop.to_csv(template_path / "initial_dvpop.csv")
    # tell PESTPP-MOU about the new file
    pst.pestpp_options["mou_dv_population_file"] = 'initial_dvpop.csv'
    # reset the decision variable bounds
    pars.loc[pars.pargp=="decvars","parlbnd"] = pars.loc[pars.pargp=="decvars", "parval1"]*pump_lbound_fraction 
    pars.loc[pars.pargp=="decvars","parubnd"] = pars.loc[pars.pargp=="decvars", "parval1"]*pump_ubound_fraction


    # ### now we need to identify the (competing) objectives. This is a bit complex now, based on the user requests
    
    # start with the most basic scenario - depletion against combined pumping
    if objectives == 'depletion_q':
        
        # first create a prior information equation aggregating all the pumping in the decision variables
        pst.add_pi_equation(pars.loc[pars.pargp=='decvars','parnme'], # parameter names to include in the equation
                            pilbl="obj_well",  # the prior information equation name
                            obs_group="greater_than_pumping") # note the "greater_" prefix.   
        
        # next identify the total depletion as a distinct observation group
        obs.loc[obs.index.str.contains("total_combined"), "obgnme"] = "less_than_depletion"
        # now reset all weights except this one to be 0
        obs.weight = 0
        obs.loc[obs.obgnme=="less_than_depletion", "weight"] = 1.0
        pst.pestpp_options["mou_objectives"] = ["obj_well",
                                        "lpr:total_combined:bdpl"]


        forward_script = 'run_pycap_standalone_opt_mou.py'

        
    elif objectives == 'fish_q':
        # first create a prior information equation aggregating all the pumping in the decision variables
        pst.add_pi_equation(pars.loc[pars.pargp=='decvars','parnme'], # parameter names to include in the equation
                            pilbl="obj_well",  # the prior information equation name
                            obs_group="greater_than_pumping") # note the "greater_" prefix.   
        
        # next identify the fish probability as a distinct observation group
        obs.loc[obs.index.str.contains("fish_prob"), "obgnme"] = "greater_than_fishprob"
        # now reset all weights except this one to be 0
        obs.weight = 0
        obs.loc[obs.obgnme=="greater_than_fishprob", "weight"] = 1.0
        pst.pestpp_options["mou_objectives"] = ["obj_well",
                                            "fish_prob"]
        forward_script = 'run_pycap_standalone_opt_mou_fish.py'
        
    elif objectives == 'fish_dollars':
        obs.loc[obs.index.str.contains("receipt"), "obgnme"] = "greater_than_receipts"
        # now reset all weights except this one to be 0
        obs.weight = 0
        obs.loc[obs.obgnme=="greater_than_receipts", "weight"] = 1.0         
        # next identify the fish probability as a distinct observation group
        obs.loc[obs.index.str.contains("fish_prob"), "obgnme"] = "greater_than_fishprob"
        # now reset all weights except this one to be 0
        obs.loc[obs.obgnme=="greater_than_fishprob", "weight"] = 1.0
        pst.pestpp_options["mou_objectives"] = ["ag_receipts",
                                        "fish_prob"]
        
   
        

        forward_script = 'run_pycap_standalone_opt_mou_fish_dollars.py'



    # some additional PESTPP-MOU options:
    pst.pestpp_options["mou_population_size"] = num_reals #twice the number of decision variables
    pst.pestpp_options["mou_save_population_every"] = 1 # save lots of files! 
                                                        # but this way we can inspect how MOU progressed    


    pst.control_data.noptmax = 50
    pst.model_command = [f'python {forward_script} {scenario_name}']
    pst.write(str(template_path / f"{scenario_name}.pst"), version=2)

    # Copy over directories
    if run_path.exists():
        shutil.rmtree(run_path)
    # copy over the binaries into template first so they get distributed

    configs_path = Path('./configurations')
    if not configs_path.exists():
        configs_path.mkdir()
    # save out all the options to a YML file
    run_options = {
        'scenario_name': scenario_name,
        'objectives': objectives,
        'pump_lbound_fraction':pump_lbound_fraction,
        'pump_ubound_fraction':pump_ubound_fraction,
        'depletion_potential_threshold': depletion_potential_threshold,
        'run_path': str(run_path),
        'ref_q': ref_q,
        'num_reals': num_reals     
    }
    
    #copy over the correct binary
    if "window" in platform.platform().lower():
        shutil.copy2('../../binaries/PESTPP/windows/pestpp-mou.exe', template_path / 'pestpp-mou.exe')
    elif "linux" in platform.platform().lower():
        shutil.copy2('../../binaries/PESTPP/linux/pestpp-mou', template_path / 'pestpp-mou')
    elif "mac" in platform.platform().lower():
        shutil.copy2('../../binaries/PESTPP/mac/pestpp-mou', template_path / 'pestpp-mou')

    # and we need the base yml file
    shutil.copy2(base_run_path / 'LPR_Redux.yml',
                template_path / 'LPR_Redux.yml')
    # we also need the forward run script
    shutil.copy2(script_path / forward_script, 
                template_path / forward_script)
    if 'fish' in objectives:
        # we need the fish file
        shutil.copy2(fish_path / 'Brook.csv',
                template_path / 'Brook.csv')         
        
    # finally populate the run path with the template_path files
    shutil.copytree(template_path, run_path)
    os.remove(run_path / 'allobs.out')

    print(f"ALL READY TO RUN:\n scenario: {scenario_name}\n directory: {run_path}")
    return scenario_name, run_path

#######################################################################################
# function to summarize MOU output and only include feasible, non-dominated solutions #
#######################################################################################
def postprocess_MOU(run_name, run_path):
    pareto_df = pd.read_csv(run_path / f"{run_name}.pareto.archive.summary.csv.zip")

    # subset to the feasible and non-dominated members
    pareto_df = pareto_df.loc[pareto_df.apply(lambda x: x.nsga2_front==1 and 
                                              x.is_feasible==1,
                                              axis=1),:]
    
    pareto_df.member = [str(i) for i in pareto_df.member]
    # let's clean up some column names
    for ccol,newcol in zip(
        ['lpr:total_combined:bdpl','obj_well', 'ag_receipts','fish_prob'],
        ['Depletion (cfs)', 'Total Pumping (cfs)', 
         'Total Agriculture Receipts ($)',
         'Trout Likelihood (%)'],
        ):
        if ccol in pareto_df.columns:
            pareto_df = pareto_df.rename(columns={ccol:newcol})
            if 'total pumping' in newcol.lower():
                pareto_df[newcol] *=  GPM2CFS #need to convert from GPM to CFS here!
    return pareto_df
#######################################################################################################
# function to support interactive plotting of a pareto curve highlighting evoluation over generations #
#######################################################################################################
def plot_pareto(currgen, pareto_df):
    x_ax = pareto_df.columns[3]
    y_ax = pareto_df.columns[2]
    fig,ax = plt.subplots()
    ax.scatter(pareto_df[x_ax], pareto_df[y_ax], c='.5', marker='.', alpha=.4)
    currdf = pareto_df.loc[pareto_df.generation==currgen]
    ax.scatter(currdf[x_ax], currdf[y_ax], c='b', marker='.')
    ax.set_title(f'Pareto Tradeoff for Generation {currgen}')
    ax.set_xlabel(x_ax)
    ax.set_ylabel(y_ax)
    plt.show()

####################################################################################################
# function to align objective values on the pareto frontier with the associated decision variables #
####################################################################################################
def prep_for_viz(pareto_df, final_generation, run_path, run_name, dollar_objective):
    #### Base directory for runs
    parent_run_path = Path("../pycap_runs")
    econ_path = Path('../econ')
    #### depletion potential calculations directory
    base_run_path = parent_run_path / "pycap_base"
    dp = gp.read_file(base_run_path / 'depletion_potential.json')
    dp.set_index('index', inplace=True)

    pst = pyemu.Pst(str(run_path / f"{run_name}.pst"))
    pars = pst.parameter_data
    
    # create DataFrame of parameters
    pars_df = pars['parval1'].to_frame()

    # subset decision variables to the realizations that are included in the pareto curve
    pareto_df_final = pareto_df.loc[pareto_df.generation==final_generation]
    # read in all the decision variable pumping rates
    dv_df = pd.concat([pd.read_csv(i, index_col=0)
                   for i in run_path.glob("*dv_pop*")])
 

    dv_df = pd.concat([dv_df, pd.read_csv(run_path / 'initial_dvpop.csv', index_col=0)])
    dv_df.index = [str(i) for i in dv_df.index]
    dv_df = dv_df[~dv_df.index.duplicated(keep='first')]
    dv_df = dv_df.loc[pareto_df_final.member]
    dv_df.columns = [i.split("__")[0] for i in dv_df.columns]
    dv_df = dv_df.T
    dv_df['geometry'] = dp.loc[dv_df.index, 'geometry']
    dv_df = gp.GeoDataFrame(data=dv_df, crs=dp.crs)

    pars_df.index = [i.split('__')[0] for i in pars_df.index]
    pars_df['geometry'] = dv_df.loc[pars_df.index,'geometry']
    pars_df = gp.GeoDataFrame(pars_df, crs=dv_df.crs)
    if dollar_objective:
        for cc in dv_df.columns:
            if 'geometry' not in cc:
                dv_df.loc[dv_df[cc]<=pars.loc[pars.wellname==dv_df[cc].index,'parval1'].values*.7,cc]=0
        receipts = pd.read_csv(econ_path / 'total_receipts.csv', index_col=0)
        pars_df['wellno'] = [int(i.split('_')[1]) for i in pars_df.index]
        pars_df=pars_df.merge(receipts['total_receipts'], 
                            left_on = 'wellno', 
                            right_index=True, 
                            how='outer').fillna(0)
    for cc in dv_df.columns:
        if 'geometry' not in cc:
            dv_df[cc] /= pars_df.loc[dv_df.index,'parval1'].values
    return pareto_df_final, dv_df
##############################################################################################################
# function to set up interactive plot of pareto frontier with maps of decision variable pumping arrangements #
##############################################################################################################

def create_viz_app(pareto_df_final, dv_df):
    app = dash.Dash(__name__)
    x_ax = pareto_df_final.columns[3]
    y_ax = pareto_df_final.columns[2]
    @app.callback(
        Output('scatter-plot', 'figure'),
        Input('scatter-plot', 'clickData')
    )
    def update_scatter_highlight(clickData):
        fig = px.scatter(
            pareto_df_final, x=x_ax, y=y_ax, title="Pareto Tradeoff"
        )
        if clickData:
            index = clickData['points'][0]['pointIndex']
            fig.add_trace(
                px.scatter(
                    pareto_df_final.iloc[[index]],x=x_ax, y=y_ax
                ).update_traces(
                    marker=dict(size=15, color='red', line=dict(width=2, color='black'))
                ).data[0]
            )
        return fig

    app.layout = html.Div([
        html.Div([
            dcc.Graph(id='scatter-plot'),
        ], style={'width': '50%', 'display': 'inline-block'}),

        html.Div([
            html.Iframe(id='map-plot', width='100%', height='600')
        ], style={'width': '50%', 'display': 'inline-block'})
    ])

    @app.callback(
        Output('map-plot', 'srcDoc'),
        Input('scatter-plot', 'clickData')
    )
    def display_map(clickData):
        if clickData is None:
            return ""

        index = clickData['points'][0]['pointIndex']

        val_array = dv_df.iloc[:, index].values
        min_val, max_val = val_array.min(), val_array.max()
        scaled_values = 3 + 12 * (val_array - min_val) / (max_val - min_val + 1e-6)
        norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
        cmap = matplotlib.colormaps['magma']

        geometries = dv_df["geometry"]
        points = list(zip(
            geometries,
            scaled_values,
            [mcolors.to_hex(cmap(norm(v))) for v in val_array],
            val_array
        ))

        if not points:
            return ""

        centroid = MultiPoint([pt for pt, _, _, _ in points]).centroid
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=11)

        for pt, radius, ccolor, val in points:
            folium.CircleMarker(
                location=[pt.y, pt.x], weight=1, radius=radius/2,
                color=ccolor, fill=True, fill_color=ccolor, fill_opacity=0.7,
                popup=f"% pumping: {val:.2f}"
            ).add_to(m)

        return m.get_root().render()

    # Expose for WSGI if needed
    app.server = app.server
    return app


def plot_pareto_with_scenarios(pareto_df, scenarios=None):

    x_ax = pareto_df.columns[3]
    y_ax = pareto_df.columns[2]
    fig,ax = plt.subplots()
    ax.scatter(pareto_df[x_ax], pareto_df[y_ax], c='.5', marker='.', alpha=.4)
    currdf = pareto_df.loc[pareto_df.generation==pareto_df.generation.max()]
    ax.scatter(currdf[x_ax], currdf[y_ax], c='b', marker='.')
    ax.set_title(f'Pareto Tradeoff and Scenarios')

    if scenarios is None:
        print('no scenarios provided')
    else:
        if not isinstance(scenarios,list):
            scenarios = [scenarios]
        scen_dict = {cscen: pd.read_csv(f'../pycap_runs/student_run/{cscen}_results.csv') 
                     for cscen in scenarios}
        # we need to set up whether considering threshold drops or not as indicated by "receipts"
        mounames = ['Depletion (cfs)', 'Total Pumping (cfs)', 
            'Total Agriculture Receipts ($)',
            'Trout Likelihood (%)']
        if ('receipt' in x_ax.lower()) or ('receipt' in y_ax.lower()):
            scennames = ['truncated_depletion', 'wells_total_q_ag',
                         'receipts','fish_prob']
        else:
            scennames = ['total_depletion', 'wells_total_q',
                         'receipts','fish_prob']
        for mouname, scenname in zip(mounames, scennames):
            if mouname in x_ax:
                x_col = scenname
            if mouname in y_ax:
                y_col = scenname


        for scen,dat in scen_dict.items():
            ax.scatter(dat[x_col],dat[y_col],marker='x', c='red')


    ax.set_xlabel(x_ax)
    ax.set_ylabel(y_ax)
    plt.show()

    