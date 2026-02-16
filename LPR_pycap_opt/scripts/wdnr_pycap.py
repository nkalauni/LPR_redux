import pandas as pd
import yaml
from pycap.analysis_project import Project
import numpy as np
GPM2CFS = 0.002228

def pycap_metrics(depl, well_dict, combdict, base_run_path, pycap_run_name):
    # we need some extra information
    fishcurve = pd.read_csv('../Inputs/fish_curves/Brook.csv', index_col=1)['POmeasure']
    receipts = pd.read_csv('../econ/total_receipts.csv', index_col=0)
    orig_q = pd.read_excel('../Inputs/LPR_Prepped_IDW.xlsx', sheet_name='HCW_Inputs', index_col=0)
    orig_q.index = [f'well_{i}__q' for i in orig_q.index]

    # well-by-well pumping rates
    well_keys = [i for i in well_dict.keys() if i.startswith('well_')]

    pars = list()
    parvals = list()

    # then again for pumping rate Q
    for k in well_keys:
        cpar = f'{k}__q'
        pars.append(cpar)
        parvals.append(well_dict[k]['Q'])
        
    pars_df = pd.DataFrame(index = pars, data= {'parval1':parvals})

    # get the total pumping in all wells in GPM regardless of censoring
    wells_total_q = pars_df['parval1'].sum()

    pars_df['wellno'] = [int(i.split('_')[1]) for i in pars_df.index]
    pars_df=pars_df.merge(receipts['total_receipts'], 
                      left_on = 'wellno', 
                      right_index=True, 
                      how='outer').fillna(0)
    pars_df = pars_df.merge(orig_q['Q_gpm'],left_index=True,
                  right_index=True).rename(columns={'Q_gpm':'orig_q'})
    # set all pumping rates <= 0.7 times original pumping rate to 0
    pars_df.loc[pars_df.parval1 <= 0.7*pars_df.orig_q, 'parval1'] = 0
    # then get a new total pumping set where low rates are cut off
    wells_total_q_ag = pars_df['parval1'].sum()

    # now we need to know the receipts value
    total_receipts = (pars_df.parval1 / pars_df.orig_q * pars_df.total_receipts).sum()

    # now get the total depletion
    total_depletion = depl.loc['total_combined','LPR']

    # finally, we need to recalculate depletion after having turned off some of the wells
    # because they fell below the 70% threshold
    pars_df['yml_wellname'] = [i.split('__')[0] for i in pars_df.index]
    for i in pars_df.yml_wellname:
        cQ = pars_df.loc[pars_df.yml_wellname==i,'parval1'].values[0]
        combdict[i]['Q'] = float(cQ)
    with open(base_run_path / 'tmprun.yml', "w") as file:
        documents = yaml.dump(combdict, file, default_flow_style=False, sort_keys=False)
    ap = Project(base_run_path /  'tmprun.yml')
    ap.report_responses()
    ap.write_responses_csv()
    new_depl = pd.read_csv(
            base_run_path / f"output/tmprun.table_report.base_stream_depletion.csv",
                        index_col = 0).loc['total_combined','LPR']
    ref_flow=8.6
    fish_prob =  np.interp(ref_flow - new_depl,
                  fishcurve.index,
                  fishcurve.values)
    retvals = pd.DataFrame(index=[pycap_run_name],
                           data={'wells_total_q':[wells_total_q*GPM2CFS], # convert here to CFS
                                 'wells_total_q_ag':[wells_total_q_ag*GPM2CFS], # convert here to CFS
                                 'receipts':[total_receipts],
                                 'total_depletion':[total_depletion],
                                 'truncated_depletion':[new_depl],
                                 'fish_prob':[fish_prob]})
    return retvals

def Excel2YML(pycap_inputs_excel, pycap_run_name, pycap_run_path):
    # read in the raw files from excel
    raw_global = pd.read_excel(pycap_inputs_excel, sheet_name = 'Global_Inputs')
    raw_hcw = pd.read_excel(pycap_inputs_excel, sheet_name = 'HCW_Inputs')
    raw_dd = pd.read_excel(pycap_inputs_excel, sheet_name = 'Drawdown_Inputs')
    raw_depl = pd.read_excel(pycap_inputs_excel, sheet_name = 'Depletion_Inputs')

    # munging to make everything readable by python
    raw_hcw["HCW"] = raw_hcw["HCW"].astype(str)
    raw_dd["HCW"] = raw_dd["HCW"].astype(str)
    raw_depl["HCW"] = raw_depl["HCW"].astype(str)
    raw_dd["pycap_resource_name"] = raw_dd["Resource_Name"]
    raw_depl["pycap_resource_name"] = raw_depl["Resource_Name"]

    raw_dd["Resource_Name"] = [i.replace(" ", "") for i in raw_dd["Resource_Name"]]
    raw_dd["pycap_resource_name"] = [
        ":".join((rn, hcw)) for rn, hcw in zip(raw_dd["Resource_Name"], raw_dd["HCW"])
    ]

    raw_depl["Resource_Name"] = [i.replace(" ", "") for i in raw_depl["Resource_Name"]]
    raw_depl["pycap_resource_name"] = [
        ":".join((rn, hcw)) for rn, hcw in zip(raw_depl["Resource_Name"], raw_depl["HCW"])
    ]    

    # create a project properties dictionary
    project_dict = dict()
    project_dict['project_properties'] = {
        'name':pycap_run_name,
        'T':float(raw_global['Transmissivity_ft2d'][0]),
        'Max_T':float(raw_global['Max_Trans_ft2d'][0]),
        'Min_T':float(raw_global['Min_Trans_ft2d'][0]),
        'S':float(raw_global['Storage_Coeff'][0]),
        'Max_S':float(raw_global['Max_S'][0]),
        'Min_S':float(raw_global['Min_S'][0]),
        'Max_FracInt':float(raw_global['Max_FractInt'][0]),
        'Min_FracInt':float(raw_global['Min_FractInt'][0]),
        'default_dd_days':float(raw_global['Default_dd_days'][0]),
        'default_depletion_years':float(raw_global['Default_depletion_years'].values[0]),
        'default_pumping_days':float(raw_global['Default_pumping_days'].values[0])
    }    
 
    stream_dict = dict()
    stream_dict = {
        i: {
            "HCW": hcw,
            "stream_apportionment": {
                "name": pycap_resource_name,
                "apportionment": float(frac_intercept),
            },
        }
        for i, hcw, pycap_resource_name, frac_intercept in zip(
            range(len(raw_depl)),
            raw_depl["HCW"],
            raw_depl["pycap_resource_name"],
            raw_depl["Fraction_Intercept"],
        )
    }

    welll_dict = dict()
    well_dict = {
        i: {
            "name": hcw,
            "status": well_stat.lower(),
            "loc": {
                "x": float(well_long),
                "y": float(well_lat),
            },
            "Q": float(Q_gpm),
            "pumping_days": int(pump_days),
        }
        for i, hcw, well_stat, well_long, well_lat, Q_gpm, pump_days in zip(
            range(len(raw_hcw)),
            raw_hcw["HCW"],
            raw_hcw["Well_Status"],
            raw_hcw["Well_Long"],
            raw_hcw["Well_Lat"],
            raw_hcw["Q_gpm"],
            raw_hcw["Pumping_Days"],
        )
    }

    stream_dict_df = pd.DataFrame.from_dict(stream_dict, orient="index")
    stream_dict_df = stream_dict_df.rename("stream_apportionment{}".format)

    well_dict = {i["name"]: i for _, i in well_dict.items()}

    if len(raw_depl["HCW"]) > 0:
    # bring in the stream apportionment values
        for j in well_dict:
            well_dict[j].update(
                stream_dict_df.loc[stream_dict_df["HCW"] == j]["stream_apportionment"]
            )

    # bring in the stream and drawdown response information
    for j in well_dict:
        well_dict[j].update(
            {
                "stream_response": (
                    list(raw_depl.loc[raw_depl["HCW"] == j]["pycap_resource_name"])
                )
            }
        )
        well_dict[j].update(
            {"dd_response": (list(raw_dd.loc[raw_dd["HCW"] == j]["Resource_Name"]))}
        )

    # rename the keys again
    well_dict = {f"well_{k}": v for k, v in well_dict.items()}

    streamresp_dict = {
        f"stream_response{i}": {
            "name": pycap_resource_name,
            "loc": {
                "x": float(res_long),
                "y": float(res_lat),
            },
        }
        for i, pycap_resource_name, res_long, res_lat in zip(
            range(len(raw_depl)),
            raw_depl["pycap_resource_name"],
            raw_depl["Resource_Long"],
            raw_depl["Resource_Lat"],
        )
    }

    raw_dd_unique = raw_dd.drop_duplicates(subset=["Resource_Name"]).reset_index()

    ddresp_dict = dict()
    ddresp_dict = {
        f"dd_response{i}": {
            "name": ResName,
            "loc": {
                "x": float(res_long),
                "y": float(res_lat),
            },
        }
        for i, ResName, res_long, res_lat in zip(
            range(len(raw_dd_unique)),
            raw_dd_unique["Resource_Name"],
            raw_dd_unique["Resource_Long"],
            raw_dd_unique["Resource_Lat"],
        )
    }

    combdict = {**project_dict, **well_dict, **ddresp_dict, **streamresp_dict}
    yml_name = pycap_run_name + ".yml"

    with open(pycap_run_path / yml_name, "w") as file:
        documents = yaml.dump(combdict, file, default_flow_style=False, sort_keys=False)
    return project_dict, stream_dict, well_dict, ddresp_dict, combdict