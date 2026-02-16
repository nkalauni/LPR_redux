import sys, os, yaml
import pyemu
from pycap.analysis_project import Project
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


def instantiate():
    # instantiate the project with paths and filenames
    datapath = Path('./')
    pestpath = Path('./')
    with open(datapath / 'LPR_Redux.yml','r') as ifp:
        initial_dict = yaml.safe_load(ifp)

    obsnames = [i.split("!")[1].lower() for i in open(pestpath/'allobs.out.ins', 'r').readlines()[1:]]
    bdplobs = [i.lower() for i in obsnames if i.endswith('bdpl')]


    """Instantiate the project with paths and filenames."""
    # only read out year 5 for time series ::: hard coded
    return initial_dict, bdplobs

def get_results(pars,  obsnames, initial_dict, bdplobs,  write_csv= False):
    
    # make sure parameter indices are cast to lower()
    pars.index = [i.lower() for i in pars.index]

    # parse and update parameter values
    ### note that, for OPT and MOU, this is only Q values
    qpars = pars.loc[pars.index.str.contains('_q')]

    # make an updated dict copy 
    upd_dict = initial_dict.copy()
   
    # set Q
    for idx,val in qpars.items():
        upd_dict[idx.split('__')[0]]['Q'] = val


    ap = Project(None, write_csv, upd_dict)
    ap.aggregate_results()
    ap.write_responses_csv()



    bdf = ap.agg_base_stream_df
    bdf.index=[f"lpr:{i}:bdpl" for i in bdf.index]
    bdf = bdf.loc[bdplobs]
    bdf['variable']=bdf.index
    bdf.rename(columns={'LPR':'value'}, inplace=True)

    return bdf.loc[obsnames, 'value']    




def process_realization(args):
    creal, row, obs_names, initial_dict, bdplobs,  par_names = args
    allout = get_results(row[par_names], obs_names, initial_dict, bdplobs)
    return creal, allout.loc[obs_names].to_numpy()

def standalone_worker(initial_dict, bdplobs, rootname='mou'):
    # Read RNS run data
    rns = pyemu.helpers.RunStor(f'./{rootname}.rns')
    runs_df = rns.get_data()
    _, par_names, obs_names = rns.file_info(f'./{rootname}.rns')

    # Prepare arguments for parallel processing
    args_list = [
        (creal, runs_df.loc[creal], obs_names, initial_dict, bdplobs, par_names)
        for creal in runs_df.index
    ]

    # Parallel execution
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_realization, args_list))

    # Populate results
    obsvals = np.zeros((len(runs_df), len(obs_names)))
    for i, (creal, obs_array) in enumerate(results):
        obsvals[i, :] = obs_array

    runs_df.loc[:, obs_names] = obsvals
    rns.update(runs_df)

if __name__== "__main__":
    initial_dict, bdplobs = instantiate()
    standalone_worker(initial_dict, bdplobs,rootname=sys.argv[1])




