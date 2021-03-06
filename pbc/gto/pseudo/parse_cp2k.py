#!/usr/bin/env python
# -*- coding: utf-8
# Author: Timothy Berkelbach <tim.berkelbach@gmail.com>
#
# parse CP2K PP format, following parse_nwchem.py

import sys
import numpy as np

def parse(string):
    '''Parse the pseudo text *string* which is in CP2K format, return an internal
    basis format which can be assigned to :attr:`Cell.pseudo`
    Lines started with # are ignored.
    '''
    pseudotxt = [x.strip() for x in string.split('\n')
                 if x.strip() and 'END' not in x and '#PSEUDOPOTENTIAL' not in x]
    return _parse(pseudotxt)

def load(pseudofile, symb):
    '''Parse the *pseudofile's entry* for atom 'symb', return an internal
    pseudo format which can be assigned to :attr:`Cell.pseudo`
    '''
    return _parse(search_seg(pseudofile, symb))

def _parse(plines):
    header_ln = plines.pop(0)
    nelecs = [ int(nelec) for nelec in plines.pop(0).split() ]
    rnc_ppl = plines.pop(0).split()
    rloc = float(rnc_ppl[0])
    nexp = int(rnc_ppl[1])
    cexp = [ float(c) for c in rnc_ppl[2:] ]
    nproj_types = int(plines.pop(0))
    r = []
    nproj = []
    hproj = []
    for p in range(nproj_types):
        rnh_ppnl = plines.pop(0).split()
        r.append(float(rnh_ppnl[0]))
        nproj.append(int(rnh_ppnl[1]))
        hproj_p_ij = []
        for h in rnh_ppnl[2:]:
            hproj_p_ij.append(float(h))
        for i in range(1,nproj[-1]):
            for h in plines.pop(0).split():
                hproj_p_ij.append(float(h))
        hproj_p = np.zeros((nproj[-1],nproj[-1]))
        hproj_p[np.triu_indices(nproj[-1])] = [ h for h in hproj_p_ij ]
        hproj_p_symm = hproj_p + hproj_p.T - np.diag(hproj_p.diagonal())
        hproj.append(hproj_p_symm.tolist())

    pseudo_params = [nelecs,
                     rloc, nexp, cexp,
                     nproj_types]
    for ri,ni,hi in zip(r,nproj,hproj):
        pseudo_params.append([ri, ni, hi])
    return pseudo_params

def search_seg(pseudofile, symb):
    '''
    Find the pseudopotential entry for atom 'symb' in file 'pseudofile'
    '''
    fin = open(pseudofile, 'r')
    fdata = fin.read().split('#PSEUDOPOTENTIAL')
    fin.close()
    for dat in fdata[1:]:
        if symb+' ' in dat:
            # remove blank lines
            return [x.strip() for x in dat.split('\n')[1:]
                    if x.strip() and 'END' not in x]
    raise RuntimeError('Pseudopotential not found for  %s  in  %s' % (symb, pseudofile))

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 2:
        ppfile = args[0]
        atom = args[1]
    else:
        print('usage: ppfile atomlabel ')
        sys.exit(1)

    print("Testing search_seg():")
    print(search_seg(ppfile,atom))

    print("Testing load() [[from file]]:")
    load(ppfile,atom)

    print("Testing parse():")
    parse("""
    #PSEUDOPOTENTIAL
    C GTH-BLYP-q4
        2    2
         0.33806609    2    -9.13626871     1.42925956
        2
         0.30232223    1     9.66551228
         0.28637912    0
    """
    )
