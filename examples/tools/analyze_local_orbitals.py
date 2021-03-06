#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
import re
from pyscf import lib
from pyscf import tools
from pyscf.tools import mo_mapping

'''
Read localized orbitals from molden, then find out C 2py and 2pz orbitals
'''


mol, mo_energy, mo_coeff, mo_occ, irrep_labels, spins = \
        tools.molden.load('benzene-631g-boys.molden')

#
# Note mol._cart_gto is not a built-in attribute defined in Mole class.  It is
# tagged in tools.molden.load function to indicate whether the orbitals read
# from molden file are based on cartesian gaussians (6d 10f).
#
if mol._cart_gto:
    # If molden file does not have 5d,9g label, it's in Cartesian Gaussian
    label = mol.cart_labels(True)
    comp = mo_mapping.mo_comps(lambda x: re.search('C.*2p[yz]', x),
                               mol, mo_coeff, cart=True)
else:
    label = mol.spheric_labels(True)
    comp = mo_mapping.mo_comps(lambda x: re.search('C.*2p[yz]', x),
                               mol, mo_coeff)
#
# For orbitals generated by pyscf (or read from pyscf chkfile), the orbitals
# are all represented on spheric gaussians (5d 7f).
#
mol = lib.chkfile.load_mol('benzene-631g.chk')
mo = lib.chkfile.load('benzene-631g.chk', 'scf/mo_coeff')
label = mol.spheric_labels(True)
idx = [i for i,s in enumerate(labels) if 'C 2p' in s]
comp = mo_mapping.mo_comps(idx, mol, mo_coeff)

#tools.dump_mat.dump_rec(mol.stdout, mo_coeff, label, start=1)

print('rank   MO-id    components')
for i,j in enumerate(numpy.argsort(-comp)):
    print('%3d    %3d      %.10f' % (i, j, comp[j]))
