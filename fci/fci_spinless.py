#!/usr/bin/env python
# Spinless Fermions
# Author: Ushnish Ray
#

import numpy
import pyscf.lib
from pyscf.fci import cistring
import ctypes
libfci = pyscf.lib.load_library('libfci')


def contract_1e(f1e, fcivec, norb, nelec):
    
    link_indexa = cistring.gen_linkstr_index_o0(range(norb), nelec)
    na = cistring.num_strings(norb, nelec)
    
    t1 = numpy.zeros((norb,norb,na),dtype=f1e.dtype)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            t1[a,i,str1] += sign * fcivec[str0]
    
    fcinew = numpy.dot(f1e.reshape(-1), t1.reshape(-1,na))
    return fcinew.reshape(fcivec.shape)


def _unpack(norb, nelec, link_index):
    if link_index is None:
        link_indexa = cistring.gen_linkstr_index(range(norb), nelec)
        link_indexb = cistring.gen_linkstr_index(range(norb), 0)
        return link_indexa, link_indexb
    else:
        return link_index

def contract_2e(eri, fcivec, norb, nelec, link_index = None, opt=None):
    
    fcivec = numpy.asarray(fcivec, order='C')
    link_indexa, link_indexb = _unpack(norb, nelec, link_index)

    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]

    if(numpy.isrealobj(eri)):
        citemp = numpy.empty_like(fcivec,dtype=numpy.float64)
        libfci.FCIcontract_2es1(eri.ctypes.data_as(ctypes.c_void_p),
                            fcivec.ctypes.data_as(ctypes.c_void_p),
                            citemp.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(norb),
                            ctypes.c_int(na), ctypes.c_int(nb),
                            ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                            link_indexa.ctypes.data_as(ctypes.c_void_p),
                            link_indexb.ctypes.data_as(ctypes.c_void_p))
        return citemp
    else:
        
        eri_r = eri.real()
        eri_i = eri.imag()
        fcivecr = fcivec.real()
        fciveci = fcivec.imag()

        citemp = numpy.empty_like(fcivec,dtype=numpy.float64)
        libfci.FCIcontract_2es1(eri_r.ctypes.data_as(ctypes.c_void_p),
                            fcivecr.ctypes.data_as(ctypes.c_void_p),
                            citemp.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(norb),
                            ctypes.c_int(na), ctypes.c_int(nb),
                            ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                            link_indexa.ctypes.data_as(ctypes.c_void_p),
                            link_indexb.ctypes.data_as(ctypes.c_void_p))
        fcivec = citemp

        citemp = numpy.empty_like(fcivec,dtype=numpy.float64)
        libfci.FCIcontract_2es1(eri_i.ctypes.data_as(ctypes.c_void_p),
                            fciveci.ctypes.data_as(ctypes.c_void_p),
                            citemp.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(norb),
                            ctypes.c_int(na), ctypes.c_int(nb),
                            ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                            link_indexa.ctypes.data_as(ctypes.c_void_p),
                            link_indexb.ctypes.data_as(ctypes.c_void_p))
        fcivec -= citemp        
        
        citemp = numpy.empty_like(fcivec,dtype=numpy.float64)
        libfci.FCIcontract_2es1(eri_r.ctypes.data_as(ctypes.c_void_p),
                            fciveci.ctypes.data_as(ctypes.c_void_p),
                            citemp.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(norb),
                            ctypes.c_int(na), ctypes.c_int(nb),
                            ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                            link_indexa.ctypes.data_as(ctypes.c_void_p),
                            link_indexb.ctypes.data_as(ctypes.c_void_p))
        fcivec += 1j*citemp 

        citemp = numpy.empty_like(fcivec,dtype=numpy.float64)
        libfci.FCIcontract_2es1(eri_i.ctypes.data_as(ctypes.c_void_p),
                            fcivecr.ctypes.data_as(ctypes.c_void_p),
                            citemp.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(norb),
                            ctypes.c_int(na), ctypes.c_int(nb),
                            ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                            link_indexa.ctypes.data_as(ctypes.c_void_p),
                            link_indexb.ctypes.data_as(ctypes.c_void_p))
        fcivec += 1j*citemp 
        return fcivec

def absorb_h1e(h1e, g2e, norb, nelec, fac=1):
    '''Modify 2e Hamiltonian to include 1e Hamiltonian contribution.
    '''
    if(h1e.dtype == numpy.complex128):
        h2e = g2e.copy().astype(numpy.complex128) 
    else:
        h2e = g2e.copy() 

#   Expect g2e in full form
#   h2e = pyscf.ao2mo.restore(1, eri, norb)
    f1e = h1e - numpy.einsum('jiik->jk', g2e) * .5
    f1e = f1e * (1./(nelec+1e-100))
#   f1e = f1e.astype(numpy.complex128)
	
    for k in range(norb):
        h2e[k,k,:,:] += f1e
        h2e[:,:,k,k] += f1e
    return h2e * fac

def make_hdiag(h1e, g2e, norb, nelec, opt=None):
    
    link_indexa = cistring.gen_linkstr_index_o0(range(norb), nelec)
    occslista = [tab[:nelec,0] for tab in link_indexa]
   #g2e = pyscf.ao2mo.restore(1, g2e, norb)
    diagj = numpy.einsum('iijj->ij',g2e)
    diagk = numpy.einsum('ijji->ij',g2e)

    hdiag = []
    for aocc in occslista:
            e1 = h1e[aocc,aocc].sum() 
            e2 = diagj[aocc][:,aocc].sum() - diagk[aocc][:,aocc].sum() 
            hdiag.append(e1 + e2*.5)

    return numpy.array(hdiag)

def kernel(h1e, g2e, norb, nelec):

    na = cistring.num_strings(norb, nelec)   
    h2e = absorb_h1e(h1e, g2e, norb, nelec, .5)    

    def hop(c):
	hc = contract_2e(h2e, c, norb, nelec, link_indexa, na)
	return hc.reshape(-1)
    hdiag = make_hdiag(h1e, g2e, norb, nelec)
    precond = lambda x, e, *args: x/(hdiag-e+1e-4)
    
    ci0 = numpy.random.random(na)
    ci0 /= numpy.linalg.norm(ci0)

    #with PyCallGraph(output=GraphvizOutput()):
    e, c = pyscf.lib.davidson(hop, ci0, precond)
    #e, c = pyscf.lib.davidson(hop, ci0, precond, max_space=100)

    return e, c


# dm_pq = <|p^+ q|>
def make_rdm1(fcivec, norb, nelec, opt=None):
    link_index = cistring.gen_linkstr_index(range(norb), nelec)
    na = cistring.num_strings(norb, nelec)
    #fcivec = fcivec.reshape(na,na)
    rdm1 = numpy.zeros((norb,norb),dtype=fcivec.dtype)
    
    for str0, tab in enumerate(link_index):
        for a, i, str1, sign in link_index[str0]:
            rdm1[a,i] += sign * numpy.dot(fcivec[str1].conj(),fcivec[str0])
    for str0, tab in enumerate(link_index):
        for a, i, str1, sign in link_index[str0]:
            rdm1[a,i] += sign * numpy.dot(fcivec[:,str1].conj(),fcivec[:,str0])
    return rdm1

# dm_pq,rs = <|p^+ q r^+ s|>
def make_rdm12(fcivec, norb, nelec, opt=None):
    link_index = cistring.gen_linkstr_index(range(norb), nelec)
    na = cistring.num_strings(norb, nelec)

    rdm1 = numpy.zeros((norb,norb),dtype=fcivec.dtype)
    rdm2 = numpy.zeros((norb,norb,norb,norb), dtype=fcivec.dtype)
    t1 = numpy.zeros((na,norb,norb),dtype=fcivec.dtype)

    for str0, tab in enumerate(link_index):     
        for a, i, str1, sign in link_index[str0]:
            t1[str1,i,a] += sign * fcivec[str0]

    rdm1 += numpy.einsum('m,mij->ij', fcivec.conj(), t1)
    #i^+ j|0> => <0|j^+ i, so swap i and j
    rdm2 += numpy.einsum('mij,mkl->jikl', t1.conj(), t1)
    
    return reorder_rdm(rdm1, rdm2)
    
def reorder_rdm(rdm1, rdm2):
    '''reorder from rdm2(pq,rs) = <E^p_q E^r_s> to rdm2(pq,rs) = <e^{pr}_{qs}>.
    Although the "reoredered rdm2" is still in Mulliken order (rdm2[e1,e1,e2,e2]),
    it is the right 2e DM (dotting it with int2e gives the energy of 2e parts)
    '''
    nmo = rdm1.shape[0]
    #if inplace:
    rdm2 = rdm2.reshape(nmo,nmo,nmo,nmo)
    #else:
    #    rdm2 = rdm2.copy().reshape(nmo,nmo,nmo,nmo)
    for k in range(nmo):
        rdm2[:,k,k,:] -= rdm1
    return rdm1, rdm2


if __name__ == '__main__':
    from functools import reduce
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo

    h1e = numpy.zeros([4,4]) 
    for i in range(0,4):
	h1e[i,i] = 0.01
 	h1e[i,(i+1)%4] = -1.0
 	h1e[i,(i-1)%4] = -1.0

    eri = numpy.zeros([4,4,4,4])
    for i in range(0,4):
 	j = (i+1)%4
        eri[i,j,i,j] = 0.0

    e1,c = kernel(h1e, eri, 4, 2)
   
    print "xxxxxxxxxxxxxxxxxxx" 
    print e1
    print c


