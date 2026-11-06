# RUN: llvm-mc --verify -triple riscv64 -mattr=+h < %s
# RUN: llvm-mc --verify=check,expected -triple riscv64 < %s

hfence.gvma zero, zero # check-error:1 {{instruction requires the following: 'H' (Hypervisor)}}

hfence.gvma zero # check-error:1 {{instruction requires the following: 'H' (Hypervisor)}}

hfence.gvma # check-error:1 {{instruction requires the following: 'H' (Hypervisor)}}

hfence.vvma zero, zero # check-error:1 {{instruction requires the following: 'H' (Hypervisor)}}

hfence.vvma zero # check-error:1 {{instruction requires the following: 'H' (Hypervisor)}}

hfence.vvma # check-error:1 {{instruction requires the following: 'H' (Hypervisor)}}

hlv.h   a0, 0(a1) # check-error:1 {{instruction requires the following: 'H' (Hypervisor)}}

hlv.wu   a0, 0(a1) # check-error:1 {{instruction requires the following: 'H' (Hypervisor)}}

hlv.b   a0, 100(a1) # expected-error:13 {{optional integer offset must be 0}}
