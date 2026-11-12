# RUN: not llvm-mc -triple riscv64 -mattr=+h < %s 2>&1 \
# RUN:     | FileCheck --match-full-lines %s -check-prefixes=CHECK-OFFSET
# RUN: not llvm-mc -triple riscv64 < %s 2>&1 \
# RUN:     | FileCheck --match-full-lines %s -check-prefixes=CHECK,CHECK-OFFSET

hfence.gvma zero, zero # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: 'H' (Hypervisor)

hfence.gvma zero # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: 'H' (Hypervisor)

hfence.gvma # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: 'H' (Hypervisor)

hfence.vvma zero, zero # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: 'H' (Hypervisor)

hfence.vvma zero # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: 'H' (Hypervisor)

hfence.vvma # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: 'H' (Hypervisor)

hlv.h   a0, 0(a1) # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: 'H' (Hypervisor)

hlv.wu   a0, 0(a1) # CHECK: <stdin>:[[@LINE]]:1: error: instruction requires the following: 'H' (Hypervisor)

hlv.b   a0, 100(a1) # CHECK-OFFSET: <stdin>:[[@LINE]]:13: error: optional integer offset must be 0
