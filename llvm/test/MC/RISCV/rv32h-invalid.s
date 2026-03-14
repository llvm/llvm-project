# RUN: not llvm-mc -triple riscv32 -mattr=+h < %s 2>&1 \
# RUN:     | FileCheck %s -check-prefixes=CHECK-OFFSET
# RUN: not llvm-mc -triple riscv32 < %s 2>&1 \
# RUN:     | FileCheck %s -check-prefixes=CHECK,CHECK-OFFSET

hfence.vvma zero, zero # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'H' (Hypervisor)

hlv.h   a0, 0(a1) # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'H' (Hypervisor)

hlv.wu   a0, 0(a1) # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'H' (Hypervisor), RV64I Base Instruction Set

hlv.b   a0, 100(a1) # CHECK-OFFSET: :[[@LINE]]:13: error: optional integer offset must be 0
