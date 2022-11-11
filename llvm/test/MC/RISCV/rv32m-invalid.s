# RUN: not llvm-mc -triple riscv32 -mattr=+m < %s 2>&1 | FileCheck %s

# RV64M instructions can't be used for RV32
mulw ra, sp, gp # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
divw tp, t0, t1 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
divuw t2, s0, s2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
remw a0, a1, a2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
remuw a3, a4, a5 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}

