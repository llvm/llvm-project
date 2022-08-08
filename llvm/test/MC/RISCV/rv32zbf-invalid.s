# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-zbf < %s 2>&1 | FileCheck %s

# Too few operands
bfp t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
bfpw t0, t1, t2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
