# RUN: not llvm-mc -triple riscv32 -mattr=+zba < %s 2>&1 | FileCheck %s

# Too few operands
sh1add t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
sh2add t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
sh3add t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
slli.uw t0, t1, 0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
add.uw t0, t1, t2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
sh1add.uw t0, t1, t2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
sh2add.uw t0, t1, t2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
sh3add.uw t0, t1, t2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
