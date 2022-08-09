# RUN: not llvm-mc -triple riscv32 -mattr=+zbb < %s 2>&1 | FileCheck %s

# Too many operands
clz t0, t1, t2 # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
# Too many operands
ctz t0, t1, t2 # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
# Too many operands
cpop t0, t1, t2 # CHECK: :[[@LINE]]:14: error: invalid operand for instruction
# Too many operands
sext.b t0, t1, t2 # CHECK: :[[@LINE]]:16: error: invalid operand for instruction
# Too many operands
sext.h t0, t1, t2 # CHECK: :[[@LINE]]:16: error: invalid operand for instruction
# Too few operands
min t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
max t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
minu t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
maxu t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
clzw t0, t1 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
ctzw t0, t1 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
cpopw t0, t1 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
