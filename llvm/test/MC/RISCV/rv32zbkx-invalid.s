# RUN: not llvm-mc -triple riscv32 -mattr=+zbkx < %s 2>&1 | FileCheck %s

# Too few operands
xperm8 t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
xperm4 t0, t1 # CHECK: :[[@LINE]]:1: error: too few operands for instruction

# Undefined Zbp instruction in Zbkx
xperm.h t0, t1, t2 # CHECK: :[[@LINE]]:1: error: instruction requires the following: 'Zbp' (Permutation 'Zb' Instructions){{$}}
