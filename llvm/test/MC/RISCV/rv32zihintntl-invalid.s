# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-zihintntl < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple riscv64 -mattr=+experimental-zihintntl < %s 2>&1 | FileCheck %s

ntl.p1 1 # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
ntl.pall 2 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
ntl.s1 3 # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
ntl.all 4 # CHECK: :[[@LINE]]:9: error: invalid operand for instruction

ntl.p1 t0, t1 # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
ntl.pall t0, t1 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
ntl.s1 t0, t1 # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
ntl.all t0, t1 # CHECK: :[[@LINE]]:9: error: invalid operand for instruction

