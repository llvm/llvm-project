# RUN: not llvm-mc -triple riscv32 -mattr=+zihintntl,+c < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple riscv64 -mattr=+zihintntl,+c < %s 2>&1 | FileCheck %s

c.ntl.p1 1 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
c.ntl.pall 2 # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
c.ntl.s1 3 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
c.ntl.all 4 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction

c.ntl.p1 t0, t1 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
c.ntl.pall t0, t1 # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
c.ntl.s1 t0, t1 # CHECK: :[[@LINE]]:10: error: invalid operand for instruction
c.ntl.all t0, t1 # CHECK: :[[@LINE]]:11: error: invalid operand for instruction

