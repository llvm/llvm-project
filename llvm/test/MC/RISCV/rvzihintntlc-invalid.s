# RUN: not llvm-mc -triple riscv32 -mattr=+zihintntl,+c < %s 2>&1 | FileCheck --match-full-lines %s
# RUN: not llvm-mc -triple riscv64 -mattr=+zihintntl,+c < %s 2>&1 | FileCheck --match-full-lines %s

c.ntl.p1 1 # CHECK: <stdin>:[[@LINE]]:10: error: unexpected extra operand for instruction
c.ntl.pall 2 # CHECK: <stdin>:[[@LINE]]:12: error: unexpected extra operand for instruction
c.ntl.s1 3 # CHECK: <stdin>:[[@LINE]]:10: error: unexpected extra operand for instruction
c.ntl.all 4 # CHECK: <stdin>:[[@LINE]]:11: error: unexpected extra operand for instruction

c.ntl.p1 t0, t1 # CHECK: <stdin>:[[@LINE]]:10: error: unexpected extra operand for instruction
c.ntl.pall t0, t1 # CHECK: <stdin>:[[@LINE]]:12: error: unexpected extra operand for instruction
c.ntl.s1 t0, t1 # CHECK: <stdin>:[[@LINE]]:10: error: unexpected extra operand for instruction
c.ntl.all t0, t1 # CHECK: <stdin>:[[@LINE]]:11: error: unexpected extra operand for instruction

