# RUN: not llvm-mc -triple riscv32 -mattr=+zicond < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple riscv64 -mattr=+zicond < %s 2>&1 | FileCheck %s

# Use of operand modifier on register name
czero.eqz t1, %lo(t2), t3 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction

# Invalid register name
czero.nez a4, a3, foo # CHECK: :[[@LINE]]:19: error: invalid operand for instruction

# Invalid operand type
czero.eqz t1, 2, t3 # CHECK: :[[@LINE]]:15: error: invalid operand for instruction

# Too many operands
czero.eqz t1, t2, t3, t4 # CHECK: :[[@LINE]]:23: error: invalid operand for instruction
czero.nez t1, t2, t3, 4 # CHECK: :[[@LINE]]:23: error: invalid operand for instruction

# Too few operands
czero.eqz t1, t2 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
