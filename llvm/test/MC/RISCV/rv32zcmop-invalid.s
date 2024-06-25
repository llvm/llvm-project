# RUN: not llvm-mc -triple riscv32 -mattr=+zcmop < %s 2>&1 | FileCheck %s

c.mop.0 # CHECK: :[[@LINE]]:1: error: unrecognized instruction mnemonic

c.mop.1 t0 # CHECK: :[[@LINE]]:9: error: invalid operand for instruction

c.mop.1 0x0 # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
