# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-zcmop < %s 2>&1 | FileCheck %s

cmop.0 # CHECK: :[[@LINE]]:1: error: unrecognized instruction mnemonic

cmop.1 t0 # CHECK: :[[@LINE]]:8: error: invalid operand for instruction

cmop.1 0x0 # CHECK: :[[@LINE]]:8: error: invalid operand for instruction
