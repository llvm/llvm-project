# RUN: not llvm-mc %s -triple=riscv64 -riscv-no-aliases 2>&1 | FileCheck %s
# RUN: not llvm-mc %s -triple=riscv64 2>&1 | FileCheck %s

li t5, 0x10000000000000000 # CHECK: :[[@LINE]]:8: error: unknown operand
li t4, foo                 # CHECK: :[[@LINE]]:8: error: operand must be a constant 64-bit integer

rdinstreth x29 # CHECK: :[[@LINE]]:1: error: instruction use requires an option to be enabled
rdcycleh x27   # CHECK: :[[@LINE]]:1: error: instruction use requires an option to be enabled
rdtimeh x28    # CHECK: :[[@LINE]]:1: error: instruction use requires an option to be enabled

foo:
  .space 8
