# RUN: not llvm-mc %s -triple=riscv64 -riscv-no-aliases 2>&1 | FileCheck %s
# RUN: not llvm-mc %s -triple=riscv64 2>&1 | FileCheck %s

li t5, 0x10000000000000000 # CHECK: :[[@LINE]]:8: error: unknown operand
li t4, foo                 # CHECK: :[[@LINE]]:8: error: operand must be a constant 64-bit integer

rdinstreth x29 # CHECK: :[[@LINE]]:1: error: instruction use requires an option to be enabled
rdcycleh x27   # CHECK: :[[@LINE]]:1: error: instruction use requires an option to be enabled
rdtimeh x28    # CHECK: :[[@LINE]]:1: error: instruction use requires an option to be enabled

sll x2, x3, 64  # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 63]
srl x2, x3, 64  # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 63]
sra x2, x3, 64  # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 63]

sll x2, x3, -1  # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 63]
srl x2, x3, -2  # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 63]
sra x2, x3, -3  # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 63]

sllw x2, x3, 32  # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 31]
srlw x2, x3, 32  # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 31]
sraw x2, x3, 32  # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 31]

sllw x2, x3, -1  # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 31]
srlw x2, x3, -2  # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 31]
sraw x2, x3, -3  # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 31]

foo:
  .space 8
