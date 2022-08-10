# UNSUPPORTED: windows
# RUN: not llvm-mc -triple=riscv64 -riscv-no-aliases < %s -o /dev/null 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple=riscv64 < %s 2>&1 -o /dev/null | FileCheck %s

li t5, 0x10000000000000000 # CHECK: :[[@LINE]]:8: error: unknown operand
li t4, foo                 # CHECK: :[[@LINE]]:8: error: operand must be a constant 64-bit integer

rdinstreth x29 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set{{$}}
rdcycleh x27   # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set{{$}}
rdtimeh x28    # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV32I Base Instruction Set{{$}}

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
