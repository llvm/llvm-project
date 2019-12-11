# UNSUPPORTED: windows
# RUN: not llvm-mc -triple=riscv32 -riscv-no-aliases < %s -o /dev/null 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple=riscv32 < %s -o /dev/null 2>&1 | FileCheck %s

# TODO ld
# TODO sd

li x0, 4294967296   # CHECK: :[[@LINE]]:8: error: immediate must be an integer in the range [-2147483648, 4294967295]
li x0, -2147483649  # CHECK: :[[@LINE]]:8: error: immediate must be an integer in the range [-2147483648, 4294967295]
li t4, foo          # CHECK: :[[@LINE]]:8: error: immediate must be an integer in the range [-2147483648, 4294967295]

negw x1, x2   # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
sext.w x3, x4 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set

sll x2, x3, 32  # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 31]
srl x2, x3, 32  # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 31]
sra x2, x3, 32  # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 31]

sll x2, x3, -1  # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 31]
srl x2, x3, -2  # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 31]
sra x2, x3, -3  # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 31]

addi x1, .      # CHECK: :[[@LINE]]:10: error: invalid operand for instruction

foo:
  .space 4
