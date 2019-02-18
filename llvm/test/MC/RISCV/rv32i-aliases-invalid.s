# RUN: not llvm-mc %s -triple=riscv32 -riscv-no-aliases 2>&1 | FileCheck %s
# RUN: not llvm-mc %s -triple=riscv32 2>&1 | FileCheck %s

# TODO ld
# TODO sd

li x0, 4294967296   # CHECK: :[[@LINE]]:8: error: immediate must be an integer in the range [-2147483648, 4294967295]
li x0, -2147483649  # CHECK: :[[@LINE]]:8: error: immediate must be an integer in the range [-2147483648, 4294967295]
li t4, foo          # CHECK: :[[@LINE]]:8: error: immediate must be an integer in the range [-2147483648, 4294967295]

negw x1, x2   # CHECK: :[[@LINE]]:1: error: instruction use requires an option to be enabled
sext.w x3, x4 # CHECK: :[[@LINE]]:1: error: instruction use requires an option to be enabled

sll x2, x3, 32  # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 31]
srl x2, x3, 32  # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 31]
sra x2, x3, 32  # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 31]

sll x2, x3, -1  # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 31]
srl x2, x3, -2  # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 31]
sra x2, x3, -3  # CHECK: :[[@LINE]]:13: error: immediate must be an integer in the range [0, 31]

foo:
  .space 4
