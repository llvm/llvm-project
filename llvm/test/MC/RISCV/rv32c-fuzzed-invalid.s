# REQUIRES: asserts
# RUN: not llvm-mc -triple riscv32 -mattr=+c,+f,+d < %s 2>&1 | FileCheck %s
#
# Fuzzed test cases produced by a LLVM MC Assembler
# Protocol Buffer Fuzzer for the RISC-V assembly language.
#

c.addi x13,f30,0    # CHECK: error: immediate must be non-zero in the range [-32, 31]
c.swsp x0,(f14)     # CHECK: error: immediate must be a multiple of 4 bytes in the range [0, 252]
c.lui x4,x0         # CHECK: error: immediate must be in [0xfffe0, 0xfffff] or [1, 31]
c.li x6,x6,x0,x0    # CHECK: error: immediate must be an integer in the range [-32, 31]
c.addi16sp 2,(x0)   # CHECK: error: invalid operand for instruction
c.fsdsp f9,x0,0     # CHECK: error: immediate must be a multiple of 8 bytes in the range [0, 504]
c.flw f15,x14,x0    # CHECK: error: immediate must be a multiple of 4 bytes in the range [0, 124]
c.fld f8,f30,x17    # CHECK: error: immediate must be a multiple of 8 bytes in the range [0, 248]
c.addi4spn x8,x2,x8 # CHECK: error: immediate must be a multiple of 4 bytes in the range [4, 1020]

