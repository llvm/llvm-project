# RUN: not llvm-mc -triple riscv64 < %s 2>&1 | FileCheck %s

# Out of range immediates
## uimm5
slliw a0, a0, 32 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 31]
srliw a0, a0, -1 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 31]
sraiw a0, a0, -19 # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 31]

## uimm6
slli a0, a0, 64 # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 63]
srli a0, a0, -1 # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 63]
srai a0, a0, -19 # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 63]

## simm12
addiw a0, a1, -2049 # CHECK: :[[@LINE]]:15: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]
ld ra, 2048(sp) # CHECK: :[[@LINE]]:8: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]

# Illegal operand modifier
## uimm5
slliw a0, a0, %lo(1) # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 31]
srliw a0, a0, %lo(a) # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 31]
sraiw a0, a0, %hi(2) # CHECK: :[[@LINE]]:15: error: immediate must be an integer in the range [0, 31]

## uimm6
slli a0, a0, %lo(1) # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 63]
srli a0, a0, %lo(a) # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 63]
srai a0, a0, %hi(2) # CHECK: :[[@LINE]]:14: error: immediate must be an integer in the range [0, 63]

## simm12
addiw a0, a1, %hi(foo) # CHECK: :[[@LINE]]:15: error: operand must be a symbol with %lo/%pcrel_lo/%tprel_lo modifier or an integer in the range [-2048, 2047]
