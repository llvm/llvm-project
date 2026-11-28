# RUN: llvm-mc --verify -triple riscv64 < %s

# Out of range immediates
## uimm5
slliw a0, a0, 32 # expected-error:15 {{immediate must be an integer in the range [0, 31]}}
srliw a0, a0, -1 # expected-error:15 {{immediate must be an integer in the range [0, 31]}}
sraiw a0, a0, -19 # expected-error:15 {{immediate must be an integer in the range [0, 31]}}

## uimm6
slli a0, a0, 64 # expected-error:14 {{immediate must be an integer in the range [0, 63]}}
srli a0, a0, -1 # expected-error:14 {{immediate must be an integer in the range [0, 63]}}
srai a0, a0, -19 # expected-error:14 {{immediate must be an integer in the range [0, 63]}}

## simm12
addiw a0, a1, -2049 # expected-error:15 {{operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]}}
ld ra, 2048(sp) # expected-error:8 {{operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]}}

# Illegal operand modifier
## uimm5
slliw a0, a0, %lo(1) # expected-error:15 {{immediate must be an integer in the range [0, 31]}}
srliw a0, a0, %lo(a) # expected-error:15 {{immediate must be an integer in the range [0, 31]}}
sraiw a0, a0, %hi(2) # expected-error:15 {{immediate must be an integer in the range [0, 31]}}

## uimm6
slli a0, a0, %lo(1) # expected-error:14 {{immediate must be an integer in the range [0, 63]}}
srli a0, a0, %lo(a) # expected-error:14 {{immediate must be an integer in the range [0, 63]}}
srai a0, a0, %hi(2) # expected-error:14 {{immediate must be an integer in the range [0, 63]}}

## simm12
addiw a0, a1, %hi(foo) # expected-error:15 {{operand must be a symbol with %lo/%pcrel_lo/%tprel_lo specifier or an integer in the range [-2048, 2047]}}
