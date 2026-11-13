# RUN: llvm-mc --verify -triple=riscv64 -mattr=+experimental-zvfbdota32f %s

# Invalid vsetvli
vsetvli a0, zero, e888, m1, ta, ma # expected-error {{operand must be e[8|8alt|16|16alt|32|64],m[1|2|4|8|f2|f4|f8],[ta|tu],[ma|mu]}}

# Invalid ci
vfbdota.vv v8, v16, v12, 4 # expected-error {{immediate must be a multiple of 8 in the range [0, 56]}}

vfbdota.vv v8, v16, v12, 64 # expected-error {{immediate must be a multiple of 8 in the range [0, 56]}}

# Invalid vs2
vfbdota.vv v8, v17, v12, 8 # expected-error {{invalid operand for instruction}}

# Invalid vs2 and ci
vfbdota.vv v8, v17, v12, 4 # expected-error:16 {{invalid operand for instruction}}
