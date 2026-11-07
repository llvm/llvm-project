# RUN: llvm-mc --verify -triple=riscv64 -mattr=+experimental-zvfwbdota16bf %s

# Invalid vsetvli
# expected-error@+1 {{operand must be e[8|8alt|16|16alt|32|64],m[1|2|4|8|f2|f4|f8],[ta|tu],[ma|mu]}}
vsetvli a0, zero, e888, m1, ta, ma

# Invalid ci
# expected-error@+1 {{immediate must be a multiple of 8 in the range [0, 56]}}
vfwbdota.vv v8, v16, v12, 4

# expected-error@+1 {{immediate must be a multiple of 8 in the range [0, 56]}}
vfwbdota.vv v8, v16, v12, 64

# Invalid vs2
# expected-error@+1 {{invalid operand for instruction}}
vfwbdota.vv v8, v17, v12, 8

# Invalid vs2 and ci
# expected-error@+1:17 {{invalid operand for instruction}}
vfwbdota.vv v8, v17, v12, 4
