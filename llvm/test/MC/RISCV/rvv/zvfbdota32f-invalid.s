# RUN: not llvm-mc -triple=riscv64 -mattr=+experimental-zvfbdota32f %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-ERROR

# Invalid vsetvli
# CHECK-ERROR: operand must be e[8|16|32|64],m[1|2|4|8|f2|f4|f8],[ta|tu],[ma|mu]
vsetvli a0, zero, e888, m1, ta, ma

# Invalid ci
# CHECK-ERROR: immediate must be an integer in the range [0, 7]{{$}}
vfbdota.vv v8, v16, v12, 8

# Invalid vs2
# CHECK-ERROR: invalid operand for instruction{{$}}
vfbdota.vv v8, v17, v12, 1

# Invalid vs2 and ci
# CHECK-ERROR: :[[@LINE+1]]:16: error: invalid operand for instruction{{$}}
vfbdota.vv v8, v17, v12, 8
