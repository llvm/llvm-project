# RUN: not llvm-mc -triple=riscv64 -mattr=+experimental-zvfwbdota16bf %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-ERROR

# Invalid vsetvli
# CHECK-ERROR: invalid instruction{{$}}
vsetvli a0, zero, e888, m1, ta, ma

# Invalid ci
# CHECK-ERROR: immediate must be an integer in the range [0, 7]{{$}}
vfwbdota.vv v8, v16, v12, 8

# Invalid vs2
# CHECK-ERROR: invalid operand for instruction{{$}}
vfwbdota.vv v8, v17, v12, 1

# Invalid vs2 and ci
# CHECK-ERROR: :[[@LINE+1]]:1: error: invalid instruction{{$}}
vfwbdota.vv v8, v17, v12, 8
