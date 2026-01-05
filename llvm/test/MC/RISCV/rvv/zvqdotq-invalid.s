# RUN: not llvm-mc -triple=riscv64 --mattr=+experimental-zvqdotq %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

vqdot.vv v0, v2, v4, v0.t
# CHECK-ERROR: the destination vector register group cannot overlap the mask register
# CHECK-ERROR-LABEL: vqdot.vv v0, v2, v4, v0.t

vqdot.vx v0, v2, a0, v0.t
# CHECK-ERROR: the destination vector register group cannot overlap the mask register
# CHECK-ERROR-LABEL: vqdot.vx v0, v2, a0, v0.t
