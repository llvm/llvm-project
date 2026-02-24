# RUN: not llvm-mc -triple=riscv64 --mattr=+experimental-zvdot4a8i %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

vdota4.vv v0, v2, v4, v0.t
# CHECK-ERROR: the destination vector register group cannot overlap the mask register
# CHECK-ERROR-LABEL: vdota4.vv v0, v2, v4, v0.t

vdota4.vx v0, v2, a0, v0.t
# CHECK-ERROR: the destination vector register group cannot overlap the mask register
# CHECK-ERROR-LABEL: vdota4.vx v0, v2, a0, v0.t
