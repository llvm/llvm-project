# RUN: not llvm-mc -triple=riscv64 --mattr=+experimental-zvzip %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

vzip.vv v0, v0, v4
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vzip.vv v0, v0, v4

vpaire.vv v0, v0, v4
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vpaire.vv v0, v0, v4

vpairo.vv v0, v0, v4
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vpairo.vv v0, v0, v4

vunzipe.v v0, v0
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vunzipe.v v0, v0

vunzipo.v v0, v0
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vunzipo.v v0, v0

vzip.vv v0, v2, v4, v0.t
# CHECK-ERROR: the destination vector register group cannot overlap the mask register
# CHECK-ERROR-LABEL: vzip.vv v0, v2, v4, v0.t

vpaire.vv v0, v2, v4, v0.t
# CHECK-ERROR: the destination vector register group cannot overlap the mask register
# CHECK-ERROR-LABEL: vpaire.vv v0, v2, v4, v0.t

vpairo.vv v0, v2, v4, v0.t
# CHECK-ERROR: the destination vector register group cannot overlap the mask register
# CHECK-ERROR-LABEL: vpairo.vv v0, v2, v4, v0.t
