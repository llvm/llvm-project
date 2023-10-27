# RUN: not llvm-mc -triple=riscv64 --mattr=+zve64x --mattr=+experimental-zvbb %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

vwsll.vv v2, v2, v4
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwsll.vv v2, v2, v4

vwsll.vx v2, v2, x10
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwsll.vx v2, v2, x10

vwsll.vi v2, v2, 1
# CHECK-ERROR: The destination vector register group cannot overlap the source vector register group.
# CHECK-ERROR-LABEL: vwsll.vi v2, v2, 1
