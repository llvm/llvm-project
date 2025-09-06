# RUN: not llvm-mc -triple=riscv64 --mattr=+zve64x --mattr=+experimental-zvabd %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

vwabdacc.vv v9, v9, v8
# CHECK-ERROR: [[@LINE-1]]:13: error: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vwabdacc.vv v9, v9, v8

vwabdacc.vx v9, v9, a0
# CHECK-ERROR: [[@LINE-1]]:13: error: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vwabdacc.vx v9, v9, a0

vwabdaccu.vv v9, v9, v8
# CHECK-ERROR: [[@LINE-1]]:14: error: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vwabdaccu.vv v9, v9, v8

vwabdaccu.vx v9, v9, a0
# CHECK-ERROR: [[@LINE-1]]:14: error: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vwabdaccu.vx v9, v9, a0
