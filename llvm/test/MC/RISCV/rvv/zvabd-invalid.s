# RUN: not llvm-mc -triple=riscv64 --mattr=+zve64x --mattr=+experimental-zvabd %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

vwabda.vv v9, v9, v8
# CHECK-ERROR: [[@LINE-1]]:11: error: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vwabda.vv v9, v9, v8

vwabdau.vv v9, v9, v8
# CHECK-ERROR: [[@LINE-1]]:12: error: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vwabdau.vv v9, v9, v8
