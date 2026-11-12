# RUN: not llvm-mc -triple=riscv64 --mattr=+zve64x --mattr=+zvksh %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

vsm3me.vv v10, v10, v8
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vsm3me.vv v10, v10, v8

vsm3c.vi v9, v9, 7
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vsm3c.vi v9, v9, 7
