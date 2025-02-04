# RUN: not llvm-mc -triple=riscv64 --mattr=+zve64x --mattr=+zvknha %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

vsha2ms.vv v10, v10, v11
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vsha2ms.vv v10, v10, v11

vsha2ms.vv v11, v10, v11
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vsha2ms.vv v11, v10, v11

vsha2ch.vv v12, v12, v11
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vsha2ch.vv v12, v12, v11

vsha2ch.vv v11, v12, v11
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vsha2ch.vv v11, v12, v11

vsha2cl.vv v13, v13, v15
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vsha2cl.vv v13, v13, v15

vsha2cl.vv v15, v13, v15
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vsha2cl.vv v15, v13, v15
