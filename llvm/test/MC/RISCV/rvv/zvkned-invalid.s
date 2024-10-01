# RUN: not llvm-mc -triple=riscv64 --mattr=+zve64x --mattr=+zvkned %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

vaesdf.vs v10, v10
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vaesdf.vs v10, v10

vaesef.vs v11, v11
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vaesef.vs v11, v11

vaesdm.vs v12, v12
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vaesdm.vs v12, v12

vaesem.vs v13, v13
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vaesem.vs v13, v13

vaesz.vs v14, v14
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vaesz.vs v14, v14

