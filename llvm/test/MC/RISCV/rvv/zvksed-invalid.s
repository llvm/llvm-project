# RUN: not llvm-mc -triple=riscv64 --mattr=+zve64x --mattr=+zvksed %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

vsm4r.vs v10, v10
# CHECK-ERROR: the destination vector register group cannot overlap the source vector register group
# CHECK-ERROR-LABEL: vsm4r.vs v10, v10
