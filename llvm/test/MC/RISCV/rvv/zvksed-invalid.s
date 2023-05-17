# RUN: not llvm-mc -triple=riscv32 --mattr=+zve32x --mattr=+experimental-zvksed -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

vsm4k.vi v10, v9, 8
# CHECK-ERROR: immediate must be an integer in the range [0, 7]
