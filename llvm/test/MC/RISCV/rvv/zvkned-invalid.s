# RUN: not llvm-mc -triple=riscv32 -show-encoding --mattr=+zve32x --mattr=+experimental-zvkned %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

vaeskf1.vi v10, v9, 0
# CHECK-ERROR: immediate must be an integer in the range [1, 10]

vaeskf2.vi v10, v9, 0
# CHECK-ERROR: immediate must be an integer in the range [2, 14]
