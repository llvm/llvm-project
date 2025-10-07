# RUN: not llvm-mc -triple xtensa --mattr=+clamps %s 2>&1 | FileCheck %s

.align	4

# Out of range immediates

# imm7_22
clamps a3, a4, 5
# CHECK: :[[#@LINE-1]]:16: error: expected immediate in range [7, 22]
