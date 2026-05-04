# RUN: not llvm-mc -triple xtensa --mattr=+sext %s 2>&1 | FileCheck %s

.align	4

# Out of range immediates

# imm7_22
sext a3, a4, 6
# CHECK: :[[#@LINE-1]]:14: error: expected immediate in range [7, 22]
