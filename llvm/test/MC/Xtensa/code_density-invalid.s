# RUN: not llvm-mc -triple xtensa --mattr=+density %s 2>&1 | FileCheck %s

LBL0:

# Out of range immediates

# imm1n_15
addi.n a2, a3, 20
# CHECK: :[[#@LINE-1]]:16: error: expected immediate in range [-1, 15] except 0

# imm1n_15
addi.n a2, a3, 0
# CHECK: :[[#@LINE-1]]:16: error: expected immediate in range [-1, 15] except 0

# imm32n_95
movi.n a2, 100
# CHECK: :[[#@LINE-1]]:12: error: expected immediate in range [-32, 95]

# Offset4m32
l32i.n a2, a3, 100
# CHECK: :[[#@LINE-1]]:16: error: expected immediate in range [0, 60], first 2 bits should be zero
