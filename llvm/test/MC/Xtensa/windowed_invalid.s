# RUN: not llvm-mc -triple xtensa --mattr=+windowed %s 2>&1 | FileCheck %s

# Out of range immediates

# entry_imm12
entry a3, -1
# CHECK: :[[#@LINE-1]]:11: error: expected immediate in range [0, 32760], first 3 bits should be zero

# entry_imm12
entry a3, 32764
# CHECK: :[[#@LINE-1]]:11: error: expected immediate in range [0, 32760], first 3 bits should be zero

# entry_imm12
entry a3, 4
# CHECK: :[[#@LINE-1]]:11: error: expected immediate in range [0, 32760], first 3 bits should be zero

# imm8n_7
rotw 100
# CHECK: :[[#@LINE-1]]:6: error: expected immediate in range [-8, 7]

# imm64n_4n
l32e a3, a4, -100
# CHECK: :[[#@LINE-1]]:14: error: expected immediate in range [-64, -4]
