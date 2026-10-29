# RUN: not llvm-mc %s -triple=xtensa  -mcpu=esp32s3  2>&1 | FileCheck %s

.align	4

LBL0:

# Out of range immediates

# Select_2
ee.fft.r2bf.s16 q7, q1, q3, q6, 6
# CHECK: :[[#@LINE-1]]:33: error: expected immediate in range [0, 1]

# Select_4
ee.fft.r2bf.s16.st.incp q7, q3, q7, a2, 4
# CHECK: :[[#@LINE-1]]:41: error: expected immediate in range [0, 3]

# Select_8
ee.ldxq.32 q2, q6, a11, 2, 9
# CHECK: :[[#@LINE-1]]:28: error: expected immediate in range [0, 7]

# Select_16
ee.slci.2q q7, q4, 17
# CHECK: :[[#@LINE-1]]:20: error: expected immediate in range [0, 15]

# Select_256
ee.set_bit_gpio_out 300
# CHECK: :[[#@LINE-1]]:21: error: expected immediate in range [0, 255]

# Offset_16_16
ee.ldf.128.ip f3, f5, f8, f0, a13, 120
# CHECK: :[[#@LINE-1]]:36: error: expected immediate in range [-128, 112], first 4 bits should be zero

# Offset_256_8
ee.ldf.64.ip f6, f5, a1, 2000
# CHECK: :[[#@LINE-1]]:26: error: expected immediate in range [-1024, 1016], first 3 bits should be zero

# Offset_256_16
ee.ldqa.s16.128.ip a11, 3000
# CHECK: :[[#@LINE-1]]:25: error: expected immediate in range [-2048, 2032], first 4 bits should be zero

# Offset_256_4
ee.ld.qacc_h.h.32.ip a6, -600
# CHECK: :[[#@LINE-1]]:26: error: expected immediate in range [-512, 508], first 2 bits should be zero

# Offset_128_2
ee.vldbc.16.ip q6, a4, 300
# CHECK: :[[#@LINE-1]]:24: error: expected immediate in range [0, 254], first bit should be zero

# Offset_128_1
ee.vldbc.8.ip q3, a3, 140
# CHECK: :[[#@LINE-1]]:23: error: expected immediate in range [0, 127]

# Offset_64_16
ee.vmulas.s16.accx.ld.ip.qup q5, a14, 600, q0, q2, q0, q2
# CHECK: :[[#@LINE-1]]:39: error: expected immediate in range [-512, 496], first 4 bits should be zero
