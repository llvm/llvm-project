# RUN: llvm-mc -triple i386 -show-encoding %s | FileCheck %s

# CHECK: vbmacor16x16x16 %ymm4, %ymm5, %ymm6
# CHECK: encoding: [0x62,0xf6,0x54,0x28,0x80,0xf4]
         vbmacor16x16x16 %ymm4, %ymm5, %ymm6

# CHECK: vbmacor16x16x16 %zmm4, %zmm5, %zmm6
# CHECK: encoding: [0x62,0xf6,0x54,0x48,0x80,0xf4]
         vbmacor16x16x16 %zmm4, %zmm5, %zmm6

# CHECK: vbmacor16x16x16 (%edx), %ymm5, %ymm6
# CHECK: encoding: [0x62,0xf6,0x54,0x28,0x80,0x32]
         vbmacor16x16x16 (%edx), %ymm5, %ymm6

# CHECK: vbmacor16x16x16 291(%esp,%esi,8), %zmm5, %zmm6
# CHECK: encoding: [0x62,0xf6,0x54,0x48,0x80,0xb4,0xf4,0x23,0x01,0x00,0x00]
         vbmacor16x16x16 291(%esp,%esi,8), %zmm5, %zmm6

# CHECK: vbmacxor16x16x16 %ymm4, %ymm5, %ymm6
# CHECK: encoding: [0x62,0xf6,0xd4,0x28,0x80,0xf4]
         vbmacxor16x16x16 %ymm4, %ymm5, %ymm6

# CHECK: vbmacxor16x16x16 %zmm4, %zmm5, %zmm6
# CHECK: encoding: [0x62,0xf6,0xd4,0x48,0x80,0xf4]
         vbmacxor16x16x16 %zmm4, %zmm5, %zmm6

# CHECK: vbmacxor16x16x16 (%edx), %ymm5, %ymm6
# CHECK: encoding: [0x62,0xf6,0xd4,0x28,0x80,0x32]
         vbmacxor16x16x16 (%edx), %ymm5, %ymm6

# CHECK: vbmacxor16x16x16 291(%esp,%esi,8), %zmm5, %zmm6
# CHECK: encoding: [0x62,0xf6,0xd4,0x48,0x80,0xb4,0xf4,0x23,0x01,0x00,0x00]
         vbmacxor16x16x16 291(%esp,%esi,8), %zmm5, %zmm6

# CHECK: vbitrevb %xmm5, %xmm6
# CHECK: encoding: [0x62,0xf6,0x7c,0x08,0x81,0xf5]
         vbitrevb %xmm5, %xmm6

# CHECK: vbitrevb %ymm5, %ymm6
# CHECK: encoding: [0x62,0xf6,0x7c,0x28,0x81,0xf5]
         vbitrevb %ymm5, %ymm6

# CHECK: vbitrevb %zmm5, %zmm6
# CHECK: encoding: [0x62,0xf6,0x7c,0x48,0x81,0xf5]
         vbitrevb %zmm5, %zmm6

# CHECK: vbitrevb %zmm5, %zmm6 {%k7}
# CHECK: encoding: [0x62,0xf6,0x7c,0x4f,0x81,0xf5]
         vbitrevb %zmm5, %zmm6 {%k7}

# CHECK: vbitrevb %zmm5, %zmm6 {%k7} {z}
# CHECK: encoding: [0x62,0xf6,0x7c,0xcf,0x81,0xf5]
         vbitrevb %zmm5, %zmm6 {%k7} {z}

# CHECK: vbitrevb (%edx), %xmm6
# CHECK: encoding: [0x62,0xf6,0x7c,0x08,0x81,0x32]
         vbitrevb (%edx), %xmm6

# CHECK: vbitrevb (%ecx), %ymm6
# CHECK: encoding: [0x62,0xf6,0x7c,0x28,0x81,0x31]
         vbitrevb (%ecx), %ymm6

# CHECK: vbitrevb (%edx), %zmm6
# CHECK: encoding: [0x62,0xf6,0x7c,0x48,0x81,0x32]
         vbitrevb (%edx), %zmm6

# CHECK: vbitrevb 291(%esp,%esi,8), %zmm6 {%k7}
# CHECK: encoding: [0x62,0xf6,0x7c,0x4f,0x81,0xb4,0xf4,0x23,0x01,0x00,0x00]
         vbitrevb 291(%esp,%esi,8), %zmm6 {%k7}
