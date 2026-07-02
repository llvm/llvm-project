# RUN: llvm-mc -triple x86_64 -show-encoding %s | FileCheck %s

# CHECK: vbmacor16x16x16 %ymm4, %ymm5, %ymm6
# CHECK: encoding: [0x62,0xf6,0x54,0x28,0x80,0xf4]
         vbmacor16x16x16 %ymm4, %ymm5, %ymm6

# CHECK: vbmacor16x16x16 %zmm4, %zmm5, %zmm6
# CHECK: encoding: [0x62,0xf6,0x54,0x48,0x80,0xf4]
         vbmacor16x16x16 %zmm4, %zmm5, %zmm6

# CHECK: vbmacor16x16x16 %ymm28, %ymm29, %ymm30
# CHECK: encoding: [0x62,0x06,0x14,0x20,0x80,0xf4]
         vbmacor16x16x16 %ymm28, %ymm29, %ymm30

# CHECK: vbmacor16x16x16 %zmm28, %zmm29, %zmm30
# CHECK: encoding: [0x62,0x06,0x14,0x40,0x80,0xf4]
         vbmacor16x16x16 %zmm28, %zmm29, %zmm30

# CHECK: vbmacor16x16x16 (%rcx), %ymm5, %ymm6
# CHECK: encoding: [0x62,0xf6,0x54,0x28,0x80,0x31]
         vbmacor16x16x16 (%rcx), %ymm5, %ymm6

# CHECK: vbmacor16x16x16 291(%rax,%r14,8), %zmm5, %zmm6
# CHECK: encoding: [0x62,0xb6,0x54,0x48,0x80,0xb4,0xf0,0x23,0x01,0x00,0x00]
         vbmacor16x16x16 291(%rax,%r14,8), %zmm5, %zmm6

# CHECK: vbmacxor16x16x16 %ymm4, %ymm5, %ymm6
# CHECK: encoding: [0x62,0xf6,0xd4,0x28,0x80,0xf4]
         vbmacxor16x16x16 %ymm4, %ymm5, %ymm6

# CHECK: vbmacxor16x16x16 %zmm4, %zmm5, %zmm6
# CHECK: encoding: [0x62,0xf6,0xd4,0x48,0x80,0xf4]
         vbmacxor16x16x16 %zmm4, %zmm5, %zmm6

# CHECK: vbmacxor16x16x16 %zmm28, %zmm29, %zmm30
# CHECK: encoding: [0x62,0x06,0x94,0x40,0x80,0xf4]
         vbmacxor16x16x16 %zmm28, %zmm29, %zmm30

# CHECK: vbmacxor16x16x16 (%rcx), %ymm5, %ymm6
# CHECK: encoding: [0x62,0xf6,0xd4,0x28,0x80,0x31]
         vbmacxor16x16x16 (%rcx), %ymm5, %ymm6

# CHECK: vbmacxor16x16x16 291(%rax,%r14,8), %zmm5, %zmm6
# CHECK: encoding: [0x62,0xb6,0xd4,0x48,0x80,0xb4,0xf0,0x23,0x01,0x00,0x00]
         vbmacxor16x16x16 291(%rax,%r14,8), %zmm5, %zmm6

# CHECK: vbitrevb %xmm5, %xmm6
# CHECK: encoding: [0x62,0xf6,0x7c,0x08,0x81,0xf5]
         vbitrevb %xmm5, %xmm6

# CHECK: vbitrevb %ymm5, %ymm6
# CHECK: encoding: [0x62,0xf6,0x7c,0x28,0x81,0xf5]
         vbitrevb %ymm5, %ymm6

# CHECK: vbitrevb %zmm5, %zmm6
# CHECK: encoding: [0x62,0xf6,0x7c,0x48,0x81,0xf5]
         vbitrevb %zmm5, %zmm6

# CHECK: vbitrevb %zmm29, %zmm30
# CHECK: encoding: [0x62,0x06,0x7c,0x48,0x81,0xf5]
         vbitrevb %zmm29, %zmm30

# CHECK: vbitrevb %zmm5, %zmm6 {%k7}
# CHECK: encoding: [0x62,0xf6,0x7c,0x4f,0x81,0xf5]
         vbitrevb %zmm5, %zmm6 {%k7}

# CHECK: vbitrevb %zmm5, %zmm6 {%k7} {z}
# CHECK: encoding: [0x62,0xf6,0x7c,0xcf,0x81,0xf5]
         vbitrevb %zmm5, %zmm6 {%k7} {z}

# CHECK: vbitrevb (%rcx), %xmm6
# CHECK: encoding: [0x62,0xf6,0x7c,0x08,0x81,0x31]
         vbitrevb (%rcx), %xmm6

# CHECK: vbitrevb (%rcx), %ymm6
# CHECK: encoding: [0x62,0xf6,0x7c,0x28,0x81,0x31]
         vbitrevb (%rcx), %ymm6

# CHECK: vbitrevb (%rcx), %zmm6
# CHECK: encoding: [0x62,0xf6,0x7c,0x48,0x81,0x31]
         vbitrevb (%rcx), %zmm6

# CHECK: vbitrevb 291(%rax,%r14,8), %zmm6 {%k7}
# CHECK: encoding: [0x62,0xb6,0x7c,0x4f,0x81,0xb4,0xf0,0x23,0x01,0x00,0x00]
         vbitrevb 291(%rax,%r14,8), %zmm6 {%k7}
