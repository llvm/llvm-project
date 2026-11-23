# RUN: llvm-mc -triple x86_64 -show-encoding -x86-asm-syntax=intel -output-asm-variant=1 %s | FileCheck %s

# CHECK: vbmacor16x16x16 ymm6, ymm5, ymm4
# CHECK: encoding: [0x62,0xf6,0x54,0x28,0x80,0xf4]
         vbmacor16x16x16 ymm6, ymm5, ymm4

# CHECK: vbmacor16x16x16 zmm6, zmm5, zmm4
# CHECK: encoding: [0x62,0xf6,0x54,0x48,0x80,0xf4]
         vbmacor16x16x16 zmm6, zmm5, zmm4

# CHECK: vbmacor16x16x16 ymm30, ymm29, ymm28
# CHECK: encoding: [0x62,0x06,0x14,0x20,0x80,0xf4]
         vbmacor16x16x16 ymm30, ymm29, ymm28

# CHECK: vbmacor16x16x16 zmm30, zmm29, zmm28
# CHECK: encoding: [0x62,0x06,0x14,0x40,0x80,0xf4]
         vbmacor16x16x16 zmm30, zmm29, zmm28

# CHECK: vbmacor16x16x16 ymm6, ymm5, ymmword ptr [rcx]
# CHECK: encoding: [0x62,0xf6,0x54,0x28,0x80,0x31]
         vbmacor16x16x16 ymm6, ymm5, ymmword ptr [rcx]

# CHECK: vbmacor16x16x16 zmm6, zmm5, zmmword ptr [rax + 8*r14 + 291]
# CHECK: encoding: [0x62,0xb6,0x54,0x48,0x80,0xb4,0xf0,0x23,0x01,0x00,0x00]
         vbmacor16x16x16 zmm6, zmm5, zmmword ptr [rax + 8*r14 + 291]

# CHECK: vbmacxor16x16x16 ymm6, ymm5, ymm4
# CHECK: encoding: [0x62,0xf6,0xd4,0x28,0x80,0xf4]
         vbmacxor16x16x16 ymm6, ymm5, ymm4

# CHECK: vbmacxor16x16x16 zmm6, zmm5, zmm4
# CHECK: encoding: [0x62,0xf6,0xd4,0x48,0x80,0xf4]
         vbmacxor16x16x16 zmm6, zmm5, zmm4

# CHECK: vbmacxor16x16x16 zmm30, zmm29, zmm28
# CHECK: encoding: [0x62,0x06,0x94,0x40,0x80,0xf4]
         vbmacxor16x16x16 zmm30, zmm29, zmm28

# CHECK: vbmacxor16x16x16 ymm6, ymm5, ymmword ptr [rcx]
# CHECK: encoding: [0x62,0xf6,0xd4,0x28,0x80,0x31]
         vbmacxor16x16x16 ymm6, ymm5, ymmword ptr [rcx]

# CHECK: vbmacxor16x16x16 zmm6, zmm5, zmmword ptr [rax + 8*r14 + 291]
# CHECK: encoding: [0x62,0xb6,0xd4,0x48,0x80,0xb4,0xf0,0x23,0x01,0x00,0x00]
         vbmacxor16x16x16 zmm6, zmm5, zmmword ptr [rax + 8*r14 + 291]

# CHECK: vbitrevb xmm6, xmm5
# CHECK: encoding: [0x62,0xf6,0x7c,0x08,0x81,0xf5]
         vbitrevb xmm6, xmm5

# CHECK: vbitrevb ymm6, ymm5
# CHECK: encoding: [0x62,0xf6,0x7c,0x28,0x81,0xf5]
         vbitrevb ymm6, ymm5

# CHECK: vbitrevb zmm6, zmm5
# CHECK: encoding: [0x62,0xf6,0x7c,0x48,0x81,0xf5]
         vbitrevb zmm6, zmm5

# CHECK: vbitrevb zmm30, zmm29
# CHECK: encoding: [0x62,0x06,0x7c,0x48,0x81,0xf5]
         vbitrevb zmm30, zmm29

# CHECK: vbitrevb zmm6 {k7}, zmm5
# CHECK: encoding: [0x62,0xf6,0x7c,0x4f,0x81,0xf5]
         vbitrevb zmm6 {k7}, zmm5

# CHECK: vbitrevb zmm6 {k7} {z}, zmm5
# CHECK: encoding: [0x62,0xf6,0x7c,0xcf,0x81,0xf5]
         vbitrevb zmm6 {k7} {z}, zmm5

# CHECK: vbitrevb xmm6, xmmword ptr [rcx]
# CHECK: encoding: [0x62,0xf6,0x7c,0x08,0x81,0x31]
         vbitrevb xmm6, xmmword ptr [rcx]

# CHECK: vbitrevb ymm6, ymmword ptr [rcx]
# CHECK: encoding: [0x62,0xf6,0x7c,0x28,0x81,0x31]
         vbitrevb ymm6, ymmword ptr [rcx]

# CHECK: vbitrevb zmm6, zmmword ptr [rcx]
# CHECK: encoding: [0x62,0xf6,0x7c,0x48,0x81,0x31]
         vbitrevb zmm6, zmmword ptr [rcx]

# CHECK: vbitrevb zmm6 {k7}, zmmword ptr [rax + 8*r14 + 291]
# CHECK: encoding: [0x62,0xb6,0x7c,0x4f,0x81,0xb4,0xf0,0x23,0x01,0x00,0x00]
         vbitrevb zmm6 {k7}, zmmword ptr [rax + 8*r14 + 291]
