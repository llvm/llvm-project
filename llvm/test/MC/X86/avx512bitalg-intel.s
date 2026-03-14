# RUN: llvm-mc -triple x86_64 -show-encoding -x86-asm-syntax=intel -output-asm-variant=1 %s | FileCheck %s

# CHECK: vpopcntb	zmm21, zmm23
# CHECK: encoding: [0x62,0xa2,0x7d,0x48,0x54,0xef]
         vpopcntb	zmm21, zmm23

# CHECK: vpopcntw	zmm21, zmm23
# CHECK: encoding: [0x62,0xa2,0xfd,0x48,0x54,0xef]
         vpopcntw	zmm21, zmm23

# CHECK: vpopcntb	zmm1 {k2}, zmm3
# CHECK: encoding: [0x62,0xf2,0x7d,0x4a,0x54,0xcb]
         vpopcntb	zmm1 {k2}, zmm3

# CHECK: vpopcntw	zmm1 {k2}, zmm3
# CHECK: encoding: [0x62,0xf2,0xfd,0x4a,0x54,0xcb]
         vpopcntw	zmm1 {k2}, zmm3

# CHECK: vpopcntb	zmm1, zmmword ptr [rcx]
# CHECK: encoding: [0x62,0xf2,0x7d,0x48,0x54,0x09]
         vpopcntb	zmm1, zmmword ptr [rcx]

# CHECK: vpopcntb	zmm1, zmmword ptr [rsp - 256]
# CHECK: encoding: [0x62,0xf2,0x7d,0x48,0x54,0x4c,0x24,0xfc]
         vpopcntb	zmm1, zmmword ptr [rsp - 256]

# CHECK: vpopcntb	zmm1, zmmword ptr [rsp + 256]
# CHECK: encoding: [0x62,0xf2,0x7d,0x48,0x54,0x4c,0x24,0x04]
         vpopcntb	zmm1, zmmword ptr [rsp + 256]

# CHECK: vpopcntb	zmm1, zmmword ptr [rcx + 8*r14 + 268435456]
# CHECK: encoding: [0x62,0xb2,0x7d,0x48,0x54,0x8c,0xf1,0x00,0x00,0x00,0x10]
         vpopcntb	zmm1, zmmword ptr [rcx + 8*r14 + 268435456]

# CHECK: vpopcntb	zmm1, zmmword ptr [rcx + 8*r14 - 536870912]
# CHECK: encoding: [0x62,0xb2,0x7d,0x48,0x54,0x8c,0xf1,0x00,0x00,0x00,0xe0]
         vpopcntb	zmm1, zmmword ptr [rcx + 8*r14 - 536870912]

# CHECK: vpopcntb	zmm1, zmmword ptr [rcx + 8*r14 - 536870910]
# CHECK: encoding: [0x62,0xb2,0x7d,0x48,0x54,0x8c,0xf1,0x02,0x00,0x00,0xe0]
         vpopcntb	zmm1, zmmword ptr [rcx + 8*r14 - 536870910]

# CHECK: vpopcntw	zmm1, zmmword ptr [rcx]
# CHECK: encoding: [0x62,0xf2,0xfd,0x48,0x54,0x09]
         vpopcntw	zmm1, zmmword ptr [rcx]

# CHECK: vpopcntw	zmm1, zmmword ptr [rsp - 256]
# CHECK: encoding: [0x62,0xf2,0xfd,0x48,0x54,0x4c,0x24,0xfc]
         vpopcntw	zmm1, zmmword ptr [rsp - 256]

# CHECK: vpopcntw	zmm1, zmmword ptr [rsp + 256]
# CHECK: encoding: [0x62,0xf2,0xfd,0x48,0x54,0x4c,0x24,0x04]
         vpopcntw	zmm1, zmmword ptr [rsp + 256]

# CHECK: vpopcntw	zmm1, zmmword ptr [rcx + 8*r14 + 268435456]
# CHECK: encoding: [0x62,0xb2,0xfd,0x48,0x54,0x8c,0xf1,0x00,0x00,0x00,0x10]
         vpopcntw	zmm1, zmmword ptr [rcx + 8*r14 + 268435456]

# CHECK: vpopcntw	zmm1, zmmword ptr [rcx + 8*r14 - 536870912]
# CHECK: encoding: [0x62,0xb2,0xfd,0x48,0x54,0x8c,0xf1,0x00,0x00,0x00,0xe0]
         vpopcntw	zmm1, zmmword ptr [rcx + 8*r14 - 536870912]

# CHECK: vpopcntw	zmm1, zmmword ptr [rcx + 8*r14 - 536870910]
# CHECK: encoding: [0x62,0xb2,0xfd,0x48,0x54,0x8c,0xf1,0x02,0x00,0x00,0xe0]
         vpopcntw	zmm1, zmmword ptr [rcx + 8*r14 - 536870910]

# CHECK: vpopcntb	zmm21 {k2}, zmmword ptr [rcx]
# CHECK: encoding: [0x62,0xe2,0x7d,0x4a,0x54,0x29]
         vpopcntb	zmm21 {k2}, zmmword ptr [rcx]

# CHECK: vpopcntb	zmm21 {k2}, zmmword ptr [rsp - 256]
# CHECK: encoding: [0x62,0xe2,0x7d,0x4a,0x54,0x6c,0x24,0xfc]
         vpopcntb	zmm21 {k2}, zmmword ptr [rsp - 256]

# CHECK: vpopcntb	zmm21 {k2}, zmmword ptr [rsp + 256]
# CHECK: encoding: [0x62,0xe2,0x7d,0x4a,0x54,0x6c,0x24,0x04]
         vpopcntb	zmm21 {k2}, zmmword ptr [rsp + 256]

# CHECK: vpopcntb	zmm21 {k2}, zmmword ptr [rcx + 8*r14 + 268435456]
# CHECK: encoding: [0x62,0xa2,0x7d,0x4a,0x54,0xac,0xf1,0x00,0x00,0x00,0x10]
         vpopcntb	zmm21 {k2}, zmmword ptr [rcx + 8*r14 + 268435456]

# CHECK: vpopcntb	zmm21 {k2}, zmmword ptr [rcx + 8*r14 - 536870912]
# CHECK: encoding: [0x62,0xa2,0x7d,0x4a,0x54,0xac,0xf1,0x00,0x00,0x00,0xe0]
         vpopcntb	zmm21 {k2}, zmmword ptr [rcx + 8*r14 - 536870912]

# CHECK: vpopcntb	zmm21 {k2}, zmmword ptr [rcx + 8*r14 - 536870910]
# CHECK: encoding: [0x62,0xa2,0x7d,0x4a,0x54,0xac,0xf1,0x02,0x00,0x00,0xe0]
         vpopcntb	zmm21 {k2}, zmmword ptr [rcx + 8*r14 - 536870910]

# CHECK: vpopcntw	zmm21 {k2}, zmmword ptr [rcx]
# CHECK: encoding: [0x62,0xe2,0xfd,0x4a,0x54,0x29]
         vpopcntw	zmm21 {k2}, zmmword ptr [rcx]

# CHECK: vpopcntw	zmm21 {k2}, zmmword ptr [rsp - 256]
# CHECK: encoding: [0x62,0xe2,0xfd,0x4a,0x54,0x6c,0x24,0xfc]
         vpopcntw	zmm21 {k2}, zmmword ptr [rsp - 256]

# CHECK: vpopcntw	zmm21 {k2}, zmmword ptr [rsp + 256]
# CHECK: encoding: [0x62,0xe2,0xfd,0x4a,0x54,0x6c,0x24,0x04]
         vpopcntw	zmm21 {k2}, zmmword ptr [rsp + 256]

# CHECK: vpopcntw	zmm21 {k2}, zmmword ptr [rcx + 8*r14 + 268435456]
# CHECK: encoding: [0x62,0xa2,0xfd,0x4a,0x54,0xac,0xf1,0x00,0x00,0x00,0x10]
         vpopcntw	zmm21 {k2}, zmmword ptr [rcx + 8*r14 + 268435456]

# CHECK: vpopcntw	zmm21 {k2}, zmmword ptr [rcx + 8*r14 - 536870912]
# CHECK: encoding: [0x62,0xa2,0xfd,0x4a,0x54,0xac,0xf1,0x00,0x00,0x00,0xe0]
         vpopcntw	zmm21 {k2}, zmmword ptr [rcx + 8*r14 - 536870912]

# CHECK: vpopcntw	zmm21 {k2}, zmmword ptr [rcx + 8*r14 - 536870910]
# CHECK: encoding: [0x62,0xa2,0xfd,0x4a,0x54,0xac,0xf1,0x02,0x00,0x00,0xe0]
         vpopcntw	zmm21 {k2}, zmmword ptr [rcx + 8*r14 - 536870910]

# CHECK: vpshufbitqmb	k1, zmm23, zmm2
# CHECK: encoding: [0x62,0xf2,0x45,0x40,0x8f,0xca]
         vpshufbitqmb	k1, zmm23, zmm2

# CHECK: vpshufbitqmb	k1 {k2}, zmm23, zmm2
# CHECK: encoding: [0x62,0xf2,0x45,0x42,0x8f,0xca]
         vpshufbitqmb	k1 {k2}, zmm23, zmm2

# CHECK: vpshufbitqmb	k1, zmm23, zmmword ptr [rcx]
# CHECK: encoding: [0x62,0xf2,0x45,0x40,0x8f,0x09]
         vpshufbitqmb	k1, zmm23, zmmword ptr [rcx]

# CHECK: vpshufbitqmb	k1, zmm23, zmmword ptr [rsp - 256]
# CHECK: encoding: [0x62,0xf2,0x45,0x40,0x8f,0x4c,0x24,0xfc]
         vpshufbitqmb	k1, zmm23, zmmword ptr [rsp - 256]

# CHECK: vpshufbitqmb	k1, zmm23, zmmword ptr [rsp + 256]
# CHECK: encoding: [0x62,0xf2,0x45,0x40,0x8f,0x4c,0x24,0x04]
         vpshufbitqmb	k1, zmm23, zmmword ptr [rsp + 256]

# CHECK: vpshufbitqmb	k1, zmm23, zmmword ptr [rcx + 8*r14 + 268435456]
# CHECK: encoding: [0x62,0xb2,0x45,0x40,0x8f,0x8c,0xf1,0x00,0x00,0x00,0x10]
         vpshufbitqmb	k1, zmm23, zmmword ptr [rcx + 8*r14 + 268435456]

# CHECK: vpshufbitqmb	k1, zmm23, zmmword ptr [rcx + 8*r14 - 536870912]
# CHECK: encoding: [0x62,0xb2,0x45,0x40,0x8f,0x8c,0xf1,0x00,0x00,0x00,0xe0]
         vpshufbitqmb	k1, zmm23, zmmword ptr [rcx + 8*r14 - 536870912]

# CHECK: vpshufbitqmb	k1, zmm23, zmmword ptr [rcx + 8*r14 - 536870910]
# CHECK: encoding: [0x62,0xb2,0x45,0x40,0x8f,0x8c,0xf1,0x02,0x00,0x00,0xe0]
         vpshufbitqmb	k1, zmm23, zmmword ptr [rcx + 8*r14 - 536870910]

# CHECK: vpshufbitqmb	k1 {k2}, zmm23, zmmword ptr [rcx]
# CHECK: encoding: [0x62,0xf2,0x45,0x42,0x8f,0x09]
         vpshufbitqmb	k1 {k2}, zmm23, zmmword ptr [rcx]

# CHECK: vpshufbitqmb	k1 {k2}, zmm23, zmmword ptr [rsp - 256]
# CHECK: encoding: [0x62,0xf2,0x45,0x42,0x8f,0x4c,0x24,0xfc]
         vpshufbitqmb	k1 {k2}, zmm23, zmmword ptr [rsp - 256]

# CHECK: vpshufbitqmb	k1 {k2}, zmm23, zmmword ptr [rsp + 256]
# CHECK: encoding: [0x62,0xf2,0x45,0x42,0x8f,0x4c,0x24,0x04]
         vpshufbitqmb	k1 {k2}, zmm23, zmmword ptr [rsp + 256]

# CHECK: vpshufbitqmb	k1 {k2}, zmm23, zmmword ptr [rcx + 8*r14 + 268435456]
# CHECK: encoding: [0x62,0xb2,0x45,0x42,0x8f,0x8c,0xf1,0x00,0x00,0x00,0x10]
         vpshufbitqmb	k1 {k2}, zmm23, zmmword ptr [rcx + 8*r14 + 268435456]

# CHECK: vpshufbitqmb	k1 {k2}, zmm23, zmmword ptr [rcx + 8*r14 - 536870912]
# CHECK: encoding: [0x62,0xb2,0x45,0x42,0x8f,0x8c,0xf1,0x00,0x00,0x00,0xe0]
         vpshufbitqmb	k1 {k2}, zmm23, zmmword ptr [rcx + 8*r14 - 536870912]

# CHECK: vpshufbitqmb	k1 {k2}, zmm23, zmmword ptr [rcx + 8*r14 - 536870910]
# CHECK: encoding: [0x62,0xb2,0x45,0x42,0x8f,0x8c,0xf1,0x02,0x00,0x00,0xe0]
         vpshufbitqmb	k1 {k2}, zmm23, zmmword ptr [rcx + 8*r14 - 536870910]
