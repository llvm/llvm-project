// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32 -show-encoding %s | FileCheck -check-prefix=GFX11 %s
// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize64 -show-encoding %s | FileCheck -check-prefix=GFX11 %s

v_cvt_pknorm_i16_f16 v5, v1, v2
// GFX11: v_cvt_pk_norm_i16_f16 v5, v1, v2        ; encoding: [0x05,0x00,0x12,0xd7,0x01,0x05,0x02,0x00]

v_cvt_pknorm_u16_f16 v5, v1, v2
// GFX11: v_cvt_pk_norm_u16_f16 v5, v1, v2        ; encoding: [0x05,0x00,0x13,0xd7,0x01,0x05,0x02,0x00]

v_add3_nc_u32 v5, v1, v2, s3
// GFX11: v_add3_u32 v5, v1, v2, s3               ; encoding: [0x05,0x00,0x55,0xd6,0x01,0x05,0x0e,0x00]

v_xor_add_u32 v5, v1, v2, s3
// GFX11: v_xad_u32 v5, v1, v2, s3                ; encoding: [0x05,0x00,0x45,0xd6,0x01,0x05,0x0e,0x00]
