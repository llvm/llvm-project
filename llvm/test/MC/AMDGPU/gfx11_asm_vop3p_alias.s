// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1100 -show-encoding %s | FileCheck -check-prefix=GFX11 %s

v_dot4_i32_i8 v5, v1, v2, s3
// GFX11: v_dot4_i32_iu8 v5, v1, v2, s3           ; encoding: [0x05,0x40,0x16,0xcc,0x01,0x05,0x0e,0x18]

v_dot8_i32_i4 v5, v1, v2, s3
// GFX11: v_dot8_i32_iu4 v5, v1, v2, s3           ; encoding: [0x05,0x40,0x18,0xcc,0x01,0x05,0x0e,0x18]
