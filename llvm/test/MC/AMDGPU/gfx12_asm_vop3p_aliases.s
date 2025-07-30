// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1200 -mattr=+wavefrontsize32 -show-encoding %s | FileCheck --check-prefixes=GFX12 %s

v_pk_min_f16 v0, v1, v2
// GFX12: v_pk_min_num_f16 v0, v1, v2             ; encoding: [0x00,0x40,0x1b,0xcc,0x01,0x05,0x02,0x18]

v_pk_max_f16 v0, v1, v2
// GFX12: v_pk_max_num_f16 v0, v1, v2             ; encoding: [0x00,0x40,0x1c,0xcc,0x01,0x05,0x02,0x18]

v_dot4_i32_i8 v5, v1, v2, s3
// GFX12: v_dot4_i32_iu8 v5, v1, v2, s3           ; encoding: [0x05,0x40,0x16,0xcc,0x01,0x05,0x0e,0x18]

v_dot8_i32_i4 v5, v1, v2, s3
// GFX12: v_dot8_i32_iu4 v5, v1, v2, s3           ; encoding: [0x05,0x40,0x18,0xcc,0x01,0x05,0x0e,0x18]
