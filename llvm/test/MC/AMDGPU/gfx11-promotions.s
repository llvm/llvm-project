// RUN: llvm-mc -triple=amdgcn -show-encoding -mcpu=gfx1100 -mattr=+wavefrontsize32,-wavefrontsize64 %s | FileCheck --check-prefix=GFX11 %s

// Check opcode promotions and forced suffices.
// 1. When a suffix is optional, check that it may be omitted.
// 2. When a suffix is optional, check that it may be specified w/o any effect.
// 3. When a suffix is required, check that specifying it enforces opcode promotion.
// 4. When a suffix is required, check that omitting the suffix results in a different encoding.

//===----------------------------------------------------------------------===//
// VOP1.
//===----------------------------------------------------------------------===//

v_mov_b32 v0, v1
// GFX11: v_mov_b32_e32 v0, v1                    ; encoding: [0x01,0x03,0x00,0x7e]

v_mov_b32_e32 v0, v1
// GFX11: v_mov_b32_e32 v0, v1                    ; encoding: [0x01,0x03,0x00,0x7e]

//===----------------------------------------------------------------------===//
// VOP2.
//===----------------------------------------------------------------------===//

v_add_f16 v5, v1, v2
// GFX11: v_add_f16_e32 v5, v1, v2                ; encoding: [0x01,0x05,0x0a,0x64]

v_add_f16_e32 v5, v1, v2
// GFX11: v_add_f16_e32 v5, v1, v2                ; encoding: [0x01,0x05,0x0a,0x64]

//===----------------------------------------------------------------------===//
// VOPC.
//===----------------------------------------------------------------------===//

v_cmp_lt_f32 vcc_lo, v1, v2
// GFX11: v_cmp_lt_f32_e32 vcc_lo, v1, v2         ; encoding: [0x01,0x05,0x22,0x7c]

v_cmp_lt_f32_e32 vcc_lo, v1, v2
// GFX11: v_cmp_lt_f32_e32 vcc_lo, v1, v2         ; encoding: [0x01,0x05,0x22,0x7c]

//===----------------------------------------------------------------------===//
// VOPCX.
//===----------------------------------------------------------------------===//

v_cmpx_class_f16 v1, v2
// GFX11: v_cmpx_class_f16_e32 v1, v2             ; encoding: [0x01,0x05,0xfa,0x7d]

v_cmpx_class_f16_e32 v1, v2
// GFX11: v_cmpx_class_f16_e32 v1, v2             ; encoding: [0x01,0x05,0xfa,0x7d]

//===----------------------------------------------------------------------===//
// VOP1.DPP8.
//===----------------------------------------------------------------------===//

v_bfrev_b32 v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_bfrev_b32_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0xe9,0x70,0x0a,0x7e,0x01,0x77,0x39,0x05]

v_bfrev_b32_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_bfrev_b32_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0xe9,0x70,0x0a,0x7e,0x01,0x77,0x39,0x05]

//===----------------------------------------------------------------------===//
// VOP1.DPP16.
//===----------------------------------------------------------------------===//

v_bfrev_b32 v5, v1 quad_perm:[3,2,1,0]
// GFX11: v_bfrev_b32_dpp v5, v1 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x70,0x0a,0x7e,0x01,0x1b,0x00,0xff]

v_bfrev_b32_dpp v5, v1 quad_perm:[3,2,1,0]
// GFX11: v_bfrev_b32_dpp v5, v1 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x70,0x0a,0x7e,0x01,0x1b,0x00,0xff]

//===----------------------------------------------------------------------===//
// VOP2.DPP8.
//===----------------------------------------------------------------------===//

v_add_f16 v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_add_f16_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0xe9,0x04,0x0a,0x64,0x01,0x77,0x39,0x05]

v_add_f16_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_add_f16_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0xe9,0x04,0x0a,0x64,0x01,0x77,0x39,0x05]

//===----------------------------------------------------------------------===//
// VOP2.DPP16.
//===----------------------------------------------------------------------===//

v_add_f16 v5, v1, v2 quad_perm:[3,2,1,0]
// GFX11: v_add_f16_dpp v5, v1, v2 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x04,0x0a,0x64,0x01,0x1b,0x00,0xff]

v_add_f16_dpp v5, v1, v2 quad_perm:[3,2,1,0]
// GFX11: v_add_f16_dpp v5, v1, v2 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x04,0x0a,0x64,0x01,0x1b,0x00,0xff]

//===----------------------------------------------------------------------===//
// VOPC.DPP8.
//===----------------------------------------------------------------------===//

v_cmp_le_u16 v1, v2 dpp8:[7,7,7,3,4,4,6,7] fi:1
// GFX11: v_cmp_le_u16 vcc_lo, v1, v2 dpp8:[7,7,7,3,4,4,6,7] fi:1 ; encoding: [0xea,0x04,0x76,0x7c,0x01,0xff,0x47,0xfa]

v_cmp_le_u16_dpp v1, v2 dpp8:[7,7,7,3,4,4,6,7] fi:1
// GFX11: v_cmp_le_u16 vcc_lo, v1, v2 dpp8:[7,7,7,3,4,4,6,7] fi:1 ; encoding: [0xea,0x04,0x76,0x7c,0x01,0xff,0x47,0xfa]

//===----------------------------------------------------------------------===//
// VOPC.DPP16.
//===----------------------------------------------------------------------===//

v_cmp_gt_u16 v1, v2 row_shl:0x7 row_mask:0x0 bank_mask:0x0 fi:1
// GFX11: v_cmp_gt_u16 vcc_lo, v1, v2 row_shl:7 row_mask:0x0 bank_mask:0x0 fi:1 ; encoding: [0xfa,0x04,0x78,0x7c,0x01,0x07,0x05,0x00]

v_cmp_gt_u16_dpp v1, v2 row_shl:0x7 row_mask:0x0 bank_mask:0x0 fi:1
// GFX11: v_cmp_gt_u16 vcc_lo, v1, v2 row_shl:7 row_mask:0x0 bank_mask:0x0 fi:1 ; encoding: [0xfa,0x04,0x78,0x7c,0x01,0x07,0x05,0x00]

//===----------------------------------------------------------------------===//
// VOPCX.DPP8.
//===----------------------------------------------------------------------===//

v_cmpx_class_f16 v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_class_f16 v1, v2 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0xe9,0x04,0xfa,0x7d,0x01,0x77,0x39,0x05]

v_cmpx_class_f16_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_class_f16 v1, v2 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0xe9,0x04,0xfa,0x7d,0x01,0x77,0x39,0x05]

//===----------------------------------------------------------------------===//
// VOPCX.DPP16.
//===----------------------------------------------------------------------===//

v_cmpx_class_f16 v1, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_class_f16 v1, v2 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x04,0xfa,0x7d,0x01,0x1b,0x00,0xff]

v_cmpx_class_f16_dpp v1, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_class_f16 v1, v2 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x04,0xfa,0x7d,0x01,0x1b,0x00,0xff]

//===----------------------------------------------------------------------===//
// VOP1 -> VOP3.
//===----------------------------------------------------------------------===//

v_sin_f32 v5, 0.5 mul:2
// GFX11: v_sin_f32_e64 v5, 0.5 mul:2             ; encoding: [0x05,0x00,0xb5,0xd5,0xf0,0x00,0x00,0x08]

v_sin_f32_e64 v5, 0.5 mul:2
// GFX11: v_sin_f32_e64 v5, 0.5 mul:2             ; encoding: [0x05,0x00,0xb5,0xd5,0xf0,0x00,0x00,0x08]

v_sin_f32_e64 v5, v1
// GFX11: v_sin_f32_e64 v5, v1                    ; encoding: [0x05,0x00,0xb5,0xd5,0x01,0x01,0x00,0x00]

v_sin_f32 v5, v1
// GFX11: v_sin_f32_e32 v5, v1                    ; encoding: [0x01,0x6b,0x0a,0x7e]

//===----------------------------------------------------------------------===//
// VOP2 -> VOP3.
//===----------------------------------------------------------------------===//

v_add_f32 v5, v1, -v2
// GFX11: v_add_f32_e64 v5, v1, -v2               ; encoding: [0x05,0x00,0x03,0xd5,0x01,0x05,0x02,0x40]

v_add_f32_e64 v5, v1, -v2
// GFX11: v_add_f32_e64 v5, v1, -v2               ; encoding: [0x05,0x00,0x03,0xd5,0x01,0x05,0x02,0x40]

v_add_f32_e64 v5, v1, v2
// GFX11: v_add_f32_e64 v5, v1, v2                ; encoding: [0x05,0x00,0x03,0xd5,0x01,0x05,0x02,0x00]

v_add_f32 v5, v1, v2
// GFX11: v_add_f32_e32 v5, v1, v2                ; encoding: [0x01,0x05,0x0a,0x06]

//===----------------------------------------------------------------------===//
// VOPC -> VOP3.
//===----------------------------------------------------------------------===//

v_cmp_f_f32 s10, -v1, v2
// GFX11: v_cmp_f_f32_e64 s10, -v1, v2            ; encoding: [0x0a,0x00,0x10,0xd4,0x01,0x05,0x02,0x20]

v_cmp_f_f32_e64 s10, -v1, v2
// GFX11: v_cmp_f_f32_e64 s10, -v1, v2            ; encoding: [0x0a,0x00,0x10,0xd4,0x01,0x05,0x02,0x20]

v_cmp_f_f32_e64 vcc_lo, v1, v2
// GFX11: v_cmp_f_f32_e64 vcc_lo, v1, v2          ; encoding: [0x6a,0x00,0x10,0xd4,0x01,0x05,0x02,0x00]

v_cmp_f_f32 vcc_lo, v1, v2
// GFX11: v_cmp_f_f32_e32 vcc_lo, v1, v2          ; encoding: [0x01,0x05,0x20,0x7c]

//===----------------------------------------------------------------------===//
// VOPCX -> VOP3.
//===----------------------------------------------------------------------===//

v_cmpx_f_f32 -v1, v2
// GFX11: v_cmpx_f_f32_e64 -v1, v2                ; encoding: [0x7e,0x00,0x90,0xd4,0x01,0x05,0x02,0x20]

v_cmpx_f_f32_e64 -v1, v2
// GFX11: v_cmpx_f_f32_e64 -v1, v2                ; encoding: [0x7e,0x00,0x90,0xd4,0x01,0x05,0x02,0x20]

v_cmpx_f_f32_e64 v1, v2
// GFX11: v_cmpx_f_f32_e64 v1, v2                 ; encoding: [0x7e,0x00,0x90,0xd4,0x01,0x05,0x02,0x00]

v_cmpx_f_f32 v1, v2
// GFX11: v_cmpx_f_f32_e32 v1, v2                 ; encoding: [0x01,0x05,0x20,0x7d]

//===----------------------------------------------------------------------===//
// VOP3.
//===----------------------------------------------------------------------===//

v_add3_u32 v5, v1, v2, s3
// GFX11: v_add3_u32 v5, v1, v2, s3               ; encoding: [0x05,0x00,0x55,0xd6,0x01,0x05,0x0e,0x00]

v_add3_u32_e64 v5, v1, v2, s3
// GFX11: v_add3_u32 v5, v1, v2, s3               ; encoding: [0x05,0x00,0x55,0xd6,0x01,0x05,0x0e,0x00]

//===----------------------------------------------------------------------===//
// VOP1 -> VOP3.DPP8.
//===----------------------------------------------------------------------===//

v_sin_f32 v5, v1 div:2 dpp8:[0,0,0,0,0,0,0,0]
// GFX11: v_sin_f32_e64_dpp v5, v1 div:2 dpp8:[0,0,0,0,0,0,0,0] ; encoding: [0x05,0x00,0xb5,0xd5,0xe9,0x00,0x00,0x18,0x01,0x00,0x00,0x00]

v_sin_f32_e64_dpp v5, v1 div:2 dpp8:[0,0,0,0,0,0,0,0]
// GFX11: v_sin_f32_e64_dpp v5, v1 div:2 dpp8:[0,0,0,0,0,0,0,0] ; encoding: [0x05,0x00,0xb5,0xd5,0xe9,0x00,0x00,0x18,0x01,0x00,0x00,0x00]

v_sin_f32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_sin_f32_e64_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x00,0xb5,0xd5,0xe9,0x00,0x00,0x00,0x01,0x77,0x39,0x05]

v_sin_f32 v5, v1 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_sin_f32_dpp v5, v1 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0xe9,0x6a,0x0a,0x7e,0x01,0x77,0x39,0x05]

//===----------------------------------------------------------------------===//
// VOP2 -> VOP3.DPP8.
//===----------------------------------------------------------------------===//

v_add_f32 v5, v1, v2 mul:4 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_add_f32_e64_dpp v5, v1, v2 mul:4 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x00,0x03,0xd5,0xe9,0x04,0x02,0x10,0x01,0x77,0x39,0x05]

v_add_f32_e64_dpp v5, v1, v2 mul:4 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_add_f32_e64_dpp v5, v1, v2 mul:4 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x00,0x03,0xd5,0xe9,0x04,0x02,0x10,0x01,0x77,0x39,0x05]

v_add_f32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_add_f32_e64_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x00,0x03,0xd5,0xe9,0x04,0x02,0x00,0x01,0x77,0x39,0x05]

v_add_f32 v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_add_f32_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0xe9,0x04,0x0a,0x06,0x01,0x77,0x39,0x05]

//===----------------------------------------------------------------------===//
// VOPC -> VOP3.DPP8.
//===----------------------------------------------------------------------===//

v_cmp_class_f32 s5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmp_class_f32_e64_dpp s5, v1, v2 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x00,0x7e,0xd4,0xe9,0x04,0x02,0x00,0x01,0x77,0x39,0x05]

v_cmp_class_f32_e64_dpp s5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmp_class_f32_e64_dpp s5, v1, v2 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x05,0x00,0x7e,0xd4,0xe9,0x04,0x02,0x00,0x01,0x77,0x39,0x05]

v_cmp_class_f32_e64_dpp vcc_lo, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmp_class_f32_e64_dpp vcc_lo, v1, v2 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x6a,0x00,0x7e,0xd4,0xe9,0x04,0x02,0x00,0x01,0x77,0x39,0x05]

v_cmp_class_f32 vcc_lo, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmp_class_f32 vcc_lo, v1, v2 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0xe9,0x04,0xfc,0x7c,0x01,0x77,0x39,0x05]

//===----------------------------------------------------------------------===//
// VOPCX -> VOP3.DPP8.
//===----------------------------------------------------------------------===//

v_cmpx_class_f32 -v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_class_f32_e64_dpp -v1, v2 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x7e,0x00,0xfe,0xd4,0xe9,0x04,0x02,0x20,0x01,0x77,0x39,0x05]

v_cmpx_class_f32_e64_dpp -v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_class_f32_e64_dpp -v1, v2 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x7e,0x00,0xfe,0xd4,0xe9,0x04,0x02,0x20,0x01,0x77,0x39,0x05]

v_cmpx_class_f32_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_class_f32_e64_dpp v1, v2 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x7e,0x00,0xfe,0xd4,0xe9,0x04,0x02,0x00,0x01,0x77,0x39,0x05]

v_cmpx_class_f32 v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_cmpx_class_f32 v1, v2 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0xe9,0x04,0xfc,0x7d,0x01,0x77,0x39,0x05]

//===----------------------------------------------------------------------===//
// VOP1 -> VOP3.DPP16.
//===----------------------------------------------------------------------===//

v_sin_f32 v5, v1 div:2 row_xmask:15
// GFX11: v_sin_f32_e64_dpp v5, v1 div:2 row_xmask:15 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0xb5,0xd5,0xfa,0x00,0x00,0x18,0x01,0x6f,0x01,0xff]

v_sin_f32_e64_dpp v5, v1 div:2 row_xmask:15
// GFX11: v_sin_f32_e64_dpp v5, v1 div:2 row_xmask:15 row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0xb5,0xd5,0xfa,0x00,0x00,0x18,0x01,0x6f,0x01,0xff]

v_sin_f32_e64_dpp v5, v1 quad_perm:[3,2,1,0]
// GFX11: v_sin_f32_e64_dpp v5, v1 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0xb5,0xd5,0xfa,0x00,0x00,0x00,0x01,0x1b,0x00,0xff]

v_sin_f32 v5, v1 quad_perm:[3,2,1,0]
// GFX11: v_sin_f32_dpp v5, v1 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x6a,0x0a,0x7e,0x01,0x1b,0x00,0xff]

//===----------------------------------------------------------------------===//
// VOP2 -> VOP3.DPP16.
//===----------------------------------------------------------------------===//

v_add_f32 v5, v1, v2 div:2 quad_perm:[3,2,1,0]
// GFX11: v_add_f32_e64_dpp v5, v1, v2 div:2 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x03,0xd5,0xfa,0x04,0x02,0x18,0x01,0x1b,0x00,0xff]

v_add_f32_e64_dpp v5, v1, v2 div:2 quad_perm:[3,2,1,0]
// GFX11: v_add_f32_e64_dpp v5, v1, v2 div:2 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x03,0xd5,0xfa,0x04,0x02,0x18,0x01,0x1b,0x00,0xff]

v_add_f32_e64_dpp v5, v1, v2 quad_perm:[3,2,1,0]
// GFX11: v_add_f32_e64_dpp v5, v1, v2 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x03,0xd5,0xfa,0x04,0x02,0x00,0x01,0x1b,0x00,0xff]

v_add_f32 v5, v1, v2 quad_perm:[3,2,1,0]
// GFX11: v_add_f32_dpp v5, v1, v2 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x04,0x0a,0x06,0x01,0x1b,0x00,0xff]

//===----------------------------------------------------------------------===//
// VOPC -> VOP3.DPP16.
//===----------------------------------------------------------------------===//

v_cmp_class_f32 s5, v1, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmp_class_f32_e64_dpp s5, v1, v2 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x7e,0xd4,0xfa,0x04,0x02,0x00,0x01,0x1b,0x00,0xff]

v_cmp_class_f32_e64_dpp s5, v1, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmp_class_f32_e64_dpp s5, v1, v2 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf ; encoding: [0x05,0x00,0x7e,0xd4,0xfa,0x04,0x02,0x00,0x01,0x1b,0x00,0xff]

v_cmp_class_f32_e64_dpp vcc_lo, v1, v2 row_share:0 row_mask:0xf bank_mask:0xf
// GFX11: v_cmp_class_f32_e64_dpp vcc_lo, v1, v2 row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0x6a,0x00,0x7e,0xd4,0xfa,0x04,0x02,0x00,0x01,0x50,0x01,0xff]

v_cmp_class_f32 vcc_lo, v1, v2 row_share:0 row_mask:0xf bank_mask:0xf
// GFX11: v_cmp_class_f32 vcc_lo, v1, v2 row_share:0 row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x04,0xfc,0x7c,0x01,0x50,0x01,0xff]

//===----------------------------------------------------------------------===//
// VOPCX -> VOP3.DPP16.
//===----------------------------------------------------------------------===//

v_cmpx_class_f32_e64_dpp v1, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_class_f32_e64_dpp v1, v2 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf ; encoding: [0x7e,0x00,0xfe,0xd4,0xfa,0x04,0x02,0x00,0x01,0x1b,0x00,0xff]

v_cmpx_class_f32 v1, v2 quad_perm:[3,2,1,0]
// GFX11: v_cmpx_class_f32 v1, v2 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf ; encoding: [0xfa,0x04,0xfc,0x7d,0x01,0x1b,0x00,0xff]

//===----------------------------------------------------------------------===//
// VOP3P.
//===----------------------------------------------------------------------===//

v_dot2_f32_f16 v0, v1, v2, v3
// GFX11: v_dot2_f32_f16 v0, v1, v2, v3           ; encoding: [0x00,0x40,0x13,0xcc,0x01,0x05,0x0e,0x1c]

v_dot2_f32_f16_e64 v0, v1, v2, v3
// GFX11: v_dot2_f32_f16 v0, v1, v2, v3           ; encoding: [0x00,0x40,0x13,0xcc,0x01,0x05,0x0e,0x1c]

//===----------------------------------------------------------------------===//
// VOP3P.DPP8.
//===----------------------------------------------------------------------===//

v_dot2_f32_f16 v0, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_dot2_f32_f16_e64_dpp v0, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x00,0x00,0x13,0xcc,0xe9,0x04,0x0e,0x04,0x01,0x77,0x39,0x05]

v_dot2_f32_f16_e64_dpp v0, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0]
// GFX11: v_dot2_f32_f16_e64_dpp v0, v1, v2, v3 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0x00,0x00,0x13,0xcc,0xe9,0x04,0x0e,0x04,0x01,0x77,0x39,0x05]

//===----------------------------------------------------------------------===//
// VOP3P.DPP16.
//===----------------------------------------------------------------------===//

v_dot2_f32_f16 v0, v1, v2, v3 quad_perm:[1,2,3,0]
// GFX11: v_dot2_f32_f16_e64_dpp v0, v1, v2, v3 quad_perm:[1,2,3,0] row_mask:0xf bank_mask:0xf ; encoding: [0x00,0x00,0x13,0xcc,0xfa,0x04,0x0e,0x04,0x01,0x39,0x00,0xff]

v_dot2_f32_f16_e64_dpp v0, v1, v2, v3 quad_perm:[1,2,3,0]
// GFX11: v_dot2_f32_f16_e64_dpp v0, v1, v2, v3 quad_perm:[1,2,3,0] row_mask:0xf bank_mask:0xf ; encoding: [0x00,0x00,0x13,0xcc,0xfa,0x04,0x0e,0x04,0x01,0x39,0x00,0xff]
