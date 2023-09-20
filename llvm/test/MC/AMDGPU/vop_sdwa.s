// RUN: not llvm-mc -triple=amdgcn -mcpu=tonga -show-encoding %s | FileCheck %s --check-prefixes=VI,GFX89
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx900 -show-encoding %s | FileCheck %s --check-prefixes=GFX9,GFX89

// RUN: not llvm-mc -triple=amdgcn %s 2>&1 | FileCheck %s --check-prefixes=NOSI,NOSICI --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck %s --check-prefixes=NOSI,NOSICI --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgcn -mcpu=bonaire %s 2>&1 | FileCheck %s --check-prefixes=NOCI,NOSICI --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgcn -mcpu=tonga %s 2>&1 | FileCheck %s --check-prefixes=NOVI,NOGFX89 --implicit-check-not=error:
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck %s --check-prefixes=NOGFX9,NOGFX89 --implicit-check-not=error:

//---------------------------------------------------------------------------//
// Check SDWA operands
//---------------------------------------------------------------------------//

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_mov_b32_sdwa v1, v2 dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD ; encoding: [0xf9,0x02,0x02,0x7e,0x02,0x10,0x06,0x00]
v_mov_b32 v1, v2 dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_mov_b32_sdwa v3, v4 dst_sel:BYTE_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 ; encoding: [0xf9,0x02,0x06,0x7e,0x04,0x11,0x05,0x00]
v_mov_b32 v3, v4 dst_sel:BYTE_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_mov_b32_sdwa v15, v99 dst_sel:BYTE_2 dst_unused:UNUSED_SEXT src0_sel:WORD_0 ; encoding: [0xf9,0x02,0x1e,0x7e,0x63,0x0a,0x04,0x00]
v_mov_b32 v15, v99 dst_sel:BYTE_2 dst_unused:UNUSED_SEXT src0_sel:WORD_0

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_min_u32_sdwa v194, v13, v1 dst_sel:BYTE_3 dst_unused:UNUSED_SEXT src0_sel:BYTE_3 src1_sel:BYTE_2 ; encoding: [0xf9,0x02,0x84,0x1d,0x0d,0x0b,0x03,0x02]
v_min_u32 v194, v13, v1 dst_sel:BYTE_3 dst_unused:UNUSED_SEXT src0_sel:BYTE_3 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_min_u32_sdwa v255, v4, v1 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:WORD_1 ; encoding: [0xf9,0x02,0xfe,0x1d,0x04,0x04,0x02,0x05]
v_min_u32 v255, v4, v1 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_min_u32_sdwa v200, v200, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:DWORD ; encoding: [0xf9,0x02,0x90,0x1d,0xc8,0x05,0x01,0x06]
v_min_u32 v200, v200, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:DWORD

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_min_u32_sdwa v1, v1, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD ; encoding: [0xf9,0x02,0x02,0x1c,0x01,0x06,0x00,0x06]
v_min_u32 v1, v1, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD

//---------------------------------------------------------------------------//
// Check optional operands
//---------------------------------------------------------------------------//

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_cvt_u32_f32_sdwa v0, v0 clamp dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD ; encoding: [0xf9,0x0e,0x00,0x7e,0x00,0x36,0x06,0x00]
v_cvt_u32_f32 v0, v0 clamp dst_sel:DWORD

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_fract_f32_sdwa v0, v0 clamp dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD ; encoding: [0xf9,0x36,0x00,0x7e,0x00,0x26,0x06,0x00]
v_fract_f32 v0, v0 clamp dst_sel:DWORD dst_unused:UNUSED_PAD

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_sin_f32_sdwa v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x52,0x00,0x7e,0x00,0x06,0x05,0x00]
v_sin_f32 v0, v0 dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_mov_b32_sdwa v1, v0 clamp dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 ; encoding: [0xf9,0x02,0x02,0x7e,0x00,0x36,0x05,0x00]
v_mov_b32 v1, v0 clamp src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_trunc_f32_sdwa v1, v0 clamp dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 ; encoding: [0xf9,0x38,0x02,0x7e,0x00,0x36,0x05,0x00]
v_trunc_f32 v1, v0 clamp dst_sel:DWORD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// GFX89: v_mov_b32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD ; encoding: [0xf9,0x02,0x02,0x7e,0x00,0x16,0x06,0x00]
v_mov_b32_sdwa v1, v0

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// GFX89: v_add_f32_sdwa v0, v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD ; encoding: [0xf9,0x00,0x00,0x02,0x00,0x06,0x05,0x06]
v_add_f32_sdwa v0, v0, v0 dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_min_f32_sdwa v0, v0, v0 clamp dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:BYTE_2 ; encoding: [0xf9,0x00,0x00,0x14,0x00,0x36,0x06,0x02]
v_min_f32 v0, v0, v0 clamp dst_sel:DWORD src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_and_b32_sdwa v0, v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_2 ; encoding: [0xf9,0x00,0x00,0x26,0x00,0x06,0x06,0x02]
v_and_b32 v0, v0, v0 dst_unused:UNUSED_PAD src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// GFX89: v_mul_i32_i24_sdwa v1, v2, v3 clamp dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x06,0x02,0x0c,0x02,0x36,0x06,0x06]
v_mul_i32_i24_sdwa v1, v2, v3 clamp

//===----------------------------------------------------------------------===//
// Check modifiers
//===----------------------------------------------------------------------===//

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_fract_f32_sdwa v0, |v0| dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x36,0x00,0x7e,0x00,0x06,0x25,0x00]
v_fract_f32 v0, |v0| dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_sin_f32_sdwa v0, -|v0| dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x52,0x00,0x7e,0x00,0x06,0x35,0x00]
v_sin_f32 v0, -abs(v0) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_add_f32_sdwa v0, -|v0|, -v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x00,0x00,0x02,0x00,0x06,0x35,0x12]
v_add_f32 v0, -|v0|, -v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_min_f32_sdwa v0, |v0|, -v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x00,0x00,0x14,0x00,0x06,0x25,0x12]
v_min_f32 v0, abs(v0), -v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// GFX89: v_mov_b32_sdwa v1, sext(v0) dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD ; encoding: [0xf9,0x02,0x02,0x7e,0x00,0x16,0x0e,0x00]
v_mov_b32_sdwa v1, sext(v0)

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_and_b32_sdwa v0, sext(v0), sext(v0) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_2 ; encoding: [0xf9,0x00,0x00,0x26,0x00,0x06,0x0e,0x0a]
v_and_b32 v0, sext(v0), sext(v0) dst_unused:UNUSED_PAD src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// VI: v_cmp_class_f32 vcc, -v1, sext(v2) src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0x20,0x7c,0x01,0x00,0x12,0x0c]
// GFX9: v_cmp_class_f32_sdwa vcc, -v1, sext(v2) src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0x20,0x7c,0x01,0x00,0x12,0x0c]
v_cmp_class_f32_sdwa vcc, -v1, sext(v2) src0_sel:BYTE_2 src1_sel:WORD_0

//===----------------------------------------------------------------------===//
// Check VOP1 opcodes
//===----------------------------------------------------------------------===//

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// GFX89: v_nop ; encoding: [0xf9,0x00,0x00,0x7e,0x00,0x00,0x00,0x00]
v_nop_sdwa

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_cvt_u32_f32_sdwa v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x0e,0x00,0x7e,0x00,0x06,0x05,0x00]
v_cvt_u32_f32 v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_fract_f32_sdwa v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x36,0x00,0x7e,0x00,0x06,0x05,0x00]
v_fract_f32 v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_sin_f32_sdwa v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x52,0x00,0x7e,0x00,0x06,0x05,0x00]
v_sin_f32 v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_mov_b32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x02,0x02,0x7e,0x00,0x06,0x05,0x00]
v_mov_b32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_cvt_f32_i32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x0a,0x02,0x7e,0x00,0x06,0x05,0x00]
v_cvt_f32_i32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_cvt_f32_u32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x0c,0x02,0x7e,0x00,0x06,0x05,0x00]
v_cvt_f32_u32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_cvt_i32_f32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x10,0x02,0x7e,0x00,0x06,0x05,0x00]
v_cvt_i32_f32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_cvt_f16_f32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x14,0x02,0x7e,0x00,0x06,0x05,0x00]
v_cvt_f16_f32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_cvt_f32_f16_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x16,0x02,0x7e,0x00,0x06,0x05,0x00]
v_cvt_f32_f16 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_cvt_rpi_i32_f32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x18,0x02,0x7e,0x00,0x06,0x05,0x00]
v_cvt_rpi_i32_f32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_cvt_flr_i32_f32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x1a,0x02,0x7e,0x00,0x06,0x05,0x00]
v_cvt_flr_i32_f32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_cvt_off_f32_i4_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x1c,0x02,0x7e,0x00,0x06,0x05,0x00]
v_cvt_off_f32_i4 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_cvt_f32_ubyte0_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x22,0x02,0x7e,0x00,0x06,0x05,0x00]
v_cvt_f32_ubyte0 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_cvt_f32_ubyte1_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x24,0x02,0x7e,0x00,0x06,0x05,0x00]
v_cvt_f32_ubyte1 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_cvt_f32_ubyte2_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x26,0x02,0x7e,0x00,0x06,0x05,0x00]
v_cvt_f32_ubyte2 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_cvt_f32_ubyte3_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x28,0x02,0x7e,0x00,0x06,0x05,0x00]
v_cvt_f32_ubyte3 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_trunc_f32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x38,0x02,0x7e,0x00,0x06,0x05,0x00]
v_trunc_f32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_ceil_f32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x3a,0x02,0x7e,0x00,0x06,0x05,0x00]
v_ceil_f32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_rndne_f32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x3c,0x02,0x7e,0x00,0x06,0x05,0x00]
v_rndne_f32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_floor_f32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x3e,0x02,0x7e,0x00,0x06,0x05,0x00]
v_floor_f32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_exp_f32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x40,0x02,0x7e,0x00,0x06,0x05,0x00]
v_exp_f32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_log_f32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x42,0x02,0x7e,0x00,0x06,0x05,0x00]
v_log_f32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_rcp_f32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x44,0x02,0x7e,0x00,0x06,0x05,0x00]
v_rcp_f32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_rcp_iflag_f32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x46,0x02,0x7e,0x00,0x06,0x05,0x00]
v_rcp_iflag_f32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_rsq_f32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x48,0x02,0x7e,0x00,0x06,0x05,0x00]
v_rsq_f32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_sqrt_f32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x4e,0x02,0x7e,0x00,0x06,0x05,0x00]
v_sqrt_f32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_cos_f32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x54,0x02,0x7e,0x00,0x06,0x05,0x00]
v_cos_f32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_not_b32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x56,0x02,0x7e,0x00,0x06,0x05,0x00]
v_not_b32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_bfrev_b32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x58,0x02,0x7e,0x00,0x06,0x05,0x00]
v_bfrev_b32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_ffbh_u32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x5a,0x02,0x7e,0x00,0x06,0x05,0x00]
v_ffbh_u32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_ffbl_b32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x5c,0x02,0x7e,0x00,0x06,0x05,0x00]
v_ffbl_b32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_ffbh_i32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x5e,0x02,0x7e,0x00,0x06,0x05,0x00]
v_ffbh_i32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_frexp_exp_i32_f32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x66,0x02,0x7e,0x00,0x06,0x05,0x00]
v_frexp_exp_i32_f32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_frexp_mant_f32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x68,0x02,0x7e,0x00,0x06,0x05,0x00]
v_frexp_mant_f32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// GFX89: v_log_legacy_f32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x98,0x02,0x7e,0x00,0x06,0x05,0x00]
// NOSI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOCI: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand.
v_log_legacy_f32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// GFX89: v_exp_legacy_f32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x96,0x02,0x7e,0x00,0x06,0x05,0x00]
// NOSI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOCI: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand.
v_exp_legacy_f32 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_cvt_f16_u16_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x72,0x02,0x7e,0x00,0x06,0x05,0x00]
v_cvt_f16_u16 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_cvt_f16_i16_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x74,0x02,0x7e,0x00,0x06,0x05,0x00]
v_cvt_f16_i16 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_cvt_u16_f16_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x76,0x02,0x7e,0x00,0x06,0x05,0x00]
v_cvt_u16_f16 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_cvt_i16_f16_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x78,0x02,0x7e,0x00,0x06,0x05,0x00]
v_cvt_i16_f16 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_rcp_f16_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x7a,0x02,0x7e,0x00,0x06,0x05,0x00]
v_rcp_f16 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_sqrt_f16_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x7c,0x02,0x7e,0x00,0x06,0x05,0x00]
v_sqrt_f16 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_rsq_f16_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x7e,0x02,0x7e,0x00,0x06,0x05,0x00]
v_rsq_f16 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_log_f16_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x80,0x02,0x7e,0x00,0x06,0x05,0x00]
v_log_f16 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_exp_f16_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x82,0x02,0x7e,0x00,0x06,0x05,0x00]
v_exp_f16 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_frexp_mant_f16_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x84,0x02,0x7e,0x00,0x06,0x05,0x00]
v_frexp_mant_f16 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_frexp_exp_i16_f16_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x86,0x02,0x7e,0x00,0x06,0x05,0x00]
v_frexp_exp_i16_f16 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_floor_f16_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x88,0x02,0x7e,0x00,0x06,0x05,0x00]
v_floor_f16 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_ceil_f16_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x8a,0x02,0x7e,0x00,0x06,0x05,0x00]
v_ceil_f16 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_trunc_f16_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x8c,0x02,0x7e,0x00,0x06,0x05,0x00]
v_trunc_f16 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_rndne_f16_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x8e,0x02,0x7e,0x00,0x06,0x05,0x00]
v_rndne_f16 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_fract_f16_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x90,0x02,0x7e,0x00,0x06,0x05,0x00]
v_fract_f16 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_sin_f16_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x92,0x02,0x7e,0x00,0x06,0x05,0x00]
v_sin_f16 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_cos_f16_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x94,0x02,0x7e,0x00,0x06,0x05,0x00]
v_cos_f16 v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// GFX9:   v_cvt_norm_i16_f16_sdwa v5, -v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD ; encoding: [0xf9,0x9a,0x0a,0x7e,0x01,0x06,0x16,0x00]
// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI:   error: instruction not supported on this GPU
v_cvt_norm_i16_f16_sdwa v5, -v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD

// GFX9:   v_cvt_norm_i16_f16_sdwa v5, |v1| dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD ; encoding: [0xf9,0x9a,0x0a,0x7e,0x01,0x06,0x26,0x00]
// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI:   error: instruction not supported on this GPU
v_cvt_norm_i16_f16_sdwa v5, |v1| dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD

// GFX9:   v_cvt_norm_u16_f16_sdwa v5, v1 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD ; encoding: [0xf9,0x9c,0x0a,0x7e,0x01,0x16,0x06,0x00]
// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI:   error: instruction not supported on this GPU
v_cvt_norm_u16_f16_sdwa v5, v1 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD

// GFX9:   v_cvt_norm_u16_f16_sdwa v5, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 ; encoding: [0xf9,0x9c,0x0a,0x7e,0x01,0x06,0x05,0x00]
// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI:   error: instruction not supported on this GPU
v_cvt_norm_u16_f16_sdwa v5, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1

// GFX9:   v_sat_pk_u8_i16_sdwa v5, sext(v1) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD ; encoding: [0xf9,0x9e,0x0a,0x7e,0x01,0x06,0x0e,0x00]
// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI:   error: instruction not supported on this GPU
v_sat_pk_u8_i16_sdwa v5, sext(v1) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD

//===----------------------------------------------------------------------===//
// Check VOP2 opcodes
//===----------------------------------------------------------------------===//

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_add_f32_sdwa v0, v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x00,0x00,0x02,0x00,0x06,0x05,0x02]
v_add_f32 v0, v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_min_f32_sdwa v0, v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x00,0x00,0x14,0x00,0x06,0x05,0x02]
v_min_f32 v0, v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_and_b32_sdwa v0, v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x00,0x00,0x26,0x00,0x06,0x05,0x02]
v_and_b32 v0, v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_mul_i32_i24_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x0c,0x02,0x06,0x05,0x02]
v_mul_i32_i24 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_sub_f32_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x04,0x02,0x06,0x05,0x02]
v_sub_f32 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_subrev_f32_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x06,0x02,0x06,0x05,0x02]
v_subrev_f32 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_mul_f32_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x0a,0x02,0x06,0x05,0x02]
v_mul_f32 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_mul_hi_i32_i24_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x0e,0x02,0x06,0x05,0x02]
v_mul_hi_i32_i24 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_mul_u32_u24_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x10,0x02,0x06,0x05,0x02]
v_mul_u32_u24 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_mul_hi_u32_u24_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x12,0x02,0x06,0x05,0x02]
v_mul_hi_u32_u24 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_max_f32_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x16,0x02,0x06,0x05,0x02]
v_max_f32 v1, v2 v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_min_i32_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x18,0x02,0x06,0x05,0x02]
v_min_i32 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_max_i32_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x1a,0x02,0x06,0x05,0x02]
v_max_i32 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_min_u32_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x1c,0x02,0x06,0x05,0x02]
v_min_u32 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_max_u32_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x1e,0x02,0x06,0x05,0x02]
v_max_u32 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_lshrrev_b32_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x20,0x02,0x06,0x05,0x02]
v_lshrrev_b32 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_ashrrev_i32_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x22,0x02,0x06,0x05,0x02]
v_ashrrev_i32 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_lshlrev_b32_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x24,0x02,0x06,0x05,0x02]
v_lshlrev_b32 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_or_b32_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x28,0x02,0x06,0x05,0x02]
v_or_b32 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX89: v_xor_b32_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x2a,0x02,0x06,0x05,0x02]
v_xor_b32 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_add_f16_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x3e,0x02,0x06,0x05,0x02]
v_add_f16 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_sub_f16_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x40,0x02,0x06,0x05,0x02]
v_sub_f16 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_subrev_f16_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x42,0x02,0x06,0x05,0x02]
v_subrev_f16 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_mul_f16_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x44,0x02,0x06,0x05,0x02]
v_mul_f16 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_add_u16_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x4c,0x02,0x06,0x05,0x02]
v_add_u16 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_sub_u16_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x4e,0x02,0x06,0x05,0x02]
v_sub_u16 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_subrev_u16_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x50,0x02,0x06,0x05,0x02]
v_subrev_u16 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_mul_lo_u16_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x52,0x02,0x06,0x05,0x02]
v_mul_lo_u16 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_lshlrev_b16_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x54,0x02,0x06,0x05,0x02]
v_lshlrev_b16 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_lshrrev_b16_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x56,0x02,0x06,0x05,0x02]
v_lshrrev_b16 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_ashrrev_i16_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x58,0x02,0x06,0x05,0x02]
v_ashrrev_i16 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_max_f16_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x5a,0x02,0x06,0x05,0x02]
v_max_f16 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_min_f16_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x5c,0x02,0x06,0x05,0x02]
v_min_f16 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_max_u16_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x5e,0x02,0x06,0x05,0x02]
v_max_u16 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_max_i16_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x60,0x02,0x06,0x05,0x02]
v_max_i16 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_min_u16_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x62,0x02,0x06,0x05,0x02]
v_min_u16 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_min_i16_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x64,0x02,0x06,0x05,0x02]
v_min_i16 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX89: v_ldexp_f16_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x66,0x02,0x06,0x05,0x02]
v_ldexp_f16 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOGFX9: :[[@LINE+2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// VI: v_add_u32_sdwa v1, vcc, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x32,0x02,0x06,0x05,0x02]
v_add_u32_sdwa v1, vcc, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOGFX9: :[[@LINE+2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// VI: v_sub_u32_sdwa v1, vcc, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x34,0x02,0x06,0x05,0x02]
v_sub_u32_sdwa v1, vcc, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOGFX9: :[[@LINE+2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// VI: v_subrev_u32_sdwa v1, vcc, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x36,0x02,0x06,0x05,0x02]
v_subrev_u32_sdwa v1, vcc, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOGFX9: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// VI: v_addc_u32_sdwa v1, vcc, v2, v3, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x38,0x02,0x06,0x05,0x02]
v_addc_u32_sdwa v1, vcc, v2, v3, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOGFX9: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// VI: v_subb_u32_sdwa v1, vcc, v2, v3, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x3a,0x02,0x06,0x05,0x02]
v_subb_u32_sdwa v1, vcc, v2, v3, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOGFX9: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// VI: v_subbrev_u32_sdwa v1, vcc, v2, v3, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x3c,0x02,0x06,0x05,0x02]
v_subbrev_u32_sdwa v1, vcc, v2, v3, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX9: v_add_co_u32_sdwa v1, vcc, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x32,0x02,0x06,0x05,0x02]
v_add_co_u32_sdwa v1, vcc, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX9: v_sub_co_u32_sdwa v1, vcc, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x34,0x02,0x06,0x05,0x02]
v_sub_co_u32_sdwa v1, vcc, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX9: v_subrev_co_u32_sdwa v1, vcc, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x36,0x02,0x06,0x05,0x02]
v_subrev_co_u32_sdwa v1, vcc, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX9: v_addc_co_u32_sdwa v1, vcc, v2, v3, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x38,0x02,0x06,0x05,0x02]
v_addc_co_u32_sdwa v1, vcc, v2, v3, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX9: v_subb_co_u32_sdwa v1, vcc, v2, v3, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x3a,0x02,0x06,0x05,0x02]
v_subb_co_u32_sdwa v1, vcc, v2, v3, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX9: v_subbrev_co_u32_sdwa v1, vcc, v2, v3, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x3c,0x02,0x06,0x05,0x02]
v_subbrev_co_u32_sdwa v1, vcc, v2, v3, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// GFX89:  v_cndmask_b32_sdwa v5, v1, v2, vcc dst_sel:BYTE_0 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x04,0x0a,0x00,0x01,0x00,0x06,0x06]
v_cndmask_b32_sdwa v5, v1, v2, vcc dst_sel:BYTE_0 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI:   error: invalid operand for instruction
// GFX9:   v_cndmask_b32_sdwa v5, -1, v2, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x04,0x0a,0x00,0xc1,0x06,0x86,0x06]
v_cndmask_b32_sdwa v5, -1, v2, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// GFX89:  v_cndmask_b32_sdwa v5, -v1, |v2|, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x04,0x0a,0x00,0x01,0x06,0x16,0x26]
v_cndmask_b32_sdwa v5, -v1, |v2|, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// GFX89:  v_cndmask_b32_sdwa v5, |v1|, -v2, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x04,0x0a,0x00,0x01,0x06,0x26,0x16]
v_cndmask_b32_sdwa v5, |v1|, -v2, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD

v_cndmask_b32_sdwa v5, vcc_lo, v2, vcc dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD
// NOSICI: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI:   error: invalid operand for instruction
// NOGFX9: :[[@LINE-3]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)

//===----------------------------------------------------------------------===//
// Check VOPC opcodes
//===----------------------------------------------------------------------===//

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// VI: v_cmp_eq_f32 vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0x84,0x7c,0x01,0x00,0x02,0x04]
// GFX9: v_cmp_eq_f32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0x84,0x7c,0x01,0x00,0x02,0x04]
v_cmp_eq_f32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// VI: v_cmp_nle_f32 vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0x98,0x7c,0x01,0x00,0x02,0x04]
// GFX9: v_cmp_nle_f32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0x98,0x7c,0x01,0x00,0x02,0x04]
v_cmp_nle_f32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// VI: v_cmpx_gt_f32 vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0xa8,0x7c,0x01,0x00,0x02,0x04]
// GFX9: v_cmpx_gt_f32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0xa8,0x7c,0x01,0x00,0x02,0x04]
v_cmpx_gt_f32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// VI: v_cmpx_nlt_f32 vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0xbc,0x7c,0x01,0x00,0x02,0x04]
// GFX9: v_cmpx_nlt_f32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0xbc,0x7c,0x01,0x00,0x02,0x04]
v_cmpx_nlt_f32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// VI: v_cmp_lt_i32 vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0x82,0x7d,0x01,0x00,0x02,0x04]
// GFX9: v_cmp_lt_i32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0x82,0x7d,0x01,0x00,0x02,0x04]
v_cmp_lt_i32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// VI: v_cmp_t_i32 vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x00,0x02,0x04]
// GFX9: v_cmp_t_i32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0x8e,0x7d,0x01,0x00,0x02,0x04]
v_cmp_t_i32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// VI: v_cmpx_eq_i32 vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0xa4,0x7d,0x01,0x00,0x02,0x04]
// GFX9: v_cmpx_eq_i32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0xa4,0x7d,0x01,0x00,0x02,0x04]
v_cmpx_eq_i32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// VI: v_cmpx_ne_i32 vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0xaa,0x7d,0x01,0x00,0x02,0x04]
// GFX9: v_cmpx_ne_i32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0xaa,0x7d,0x01,0x00,0x02,0x04]
v_cmpx_ne_i32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// VI: v_cmp_f_u32 vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0x90,0x7d,0x01,0x00,0x02,0x04]
// GFX9: v_cmp_f_u32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0x90,0x7d,0x01,0x00,0x02,0x04]
v_cmp_f_u32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// VI: v_cmp_gt_u32 vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0x98,0x7d,0x01,0x00,0x02,0x04]
// GFX9: v_cmp_gt_u32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0x98,0x7d,0x01,0x00,0x02,0x04]
v_cmp_gt_u32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// VI: v_cmpx_le_u32 vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0xb6,0x7d,0x01,0x00,0x02,0x04]
// GFX9: v_cmpx_le_u32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0xb6,0x7d,0x01,0x00,0x02,0x04]
v_cmpx_le_u32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// VI: v_cmpx_ne_u32 vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0xba,0x7d,0x01,0x00,0x02,0x04]
// GFX9: v_cmpx_ne_u32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0xba,0x7d,0x01,0x00,0x02,0x04]
v_cmpx_ne_u32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// VI: v_cmp_class_f32 vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0x20,0x7c,0x01,0x00,0x02,0x04]
// GFX9: v_cmp_class_f32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0x20,0x7c,0x01,0x00,0x02,0x04]
v_cmp_class_f32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// VI: v_cmpx_class_f32 vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0x22,0x7c,0x01,0x00,0x02,0x04]
// GFX9: v_cmpx_class_f32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0 ; encoding: [0xf9,0x04,0x22,0x7c,0x01,0x00,0x02,0x04]
v_cmpx_class_f32_sdwa vcc, v1, v2 src0_sel:BYTE_2 src1_sel:WORD_0

//===----------------------------------------------------------------------===//
// Check GFX9-specific SDWA features
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// v_mac_f16/f32 is prohibited
//===----------------------------------------------------------------------===//

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: not a valid operand.
// VI: v_mac_f32_sdwa v3, v4, v5 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:DWORD ; encoding: [0xf9,0x0a,0x06,0x2c,0x04,0x16,0x05,0x06]
// NOGFX9: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand.
v_mac_f32 v3, v4, v5 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:WORD_1

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: not a valid operand.
// VI: v_mac_f32_sdwa v15, v99, v194 dst_sel:DWORD dst_unused:UNUSED_SEXT src0_sel:WORD_0 src1_sel:DWORD ; encoding: [0xf9,0x84,0x1f,0x2c,0x63,0x0e,0x04,0x06]
// NOGFX9: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand.
v_mac_f32 v15, v99, v194 dst_sel:DWORD dst_unused:UNUSED_SEXT src0_sel:WORD_0

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: not a valid operand.
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// NOGFX9: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand.
v_mac_f32 v194, v13, v1 dst_sel:BYTE_0 dst_unused:UNUSED_SEXT src0_sel:BYTE_3 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// VI: v_mac_f16_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x06,0x02,0x46,0x02,0x06,0x05,0x02]
// NOGFX9: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand.
v_mac_f16 v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

//===----------------------------------------------------------------------===//
// Scalar registers are allowed
//===----------------------------------------------------------------------===//

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: not a valid operand.
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_mov_b32_sdwa v1, s2 dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD ; encoding: [0xf9,0x02,0x02,0x7e,0x02,0x10,0x86,0x00]
v_mov_b32 v1, s2 dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: not a valid operand.
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_mov_b32_sdwa v1, exec_lo dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD ; encoding: [0xf9,0x02,0x02,0x7e,0x7e,0x10,0x86,0x00]
v_mov_b32 v1, exec_lo dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: register not available on this GPU
// GFX9: v_mov_b32_sdwa v1, ttmp12 dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD ; encoding: [0xf9,0x02,0x02,0x7e,0x78,0x10,0x86,0x00]
v_mov_b32_sdwa v1, ttmp12 dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: not a valid operand.
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_add_f32_sdwa v0, s0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x00,0x00,0x02,0x00,0x06,0x85,0x02]
v_add_f32 v0, s0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: not a valid operand.
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_add_f32_sdwa v0, v0, s22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x2c,0x00,0x02,0x00,0x06,0x05,0x82]
v_add_f32 v0, v0, s22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: not a valid operand.
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// NOGFX9: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_add_f32 v0, exec_lo, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: not a valid operand.
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// NOGFX9: :[[@LINE+1]]:{{[0-9]+}}: error: register not available on this GPU
v_add_f32 v0, v1, tba_lo dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: not a valid operand.
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// NOGFX9: :[[@LINE+1]]:{{[0-9]+}}: error: register not available on this GPU
v_add_f32 v0, v1, tma_hi dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_cmp_eq_f32_sdwa vcc, s1, v2 src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x04,0x84,0x7c,0x01,0x00,0x85,0x02]
v_cmp_eq_f32_sdwa vcc, s1, v2 src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_cmp_eq_f32_sdwa vcc, v1, s22 src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x2c,0x84,0x7c,0x01,0x00,0x05,0x82]
v_cmp_eq_f32_sdwa vcc, v1, s22 src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: register not available on this GPU
// GFX9: v_cmp_eq_f32_sdwa ttmp[12:13], v1, v2 src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x04,0x84,0x7c,0x01,0xf8,0x05,0x02]
v_cmp_eq_f32_sdwa ttmp[12:13], v1, v2 src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// NOGFX9: :[[@LINE+1]]:{{[0-9]+}}: error: register not available on this GPU
v_cmp_eq_f32_sdwa tba, v1, v2 src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// NOGFX9: :[[@LINE+1]]:{{[0-9]+}}: error: register not available on this GPU
v_cmp_eq_f32_sdwa tma, v1, v2 src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: register not available on this GPU
// GFX9: v_cmp_eq_f32_sdwa vcc, v1, ttmp15 src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0xf6,0x84,0x7c,0x01,0x00,0x05,0x82]
v_cmp_eq_f32_sdwa vcc, v1, ttmp15 src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// NOGFX9: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand (violates constant bus restrictions)
v_cmp_eq_f32_sdwa vcc, exec_lo, vcc_lo src0_sel:WORD_1 src1_sel:BYTE_2

// NOVI: :[[@LINE+4]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_ceil_f16_sdwa v5, flat_scratch_lo dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD ; encoding: [0xf9,0x8a,0x0a,0x7e,0x66,0x06,0x86,0x00]
// NOSI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOCI: :[[@LINE+1]]:{{[0-9]+}}: error: instruction not supported on this GPU
v_ceil_f16_sdwa v5, flat_scratch_lo dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD

//===----------------------------------------------------------------------===//
// Inline constants are allowed (though semantics is not clear yet)
//===----------------------------------------------------------------------===//

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_mov_b32_sdwa v5, 0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD ; encoding: [0xf9,0x02,0x0a,0x7e,0x80,0x06,0x86,0x00]
v_mov_b32_sdwa v5, 0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_mov_b32_sdwa v5, -1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD ; encoding: [0xf9,0x02,0x0a,0x7e,0xc1,0x06,0x86,0x00]
v_mov_b32_sdwa v5, -1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_mov_b32_sdwa v5, 0.5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD ; encoding: [0xf9,0x02,0x0a,0x7e,0xf0,0x06,0x86,0x00]
v_mov_b32_sdwa v5, 0.5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_mov_b32_sdwa v5, -4.0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD ; encoding: [0xf9,0x02,0x0a,0x7e,0xf7,0x06,0x86,0x00]
v_mov_b32_sdwa v5, -4.0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_mov_b32_sdwa v5, sext(-1) dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD ; encoding: [0xf9,0x02,0x0a,0x7e,0xc1,0x16,0x8e,0x00]
v_mov_b32_sdwa v5, sext(-1) dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_add_f32_sdwa v5, -1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x04,0x0a,0x02,0xc1,0x06,0x86,0x06]
v_add_f32_sdwa v5, -1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_add_f32_sdwa v5, |-1|, v2 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x04,0x0a,0x02,0xc1,0x16,0xa6,0x06]
v_add_f32_sdwa v5, |-1|, v2 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_add_f32_sdwa v5, neg(-1), -|v2| dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD  src1_sel:DWORD ; encoding: [0xf9,0x04,0x0a,0x02,0xc1,0x16,0x96,0x36]
v_add_f32_sdwa v5, neg(-1), -|v2| dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_add_f32_sdwa v5, -|-1|, v2 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x04,0x0a,0x02,0xc1,0x16,0xb6,0x06]
v_add_f32_sdwa v5, -|-1|, v2 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_add_f32_sdwa v5, 0.5, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x04,0x0a,0x02,0xf0,0x06,0x86,0x06]
v_add_f32_sdwa v5, 0.5, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_add_f32_sdwa v5, |-4.0|, v2 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x04,0x0a,0x02,0xf7,0x16,0xa6,0x06]
v_add_f32_sdwa v5, |-4.0|, v2 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_add_f32_sdwa v5, neg(-4.0), v2 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x04,0x0a,0x02,0xf7,0x16,0x96,0x06]
v_add_f32_sdwa v5, neg(-4.0), v2 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_add_f32_sdwa v5, -|-4.0|, v2 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x04,0x0a,0x02,0xf7,0x16,0xb6,0x06]
v_add_f32_sdwa v5, -|-4.0|, v2 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_add_f32_sdwa v5, v2, -4.0 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0xee,0x0b,0x02,0x02,0x16,0x06,0x86]
v_add_f32_sdwa v5, v2, -4.0 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_add_f32_sdwa v5, v2, |-4.0| dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0xee,0x0b,0x02,0x02,0x16,0x06,0xa6]
v_add_f32_sdwa v5, v2, |-4.0| dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_add_f32_sdwa v5, v2, neg(-4.0) dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0xee,0x0b,0x02,0x02,0x16,0x06,0x96]
v_add_f32_sdwa v5, v2, neg(-4.0) dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_add_f32_sdwa v5, v2, -|-4.0| dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0xee,0x0b,0x02,0x02,0x16,0x06,0xb6]
v_add_f32_sdwa v5, v2, -|-4.0| dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_add_f32_sdwa v5, v2, -1 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x82,0x0b,0x02,0x02,0x16,0x06,0x86]
v_add_f32_sdwa v5, v2, -1 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_add_f32_sdwa v5, v2, |-1| dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x82,0x0b,0x02,0x02,0x16,0x06,0xa6]
v_add_f32_sdwa v5, v2, |-1| dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_add_f32_sdwa v5, v2, neg(-1) dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x82,0x0b,0x02,0x02,0x16,0x06,0x96]
v_add_f32_sdwa v5, v2, neg(-1) dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_add_f32_sdwa v5, v2, -|-1| dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x82,0x0b,0x02,0x02,0x16,0x06,0xb6]
v_add_f32_sdwa v5, v2, -|-1| dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_and_b32_sdwa v5, -4.0, v2 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x04,0x0a,0x26,0xf7,0x16,0x86,0x06]
v_and_b32_sdwa v5, -4.0, v2 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_and_b32_sdwa v5, sext(-4.0), v2 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x04,0x0a,0x26,0xf7,0x16,0x8e,0x06]
v_and_b32_sdwa v5, sext(-4.0), v2 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_and_b32_sdwa v5, v2, -1 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x82,0x0b,0x26,0x02,0x16,0x06,0x86]
v_and_b32_sdwa v5, v2, -1 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_and_b32_sdwa v5, v2, sext(-1) dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x82,0x0b,0x26,0x02,0x16,0x06,0x8e]
v_and_b32_sdwa v5, v2, sext(-1) dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_exp_f16_sdwa v5, -1 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD ; encoding: [0xf9,0x82,0x0a,0x7e,0xc1,0x16,0x86,0x00]
v_exp_f16_sdwa v5, -1

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_exp_f16_sdwa v5, |-1| dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD ; encoding: [0xf9,0x82,0x0a,0x7e,0xc1,0x16,0xa6,0x00]
v_exp_f16_sdwa v5, |-1|

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_exp_f16_sdwa v5, neg(-1) dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD ; encoding: [0xf9,0x82,0x0a,0x7e,0xc1,0x16,0x96,0x00]
v_exp_f16_sdwa v5, neg(-1)

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_exp_f16_sdwa v5, -|-1| dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD ; encoding: [0xf9,0x82,0x0a,0x7e,0xc1,0x16,0xb6,0x00]
v_exp_f16_sdwa v5, -|-1|

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_exp_f16_sdwa v5, 0.5 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD ; encoding: [0xf9,0x82,0x0a,0x7e,0xf0,0x16,0x86,0x00]
v_exp_f16_sdwa v5, 0.5

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_exp_f16_sdwa v5, |0.5| dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD ; encoding: [0xf9,0x82,0x0a,0x7e,0xf0,0x16,0xa6,0x00]
v_exp_f16_sdwa v5, |0.5|

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_exp_f16_sdwa v5, neg(0.5) dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD ; encoding: [0xf9,0x82,0x0a,0x7e,0xf0,0x16,0x96,0x00]
v_exp_f16_sdwa v5, neg(0.5)

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_exp_f16_sdwa v5, -|0.5| dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD ; encoding: [0xf9,0x82,0x0a,0x7e,0xf0,0x16,0xb6,0x00]
v_exp_f16_sdwa v5, -|0.5|

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// NOGFX9: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_max_i16_sdwa v5, -4.0, v2 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// NOGFX9: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_max_i16_sdwa v5, sext(-4.0), v2 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_max_i16_sdwa v5, v2, -1 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x82,0x0b,0x60,0x02,0x16,0x06,0x86]
v_max_i16_sdwa v5, v2, -1 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_max_i16_sdwa v5, v2, sext(-1) dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x82,0x0b,0x60,0x02,0x16,0x06,0x8e]
v_max_i16_sdwa v5, v2, sext(-1) dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_cmp_eq_f32_sdwa s[6:7], -4.0, v2 src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x04,0x84,0x7c,0xf7,0x86,0x86,0x06]
v_cmp_eq_f32_sdwa s[6:7], -4.0, v2 src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_cmp_eq_f32_sdwa s[6:7], |-4.0|, v2 src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x04,0x84,0x7c,0xf7,0x86,0xa6,0x06]
v_cmp_eq_f32_sdwa s[6:7], |-4.0|, v2 src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_cmp_eq_f32_sdwa s[6:7], neg(-4.0), v2 src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x04,0x84,0x7c,0xf7,0x86,0x96,0x06]
v_cmp_eq_f32_sdwa s[6:7], neg(-4.0), v2 src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_cmp_eq_f32_sdwa s[6:7], -|-4.0|, v2 src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x04,0x84,0x7c,0xf7,0x86,0xb6,0x06]
v_cmp_eq_f32_sdwa s[6:7], -|-4.0|, v2 src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_cmp_eq_f32_sdwa s[6:7], v2, -1 src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x82,0x85,0x7c,0x02,0x86,0x06,0x86]
v_cmp_eq_f32_sdwa s[6:7], v2, -1 src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_cmp_eq_f32_sdwa s[6:7], v2, |-1| src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x82,0x85,0x7c,0x02,0x86,0x06,0xa6]
v_cmp_eq_f32_sdwa s[6:7], v2, |-1| src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_cmp_eq_f32_sdwa s[6:7], v2, neg(-1) src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x82,0x85,0x7c,0x02,0x86,0x06,0x96]
v_cmp_eq_f32_sdwa s[6:7], v2, neg(-1) src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_cmp_eq_f32_sdwa s[6:7], v2, -|-1| src0_sel:DWORD src1_sel:DWORD ; encoding: [0xf9,0x82,0x85,0x7c,0x02,0x86,0x06,0xb6]
v_cmp_eq_f32_sdwa s[6:7], v2, -|-1| src0_sel:DWORD src1_sel:DWORD

//===----------------------------------------------------------------------===//
// Literals are not allowed
//===----------------------------------------------------------------------===//

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// NOGFX89: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_add_f32 v0, v1, 3.45 src0_sel:BYTE_2 src1_sel:WORD_0

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// NOGFX89: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cmpx_class_f32 vcc, v1, 200 src0_sel:BYTE_2 src1_sel:WORD_0

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// NOGFX89: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_cmpx_class_f32 vcc, 200, v1 src0_sel:BYTE_2 src1_sel:WORD_0

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOGFX89: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_mov_b32_sdwa v5, -17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD

//===----------------------------------------------------------------------===//
// VOPC with arbitrary SGPR destination
//===----------------------------------------------------------------------===//

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// GFX9: v_cmp_eq_f32_sdwa s[2:3], v1, v2 src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x04,0x84,0x7c,0x01,0x82,0x05,0x02]
v_cmp_eq_f32_sdwa s[2:3], v1, v2 src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// GFX9: v_cmp_eq_f32_sdwa exec, v1, v2 src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x04,0x84,0x7c,0x01,0xfe,0x05,0x02]
v_cmp_eq_f32_sdwa exec, v1, v2 src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: invalid operand for instruction
// GFX9: v_cmp_eq_f32_sdwa exec, s2, v2 src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x04,0x84,0x7c,0x02,0xfe,0x85,0x02]
v_cmp_eq_f32_sdwa exec, s2, v2 src0_sel:WORD_1 src1_sel:BYTE_2

//===----------------------------------------------------------------------===//
// OMod output modifier allowed
//===----------------------------------------------------------------------===//

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: not a valid operand.
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// GFX9: v_trunc_f32_sdwa v1, v2 mul:2 dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD ; encoding: [0xf9,0x38,0x02,0x7e,0x02,0x50,0x06,0x00]
v_trunc_f32 v1, v2 mul:2 dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: not a valid operand.
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX9: v_trunc_f32_sdwa v1, v2 clamp div:2 dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD ; encoding: [0xf9,0x38,0x02,0x7e,0x02,0xf0,0x06,0x00]
v_trunc_f32 v1, v2 clamp div:2 dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: not a valid operand.
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// GFX9: v_add_f32_sdwa v0, v0, v0 mul:2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x00,0x00,0x02,0x00,0x46,0x05,0x02]
v_add_f32 v0, v0, v0 mul:2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: not a valid operand.
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// GFX9: v_add_f32_sdwa v0, v0, v0 clamp div:2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2 ; encoding: [0xf9,0x00,0x00,0x02,0x00,0xe6,0x05,0x02]
v_add_f32 v0, v0, v0 clamp div:2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

//---------------------------------------------------------------------------//
// Check Instructions
//---------------------------------------------------------------------------//

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX9: v_screen_partition_4se_b32_sdwa v5, v1 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:BYTE_0 ; encoding: [0xf9,0x6e,0x0a,0x7e,0x01,0x16,0x00,0x00]
v_screen_partition_4se_b32_sdwa v5, v1 src0_sel:BYTE_0

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOGFX89: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand.
v_cndmask_b32_sdwa v5, v1, sext(v2), vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
// NOGFX89: :[[@LINE+1]]:{{[0-9]+}}: error: not a valid operand.
v_cndmask_b32_sdwa v5, sext(v1), v2, vcc dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD

//===----------------------------------------------------------------------===//
// Validate register size checks (bug 37943)
//===----------------------------------------------------------------------===//

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// NOGFX89: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_add_f32 v0, s[0:1], v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// NOGFX89: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_add_f32 v0, s[0:3], v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// NOGFX89: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_add_f32 v0, v[0:1], v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// NOGFX89: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_add_f32 v0, v[0:2], v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// NOGFX89: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_add_f32 v0, v0, s[0:1] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// NOGFX89: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_add_f32 v0, s0, v[0:1] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand.
// NOGFX89: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_add_f32 v0, s0, v[0:3] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOGFX89: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_add_f16 v1, v[2:3], v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOGFX89: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_add_f16 v1, s[2:3], v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOGFX89: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_add_f16 v1, v2, v[2:3] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+2]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOGFX89: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_add_f16 v1, v2, s[2:3] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand
// NOGFX9: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_add_u32 v1, v[2:3], v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand
// NOGFX9: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_add_u32 v1, s[2:3], v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand
// NOGFX9: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_add_u32 v1, v3, v[2:3] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2

// NOSICI: :[[@LINE+3]]:{{[0-9]+}}: error: instruction not supported on this GPU
// NOVI: :[[@LINE+2]]:{{[0-9]+}}: error: not a valid operand
// NOGFX9: :[[@LINE+1]]:{{[0-9]+}}: error: invalid operand for instruction
v_add_u32 v1, v3, s[2:3] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:BYTE_2
