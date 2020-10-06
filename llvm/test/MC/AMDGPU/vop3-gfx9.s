// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding %s | FileCheck -check-prefix=GFX9 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck -check-prefixes=NOSI,NOSICI,NOGCN --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=hawaii %s 2>&1 | FileCheck -check-prefixes=NOCI,NOSICI,NOGCN --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck -check-prefixes=NOVI,NOGCN --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck -check-prefix=NOGFX9 --implicit-check-not=error: %s

v_lshl_add_u32 v1, v2, v3, v4
// GFX9: v_lshl_add_u32 v1, v2, v3, v4 ; encoding: [0x01,0x00,0xfd,0xd1,0x02,0x07,0x12,0x04]
// NOGCN: :1: error: instruction not supported on this GPU

v_add_lshl_u32 v1, v2, v3, v4
// GFX9: v_add_lshl_u32 v1, v2, v3, v4 ; encoding: [0x01,0x00,0xfe,0xd1,0x02,0x07,0x12,0x04]
// NOGCN: :1: error: instruction not supported on this GPU

v_add3_u32 v1, v2, v3, v4
// GFX9: v_add3_u32 v1, v2, v3, v4 ; encoding: [0x01,0x00,0xff,0xd1,0x02,0x07,0x12,0x04]
// NOGCN: :1: error: instruction not supported on this GPU

v_lshl_or_b32 v1, v2, v3, v4
// GFX9: v_lshl_or_b32 v1, v2, v3, v4 ; encoding: [0x01,0x00,0x00,0xd2,0x02,0x07,0x12,0x04]
// NOGCN: :1: error: instruction not supported on this GPU

v_and_or_b32 v1, v2, v3, v4
// GFX9: v_and_or_b32 v1, v2, v3, v4 ; encoding: [0x01,0x00,0x01,0xd2,0x02,0x07,0x12,0x04]
// NOGCN: :1: error: instruction not supported on this GPU

v_or3_b32 v1, v2, v3, v4
// GFX9: v_or3_b32 v1, v2, v3, v4 ; encoding: [0x01,0x00,0x02,0xd2,0x02,0x07,0x12,0x04]
// NOGCN: :1: error: instruction not supported on this GPU

v_pack_b32_f16 v1, v2, v3
// GFX9: v_pack_b32_f16 v1, v2, v3 ; encoding: [0x01,0x00,0xa0,0xd2,0x02,0x07,0x02,0x00]
// NOGCN: :1: error: instruction not supported on this GPU

v_pack_b32_f16 v5, v1, v2 op_sel:[1,0,0]
// GFX9: v_pack_b32_f16 v5, v1, v2 op_sel:[1,0,0] ; encoding: [0x05,0x08,0xa0,0xd2,0x01,0x05,0x02,0x00]
// NOGCN: error: instruction not supported on this GPU

v_pack_b32_f16 v5, v1, v2 op_sel:[0,1,0]
// GFX9: v_pack_b32_f16 v5, v1, v2 op_sel:[0,1,0] ; encoding: [0x05,0x10,0xa0,0xd2,0x01,0x05,0x02,0x00]
// NOGCN: error: instruction not supported on this GPU

v_pack_b32_f16 v5, v1, v2 op_sel:[0,0,1]
// GFX9: v_pack_b32_f16 v5, v1, v2 op_sel:[0,0,1] ; encoding: [0x05,0x40,0xa0,0xd2,0x01,0x05,0x02,0x00]
// NOGCN: error: instruction not supported on this GPU

v_xad_u32 v1, v2, v3, v4
// GFX9: v_xad_u32 v1, v2, v3, v4 ; encoding: [0x01,0x00,0xf3,0xd1,0x02,0x07,0x12,0x04]
// NOGCN: :1: error: instruction not supported on this GPU

v_min3_f16 v1, v2, v3, v4
// GFX9: v_min3_f16 v1, v2, v3, v4 ; encoding: [0x01,0x00,0xf4,0xd1,0x02,0x07,0x12,0x04]
// NOGCN: :1: error: instruction not supported on this GPU

v_min3_i16 v1, v2, v3, v4
// GFX9: v_min3_i16 v1, v2, v3, v4 ; encoding: [0x01,0x00,0xf5,0xd1,0x02,0x07,0x12,0x04]
// NOGCN: :1: error: instruction not supported on this GPU

v_min3_u16 v1, v2, v3, v4
// GFX9: v_min3_u16 v1, v2, v3, v4 ; encoding: [0x01,0x00,0xf6,0xd1,0x02,0x07,0x12,0x04]
// NOGCN: :1: error: instruction not supported on this GPU

v_max3_f16 v1, v2, v3, v4
// GFX9: v_max3_f16 v1, v2, v3, v4 ; encoding: [0x01,0x00,0xf7,0xd1,0x02,0x07,0x12,0x04]
// NOGCN: :1: error: instruction not supported on this GPU

v_max3_f16 v5, v1, v2, v3 op_sel:[0,0,0,0]
// GFX9: v_max3_f16 v5, v1, v2, v3 ; encoding: [0x05,0x00,0xf7,0xd1,0x01,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_max3_f16 v5, v1, v2, v3 op_sel:[1,0,0,0]
// GFX9: v_max3_f16 v5, v1, v2, v3 op_sel:[1,0,0,0] ; encoding: [0x05,0x08,0xf7,0xd1,0x01,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_max3_f16 v5, v1, v2, v3 op_sel:[0,1,0,0]
// GFX9: v_max3_f16 v5, v1, v2, v3 op_sel:[0,1,0,0] ; encoding: [0x05,0x10,0xf7,0xd1,0x01,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_max3_f16 v5, v1, v2, v3 op_sel:[0,0,1,0]
// GFX9: v_max3_f16 v5, v1, v2, v3 op_sel:[0,0,1,0] ; encoding: [0x05,0x20,0xf7,0xd1,0x01,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_max3_f16 v5, v1, v2, v3 op_sel:[0,0,0,1]
// GFX9: v_max3_f16 v5, v1, v2, v3 op_sel:[0,0,0,1] ; encoding: [0x05,0x40,0xf7,0xd1,0x01,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_max3_f16 v5, v1, v2, v3 op_sel:[1,1,1,1]
// GFX9: v_max3_f16 v5, v1, v2, v3 op_sel:[1,1,1,1] ; encoding: [0x05,0x78,0xf7,0xd1,0x01,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_max3_i16 v1, v2, v3, v4
// GFX9: v_max3_i16 v1, v2, v3, v4 ; encoding: [0x01,0x00,0xf8,0xd1,0x02,0x07,0x12,0x04]
// NOGCN: :1: error: instruction not supported on this GPU

v_max3_i16 v5, v1, v2, v3 op_sel:[0,0,0,0]
// GFX9: v_max3_i16 v5, v1, v2, v3 ; encoding: [0x05,0x00,0xf8,0xd1,0x01,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_max3_i16 v5, v1, v2, v3 op_sel:[1,0,0,0]
// GFX9: v_max3_i16 v5, v1, v2, v3 op_sel:[1,0,0,0] ; encoding: [0x05,0x08,0xf8,0xd1,0x01,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_max3_i16 v5, v1, v2, v3 op_sel:[0,1,0,0]
// GFX9: v_max3_i16 v5, v1, v2, v3 op_sel:[0,1,0,0] ; encoding: [0x05,0x10,0xf8,0xd1,0x01,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_max3_i16 v5, v1, v2, v3 op_sel:[0,0,1,0]
// GFX9: v_max3_i16 v5, v1, v2, v3 op_sel:[0,0,1,0] ; encoding: [0x05,0x20,0xf8,0xd1,0x01,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_max3_i16 v5, v1, v2, v3 op_sel:[0,0,0,1]
// GFX9: v_max3_i16 v5, v1, v2, v3 op_sel:[0,0,0,1] ; encoding: [0x05,0x40,0xf8,0xd1,0x01,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_max3_i16 v5, v1, v2, v3 op_sel:[1,1,1,1]
// GFX9: v_max3_i16 v5, v1, v2, v3 op_sel:[1,1,1,1] ; encoding: [0x05,0x78,0xf8,0xd1,0x01,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_max3_u16 v1, v2, v3, v4
// GFX9: v_max3_u16 v1, v2, v3, v4 ; encoding: [0x01,0x00,0xf9,0xd1,0x02,0x07,0x12,0x04]
// NOGCN: :1: error: instruction not supported on this GPU

v_med3_f16 v1, v2, v3, v4
// GFX9: v_med3_f16 v1, v2, v3, v4 ; encoding: [0x01,0x00,0xfa,0xd1,0x02,0x07,0x12,0x04]
// NOGCN: :1: error: instruction not supported on this GPU

v_med3_i16 v1, v2, v3, v4
// GFX9: v_med3_i16 v1, v2, v3, v4 ; encoding: [0x01,0x00,0xfb,0xd1,0x02,0x07,0x12,0x04]
// NOGCN: :1: error: instruction not supported on this GPU

v_med3_u16 v1, v2, v3, v4
// GFX9: v_med3_u16 v1, v2, v3, v4 ; encoding: [0x01,0x00,0xfc,0xd1,0x02,0x07,0x12,0x04]
// NOGCN: :1: error: instruction not supported on this GPU

v_mad_u32_u16 v5, v1, v2, v3
// GFX9: v_mad_u32_u16 v5, v1, v2, v3 ; encoding: [0x05,0x00,0xf1,0xd1,0x01,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_mad_u32_u16 v5, v1, v2, v3 op_sel:[1,0,0,0]
// GFX9: v_mad_u32_u16 v5, v1, v2, v3 op_sel:[1,0,0,0] ; encoding: [0x05,0x08,0xf1,0xd1,0x01,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_mad_u32_u16 v5, v1, v2, v3 op_sel:[0,1,0,0]
// GFX9: v_mad_u32_u16 v5, v1, v2, v3 op_sel:[0,1,0,0] ; encoding: [0x05,0x10,0xf1,0xd1,0x01,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_mad_u32_u16 v5, v1, v2, v3 op_sel:[0,0,1,0]
// GFX9: v_mad_u32_u16 v5, v1, v2, v3 op_sel:[0,0,1,0] ; encoding: [0x05,0x20,0xf1,0xd1,0x01,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_mad_u32_u16 v5, v1, v2, v3 op_sel:[0,0,0,1]
// GFX9: v_mad_u32_u16 v5, v1, v2, v3 op_sel:[0,0,0,1] ; encoding: [0x05,0x40,0xf1,0xd1,0x01,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_mad_u32_u16 v5, v1, v2, v3 op_sel:[1,1,1,1]
// GFX9: v_mad_u32_u16 v5, v1, v2, v3 op_sel:[1,1,1,1] ; encoding: [0x05,0x78,0xf1,0xd1,0x01,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_mad_i32_i16 v5, v1, v2, v3
// GFX9: v_mad_i32_i16 v5, v1, v2, v3 ; encoding: [0x05,0x00,0xf2,0xd1,0x01,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_mad_i32_i16 v5, v1, v2, v3 op_sel:[0,0,0,1]
// GFX9: v_mad_i32_i16 v5, v1, v2, v3 op_sel:[0,0,0,1] ; encoding: [0x05,0x40,0xf2,0xd1,0x01,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_cvt_pknorm_i16_f16 v5, v1, v2
// GFX9: v_cvt_pknorm_i16_f16 v5, v1, v2 ; encoding: [0x05,0x00,0x99,0xd2,0x01,0x05,0x02,0x00]
// NOGCN: error: instruction not supported on this GPU

v_cvt_pknorm_i16_f16 v5, -v1, v2
// GFX9: v_cvt_pknorm_i16_f16 v5, -v1, v2 ; encoding: [0x05,0x00,0x99,0xd2,0x01,0x05,0x02,0x20]
// NOGCN: error: instruction not supported on this GPU

v_cvt_pknorm_i16_f16 v5, v1, -v2
// GFX9: v_cvt_pknorm_i16_f16 v5, v1, -v2 ; encoding: [0x05,0x00,0x99,0xd2,0x01,0x05,0x02,0x40]
// NOGCN: error: instruction not supported on this GPU

v_cvt_pknorm_i16_f16 v5, -v1, -v2
// GFX9: v_cvt_pknorm_i16_f16 v5, -v1, -v2 ; encoding: [0x05,0x00,0x99,0xd2,0x01,0x05,0x02,0x60]
// NOGCN: error: instruction not supported on this GPU

v_cvt_pknorm_i16_f16 v5, |v1|, v2
// GFX9: v_cvt_pknorm_i16_f16 v5, |v1|, v2 ; encoding: [0x05,0x01,0x99,0xd2,0x01,0x05,0x02,0x00]
// NOGCN: error: instruction not supported on this GPU

v_cvt_pknorm_i16_f16 v5, v1, |v2|
// GFX9: v_cvt_pknorm_i16_f16 v5, v1, |v2| ; encoding: [0x05,0x02,0x99,0xd2,0x01,0x05,0x02,0x00]
// NOGCN: error: instruction not supported on this GPU

v_cvt_pknorm_i16_f16 v5, v1, v2 op_sel:[0,0,0]
// GFX9: v_cvt_pknorm_i16_f16 v5, v1, v2 ; encoding: [0x05,0x00,0x99,0xd2,0x01,0x05,0x02,0x00]
// NOGCN: error: instruction not supported on this GPU

v_cvt_pknorm_i16_f16 v5, v1, v2 op_sel:[1,0,0]
// GFX9: v_cvt_pknorm_i16_f16 v5, v1, v2 op_sel:[1,0,0] ; encoding: [0x05,0x08,0x99,0xd2,0x01,0x05,0x02,0x00]
// NOGCN: error: instruction not supported on this GPU

v_cvt_pknorm_i16_f16 v5, v1, v2 op_sel:[1,1,1]
// GFX9: v_cvt_pknorm_i16_f16 v5, v1, v2 op_sel:[1,1,1] ; encoding: [0x05,0x58,0x99,0xd2,0x01,0x05,0x02,0x00]
// NOGCN: error: instruction not supported on this GPU

v_cvt_pknorm_u16_f16 v5, -v1, -v2
// GFX9: v_cvt_pknorm_u16_f16 v5, -v1, -v2 ; encoding: [0x05,0x00,0x9a,0xd2,0x01,0x05,0x02,0x60]
// NOGCN: error: instruction not supported on this GPU

v_cvt_pknorm_u16_f16 v5, |v1|, |v2|
// GFX9: v_cvt_pknorm_u16_f16 v5, |v1|, |v2| ; encoding: [0x05,0x03,0x9a,0xd2,0x01,0x05,0x02,0x00]
// NOGCN: error: instruction not supported on this GPU

v_cvt_pknorm_u16_f16 v5, v1, v2 op_sel:[1,1,1]
// GFX9: v_cvt_pknorm_u16_f16 v5, v1, v2 op_sel:[1,1,1] ; encoding: [0x05,0x58,0x9a,0xd2,0x01,0x05,0x02,0x00]
// NOGCN: error: instruction not supported on this GPU

v_add_i16 v5, v1, v2
// GFX9: v_add_i16 v5, v1, v2 ; encoding: [0x05,0x00,0x9e,0xd2,0x01,0x05,0x02,0x00]
// NOGCN: error: instruction not supported on this GPU

v_add_i16 v5, v1, v2 op_sel:[1,1,1]
// GFX9: v_add_i16 v5, v1, v2 op_sel:[1,1,1] ; encoding: [0x05,0x58,0x9e,0xd2,0x01,0x05,0x02,0x00]
// NOGCN: error: instruction not supported on this GPU

v_sub_i16 v5, v1, v2
// GFX9: v_sub_i16 v5, v1, v2 ; encoding: [0x05,0x00,0x9f,0xd2,0x01,0x05,0x02,0x00]
// NOGCN: error: instruction not supported on this GPU

v_sub_i16 v5, v1, v2 op_sel:[1,1,1]
// GFX9: v_sub_i16 v5, v1, v2 op_sel:[1,1,1] ; encoding: [0x05,0x58,0x9f,0xd2,0x01,0x05,0x02,0x00]
// NOGCN: error: instruction not supported on this GPU

v_sub_i16 v5, v1, v2 clamp
// GFX9: v_sub_i16 v5, v1, v2 clamp ; encoding: [0x05,0x80,0x9f,0xd2,0x01,0x05,0x02,0x00]
// NOGCN: error: instruction not supported on this GPU

v_fma_f16_e64 v5, v1, v2, v3
// GFX9: v_fma_f16 v5, v1, v2, v3 ; encoding: [0x05,0x00,0x06,0xd2,0x01,0x05,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU

v_fma_f16 v5, v1, -v2, v3
// GFX9: v_fma_f16 v5, v1, -v2, v3 ; encoding: [0x05,0x00,0x06,0xd2,0x01,0x05,0x0e,0x44]
// NOSICI: error: instruction not supported on this GPU

v_fma_f16 v5, v1, v2, |v3|
// GFX9: v_fma_f16 v5, v1, v2, |v3| ; encoding: [0x05,0x04,0x06,0xd2,0x01,0x05,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU

v_fma_f16 v5, v1, v2, v3 clamp
// GFX9: v_fma_f16 v5, v1, v2, v3 clamp ; encoding: [0x05,0x80,0x06,0xd2,0x01,0x05,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU

v_fma_f16 v5, v1, v2, v3 op_sel:[1,0,0,0]
// GFX9: v_fma_f16 v5, v1, v2, v3 op_sel:[1,0,0,0] ; encoding: [0x05,0x08,0x06,0xd2,0x01,0x05,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: instruction not supported on this GPU

v_fma_f16 v5, v1, v2, v3 op_sel:[0,1,0,0]
// GFX9: v_fma_f16 v5, v1, v2, v3 op_sel:[0,1,0,0] ; encoding: [0x05,0x10,0x06,0xd2,0x01,0x05,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: instruction not supported on this GPU

v_fma_f16 v5, v1, v2, v3 op_sel:[1,1,1,1]
// GFX9: v_fma_f16 v5, v1, v2, v3 op_sel:[1,1,1,1] ; encoding: [0x05,0x78,0x06,0xd2,0x01,0x05,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: instruction not supported on this GPU

v_fma_legacy_f16_e64 v5, v1, v2, v3
// GFX9: v_fma_legacy_f16 v5, v1, v2, v3 ; encoding:  [0x05,0x00,0xee,0xd1,0x01,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_fma_legacy_f16 v5, -v1, v2, v3
// GFX9: v_fma_legacy_f16 v5, -v1, v2, v3 ; encoding:  [0x05,0x00,0xee,0xd1,0x01,0x05,0x0e,0x24]
// NOGCN: error: instruction not supported on this GPU

v_fma_legacy_f16 v5, v1, |v2|, v3
// GFX9: v_fma_legacy_f16 v5, v1, |v2|, v3 ; encoding:  [0x05,0x02,0xee,0xd1,0x01,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_fma_legacy_f16 v5, v1, v2, v3 clamp
// GFX9: v_fma_legacy_f16 v5, v1, v2, v3 clamp ; encoding:  [0x05,0x80,0xee,0xd1,0x01,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_div_fixup_f16_e64 v5, 0.5, v2, v3
// GFX9: v_div_fixup_f16 v5, 0.5, v2, v3 ; encoding: [0x05,0x00,0x07,0xd2,0xf0,0x04,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU

v_div_fixup_f16 v5, v1, 0.5, v3
// GFX9: v_div_fixup_f16 v5, v1, 0.5, v3 ; encoding: [0x05,0x00,0x07,0xd2,0x01,0xe1,0x0d,0x04]
// NOSICI: error: instruction not supported on this GPU

v_div_fixup_f16 v5, v1, v2, 0.5
// GFX9: v_div_fixup_f16 v5, v1, v2, 0.5 ; encoding: [0x05,0x00,0x07,0xd2,0x01,0x05,0xc2,0x03]
// NOSICI: error: instruction not supported on this GPU

v_div_fixup_f16 v5, -v1, v2, v3
// GFX9: v_div_fixup_f16 v5, -v1, v2, v3 ; encoding: [0x05,0x00,0x07,0xd2,0x01,0x05,0x0e,0x24]
// NOSICI: error: instruction not supported on this GPU

v_div_fixup_f16 v5, |v1|, v2, v3
// GFX9: v_div_fixup_f16 v5, |v1|, v2, v3 ; encoding: [0x05,0x01,0x07,0xd2,0x01,0x05,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU

v_div_fixup_f16 v5, v1, v2, v3 clamp
// GFX9: v_div_fixup_f16 v5, v1, v2, v3 clamp ; encoding: [0x05,0x80,0x07,0xd2,0x01,0x05,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU

v_div_fixup_f16 v5, v1, v2, v3 op_sel:[1,0,0,0]
// GFX9: v_div_fixup_f16 v5, v1, v2, v3 op_sel:[1,0,0,0] ; encoding: [0x05,0x08,0x07,0xd2,0x01,0x05,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: instruction not supported on this GPU

v_div_fixup_f16 v5, v1, v2, v3 op_sel:[0,0,1,0]
// GFX9: v_div_fixup_f16 v5, v1, v2, v3 op_sel:[0,0,1,0] ; encoding: [0x05,0x20,0x07,0xd2,0x01,0x05,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: instruction not supported on this GPU

v_div_fixup_f16 v5, v1, v2, v3 op_sel:[0,0,0,1]
// GFX9: v_div_fixup_f16 v5, v1, v2, v3 op_sel:[0,0,0,1] ; encoding: [0x05,0x40,0x07,0xd2,0x01,0x05,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: instruction not supported on this GPU

v_div_fixup_legacy_f16_e64 v5, 0.5, v2, v3
// GFX9: v_div_fixup_legacy_f16 v5, 0.5, v2, v3 ; encoding: [0x05,0x00,0xef,0xd1,0xf0,0x04,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_div_fixup_legacy_f16 v5, v1, 0.5, v3
// GFX9: v_div_fixup_legacy_f16 v5, v1, 0.5, v3 ; encoding: [0x05,0x00,0xef,0xd1,0x01,0xe1,0x0d,0x04]
// NOGCN: error: instruction not supported on this GPU

v_div_fixup_legacy_f16 v5, v1, v2, 0.5
// GFX9: v_div_fixup_legacy_f16 v5, v1, v2, 0.5 ; encoding: [0x05,0x00,0xef,0xd1,0x01,0x05,0xc2,0x03]
// NOGCN: error: instruction not supported on this GPU

v_div_fixup_legacy_f16 v5, -v1, v2, v3
// GFX9: v_div_fixup_legacy_f16 v5, -v1, v2, v3 ; encoding: [0x05,0x00,0xef,0xd1,0x01,0x05,0x0e,0x24]
// NOGCN: error: instruction not supported on this GPU

v_div_fixup_legacy_f16 v5, v1, |v2|, v3
// GFX9: v_div_fixup_legacy_f16 v5, v1, |v2|, v3 ; encoding: [0x05,0x02,0xef,0xd1,0x01,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_div_fixup_legacy_f16 v5, v1, v2, v3 clamp
// GFX9: v_div_fixup_legacy_f16 v5, v1, v2, v3 clamp ; encoding: [0x05,0x80,0xef,0xd1,0x01,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_mad_f16_e64 v5, 0.5, v2, v3
// GFX9: v_mad_f16 v5, 0.5, v2, v3 ; encoding: [0x05,0x00,0x03,0xd2,0xf0,0x04,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU

v_mad_f16 v5, v1, 0.5, v3
// GFX9: v_mad_f16 v5, v1, 0.5, v3 ; encoding: [0x05,0x00,0x03,0xd2,0x01,0xe1,0x0d,0x04]
// NOSICI: error: instruction not supported on this GPU

v_mad_f16 v5, v1, v2, 0.5
// GFX9: v_mad_f16 v5, v1, v2, 0.5 ; encoding: [0x05,0x00,0x03,0xd2,0x01,0x05,0xc2,0x03]
// NOSICI: error: instruction not supported on this GPU

v_mad_f16 v5, v1, v2, -v3
// GFX9: v_mad_f16 v5, v1, v2, -v3 ; encoding: [0x05,0x00,0x03,0xd2,0x01,0x05,0x0e,0x84]
// NOSICI: error: instruction not supported on this GPU

v_mad_f16 v5, v1, v2, |v3|
// GFX9: v_mad_f16 v5, v1, v2, |v3| ; encoding: [0x05,0x04,0x03,0xd2,0x01,0x05,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU

v_mad_f16 v5, v1, v2, v3 op_sel:[0,0,0,0]
// GFX9: v_mad_f16 v5, v1, v2, v3 ; encoding: [0x05,0x00,0x03,0xd2,0x01,0x05,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: instruction not supported on this GPU

v_mad_f16 v5, v1, v2, v3 op_sel:[1,0,0,0]
// GFX9: v_mad_f16 v5, v1, v2, v3 op_sel:[1,0,0,0] ; encoding: [0x05,0x08,0x03,0xd2,0x01,0x05,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: instruction not supported on this GPU

v_mad_f16 v5, v1, v2, v3 op_sel:[0,1,0,0]
// GFX9: v_mad_f16 v5, v1, v2, v3 op_sel:[0,1,0,0] ; encoding: [0x05,0x10,0x03,0xd2,0x01,0x05,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: instruction not supported on this GPU

v_mad_f16 v5, v1, v2, v3 op_sel:[0,0,1,0]
// GFX9: v_mad_f16 v5, v1, v2, v3 op_sel:[0,0,1,0] ; encoding: [0x05,0x20,0x03,0xd2,0x01,0x05,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: instruction not supported on this GPU

v_mad_f16 v5, v1, v2, v3 op_sel:[0,0,0,1]
// GFX9: v_mad_f16 v5, v1, v2, v3 op_sel:[0,0,0,1] ; encoding: [0x05,0x40,0x03,0xd2,0x01,0x05,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: instruction not supported on this GPU

v_mad_f16 v5, v1, v2, v3 op_sel:[1,1,1,1]
// GFX9: v_mad_f16 v5, v1, v2, v3 op_sel:[1,1,1,1] ; encoding: [0x05,0x78,0x03,0xd2,0x01,0x05,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: instruction not supported on this GPU

v_mad_f16 v5, v1, v2, v3 clamp
// GFX9: v_mad_f16 v5, v1, v2, v3 clamp ; encoding: [0x05,0x80,0x03,0xd2,0x01,0x05,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU

v_mad_i16_e64 v5, 0, v2, v3
// GFX9: v_mad_i16 v5, 0, v2, v3 ; encoding: [0x05,0x00,0x05,0xd2,0x80,0x04,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU

v_mad_i16 v5, v1, -1, v3
// GFX9: v_mad_i16 v5, v1, -1, v3 ; encoding: [0x05,0x00,0x05,0xd2,0x01,0x83,0x0d,0x04]
// NOSICI: error: instruction not supported on this GPU

v_mad_i16 v5, v1, v2, -4.0
// NOGFX9: error: invalid literal operand
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: invalid literal operand

v_mad_i16 v5, v1, v2, v3 clamp
// GFX9: v_mad_i16 v5, v1, v2, v3 clamp ; encoding: [0x05,0x80,0x05,0xd2,0x01,0x05,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU

v_mad_i16 v5, v1, v2, v3 op_sel:[0,0,0,1]
// GFX9: v_mad_i16 v5, v1, v2, v3 op_sel:[0,0,0,1] ; encoding: [0x05,0x40,0x05,0xd2,0x01,0x05,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: instruction not supported on this GPU

v_mad_i16 v5, v1, v2, v3 op_sel:[1,1,1,1]
// GFX9: v_mad_i16 v5, v1, v2, v3 op_sel:[1,1,1,1] ; encoding: [0x05,0x78,0x05,0xd2,0x01,0x05,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: instruction not supported on this GPU

v_mad_legacy_f16_e64 v5, 0.5, v2, v3
// GFX9: v_mad_legacy_f16 v5, 0.5, v2, v3 ; encoding: [0x05,0x00,0xea,0xd1,0xf0,0x04,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_mad_legacy_f16 v5, v1, 0.5, v3
// GFX9: v_mad_legacy_f16 v5, v1, 0.5, v3 ; encoding: [0x05,0x00,0xea,0xd1,0x01,0xe1,0x0d,0x04]
// NOGCN: error: instruction not supported on this GPU

v_mad_legacy_f16 v5, v1, v2, 0.5
// GFX9: v_mad_legacy_f16 v5, v1, v2, 0.5 ; encoding: [0x05,0x00,0xea,0xd1,0x01,0x05,0xc2,0x03]
// NOGCN: error: instruction not supported on this GPU

v_mad_legacy_f16 v5, v1, -v2, v3
// GFX9: v_mad_legacy_f16 v5, v1, -v2, v3 ; encoding: [0x05,0x00,0xea,0xd1,0x01,0x05,0x0e,0x44]
// NOGCN: error: instruction not supported on this GPU

v_mad_legacy_f16 v5, v1, |v2|, v3
// GFX9: v_mad_legacy_f16 v5, v1, |v2|, v3 ; encoding: [0x05,0x02,0xea,0xd1,0x01,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_mad_legacy_f16 v5, v1, v2, v3 clamp
// GFX9: v_mad_legacy_f16 v5, v1, v2, v3 clamp ; encoding: [0x05,0x80,0xea,0xd1,0x01,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_mad_legacy_i16_e64 v5, 0, v2, v3
// GFX9: v_mad_legacy_i16 v5, 0, v2, v3 ; encoding: [0x05,0x00,0xec,0xd1,0x80,0x04,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_mad_legacy_i16 v5, v1, -1, v3
// GFX9: v_mad_legacy_i16 v5, v1, -1, v3 ; encoding: [0x05,0x00,0xec,0xd1,0x01,0x83,0x0d,0x04]
// NOGCN: error: instruction not supported on this GPU

v_mad_legacy_i16 v5, v1, v2, -4.0
// NOGFX9: error: invalid literal operand
// NOGCN: error: instruction not supported on this GPU

v_mad_legacy_i16 v5, v1, v2, -4.0 clamp
// NOGFX9: error: invalid literal operand
// NOGCN: error: instruction not supported on this GPU

v_mad_legacy_u16_e64 v5, 0, v2, v3
// GFX9: v_mad_legacy_u16 v5, 0, v2, v3 ; encoding: [0x05,0x00,0xeb,0xd1,0x80,0x04,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_mad_legacy_u16 v5, v1, -1, v3
// GFX9: v_mad_legacy_u16 v5, v1, -1, v3 ; encoding: [0x05,0x00,0xeb,0xd1,0x01,0x83,0x0d,0x04]
// NOGCN: error: instruction not supported on this GPU

v_mad_legacy_u16 v5, v1, v2, -4.0
// NOGFX9: error: invalid literal operand
// NOGCN: error: instruction not supported on this GPU

v_mad_legacy_u16 v5, v1, v2, -4.0 clamp
// NOGFX9: error: invalid literal operand
// NOGCN: error: instruction not supported on this GPU

v_mad_u16_e64 v5, 0, v2, v3
// GFX9: v_mad_u16 v5, 0, v2, v3 ; encoding: [0x05,0x00,0x04,0xd2,0x80,0x04,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU

v_mad_u16 v5, v1, -1, v3
// GFX9: v_mad_u16 v5, v1, -1, v3 ; encoding: [0x05,0x00,0x04,0xd2,0x01,0x83,0x0d,0x04]
// NOSICI: error: instruction not supported on this GPU

v_mad_u16 v5, v1, v2, -4.0
// NOGFX9: error: invalid literal operand
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: invalid literal operand

v_mad_u16 v5, v1, v2, v3 clamp
// GFX9: v_mad_u16 v5, v1, v2, v3 clamp ; encoding: [0x05,0x80,0x04,0xd2,0x01,0x05,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU

v_mad_u16 v5, v1, v2, v3 op_sel:[1,0,0,0]
// GFX9: v_mad_u16 v5, v1, v2, v3 op_sel:[1,0,0,0] ; encoding: [0x05,0x08,0x04,0xd2,0x01,0x05,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: instruction not supported on this GPU

v_mad_u16 v5, v1, v2, v3 op_sel:[0,0,0,1]
// GFX9: v_mad_u16 v5, v1, v2, v3 op_sel:[0,0,0,1] ; encoding: [0x05,0x40,0x04,0xd2,0x01,0x05,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: instruction not supported on this GPU

v_mad_u16 v5, v1, v2, v3 op_sel:[1,1,1,1]
// GFX9: v_mad_u16 v5, v1, v2, v3 op_sel:[1,1,1,1] ; encoding: [0x05,0x78,0x04,0xd2,0x01,0x05,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: instruction not supported on this GPU

v_interp_p2_f16 v5, v2, attr0.x, v3
// GFX9: v_interp_p2_f16 v5, v2, attr0.x, v3 ; encoding: [0x05,0x00,0x77,0xd2,0x00,0x04,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU

v_interp_p2_f16 v5, -v2, attr0.x, v3
// GFX9: v_interp_p2_f16 v5, -v2, attr0.x, v3 ; encoding: [0x05,0x00,0x77,0xd2,0x00,0x04,0x0e,0x44]
// NOSICI: error: instruction not supported on this GPU

v_interp_p2_f16 v5, v2, attr0.x, |v3|
// GFX9: v_interp_p2_f16 v5, v2, attr0.x, |v3| ; encoding: [0x05,0x04,0x77,0xd2,0x00,0x04,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU

v_interp_p2_f16 v5, v2, attr0.w, v3
// GFX9: v_interp_p2_f16 v5, v2, attr0.w, v3 ; encoding: [0x05,0x00,0x77,0xd2,0xc0,0x04,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU

v_interp_p2_f16 v5, v2, attr0.x, v3 high
// GFX9: v_interp_p2_f16 v5, v2, attr0.x, v3 high ; encoding: [0x05,0x00,0x77,0xd2,0x00,0x05,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU

v_interp_p2_f16 v5, v2, attr0.x, v3 clamp
// GFX9: v_interp_p2_f16 v5, v2, attr0.x, v3 clamp ; encoding: [0x05,0x80,0x77,0xd2,0x00,0x04,0x0e,0x04]
// NOSICI: error: instruction not supported on this GPU

v_interp_p2_legacy_f16 v5, v2, attr31.x, v3
// GFX9: v_interp_p2_legacy_f16 v5, v2, attr31.x, v3 ; encoding: [0x05,0x00,0x76,0xd2,0x1f,0x04,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_interp_p2_legacy_f16 v5, -v2, attr0.x, v3
// GFX9: v_interp_p2_legacy_f16 v5, -v2, attr0.x, v3 ; encoding: [0x05,0x00,0x76,0xd2,0x00,0x04,0x0e,0x44]
// NOGCN: error: instruction not supported on this GPU

v_interp_p2_legacy_f16 v5, v2, attr0.x, |v3|
// GFX9: v_interp_p2_legacy_f16 v5, v2, attr0.x, |v3| ; encoding: [0x05,0x04,0x76,0xd2,0x00,0x04,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_interp_p2_legacy_f16 v5, v2, attr0.w, v3
// GFX9: v_interp_p2_legacy_f16 v5, v2, attr0.w, v3 ; encoding: [0x05,0x00,0x76,0xd2,0xc0,0x04,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_interp_p2_legacy_f16 v5, v2, attr0.x, v3 high
// GFX9: v_interp_p2_legacy_f16 v5, v2, attr0.x, v3 high ; encoding: [0x05,0x00,0x76,0xd2,0x00,0x05,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_interp_p2_legacy_f16 v5, v2, attr0.x, v3 clamp
// GFX9: v_interp_p2_legacy_f16 v5, v2, attr0.x, v3 clamp ; encoding: [0x05,0x80,0x76,0xd2,0x00,0x04,0x0e,0x04]
// NOGCN: error: instruction not supported on this GPU

v_cvt_norm_i16_f16_e64 v5, -v1
// GFX9: v_cvt_norm_i16_f16_e64 v5, -v1 ; encoding: [0x05,0x00,0x8d,0xd1,0x01,0x01,0x00,0x20]
// NOGCN: error: instruction not supported on this GPU

v_cvt_norm_i16_f16_e64 v5, |v1|
// GFX9: v_cvt_norm_i16_f16_e64 v5, |v1| ; encoding: [0x05,0x01,0x8d,0xd1,0x01,0x01,0x00,0x00]
// NOGCN: error: instruction not supported on this GPU

v_cvt_norm_u16_f16_e64 v5, -v1
// GFX9: v_cvt_norm_u16_f16_e64 v5, -v1 ; encoding: [0x05,0x00,0x8e,0xd1,0x01,0x01,0x00,0x20]
// NOGCN: error: instruction not supported on this GPU

v_cvt_norm_u16_f16_e64 v5, |v1|
// GFX9: v_cvt_norm_u16_f16_e64 v5, |v1| ; encoding: [0x05,0x01,0x8e,0xd1,0x01,0x01,0x00,0x00]
// NOGCN: error: instruction not supported on this GPU

v_sat_pk_u8_i16_e64 v5, -1
// GFX9: v_sat_pk_u8_i16_e64 v5, -1 ; encoding: [0x05,0x00,0x8f,0xd1,0xc1,0x00,0x00,0x00]
// NOGCN: error: instruction not supported on this GPU

v_sat_pk_u8_i16_e64 v5, v255
// GFX9: v_sat_pk_u8_i16_e64 v5, v255 ; encoding: [0x05,0x00,0x8f,0xd1,0xff,0x01,0x00,0x00]
// NOGCN: error: instruction not supported on this GPU

v_screen_partition_4se_b32_e64 v5, v1
// NOGCN: error: instruction not supported on this GPU
// GFX9: v_screen_partition_4se_b32_e64 v5, v1 ; encoding: [0x05,0x00,0x77,0xd1,0x01,0x01,0x00,0x00]

v_screen_partition_4se_b32_e64 v5, -1
// NOGCN: error: instruction not supported on this GPU
// GFX9: v_screen_partition_4se_b32_e64 v5, -1 ; encoding: [0x05,0x00,0x77,0xd1,0xc1,0x00,0x00,0x00]

v_add_u32 v84, v13, s31 clamp
// GFX9: v_add_u32_e64 v84, v13, s31 clamp ; encoding: [0x54,0x80,0x34,0xd1,0x0d,0x3f,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: invalid operand for instruction

v_sub_u32 v84, v13, s31 clamp
// GFX9: v_sub_u32_e64 v84, v13, s31 clamp ; encoding: [0x54,0x80,0x35,0xd1,0x0d,0x3f,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: invalid operand for instruction

v_subrev_u32 v84, v13, s31 clamp
// GFX9: v_subrev_u32_e64 v84, v13, s31 clamp ; encoding: [0x54,0x80,0x36,0xd1,0x0d,0x3f,0x00,0x00]
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: invalid operand for instruction

v_addc_co_u32 v84, s[4:5], v13, v31, vcc clamp
// GFX9: v_addc_co_u32_e64 v84, s[4:5], v13, v31, vcc clamp ; encoding: [0x54,0x84,0x1c,0xd1,0x0d,0x3f,0xaa,0x01]
// NOGCN: error: instruction not supported on this GPU

v_subb_co_u32 v84, s[2:3], v13, v31, vcc clamp
// GFX9: v_subb_co_u32_e64 v84, s[2:3], v13, v31, vcc clamp ; encoding: [0x54,0x82,0x1d,0xd1,0x0d,0x3f,0xaa,0x01]
// NOGCN: error: instruction not supported on this GPU

v_subbrev_co_u32 v84, vcc, v13, v31, s[6:7] clamp
// GFX9: v_subbrev_co_u32_e64 v84, vcc, v13, v31, s[6:7] clamp ; encoding: [0x54,0xea,0x1e,0xd1,0x0d,0x3f,0x1a,0x00]
// NOGCN: error: instruction not supported on this GPU

v_add_co_u32 v84, s[4:5], v13, v31 clamp
// GFX9: v_add_co_u32_e64 v84, s[4:5], v13, v31 clamp ; encoding: [0x54,0x84,0x19,0xd1,0x0d,0x3f,0x02,0x00]
// NOSICI: error: integer clamping is not supported on this GPU
// NOVI: error: instruction not supported on this GPU

v_sub_co_u32 v84, s[2:3], v13, v31 clamp
// GFX9: v_sub_co_u32_e64 v84, s[2:3], v13, v31 clamp ; encoding: [0x54,0x82,0x1a,0xd1,0x0d,0x3f,0x02,0x00]
// NOSICI: error: integer clamping is not supported on this GPU
// NOVI: error: instruction not supported on this GPU

v_subrev_co_u32 v84, vcc, v13, v31 clamp
// GFX9: v_subrev_co_u32_e64 v84, vcc, v13, v31 clamp ; encoding: [0x54,0xea,0x1b,0xd1,0x0d,0x3f,0x02,0x00]
// NOSICI: error: integer clamping is not supported on this GPU
// NOVI: error: instruction not supported on this GPU

v_addc_co_u32 v84, vcc, v13, v31, vcc
// GFX9: v_addc_co_u32_e32 v84, vcc, v13, v31, vcc ; encoding: [0x0d,0x3f,0xa8,0x38]
// NOGCN: error: instruction not supported on this GPU

v_subb_co_u32 v84, vcc, v13, v31, vcc
// GFX9: v_subb_co_u32_e32 v84, vcc, v13, v31, vcc ; encoding: [0x0d,0x3f,0xa8,0x3a]
// NOGCN: error: instruction not supported on this GPU

v_subbrev_co_u32 v84, vcc, v13, v31, vcc
// GFX9: v_subbrev_co_u32_e32 v84, vcc, v13, v31, vcc ; encoding: [0x0d,0x3f,0xa8,0x3c]
// NOGCN: error: instruction not supported on this GPU

v_add_co_u32 v84, vcc, v13, v31
// GFX9: v_add_co_u32_e32 v84, vcc, v13, v31 ; encoding: [0x0d,0x3f,0xa8,0x32]
// NOVI: error: instruction not supported on this GPU

v_sub_co_u32 v84, vcc, v13, v31
// GFX9: v_sub_co_u32_e32 v84, vcc, v13, v31 ; encoding: [0x0d,0x3f,0xa8,0x34]
// NOVI: error: instruction not supported on this GPU

v_subrev_co_u32 v84, vcc, v13, v31
// GFX9: v_subrev_co_u32_e32 v84, vcc, v13, v31 ; encoding: [0x0d,0x3f,0xa8,0x36]
// NOVI: error: instruction not supported on this GPU

v_add_i32 v1, v2, v3
// GFX9: v_add_i32 v1, v2, v3 ; encoding: [0x01,0x00,0x9c,0xd2,0x02,0x07,0x02,0x00]
// NOGCN: error: instruction not supported on this GPU

v_add_i32 v1, v2, v3 clamp
// GFX9: v_add_i32 v1, v2, v3 clamp ; encoding: [0x01,0x80,0x9c,0xd2,0x02,0x07,0x02,0x00]
// NOSICI: error: invalid operand for instruction
// NOVI: error: instruction not supported on this GPU

v_sub_i32 v1, v2, v3
// GFX9: v_sub_i32 v1, v2, v3 ; encoding: [0x01,0x00,0x9d,0xd2,0x02,0x07,0x02,0x00]
// NOGCN: error: instruction not supported on this GPU

v_sub_i32 v1, v2, v3 clamp
// GFX9: v_sub_i32 v1, v2, v3 clamp ; encoding: [0x01,0x80,0x9d,0xd2,0x02,0x07,0x02,0x00]
// NOSICI: error: invalid operand for instruction
// NOVI: error: instruction not supported on this GPU

//===----------------------------------------------------------------------===//
// Validate register size checks (bug 37943)
//===----------------------------------------------------------------------===//

// NOGCN: error: invalid operand for instruction
// NOGFX9: error: invalid operand for instruction
v_add_f64 v[0:1], s0, v[0:1]

// NOGCN: error: invalid operand for instruction
// NOGFX9: error: invalid operand for instruction
v_add_f64 v[0:1], s[0:3], v[0:1]

// NOGCN: error: invalid operand for instruction
// NOGFX9: error: invalid operand for instruction
v_add_f64 v[0:1], v0, v[0:1]

// NOGCN: error: invalid operand for instruction
// NOGFX9: error: invalid operand for instruction
v_add_f64 v[0:1], v[0:2], v[0:1]

// NOGCN: error: invalid operand for instruction
// NOGFX9: error: invalid operand for instruction
v_add_f64 v[0:1], v[0:3], v[0:1]

// NOGCN: error: invalid operand for instruction
// NOGFX9: error: invalid operand for instruction
v_add_f64 v[0:1], v[0:1], v0

// NOGCN: error: invalid operand for instruction
// NOGFX9: error: invalid operand for instruction
v_add_f64 v[0:1], v[0:1], s0

// NOGCN: error: invalid operand for instruction
// NOGFX9: error: invalid operand for instruction
v_add_f32 v0, s[0:1], v0

// NOGCN: error: invalid operand for instruction
// NOGFX9: error: invalid operand for instruction
v_add_f32 v0, v[0:1], v0

// NOGCN: error: invalid operand for instruction
// NOGFX9: error: invalid operand for instruction
v_add_f32 v0, v0, s[0:1]

// NOGCN: error: invalid operand for instruction
// NOGFX9: error: invalid operand for instruction
v_add_f32 v0, v0, v[0:1]

// NOGFX9: error: invalid operand for instruction
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: invalid operand for instruction
v_add_f16 v0, s[0:1], v0

// NOGFX9: error: invalid operand for instruction
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: invalid operand for instruction
v_add_f16 v0, v[0:1], v0

// NOGFX9: error: invalid operand for instruction
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: invalid operand for instruction
v_add_f16 v0, v0, s[0:1]

// NOGFX9: error: invalid operand for instruction
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: invalid operand for instruction
v_add_f16 v0, v0, v[0:1]

// NOGFX9: error: invalid operand for instruction
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: invalid operand for instruction
v_add_u16 v0, s[0:1], v0

// NOGFX9: error: invalid operand for instruction
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: invalid operand for instruction
v_add_u16 v0, v[0:1], v0

// NOGFX9: error: invalid operand for instruction
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: invalid operand for instruction
v_add_u16 v0, v0, s[0:1]

// NOGFX9: error: invalid operand for instruction
// NOSICI: error: instruction not supported on this GPU
// NOVI: error: invalid operand for instruction
v_add_u16 v0, v0, v[0:1]

