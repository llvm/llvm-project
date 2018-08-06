// RUN: not llvm-mc -arch=amdgcn -show-encoding %s | FileCheck --check-prefix=GCN --check-prefix=SICI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=fiji -show-encoding %s 2>&1 | FileCheck --check-prefix=GCN --check-prefix=VI --check-prefix=GFX89 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding %s 2>&1 | FileCheck --check-prefix=GCN --check-prefix=GFX89 --check-prefix=GFX9 %s

// RUN: not llvm-mc -arch=amdgcn -show-encoding %s 2>&1 | FileCheck --check-prefix=NOSICI --check-prefix=NOSICIVI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=fiji -show-encoding %s 2>&1 | FileCheck --check-prefix=NOVI --check-prefix=NOSICIVI --check-prefix=NOGFX89 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding %s 2>&1 | FileCheck --check-prefix=NOGFX89 %s

s_mov_b32 s1, s2
// SICI: s_mov_b32 s1, s2 ; encoding: [0x02,0x03,0x81,0xbe]
// GFX89: s_mov_b32 s1, s2 ; encoding: [0x02,0x00,0x81,0xbe]

s_mov_b32 s1, 1
// SICI: s_mov_b32 s1, 1 ; encoding: [0x81,0x03,0x81,0xbe]
// GFX89: s_mov_b32 s1, 1 ; encoding: [0x81,0x00,0x81,0xbe]

s_mov_b32 s1, 100
// SICI: s_mov_b32 s1, 0x64 ; encoding: [0xff,0x03,0x81,0xbe,0x64,0x00,0x00,0x00]
// GFX89: s_mov_b32 s1, 0x64 ; encoding: [0xff,0x00,0x81,0xbe,0x64,0x00,0x00,0x00]

// Literal constant sign bit
s_mov_b32 s1, 0x80000000
// SICI: s_mov_b32 s1, 0x80000000 ; encoding: [0xff,0x03,0x81,0xbe,0x00,0x00,0x00,0x80]
// GFX89: s_mov_b32 s1, 0x80000000 ; encoding: [0xff,0x00,0x81,0xbe,0x00,0x00,0x00,0x80]

// Negative 32-bit constant
s_mov_b32 s0, 0xfe5163ab
// SICI: s_mov_b32 s0, 0xfe5163ab ; encoding: [0xff,0x03,0x80,0xbe,0xab,0x63,0x51,0xfe]
// GFX89: s_mov_b32 s0, 0xfe5163ab ; encoding: [0xff,0x00,0x80,0xbe,0xab,0x63,0x51,0xfe]

s_mov_b64 s[2:3], s[4:5]
// SICI: s_mov_b64 s[2:3], s[4:5] ; encoding: [0x04,0x04,0x82,0xbe]
// GFX89: s_mov_b64 s[2:3], s[4:5] ; encoding: [0x04,0x01,0x82,0xbe]

s_mov_b64 s[2:3], 0xffffffffffffffff
// SICI: s_mov_b64 s[2:3], -1 ; encoding: [0xc1,0x04,0x82,0xbe]
// GFX89: s_mov_b64 s[2:3], -1 ; encoding: [0xc1,0x01,0x82,0xbe]

s_mov_b64 s[2:3], 0xffffffff
// SICI: s_mov_b64 s[2:3], 0xffffffff ; encoding: [0xff,0x04,0x82,0xbe,0xff,0xff,0xff,0xff]
// GFX89: s_mov_b64 s[2:3], 0xffffffff ; encoding: [0xff,0x01,0x82,0xbe,0xff,0xff,0xff,0xff]

s_mov_b64 s[0:1], 0x80000000
// SICI: s_mov_b64 s[0:1], 0x80000000 ; encoding: [0xff,0x04,0x80,0xbe,0x00,0x00,0x00,0x80]
// GFX89: s_mov_b64 s[0:1], 0x80000000 ; encoding: [0xff,0x01,0x80,0xbe,0x00,0x00,0x00,0x80]

s_mov_b64 s[102:103], -1
// SICI: s_mov_b64 s[102:103], -1 ; encoding: [0xc1,0x04,0xe6,0xbe]
// NOGFX89: error: not a valid operand

s_cmov_b32 s1, 200
// SICI: s_cmov_b32 s1, 0xc8 ; encoding: [0xff,0x05,0x81,0xbe,0xc8,0x00,0x00,0x00]
// GFX89: s_cmov_b32 s1, 0xc8 ; encoding: [0xff,0x02,0x81,0xbe,0xc8,0x00,0x00,0x00]

s_cmov_b32 s1, 1.0
// SICI: s_cmov_b32 s1, 1.0 ; encoding: [0xf2,0x05,0x81,0xbe]
// GFX89: s_cmov_b32 s1, 1.0 ; encoding: [0xf2,0x02,0x81,0xbe]

s_cmov_b32 s1, s2
// SICI: s_cmov_b32 s1, s2 ; encoding: [0x02,0x05,0x81,0xbe]
// GFX89: s_cmov_b32 s1, s2 ; encoding: [0x02,0x02,0x81,0xbe]

//s_cmov_b64 s[2:3], 1.0
//GCN-FIXME: s_cmov_b64 s[2:3], 1.0 ; encoding: [0xf2,0x05,0x82,0xb3]

s_cmov_b64 s[2:3], s[4:5]
// SICI: s_cmov_b64 s[2:3], s[4:5] ; encoding: [0x04,0x06,0x82,0xbe]
// GFX89: s_cmov_b64 s[2:3], s[4:5] ; encoding: [0x04,0x03,0x82,0xbe]

s_not_b32 s1, s2
// SICI: s_not_b32 s1, s2 ; encoding: [0x02,0x07,0x81,0xbe]
// GFX89: s_not_b32 s1, s2 ; encoding: [0x02,0x04,0x81,0xbe]

s_not_b64 s[2:3], s[4:5]
// SICI: s_not_b64 s[2:3], s[4:5] ; encoding: [0x04,0x08,0x82,0xbe]
// GFX89: s_not_b64 s[2:3], s[4:5] ; encoding: [0x04,0x05,0x82,0xbe]

s_wqm_b32 s1, s2
// SICI: s_wqm_b32 s1, s2 ; encoding: [0x02,0x09,0x81,0xbe]
// GFX89: s_wqm_b32 s1, s2 ; encoding: [0x02,0x06,0x81,0xbe]

s_wqm_b64 s[2:3], s[4:5]
// SICI: s_wqm_b64 s[2:3], s[4:5] ; encoding: [0x04,0x0a,0x82,0xbe]
// GFX89: s_wqm_b64 s[2:3], s[4:5] ; encoding: [0x04,0x07,0x82,0xbe]

s_brev_b32 s1, s2
// SICI: s_brev_b32 s1, s2 ; encoding: [0x02,0x0b,0x81,0xbe]
// GFX89: s_brev_b32 s1, s2 ; encoding: [0x02,0x08,0x81,0xbe]

s_brev_b64 s[2:3], s[4:5]
// SICI: s_brev_b64 s[2:3], s[4:5] ; encoding: [0x04,0x0c,0x82,0xbe]
// GFX89: s_brev_b64 s[2:3], s[4:5] ; encoding: [0x04,0x09,0x82,0xbe]

s_bcnt0_i32_b32 s1, s2
// SICI: s_bcnt0_i32_b32 s1, s2 ; encoding: [0x02,0x0d,0x81,0xbe]
// GFX89: s_bcnt0_i32_b32 s1, s2 ; encoding: [0x02,0x0a,0x81,0xbe]

s_bcnt0_i32_b64 s1, s[2:3]
// SICI: s_bcnt0_i32_b64 s1, s[2:3] ; encoding: [0x02,0x0e,0x81,0xbe]
// GFX89: s_bcnt0_i32_b64 s1, s[2:3] ; encoding: [0x02,0x0b,0x81,0xbe]

s_bcnt1_i32_b32 s1, s2
// SICI: s_bcnt1_i32_b32 s1, s2 ; encoding: [0x02,0x0f,0x81,0xbe]
// GFX89: s_bcnt1_i32_b32 s1, s2 ; encoding: [0x02,0x0c,0x81,0xbe]

s_bcnt1_i32_b64 s1, s[2:3]
// SICI: s_bcnt1_i32_b64 s1, s[2:3] ; encoding: [0x02,0x10,0x81,0xbe]
// GFX89: s_bcnt1_i32_b64 s1, s[2:3] ; encoding: [0x02,0x0d,0x81,0xbe]

s_ff0_i32_b32 s1, s2
// SICI: s_ff0_i32_b32 s1, s2 ; encoding: [0x02,0x11,0x81,0xbe]
// GFX89: s_ff0_i32_b32 s1, s2 ; encoding: [0x02,0x0e,0x81,0xbe]

s_ff0_i32_b64 s1, s[2:3]
// SICI: s_ff0_i32_b64 s1, s[2:3] ; encoding: [0x02,0x12,0x81,0xbe]
// GFX89: s_ff0_i32_b64 s1, s[2:3] ; encoding: [0x02,0x0f,0x81,0xbe]

s_ff1_i32_b32 s1, s2
// SICI: s_ff1_i32_b32 s1, s2 ; encoding: [0x02,0x13,0x81,0xbe]
// GFX89: s_ff1_i32_b32 s1, s2 ; encoding: [0x02,0x10,0x81,0xbe]

s_ff1_i32_b64 s1, s[2:3]
// SICI: s_ff1_i32_b64 s1, s[2:3] ; encoding: [0x02,0x14,0x81,0xbe]
// GFX89: s_ff1_i32_b64 s1, s[2:3] ; encoding: [0x02,0x11,0x81,0xbe]

s_flbit_i32_b32 s1, s2
// SICI: s_flbit_i32_b32 s1, s2 ; encoding: [0x02,0x15,0x81,0xbe]
// GFX89: s_flbit_i32_b32 s1, s2 ; encoding: [0x02,0x12,0x81,0xbe]

s_flbit_i32_b64 s1, s[2:3]
// SICI: s_flbit_i32_b64 s1, s[2:3] ; encoding: [0x02,0x16,0x81,0xbe]
// GFX89: s_flbit_i32_b64 s1, s[2:3] ; encoding: [0x02,0x13,0x81,0xbe]

s_flbit_i32 s1, s2
// SICI: s_flbit_i32 s1, s2 ; encoding: [0x02,0x17,0x81,0xbe]
// GFX89: s_flbit_i32 s1, s2 ; encoding: [0x02,0x14,0x81,0xbe]

s_flbit_i32_i64 s1, s[2:3]
// SICI: s_flbit_i32_i64 s1, s[2:3] ; encoding: [0x02,0x18,0x81,0xbe]
// GFX89: s_flbit_i32_i64 s1, s[2:3] ; encoding: [0x02,0x15,0x81,0xbe]

s_sext_i32_i8 s1, s2
// SICI: s_sext_i32_i8 s1, s2 ; encoding: [0x02,0x19,0x81,0xbe]
// GFX89: s_sext_i32_i8 s1, s2 ; encoding: [0x02,0x16,0x81,0xbe]

s_sext_i32_i16 s1, s2
// SICI: s_sext_i32_i16 s1, s2 ; encoding: [0x02,0x1a,0x81,0xbe]
// GFX89: s_sext_i32_i16 s1, s2 ; encoding: [0x02,0x17,0x81,0xbe]

s_bitset0_b32 s1, s2
// SICI: s_bitset0_b32 s1, s2 ; encoding: [0x02,0x1b,0x81,0xbe]
// GFX89: s_bitset0_b32 s1, s2 ; encoding: [0x02,0x18,0x81,0xbe]

s_bitset0_b64 s[2:3], s4
// SICI: s_bitset0_b64 s[2:3], s4 ; encoding: [0x04,0x1c,0x82,0xbe]
// GFX89: s_bitset0_b64 s[2:3], s4 ; encoding: [0x04,0x19,0x82,0xbe]

s_bitset1_b32 s1, s2
// SICI: s_bitset1_b32 s1, s2 ; encoding: [0x02,0x1d,0x81,0xbe]
// GFX89: s_bitset1_b32 s1, s2 ; encoding: [0x02,0x1a,0x81,0xbe]

s_bitset1_b64 s[2:3], s4
// SICI: s_bitset1_b64 s[2:3], s4 ; encoding: [0x04,0x1e,0x82,0xbe]
// GFX89: s_bitset1_b64 s[2:3], s4 ; encoding: [0x04,0x1b,0x82,0xbe]

s_getpc_b64 s[2:3]
// SICI: s_getpc_b64 s[2:3] ; encoding: [0x00,0x1f,0x82,0xbe]
// GFX89: s_getpc_b64 s[2:3] ; encoding: [0x00,0x1c,0x82,0xbe]

s_setpc_b64 s[4:5]
// SICI: s_setpc_b64 s[4:5] ; encoding: [0x04,0x20,0x80,0xbe]
// GFX89: s_setpc_b64 s[4:5] ; encoding: [0x04,0x1d,0x80,0xbe]

s_swappc_b64 s[2:3], s[4:5]
// SICI: s_swappc_b64 s[2:3], s[4:5] ; encoding: [0x04,0x21,0x82,0xbe]
// GFX89: s_swappc_b64 s[2:3], s[4:5] ; encoding: [0x04,0x1e,0x82,0xbe]

s_rfe_b64 s[4:5]
// SICI: s_rfe_b64 s[4:5] ; encoding: [0x04,0x22,0x80,0xbe]
// GFX89: s_rfe_b64 s[4:5] ; encoding: [0x04,0x1f,0x80,0xbe]

s_and_saveexec_b64 s[2:3], s[4:5]
// SICI: s_and_saveexec_b64 s[2:3], s[4:5] ; encoding: [0x04,0x24,0x82,0xbe]
// GFX89: s_and_saveexec_b64 s[2:3], s[4:5] ; encoding: [0x04,0x20,0x82,0xbe]

s_or_saveexec_b64 s[2:3], s[4:5]
// SICI: s_or_saveexec_b64 s[2:3], s[4:5] ; encoding: [0x04,0x25,0x82,0xbe]
// GFX89: s_or_saveexec_b64 s[2:3], s[4:5] ; encoding: [0x04,0x21,0x82,0xbe]

s_xor_saveexec_b64 s[2:3], s[4:5]
// SICI: s_xor_saveexec_b64 s[2:3], s[4:5] ; encoding: [0x04,0x26,0x82,0xbe]
// GFX89: s_xor_saveexec_b64 s[2:3], s[4:5] ; encoding: [0x04,0x22,0x82,0xbe]

s_andn2_saveexec_b64 s[2:3], s[4:5]
// SICI: s_andn2_saveexec_b64 s[2:3], s[4:5] ; encoding: [0x04,0x27,0x82,0xbe]
// GFX89: s_andn2_saveexec_b64 s[2:3], s[4:5] ; encoding: [0x04,0x23,0x82,0xbe]

s_orn2_saveexec_b64 s[2:3], s[4:5]
// SICI: s_orn2_saveexec_b64 s[2:3], s[4:5] ; encoding: [0x04,0x28,0x82,0xbe]
// GFX89: s_orn2_saveexec_b64 s[2:3], s[4:5] ; encoding: [0x04,0x24,0x82,0xbe]

s_nand_saveexec_b64 s[2:3], s[4:5]
// SICI: s_nand_saveexec_b64 s[2:3], s[4:5] ; encoding: [0x04,0x29,0x82,0xbe]
// GFX89: s_nand_saveexec_b64 s[2:3], s[4:5] ; encoding: [0x04,0x25,0x82,0xbe]

s_nor_saveexec_b64 s[2:3], s[4:5]
// SICI: s_nor_saveexec_b64 s[2:3], s[4:5] ; encoding: [0x04,0x2a,0x82,0xbe]
// GFX89: s_nor_saveexec_b64 s[2:3], s[4:5] ; encoding: [0x04,0x26,0x82,0xbe]

s_xnor_saveexec_b64 s[2:3], s[4:5]
// SICI: s_xnor_saveexec_b64 s[2:3], s[4:5] ; encoding: [0x04,0x2b,0x82,0xbe]
// GFX89: s_xnor_saveexec_b64 s[2:3], s[4:5] ; encoding: [0x04,0x27,0x82,0xbe]

s_quadmask_b32 s1, s2
// SICI: s_quadmask_b32 s1, s2 ; encoding: [0x02,0x2c,0x81,0xbe]
// GFX89: s_quadmask_b32 s1, s2 ; encoding: [0x02,0x28,0x81,0xbe]

s_quadmask_b64 s[2:3], s[4:5]
// SICI: s_quadmask_b64 s[2:3], s[4:5] ; encoding: [0x04,0x2d,0x82,0xbe]
// GFX89: s_quadmask_b64 s[2:3], s[4:5] ; encoding: [0x04,0x29,0x82,0xbe]

s_movrels_b32 s1, s2
// SICI: s_movrels_b32 s1, s2 ; encoding: [0x02,0x2e,0x81,0xbe]
// GFX89: s_movrels_b32 s1, s2 ; encoding: [0x02,0x2a,0x81,0xbe]

s_movrels_b64 s[2:3], s[4:5]
// SICI: s_movrels_b64 s[2:3], s[4:5] ; encoding: [0x04,0x2f,0x82,0xbe]
// GFX89: s_movrels_b64 s[2:3], s[4:5] ; encoding: [0x04,0x2b,0x82,0xbe]

s_movreld_b32 s1, s2
// SICI: s_movreld_b32 s1, s2 ; encoding: [0x02,0x30,0x81,0xbe]
// GFX89: s_movreld_b32 s1, s2 ; encoding: [0x02,0x2c,0x81,0xbe]

s_movreld_b64 s[2:3], s[4:5]
// SICI: s_movreld_b64 s[2:3], s[4:5] ; encoding: [0x04,0x31,0x82,0xbe]
// GFX89: s_movreld_b64 s[2:3], s[4:5] ; encoding: [0x04,0x2d,0x82,0xbe]

s_cbranch_join s4
// SICI: s_cbranch_join s4 ; encoding: [0x04,0x32,0x80,0xbe]
// GFX89: s_cbranch_join s4 ; encoding: [0x04,0x2e,0x80,0xbe]

s_cbranch_join 1
// NOSICI: error: invalid operand for instruction
// NOGFX89: error: invalid operand for instruction

s_cbranch_join 100
// NOSICI: error: invalid operand for instruction
// NOGFX89: error: invalid operand for instruction

s_abs_i32 s1, s2
// SICI: s_abs_i32 s1, s2 ; encoding: [0x02,0x34,0x81,0xbe]
// GFX89: s_abs_i32 s1, s2 ; encoding: [0x02,0x30,0x81,0xbe]

s_mov_fed_b32 s1, s2
// SICI: s_mov_fed_b32 s1, s2 ; encoding: [0x02,0x35,0x81,0xbe]

s_set_gpr_idx_idx s0
// GFX89: s_set_gpr_idx_idx s0 ; encoding: [0x00,0x32,0x80,0xbe]
// NOSICI: error: instruction not supported on this GPU

s_andn1_saveexec_b64 s[100:101], s[2:3]
// GFX9:     s_andn1_saveexec_b64 s[100:101], s[2:3] ; encoding: [0x02,0x33,0xe4,0xbe]
// NOSICIVI: error: instruction not supported on this GPU

s_andn1_saveexec_b64 s[10:11], s[4:5]
// GFX9:     s_andn1_saveexec_b64 s[10:11], s[4:5] ; encoding: [0x04,0x33,0x8a,0xbe]
// NOSICIVI: error: instruction not supported on this GPU

s_andn1_saveexec_b64 s[10:11], -1
// GFX9:     s_andn1_saveexec_b64 s[10:11], -1 ; encoding: [0xc1,0x33,0x8a,0xbe]
// NOSICIVI: error: instruction not supported on this GPU

s_andn1_saveexec_b64 s[10:11], 0xaf123456
// GFX9:     s_andn1_saveexec_b64 s[10:11], 0xaf123456 ; encoding: [0xff,0x33,0x8a,0xbe,0x56,0x34,0x12,0xaf]
// NOSICIVI: error: instruction not supported on this GPU

s_andn1_wrexec_b64 s[10:11], s[2:3]
// GFX9:     s_andn1_wrexec_b64 s[10:11], s[2:3] ; encoding: [0x02,0x35,0x8a,0xbe]
// NOSICIVI: error: instruction not supported on this GPU

s_andn2_wrexec_b64 s[12:13], s[2:3]
// GFX9:     s_andn2_wrexec_b64 s[12:13], s[2:3] ; encoding: [0x02,0x36,0x8c,0xbe]
// NOSICIVI: error: instruction not supported on this GPU

s_orn1_saveexec_b64 s[10:11], 0
// GFX9:     s_orn1_saveexec_b64 s[10:11], 0 ; encoding: [0x80,0x34,0x8a,0xbe]
// NOSICIVI: error: instruction not supported on this GPU

s_bitreplicate_b64_b32 s[10:11], s101
// GFX9:     s_bitreplicate_b64_b32 s[10:11], s101 ; encoding: [0x65,0x37,0x8a,0xbe]
// NOSICIVI: error: instruction not supported on this GPU

s_bitreplicate_b64_b32 s[10:11], -1
// GFX9:     s_bitreplicate_b64_b32 s[10:11], -1 ; encoding: [0xc1,0x37,0x8a,0xbe]
// NOSICIVI: error: instruction not supported on this GPU

s_bitreplicate_b64_b32 s[10:11], 0x3f717273
// GFX9:     s_bitreplicate_b64_b32 s[10:11], 0x3f717273 ; encoding: [0xff,0x37,0x8a,0xbe,0x73,0x72,0x71,0x3f]
// NOSICIVI: error: instruction not supported on this GPU
