// RUN: not llvm-mc -arch=amdgcn -show-encoding %s | FileCheck --check-prefix=GCN --check-prefix=SICI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti -show-encoding %s | FileCheck --check-prefix=GCN --check-prefix=SICI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=bonaire -show-encoding %s | FileCheck --check-prefix=GCN --check-prefix=SICI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=fiji -show-encoding %s | FileCheck --check-prefix=GCN --check-prefix=GFX89 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding %s | FileCheck --check-prefix=GCN --check-prefix=GFX89 --check-prefix=GFX9 %s

// RUN: not llvm-mc -arch=amdgcn -show-encoding %s 2>&1 | FileCheck --check-prefix=NOSICIVI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti -show-encoding %s 2>&1 | FileCheck --check-prefix=NOSICIVI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=bonaire -show-encoding %s 2>&1 | FileCheck --check-prefix=NOSICIVI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=fiji -show-encoding %s 2>&1 | FileCheck --check-prefix=NOSICIVI --check-prefix=NOVI --check-prefix=NOGFX89 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 -show-encoding %s 2>&1 | FileCheck --check-prefix=NOGFX89 %s

s_add_u32 s1, s2, s3
// GCN: s_add_u32 s1, s2, s3 ; encoding: [0x02,0x03,0x01,0x80]

s_sub_u32 s1, s2, s3
// GCN: s_sub_u32 s1, s2, s3 ; encoding: [0x02,0x03,0x81,0x80]

s_add_i32 s1, s2, s3
// GCN: s_add_i32 s1, s2, s3 ; encoding: [0x02,0x03,0x01,0x81]

s_sub_i32 s1, s2, s3
// GCN: s_sub_i32 s1, s2, s3 ; encoding: [0x02,0x03,0x81,0x81]

s_addc_u32 s1, s2, s3
// GCN: s_addc_u32 s1, s2, s3 ; encoding: [0x02,0x03,0x01,0x82]

s_subb_u32 s1, s2, s3
// GCN: s_subb_u32 s1, s2, s3 ; encoding: [0x02,0x03,0x81,0x82]

s_min_i32 s1, s2, s3
// GCN: s_min_i32 s1, s2, s3 ; encoding: [0x02,0x03,0x01,0x83]

s_min_u32 s1, s2, s3
// GCN: s_min_u32 s1, s2, s3 ; encoding: [0x02,0x03,0x81,0x83]

s_max_i32 s1, s2, s3
// GCN: s_max_i32 s1, s2, s3 ; encoding: [0x02,0x03,0x01,0x84]

s_max_u32 s1, s2, s3
// GCN: s_max_u32 s1, s2, s3 ; encoding: [0x02,0x03,0x81,0x84]

s_cselect_b32 s1, s2, s3
// GCN: s_cselect_b32 s1, s2, s3 ; encoding: [0x02,0x03,0x01,0x85]

s_cselect_b64 s[2:3], s[4:5], s[6:7]
// GCN: s_cselect_b64 s[2:3], s[4:5], s[6:7] ; encoding: [0x04,0x06,0x82,0x85]

s_and_b32 s2, s4, s6
// SICI: s_and_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x87]
// GFX89: s_and_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x86]

s_and_b64 s[2:3], s[4:5], s[6:7]
// SICI: s_and_b64 s[2:3], s[4:5], s[6:7] ; encoding: [0x04,0x06,0x82,0x87]
// GFX89: s_and_b64 s[2:3], s[4:5], s[6:7] ; encoding: [0x04,0x06,0x82,0x86]

s_or_b32 s2, s4, s6
// SICI: s_or_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x88]
// GFX89: s_or_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x87]

s_or_b64 s[2:3], s[4:5], s[6:7]
// SICI: s_or_b64 s[2:3], s[4:5], s[6:7] ; encoding: [0x04,0x06,0x82,0x88]
// GFX89: s_or_b64 s[2:3], s[4:5], s[6:7] ; encoding: [0x04,0x06,0x82,0x87]

s_xor_b32 s2, s4, s6
// SICI: s_xor_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x89]
// GFX89: s_xor_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x88]

s_xor_b64 s[2:3], s[4:5], s[6:7]
// SICI: s_xor_b64 s[2:3], s[4:5], s[6:7] ; encoding: [0x04,0x06,0x82,0x89]
// GFX89: s_xor_b64 s[2:3], s[4:5], s[6:7] ; encoding: [0x04,0x06,0x82,0x88]

s_andn2_b32 s2, s4, s6
// SICI: s_andn2_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x8a]
// GFX89: s_andn2_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x89]

s_andn2_b64 s[2:3], s[4:5], s[6:7]
// SICI: s_andn2_b64 s[2:3], s[4:5], s[6:7] ; encoding: [0x04,0x06,0x82,0x8a]
// GFX89: s_andn2_b64 s[2:3], s[4:5], s[6:7] ; encoding: [0x04,0x06,0x82,0x89]

s_orn2_b32 s2, s4, s6
// SICI: s_orn2_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x8b]
// GFX89: s_orn2_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x8a]

s_orn2_b64 s[2:3], s[4:5], s[6:7]
// SICI: s_orn2_b64 s[2:3], s[4:5], s[6:7] ; encoding: [0x04,0x06,0x82,0x8b]
// GFX89: s_orn2_b64 s[2:3], s[4:5], s[6:7] ; encoding: [0x04,0x06,0x82,0x8a]

s_nand_b32 s2, s4, s6
// SICI: s_nand_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x8c]
// GFX89: s_nand_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x8b]

s_nand_b64 s[2:3], s[4:5], s[6:7]
// SICI: s_nand_b64 s[2:3], s[4:5], s[6:7] ; encoding: [0x04,0x06,0x82,0x8c]
// GFX89: s_nand_b64 s[2:3], s[4:5], s[6:7] ; encoding: [0x04,0x06,0x82,0x8b]

s_nor_b32 s2, s4, s6
// SICI: s_nor_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x8d]
// GFX89: s_nor_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x8c]

s_nor_b64 s[2:3], s[4:5], s[6:7]
// SICI: s_nor_b64 s[2:3], s[4:5], s[6:7] ; encoding: [0x04,0x06,0x82,0x8d]
// GFX89: s_nor_b64 s[2:3], s[4:5], s[6:7] ; encoding: [0x04,0x06,0x82,0x8c]

s_xnor_b32 s2, s4, s6
// SICI: s_xnor_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x8e]
// GFX89: s_xnor_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x8d]

s_xnor_b64 s[2:3], s[4:5], s[6:7]
// SICI: s_xnor_b64 s[2:3], s[4:5], s[6:7] ; encoding: [0x04,0x06,0x82,0x8e]
// GFX89: s_xnor_b64 s[2:3], s[4:5], s[6:7] ; encoding: [0x04,0x06,0x82,0x8d]

s_lshl_b32 s2, s4, s6
// SICI: s_lshl_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x8f]
// GFX89: s_lshl_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x8e]

s_lshl_b64 s[2:3], s[4:5], s6
// SICI: s_lshl_b64 s[2:3], s[4:5], s6 ; encoding: [0x04,0x06,0x82,0x8f]
// GFX89: s_lshl_b64 s[2:3], s[4:5], s6 ; encoding: [0x04,0x06,0x82,0x8e]

s_lshr_b32 s2, s4, s6
// SICI: s_lshr_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x90]
// GFX89: s_lshr_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x8f]

s_lshr_b64 s[2:3], s[4:5], s6
// SICI: s_lshr_b64 s[2:3], s[4:5], s6 ; encoding: [0x04,0x06,0x82,0x90]
// GFX89: s_lshr_b64 s[2:3], s[4:5], s6 ; encoding: [0x04,0x06,0x82,0x8f]

s_ashr_i32 s2, s4, s6
// SICI: s_ashr_i32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x91]
// GFX89: s_ashr_i32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x90]

s_ashr_i64 s[2:3], s[4:5], s6
// SICI: s_ashr_i64 s[2:3], s[4:5], s6 ; encoding: [0x04,0x06,0x82,0x91]
// GFX89: s_ashr_i64 s[2:3], s[4:5], s6 ; encoding: [0x04,0x06,0x82,0x90]

s_bfm_b32 s2, s4, s6
// SICI: s_bfm_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x92]
// GFX89: s_bfm_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x91]

s_bfm_b64 s[2:3], s4, s6
// SICI: s_bfm_b64 s[2:3], s4, s6 ; encoding: [0x04,0x06,0x82,0x92]
// GFX89: s_bfm_b64 s[2:3], s4, s6 ; encoding: [0x04,0x06,0x82,0x91]

s_mul_i32 s2, s4, s6
// SICI: s_mul_i32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x93]
// GFX89: s_mul_i32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x92]

s_bfe_u32 s2, s4, s6
// SICI: s_bfe_u32 s2, s4, s6 ; encoding: [0x04,0x06,0x82,0x93]
// GFX89: s_bfe_u32 s2, s4, s6 ; encoding: [0x04,0x06,0x82,0x92]

s_bfe_i32 s2, s4, s6
// SICI: s_bfe_i32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x94]
// GFX89: s_bfe_i32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x93]

s_bfe_u64 s[2:3], s[4:5], s6
// SICI: s_bfe_u64 s[2:3], s[4:5], s6 ; encoding: [0x04,0x06,0x82,0x94]
// GFX89: s_bfe_u64 s[2:3], s[4:5], s6 ; encoding: [0x04,0x06,0x82,0x93]

s_bfe_i64 s[2:3], s[4:5], s6
// SICI: s_bfe_i64 s[2:3], s[4:5], s6 ; encoding: [0x04,0x06,0x02,0x95]
// GFX89: s_bfe_i64 s[2:3], s[4:5], s6 ; encoding: [0x04,0x06,0x02,0x94]

s_cbranch_g_fork s[4:5], s[6:7]
// SICI: s_cbranch_g_fork s[4:5], s[6:7] ; encoding: [0x04,0x06,0x80,0x95]
// GFX89: s_cbranch_g_fork s[4:5], s[6:7] ; encoding: [0x04,0x06,0x80,0x94]

s_cbranch_g_fork 1, s[6:7]
// SICI: s_cbranch_g_fork 1, s[6:7] ; encoding: [0x81,0x06,0x80,0x95]
// GFX89: s_cbranch_g_fork 1, s[6:7] ; encoding: [0x81,0x06,0x80,0x94]

s_cbranch_g_fork s[6:7], 2
// SICI: s_cbranch_g_fork s[6:7], 2 ; encoding: [0x06,0x82,0x80,0x95]
// GFX89: s_cbranch_g_fork s[6:7], 2 ; encoding: [0x06,0x82,0x80,0x94]

s_absdiff_i32 s2, s4, s6
// SICI: s_absdiff_i32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x96]
// GFX89: s_absdiff_i32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x95]

s_add_u32 s101, s102, s103
// SICI: s_add_u32 s101, s102, s103 ; encoding: [0x66,0x67,0x65,0x80]
// NOGFX89: error: not a valid operand

s_lshl1_add_u32 s5, s1, s2
// GFX9:     s_lshl1_add_u32 s5, s1, s2 ; encoding: [0x01,0x02,0x05,0x97]
// NOSICIVI: error

s_lshl1_add_u32 s5, -1, s2
// GFX9:     s_lshl1_add_u32 s5, -1, s2 ; encoding: [0xc1,0x02,0x05,0x97]
// NOSICIVI: error

s_lshl1_add_u32 s5, s1, 0
// GFX9:     s_lshl1_add_u32 s5, s1, 0 ; encoding: [0x01,0x80,0x05,0x97]
// NOSICIVI: error

s_lshl1_add_u32 s5, s1, 0x3f717273
// GFX9:     s_lshl1_add_u32 s5, s1, 0x3f717273 ; encoding: [0x01,0xff,0x05,0x97,0x73,0x72,0x71,0x3f]
// NOSICIVI: error

s_lshl2_add_u32 s101, s1, s2
// GFX9:     s_lshl2_add_u32 s101, s1, s2 ; encoding: [0x01,0x02,0xe5,0x97]
// NOSICIVI: error

s_lshl2_add_u32 s5, 0xaf123456, s2
// GFX9:     s_lshl2_add_u32 s5, 0xaf123456, s2 ; encoding: [0xff,0x02,0x85,0x97,0x56,0x34,0x12,0xaf]
// NOSICIVI: error

s_lshl3_add_u32 s5, 0x3f717273, s2
// GFX9:     s_lshl3_add_u32 s5, 0x3f717273, s2 ; encoding: [0xff,0x02,0x05,0x98,0x73,0x72,0x71,0x3f]
// NOSICIVI: error

s_lshl3_add_u32 s5, s1, s101
// GFX9:     s_lshl3_add_u32 s5, s1, s101 ; encoding: [0x01,0x65,0x05,0x98]
// NOSICIVI: error

s_lshl4_add_u32 s5, s1, 0xaf123456
// GFX9:     s_lshl4_add_u32 s5, s1, 0xaf123456 ; encoding: [0x01,0xff,0x85,0x98,0x56,0x34,0x12,0xaf]
// NOSICIVI: error

s_lshl4_add_u32 s5, -1, s2
// GFX9:     s_lshl4_add_u32 s5, -1, s2 ; encoding: [0xc1,0x02,0x85,0x98]
// NOSICIVI: error

s_mul_hi_i32 s5, s101, s2
// GFX9:     s_mul_hi_i32 s5, s101, s2 ; encoding: [0x65,0x02,0x85,0x96]
// NOSICIVI: error

s_mul_hi_i32 s5, 0, s2
// GFX9:     s_mul_hi_i32 s5, 0, s2 ; encoding: [0x80,0x02,0x85,0x96]
// NOSICIVI: error

s_mul_hi_i32 s5, 0x3f717273, s2
// GFX9:     s_mul_hi_i32 s5, 0x3f717273, s2 ; encoding: [0xff,0x02,0x85,0x96,0x73,0x72,0x71,0x3f]
// NOSICIVI: error

s_mul_hi_i32 s5, s1, s101
// GFX9:     s_mul_hi_i32 s5, s1, s101 ; encoding: [0x01,0x65,0x85,0x96]
// NOSICIVI: error

s_mul_hi_i32 s5, s1, 0
// GFX9:     s_mul_hi_i32 s5, s1, 0 ; encoding: [0x01,0x80,0x85,0x96]
// NOSICIVI: error

s_mul_hi_u32 s5, s1, 0x3f717273
// GFX9:     s_mul_hi_u32 s5, s1, 0x3f717273 ; encoding: [0x01,0xff,0x05,0x96,0x73,0x72,0x71,0x3f]
// NOSICIVI: error

s_mul_hi_u32 s5, s1, s101
// GFX9:     s_mul_hi_u32 s5, s1, s101 ; encoding: [0x01,0x65,0x05,0x96]
// NOSICIVI: error

s_mul_hi_u32 s5, s1, 0
// GFX9:     s_mul_hi_u32 s5, s1, 0 ; encoding: [0x01,0x80,0x05,0x96]
// NOSICIVI: error
