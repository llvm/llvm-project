// RUN: llvm-mc -arch=amdgcn -show-encoding %s | FileCheck %s
// RUN: llvm-mc -arch=amdgcn -mcpu=SI -show-encoding %s | FileCheck %s

// CHECK: s_add_u32 s1, s2, s3 ; encoding: [0x02,0x03,0x01,0x80]
s_add_u32 s1, s2, s3

// CHECK: s_sub_u32 s1, s2, s3 ; encoding: [0x02,0x03,0x81,0x80]
s_sub_u32 s1, s2, s3

// CHECK: s_add_i32 s1, s2, s3 ; encoding: [0x02,0x03,0x01,0x81]
s_add_i32 s1, s2, s3

// CHECK: s_sub_i32 s1, s2, s3 ; encoding: [0x02,0x03,0x81,0x81]
s_sub_i32 s1, s2, s3

// CHECK: s_addc_u32 s1, s2, s3 ; encoding: [0x02,0x03,0x01,0x82]
s_addc_u32 s1, s2, s3

// CHECK: s_subb_u32 s1, s2, s3 ; encoding: [0x02,0x03,0x81,0x82]
s_subb_u32 s1, s2, s3

// CHECK: s_min_i32 s1, s2, s3 ; encoding: [0x02,0x03,0x01,0x83]
s_min_i32 s1, s2, s3

// CHECK: s_min_u32 s1, s2, s3 ; encoding: [0x02,0x03,0x81,0x83]
s_min_u32 s1, s2, s3

// CHECK: s_max_i32 s1, s2, s3 ; encoding: [0x02,0x03,0x01,0x84]
s_max_i32 s1, s2, s3

// CHECK: s_max_u32 s1, s2, s3 ; encoding: [0x02,0x03,0x81,0x84]
s_max_u32 s1, s2, s3

// CHECK: s_cselect_b32 s1, s2, s3 ; encoding: [0x02,0x03,0x01,0x85]
s_cselect_b32 s1, s2, s3

// CHECK: s_cselect_b64 s[2:3], s[4:5], s[6:7] ; encoding: [0x04,0x06,0x82,0x85]
s_cselect_b64 s[2:3], s[4:5], s[6:7]

// CHECK: s_and_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x87]
s_and_b32 s2, s4, s6

// CHECK: s_and_b64 s[2:3], s[4:5], s[6:7] ; encoding: [0x04,0x06,0x82,0x87]
s_and_b64 s[2:3], s[4:5], s[6:7]

// CHECK: s_or_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x88]
s_or_b32 s2, s4, s6

// CHECK: s_or_b64 s[2:3], s[4:5], s[6:7] ; encoding: [0x04,0x06,0x82,0x88]
s_or_b64 s[2:3], s[4:5], s[6:7]

// CHECK: s_xor_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x89]
s_xor_b32 s2, s4, s6

// CHECK: s_xor_b64 s[2:3], s[4:5], s[6:7] ; encoding: [0x04,0x06,0x82,0x89]
s_xor_b64 s[2:3], s[4:5], s[6:7]

// CHECK: s_andn2_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x8a]
s_andn2_b32 s2, s4, s6

// CHECK: s_andn2_b64 s[2:3], s[4:5], s[6:7] ; encoding: [0x04,0x06,0x82,0x8a]
s_andn2_b64 s[2:3], s[4:5], s[6:7]

// CHECK: s_orn2_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x8b]
s_orn2_b32 s2, s4, s6

// CHECK: s_orn2_b64 s[2:3], s[4:5], s[6:7] ; encoding: [0x04,0x06,0x82,0x8b]
s_orn2_b64 s[2:3], s[4:5], s[6:7]

// CHECK: s_nand_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x8c]
s_nand_b32 s2, s4, s6

// CHECK: s_nand_b64 s[2:3], s[4:5], s[6:7] ; encoding: [0x04,0x06,0x82,0x8c]
s_nand_b64 s[2:3], s[4:5], s[6:7]

// CHECK: s_nor_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x8d]
s_nor_b32 s2, s4, s6

// CHECK: s_nor_b64 s[2:3], s[4:5], s[6:7] ; encoding: [0x04,0x06,0x82,0x8d]
s_nor_b64 s[2:3], s[4:5], s[6:7]

// CHECK: s_xnor_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x8e]
s_xnor_b32 s2, s4, s6

// CHECK: s_xnor_b64 s[2:3], s[4:5], s[6:7] ; encoding: [0x04,0x06,0x82,0x8e]
s_xnor_b64 s[2:3], s[4:5], s[6:7]

// CHECK: s_lshl_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x8f]
s_lshl_b32 s2, s4, s6

// CHECK: s_lshl_b64 s[2:3], s[4:5], s6 ; encoding: [0x04,0x06,0x82,0x8f]
s_lshl_b64 s[2:3], s[4:5], s6

// CHECK: s_lshr_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x90]
s_lshr_b32 s2, s4, s6

// CHECK: s_lshr_b64 s[2:3], s[4:5], s6 ; encoding: [0x04,0x06,0x82,0x90]
s_lshr_b64 s[2:3], s[4:5], s6

// CHECK: s_ashr_i32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x91]
s_ashr_i32 s2, s4, s6

// CHECK: s_ashr_i64 s[2:3], s[4:5], s6 ; encoding: [0x04,0x06,0x82,0x91]
s_ashr_i64 s[2:3], s[4:5], s6

// CHECK: s_bfm_b32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x92]
s_bfm_b32 s2, s4, s6

// CHECK: s_bfm_b64 s[2:3], s4, s6 ; encoding: [0x04,0x06,0x82,0x92]
s_bfm_b64 s[2:3], s4, s6

// CHECK: s_mul_i32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x93]
s_mul_i32 s2, s4, s6

// CHECK: s_bfe_u32 s2, s4, s6 ; encoding: [0x04,0x06,0x82,0x93]
s_bfe_u32 s2, s4, s6

// CHECK: s_bfe_i32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x94]
s_bfe_i32 s2, s4, s6

// CHECK: s_bfe_u64 s[2:3], s[4:5], s6 ; encoding: [0x04,0x06,0x82,0x94]
s_bfe_u64 s[2:3], s[4:5], s6

// CHECK: s_bfe_i64 s[2:3], s[4:5], s6 ; encoding: [0x04,0x06,0x02,0x95]
s_bfe_i64 s[2:3], s[4:5], s6

// CHECK: s_cbranch_g_fork s[4:5], s[6:7] ; encoding: [0x04,0x06,0x80,0x95]
s_cbranch_g_fork s[4:5], s[6:7]

// CHECK: s_absdiff_i32 s2, s4, s6 ; encoding: [0x04,0x06,0x02,0x96]
s_absdiff_i32 s2, s4, s6

// CHECK: s_add_u32 s101, s102, s103 ; encoding: [0x66,0x67,0x65,0x80]
s_add_u32 s101, s102, s103
