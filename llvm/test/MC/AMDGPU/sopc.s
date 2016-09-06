// RUN: llvm-mc -arch=amdgcn -show-encoding %s | FileCheck --check-prefix=GCN %s
// RUN: llvm-mc -arch=amdgcn -mcpu=SI -show-encoding %s | FileCheck --check-prefix=GCN %s
// RUN: llvm-mc -arch=amdgcn -mcpu=fiji -show-encoding %s | FileCheck --check-prefix=GCN %s

//===----------------------------------------------------------------------===//
// SOPC Instructions
//===----------------------------------------------------------------------===//

s_cmp_eq_i32 s1, s2
// GCN: s_cmp_eq_i32 s1, s2 ; encoding: [0x01,0x02,0x00,0xbf]

s_cmp_lg_i32 s1, s2
// GCN: s_cmp_lg_i32 s1, s2 ; encoding: [0x01,0x02,0x01,0xbf]

s_cmp_gt_i32 s1, s2
// GCN: s_cmp_gt_i32 s1, s2 ; encoding: [0x01,0x02,0x02,0xbf]

s_cmp_ge_i32 s1, s2
// GCN: s_cmp_ge_i32 s1, s2 ; encoding: [0x01,0x02,0x03,0xbf]

s_cmp_lt_i32 s1, s2
// GCN: s_cmp_lt_i32 s1, s2 ; encoding: [0x01,0x02,0x04,0xbf]

s_cmp_le_i32 s1, s2
// GCN: s_cmp_le_i32 s1, s2 ; encoding: [0x01,0x02,0x05,0xbf]

s_cmp_eq_u32 s1, s2
// GCN: s_cmp_eq_u32 s1, s2 ; encoding: [0x01,0x02,0x06,0xbf]

s_cmp_lg_u32 s1, s2
// GCN: s_cmp_lg_u32 s1, s2 ; encoding: [0x01,0x02,0x07,0xbf]

s_cmp_gt_u32 s1, s2
// GCN: s_cmp_gt_u32 s1, s2 ; encoding: [0x01,0x02,0x08,0xbf]

s_cmp_ge_u32 s1, s2
// GCN: s_cmp_ge_u32 s1, s2 ; encoding: [0x01,0x02,0x09,0xbf]

s_cmp_lt_u32 s1, s2
// GCN: s_cmp_lt_u32 s1, s2 ; encoding: [0x01,0x02,0x0a,0xbf]

s_cmp_le_u32 s1, s2
// GCN: s_cmp_le_u32 s1, s2 ; encoding: [0x01,0x02,0x0b,0xbf]

s_bitcmp0_b32 s1, s2
// GCN: s_bitcmp0_b32 s1, s2 ; encoding: [0x01,0x02,0x0c,0xbf]

s_bitcmp1_b32 s1, s2
// GCN: s_bitcmp1_b32 s1, s2 ; encoding: [0x01,0x02,0x0d,0xbf]

s_bitcmp0_b64 s[2:3], s4
// GCN: s_bitcmp0_b64 s[2:3], s4 ; encoding: [0x02,0x04,0x0e,0xbf]

s_bitcmp1_b64 s[2:3], s4
// GCN: s_bitcmp1_b64 s[2:3], s4 ; encoding: [0x02,0x04,0x0f,0xbf]

s_setvskip s3, s5
// GCN: s_setvskip s3, s5 ; encoding: [0x03,0x05,0x10,0xbf]
