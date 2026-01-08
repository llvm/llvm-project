// RUN: not llvm-mc -triple=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck -check-prefixes=GCN-ERR,SICIVI9-ERR,SIVICI-ERR,SI-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=tonga %s 2>&1 | FileCheck -check-prefixes=GCN-ERR,SICIVI9-ERR,SIVICI-ERR,VI-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck -check-prefixes=GCN-ERR,GFX9-ERR,SICIVI9-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck -check-prefixes=GCN-ERR,GFX10-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 %s 2>&1 | FileCheck -check-prefixes=GCN-ERR,GFX1250-ERR --implicit-check-not=error: %s

// RUN: not llvm-mc -triple=amdgcn -mcpu=tahiti -show-encoding %s | FileCheck -check-prefix=SIVICI %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=tonga -show-encoding %s | FileCheck -check-prefixes=SIVICI,CIVI9 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx900 -show-encoding %s | FileCheck -check-prefixes=GFX9,CIVI9 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1010 -show-encoding %s | FileCheck -check-prefix=GFX10 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s | FileCheck -check-prefix=GFX1250 %s

s_add_i32 s106, s0, s1
// GCN-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: register index is out of range

s_add_i32 s104, s0, s1
// SICIVI9-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: s104 register not available on this GPU
// GFX10: s_add_i32 s104, s0, s1 ; encoding:
// GFX1250: s_add_co_i32 s104, s0, s1 ; encoding:

s_add_i32 s105, s0, s1
// SICIVI9-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: s105 register not available on this GPU
// GFX10: s_add_i32 s105, s0, s1 ; encoding:
// GFX1250: s_add_co_i32 s105, s0, s1 ; encoding:

v_add_i32 v256, v0, v1
// GFX10-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX9-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: register index is out of range
// SI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: register index is out of range
// VI-ERR: :[[@LINE-4]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX1250-ERR: :[[@LINE-5]]:{{[0-9]+}}: error: register index is out of range

v_add_i32 v257, v0, v1
// GFX10-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX9-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: register index is out of range
// SI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: register index is out of range
// VI-ERR: :[[@LINE-4]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX1250-ERR: :[[@LINE-5]]:{{[0-9]+}}: error: register index is out of range

s_mov_b64 s[0:17], -1
// GCN-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid or unsupported register size

s_mov_b64 s[103:104], -1
// GCN-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register alignment

s_mov_b64 s[105:106], -1
// GCN-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register alignment

s_mov_b64 s[104:105], -1
// SICIVI9-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: s[104:105] register not available on this GPU
// GFX10: s_mov_b64 s[104:105], -1 ; encoding:
// GFX1250: s_mov_b64 s[104:105], -1 ; encoding:

s_load_dwordx4 s[102:105], s[2:3], s4
// GCN-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register alignment

s_load_dwordx4 s[104:108], s[2:3], s4
// GCN-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: register index is out of range

s_load_dwordx4 s[108:112], s[2:3], s4
// GCN-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: register index is out of range

s_load_dwordx4 s[1:4], s[2:3], s4
// GCN-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register alignment

s_load_dwordx4 s[2:5], s[2:3], s4
// GCN-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register alignment

s_load_dwordx8 s[104:111], s[2:3], s4
// GCN-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: register index is out of range

s_load_dwordx8 s[100:107], s[2:3], s4
// GCN-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: register index is out of range

s_load_dwordx8 s[108:115], s[2:3], s4
// GCN-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: register index is out of range

s_load_dwordx16 s[92:107], s[2:3], s4
// GCN-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: register index is out of range

s_load_dwordx16 s[96:111], s[2:3], s4
// GCN-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: register index is out of range

s_load_dwordx16 s[100:115], s[2:3], s4
// GCN-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: register index is out of range

s_load_dwordx16 s[104:119], s[2:3], s4
// GCN-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: register index is out of range

s_load_dwordx16 s[108:123], s[2:3], s4
// GCN-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: register index is out of range

s_mov_b32 ttmp16, 0
// GCN-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: register index is out of range

s_mov_b32 ttmp12, 0
// GFX9: s_mov_b32 ttmp12, 0 ; encoding:
// GFX10: s_mov_b32 ttmp12, 0 ; encoding:
// SIVICI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: ttmp12 register not available on this GPU
// GFX1250: s_mov_b32 ttmp12, 0 ; encoding:

s_mov_b32 ttmp15, 0
// GFX9: s_mov_b32 ttmp15, 0 ; encoding:
// GFX10: s_mov_b32 ttmp15, 0 ; encoding:
// SIVICI-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: ttmp15 register not available on this GPU
// GFX1250: s_mov_b32 ttmp15, 0 ; encoding:

s_mov_b32 flat_scratch_lo, 0
// SI-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: flat_scratch_lo register not available on this GPU
// GFX10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: flat_scratch_lo register not available on this GPU
// CIVI9: s_mov_b32 flat_scratch_lo, 0 ; encoding: [0x80,0x00,0xe6,0xbe]
// GFX1250-ERR: :[[@LINE-4]]:{{[0-9]+}}: error: flat_scratch_lo register not available on this GPU

s_mov_b32 flat_scratch_hi, 0
// SI-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: flat_scratch_hi register not available on this GPU
// GFX10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: flat_scratch_hi register not available on this GPU
// CIVI9: s_mov_b32 flat_scratch_hi, 0 ; encoding: [0x80,0x00,0xe7,0xbe]
// GFX1250-ERR: :[[@LINE-4]]:{{[0-9]+}}: error: flat_scratch_hi register not available on this GPU

s_mov_b32 tma_lo, 0
// SIVICI: s_mov_b32 tma_lo, 0 ; encoding:
// GFX9-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: tma_lo register not available on this GPU
// GFX10-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: tma_lo register not available on this GPU
// GFX1250-ERR: :[[@LINE-4]]:{{[0-9]+}}: error: tma_lo register not available on this GPU

s_mov_b32 tba_lo, 0
// SIVICI: s_mov_b32 tba_lo, 0 ; encoding:
// GFX9-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: tba_lo register not available on this GPU
// GFX10-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: tba_lo register not available on this GPU
// GFX1250-ERR: :[[@LINE-4]]:{{[0-9]+}}: error: tba_lo register not available on this GPU

v_cvt_f32_f64 v0, v[255:256]
// SICIVI9-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: register index is out of range
// GFX10-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: register index is out of range
// GFX1250-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: invalid register class: vgpr tuples must be 64 bit aligned

v_mqsad_u32_u8 v[254:257], v[0:1], -4.0, v[2:5]
// SI-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
// VI-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: register index is out of range
// GFX9-ERR: :[[@LINE-3]]:{{[0-9]+}}: error: register index is out of range
// GFX10-ERR: :[[@LINE-4]]:{{[0-9]+}}: error: register index is out of range
// GFX1250: v_mqsad_u32_u8 v[254:257], v[0:1], -4.0, v[2:5] ; encoding: [0xfe,0x00,0x3d,0xd6,0x00,0xef,0x09,0x04]
