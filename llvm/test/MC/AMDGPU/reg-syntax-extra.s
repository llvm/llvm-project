// RUN: not llvm-mc -arch=amdgcn -show-encoding %s | FileCheck --check-prefix=GCN --check-prefix=SICI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=SI -show-encoding %s | FileCheck --check-prefix=GCN --check-prefix=SICI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=SI -show-encoding %s 2>&1 | FileCheck --check-prefix=NOSICI %s
// RUN: llvm-mc -arch=amdgcn -mcpu=fiji -show-encoding %s 2>&1 | FileCheck --check-prefix=GCN --check-prefix=VI %s

s_mov_b32 [ttmp5], [ttmp3]
// SICI: s_mov_b32 ttmp5, ttmp3          ; encoding: [0x73,0x03,0xf5,0xbe]
// VI:   s_mov_b32 ttmp5, ttmp3          ; encoding: [0x73,0x00,0xf5,0xbe]

s_mov_b64 [ttmp4,ttmp5], [ttmp2,ttmp3]
// SICI: s_mov_b64 ttmp[4:5], ttmp[2:3]  ; encoding: [0x72,0x04,0xf4,0xbe]
// VI:   s_mov_b64 ttmp[4:5], ttmp[2:3]  ; encoding: [0x72,0x01,0xf4,0xbe]

s_mov_b64 ttmp[4:5], ttmp[2:3]
// SICI: s_mov_b64 ttmp[4:5], ttmp[2:3]  ; encoding: [0x72,0x04,0xf4,0xbe]
// VI:   s_mov_b64 ttmp[4:5], ttmp[2:3]  ; encoding: [0x72,0x01,0xf4,0xbe]

s_mov_b64 [s6,s7], s[8:9]
// SICI: s_mov_b64 s[6:7], s[8:9]        ; encoding: [0x08,0x04,0x86,0xbe]
// VI:   s_mov_b64 s[6:7], s[8:9]        ; encoding: [0x08,0x01,0x86,0xbe]

s_mov_b64 s[6:7], [s8,s9]
// SICI: s_mov_b64 s[6:7], s[8:9]        ; encoding: [0x08,0x04,0x86,0xbe]
// VI:   s_mov_b64 s[6:7], s[8:9]        ; encoding: [0x08,0x01,0x86,0xbe]

s_mov_b64 [exec_lo,exec_hi], s[2:3]
// SICI: s_mov_b64 exec, s[2:3]          ; encoding: [0x02,0x04,0xfe,0xbe]
// VI:   s_mov_b64 exec, s[2:3]          ; encoding: [0x02,0x01,0xfe,0xbe]

s_mov_b64 [flat_scratch_lo,flat_scratch_hi], s[2:3]
// NOSICI: error:
// VI:   s_mov_b64 flat_scratch, s[2:3]  ; encoding: [0x02,0x01,0xe6,0xbe]

s_mov_b64 [vcc_lo,vcc_hi], s[2:3]
// SICI: s_mov_b64 vcc, s[2:3]           ; encoding: [0x02,0x04,0xea,0xbe]
// VI:   s_mov_b64 vcc, s[2:3]           ; encoding: [0x02,0x01,0xea,0xbe]

s_mov_b64 [tba_lo,tba_hi], s[2:3]
// SICI:  s_mov_b64 tba, s[2:3]           ; encoding: [0x02,0x04,0xec,0xbe]
// VI:    s_mov_b64 tba, s[2:3]           ; encoding: [0x02,0x01,0xec,0xbe]

s_mov_b64 [tma_lo,tma_hi], s[2:3]
// SICI:  s_mov_b64 tma, s[2:3]           ; encoding: [0x02,0x04,0xee,0xbe]
// VI:    s_mov_b64 tma, s[2:3]           ; encoding: [0x02,0x01,0xee,0xbe]

v_mov_b32 [v1], [v2]
// GCN:  v_mov_b32_e32 v1, v2 ; encoding: [0x02,0x03,0x02,0x7e]

v_rcp_f64 [v1,v2], [v2,v3]
// SICI: v_rcp_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x5f,0x02,0x7e]
// VI:   v_rcp_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x4b,0x02,0x7e]

buffer_load_dwordx4 [v1,v2,v3,v4], [s4,s5,s6,s7], s1
// SICI: buffer_load_dwordx4 v[1:4], s[4:7], s1 ; encoding: [0x00,0x00,0x38,0xe0,0x00,0x01,0x01,0x01]
// VI:   buffer_load_dwordx4 v[1:4], s[4:7], s1 ; encoding: [0x00,0x00,0x5c,0xe0,0x00,0x01,0x01,0x01]
