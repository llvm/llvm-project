// RUN: not llvm-mc -arch=amdgcn -show-encoding %s | FileCheck --check-prefixes=GCN,SICI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti -show-encoding %s | FileCheck --check-prefixes=GCN,SICI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=fiji -show-encoding %s | FileCheck --check-prefixes=GCN,VI %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -show-encoding %s | FileCheck --check-prefixes=GCN,GFX10 %s

// RUN: not llvm-mc -arch=amdgcn -mcpu=tahiti %s 2>&1 | FileCheck --check-prefixes=NOSICI,NOSICIVI --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=fiji %s 2>&1 | FileCheck --check-prefix=NOVI --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck --check-prefix=NOGFX10 --implicit-check-not=error: %s

s_mov_b32 [ttmp5], [ttmp3]
// SICI: s_mov_b32 ttmp5, ttmp3          ; encoding: [0x73,0x03,0xf5,0xbe]
// VI:   s_mov_b32 ttmp5, ttmp3          ; encoding: [0x73,0x00,0xf5,0xbe]
// GFX10: s_mov_b32 ttmp5, ttmp3         ; encoding: [0x6f,0x03,0xf1,0xbe]

s_mov_b64 [ttmp4,ttmp5], [ttmp2,ttmp3]
// SICI: s_mov_b64 ttmp[4:5], ttmp[2:3]  ; encoding: [0x72,0x04,0xf4,0xbe]
// VI:   s_mov_b64 ttmp[4:5], ttmp[2:3]  ; encoding: [0x72,0x01,0xf4,0xbe]
// GFX10: s_mov_b64 ttmp[4:5], ttmp[2:3] ; encoding: [0x6e,0x04,0xf0,0xbe]

s_mov_b64 ttmp[4:5], ttmp[2:3]
// SICI: s_mov_b64 ttmp[4:5], ttmp[2:3]  ; encoding: [0x72,0x04,0xf4,0xbe]
// VI:   s_mov_b64 ttmp[4:5], ttmp[2:3]  ; encoding: [0x72,0x01,0xf4,0xbe]
// GFX10: s_mov_b64 ttmp[4:5], ttmp[2:3] ; encoding: [0x6e,0x04,0xf0,0xbe]

s_mov_b64 [s6,s7], s[8:9]
// SICI: s_mov_b64 s[6:7], s[8:9]        ; encoding: [0x08,0x04,0x86,0xbe]
// VI:   s_mov_b64 s[6:7], s[8:9]        ; encoding: [0x08,0x01,0x86,0xbe]
// GFX10: s_mov_b64 s[6:7], s[8:9]       ; encoding: [0x08,0x04,0x86,0xbe]

s_mov_b64 s[6:7], [s8,s9]
// SICI: s_mov_b64 s[6:7], s[8:9]        ; encoding: [0x08,0x04,0x86,0xbe]
// VI:   s_mov_b64 s[6:7], s[8:9]        ; encoding: [0x08,0x01,0x86,0xbe]
// GFX10: s_mov_b64 s[6:7], s[8:9]       ; encoding: [0x08,0x04,0x86,0xbe]

s_mov_b64 [exec_lo,exec_hi], s[2:3]
// SICI: s_mov_b64 exec, s[2:3]          ; encoding: [0x02,0x04,0xfe,0xbe]
// VI:   s_mov_b64 exec, s[2:3]          ; encoding: [0x02,0x01,0xfe,0xbe]
// GFX10: s_mov_b64 exec, s[2:3]         ; encoding: [0x02,0x04,0xfe,0xbe]

s_mov_b64 [flat_scratch_lo,flat_scratch_hi], s[2:3]
// NOSICI: error: register not available on this GPU
// VI:   s_mov_b64 flat_scratch, s[2:3]  ; encoding: [0x02,0x01,0xe6,0xbe]
// NOGFX10: error: register not available on this GPU

s_mov_b64 [vcc_lo,vcc_hi], s[2:3]
// SICI: s_mov_b64 vcc, s[2:3]           ; encoding: [0x02,0x04,0xea,0xbe]
// VI:   s_mov_b64 vcc, s[2:3]           ; encoding: [0x02,0x01,0xea,0xbe]
// GFX10: s_mov_b64 vcc, s[2:3]          ; encoding: [0x02,0x04,0xea,0xbe]

s_mov_b64 [tba_lo,tba_hi], s[2:3]
// SICI:  s_mov_b64 tba, s[2:3]           ; encoding: [0x02,0x04,0xec,0xbe]
// VI:    s_mov_b64 tba, s[2:3]           ; encoding: [0x02,0x01,0xec,0xbe]
// NOGFX10: error: register not available on this GPU

s_mov_b64 [tma_lo,tma_hi], s[2:3]
// SICI:  s_mov_b64 tma, s[2:3]           ; encoding: [0x02,0x04,0xee,0xbe]
// VI:    s_mov_b64 tma, s[2:3]           ; encoding: [0x02,0x01,0xee,0xbe]
// NOGFX10: error: register not available on this GPU

v_mov_b32_e32 [v1], [v2]
// GCN:  v_mov_b32_e32 v1, v2 ; encoding: [0x02,0x03,0x02,0x7e]

v_rcp_f64 [v1,v2], [v2,v3]
// SICI: v_rcp_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x5f,0x02,0x7e]
// VI:   v_rcp_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x4b,0x02,0x7e]
// GFX10: v_rcp_f64_e32 v[1:2], v[2:3] ; encoding: [0x02,0x5f,0x02,0x7e]

buffer_load_dwordx4 [v1,v2,v3,v4], off, [s4,s5,s6,s7], s1
// SICI: buffer_load_dwordx4 v[1:4], off, s[4:7], s1 ; encoding: [0x00,0x00,0x38,0xe0,0x00,0x01,0x01,0x01]
// VI:   buffer_load_dwordx4 v[1:4], off, s[4:7], s1 ; encoding: [0x00,0x00,0x5c,0xe0,0x00,0x01,0x01,0x01]
// GFX10: buffer_load_dwordx4 v[1:4], off, s[4:7], s1 ; encoding: [0x00,0x00,0x38,0xe0,0x00,0x01,0x01,0x01]

buffer_load_dword v1, off, [ttmp4,ttmp5,ttmp6,ttmp7], s1
// SICI: buffer_load_dword v1, off, ttmp[4:7], s1 ; encoding: [0x00,0x00,0x30,0xe0,0x00,0x01,0x1d,0x01]
// VI:   buffer_load_dword v1, off, ttmp[4:7], s1 ; encoding: [0x00,0x00,0x50,0xe0,0x00,0x01,0x1d,0x01]
// GFX10: buffer_load_dword v1, off, ttmp[4:7], s1 ; encoding: [0x00,0x00,0x30,0xe0,0x00,0x01,0x1c,0x01]

buffer_store_format_xyzw v[1:4], off, [ttmp4,ttmp5,ttmp6,ttmp7], ttmp1
// SICI: buffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1 ; encoding: [0x00,0x00,0x1c,0xe0,0x00,0x01,0x1d,0x71]
// VI:   buffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1 ; encoding: [0x00,0x00,0x1c,0xe0,0x00,0x01,0x1d,0x71]
// GFX10: buffer_store_format_xyzw v[1:4], off, ttmp[4:7], ttmp1 ; encoding: [0x00,0x00,0x1c,0xe0,0x00,0x01,0x1c,0x6d]

buffer_load_ubyte v1, off, [ttmp4,ttmp5,ttmp6,ttmp7], ttmp1
// SICI: buffer_load_ubyte v1, off, ttmp[4:7], ttmp1 ; encoding: [0x00,0x00,0x20,0xe0,0x00,0x01,0x1d,0x71]
// VI:   buffer_load_ubyte v1, off, ttmp[4:7], ttmp1 ; encoding: [0x00,0x00,0x40,0xe0,0x00,0x01,0x1d,0x71]
// GFX10: buffer_load_ubyte v1, off, ttmp[4:7], ttmp1 ; encoding: [0x00,0x00,0x20,0xe0,0x00,0x01,0x1c,0x6d]

buffer_store_dwordx4 v[1:4], off, [ttmp4,ttmp5,ttmp6,ttmp7], ttmp1
// SICI: buffer_store_dwordx4 v[1:4], off, ttmp[4:7], ttmp1 ; encoding: [0x00,0x00,0x78,0xe0,0x00,0x01,0x1d,0x71]
// VI:   buffer_store_dwordx4 v[1:4], off, ttmp[4:7], ttmp1 ; encoding: [0x00,0x00,0x7c,0xe0,0x00,0x01,0x1d,0x71]
// GFX10: buffer_store_dwordx4 v[1:4], off, ttmp[4:7], ttmp1 ; encoding: [0x00,0x00,0x78,0xe0,0x00,0x01,0x1c,0x6d]

s_load_dwordx4 [ttmp4,ttmp5,ttmp6,ttmp7], [ttmp2,ttmp3], ttmp4
// SICI: s_load_dwordx4 ttmp[4:7], ttmp[2:3], ttmp4 ; encoding: [0x74,0x72,0xba,0xc0]
// VI:	 s_load_dwordx4 ttmp[4:7], ttmp[2:3], ttmp4 ; encoding: [0x39,0x1d,0x08,0xc0,0x74,0x00,0x00,0x00]
// GFX10: s_load_dwordx4 ttmp[4:7], ttmp[2:3], ttmp4 ; encoding: [0x37,0x1c,0x08,0xf4,0x00,0x00,0x00,0xe0]

s_buffer_load_dword ttmp1, [ttmp4,ttmp5,ttmp6,ttmp7], ttmp4
// SICI: s_buffer_load_dword ttmp1, ttmp[4:7], ttmp4 ; encoding: [0x74,0xf4,0x38,0xc2]
// VI:	 s_buffer_load_dword ttmp1, ttmp[4:7], ttmp4 ; encoding: [0x7a,0x1c,0x20,0xc0,0x74,0x00,0x00,0x00]
// GFX10: s_buffer_load_dword ttmp1, ttmp[4:7], ttmp4 ; encoding: [0x78,0x1b,0x20,0xf4,0x00,0x00,0x00,0xe0]

s_buffer_load_dwordx4 [ttmp8,ttmp9,ttmp10,ttmp11], [ttmp4,ttmp5,ttmp6,ttmp7], ttmp4
// SICI: s_buffer_load_dwordx4 ttmp[8:11], ttmp[4:7], ttmp4 ; encoding: [0x74,0x74,0xbc,0xc2]
// VI:   s_buffer_load_dwordx4 ttmp[8:11], ttmp[4:7], ttmp4 ; encoding: [0x3a,0x1e,0x28,0xc0,0x74,0x00,0x00,0x00]
// GFX10: s_buffer_load_dwordx4 ttmp[8:11], ttmp[4:7], ttmp4 ; encoding: [0x38,0x1d,0x28,0xf4,0x00,0x00,0x00,0xe0]

s_buffer_load_dwordx4 [ttmp[8],ttmp[8+1],ttmp[5*2],ttmp[(3+2)*2+1]], ttmp[45/11:(33+45)/11], ttmp4
// SICI: s_buffer_load_dwordx4 ttmp[8:11], ttmp[4:7], ttmp4 ; encoding: [0x74,0x74,0xbc,0xc2]
// VI:   s_buffer_load_dwordx4 ttmp[8:11], ttmp[4:7], ttmp4 ; encoding: [0x3a,0x1e,0x28,0xc0,0x74,0x00,0x00,0x00]
// GFX10: s_buffer_load_dwordx4 ttmp[8:11], ttmp[4:7], ttmp4 ; encoding: [0x38,0x1d,0x28,0xf4,0x00,0x00,0x00,0xe0]

s_buffer_load_dwordx4 ttmp[7+1:(3+2)*2+1], [ttmp[45/11],ttmp[5],ttmp6,ttmp[(33+45)/11]], ttmp4
// SICI: s_buffer_load_dwordx4 ttmp[8:11], ttmp[4:7], ttmp4 ; encoding: [0x74,0x74,0xbc,0xc2]
// VI:   s_buffer_load_dwordx4 ttmp[8:11], ttmp[4:7], ttmp4 ; encoding: [0x3a,0x1e,0x28,0xc0,0x74,0x00,0x00,0x00]
// GFX10: s_buffer_load_dwordx4 ttmp[8:11], ttmp[4:7], ttmp4 ; encoding: [0x38,0x1d,0x28,0xf4,0x00,0x00,0x00,0xe0]

flat_load_dword v[8:8], v[2:3]
// VI:   flat_load_dword v8, v[2:3] ; encoding: [0x00,0x00,0x50,0xdc,0x02,0x00,0x00,0x08]
// GFX10: flat_load_dword v8, v[2:3] ; encoding: [0x00,0x00,0x30,0xdc,0x02,0x00,0x7d,0x08]
// NOSICI: error: instruction not supported on this GPU

flat_load_dword v[63/8+1:65/8], v[2:3]
// VI:   flat_load_dword v8, v[2:3] ; encoding: [0x00,0x00,0x50,0xdc,0x02,0x00,0x00,0x08]
// GFX10: flat_load_dword v8, v[2:3] ; encoding: [0x00,0x00,0x30,0xdc,0x02,0x00,0x7d,0x08]
// NOSICI: error: instruction not supported on this GPU

flat_load_dword v8, v[2*2-2:(3+7)/3]
// VI:   flat_load_dword v8, v[2:3] ; encoding: [0x00,0x00,0x50,0xdc,0x02,0x00,0x00,0x08]
// GFX10: flat_load_dword v8, v[2:3] ; encoding: [0x00,0x00,0x30,0xdc,0x02,0x00,0x7d,0x08]
// NOSICI: error: instruction not supported on this GPU

flat_load_dword v[63/8+1], v[2:3]
// VI:   flat_load_dword v8, v[2:3] ; encoding: [0x00,0x00,0x50,0xdc,0x02,0x00,0x00,0x08]
// GFX10: flat_load_dword v8, v[2:3] ; encoding: [0x00,0x00,0x30,0xdc,0x02,0x00,0x7d,0x08]
// NOSICI: error: instruction not supported on this GPU

flat_load_dwordx4 v[8:11], v[2*2-2:(3*3-6)]
// VI:   flat_load_dwordx4 v[8:11], v[2:3] ; encoding: [0x00,0x00,0x5c,0xdc,0x02,0x00,0x00,0x08]
// GFX10: flat_load_dwordx4 v[8:11], v[2:3] ; encoding: [0x00,0x00,0x38,0xdc,0x02,0x00,0x7d,0x08]
// NOSICI: error: instruction not supported on this GPU

flat_load_dwordx4 v[8/2+4:11/2+6], v[2:3]
// VI:   flat_load_dwordx4 v[8:11], v[2:3] ; encoding: [0x00,0x00,0x5c,0xdc,0x02,0x00,0x00,0x08]
// GFX10: flat_load_dwordx4 v[8:11], v[2:3] ; encoding: [0x00,0x00,0x38,0xdc,0x02,0x00,0x7d,0x08]
// NOSICI: error: instruction not supported on this GPU

flat_load_dwordx4   [v[8/2+4],v9,v[10],v[11/2+6]], v[2:3]
// VI:   flat_load_dwordx4 v[8:11], v[2:3] ; encoding: [0x00,0x00,0x5c,0xdc,0x02,0x00,0x00,0x08]
// GFX10: flat_load_dwordx4 v[8:11], v[2:3] ; encoding: [0x00,0x00,0x38,0xdc,0x02,0x00,0x7d,0x08]
// NOSICI: error: instruction not supported on this GPU

v_mul_f32 v0, null, v2
// NOSICIVI: error: 'null' operand is not supported on this GPU
// GFX10: v_mul_f32_e32 v0, null, v2 ; encoding: [0x7d,0x04,0x00,0x10]
// NOVI: error: 'null' operand is not supported on this GPU

v_mul_f64 v[0:1], null, null
// NOSICIVI: error: 'null' operand is not supported on this GPU
// GFX10: v_mul_f64 v[0:1], null, null ; encoding: [0x00,0x00,0x65,0xd5,0x7d,0xfa,0x00,0x00]
// NOVI: error: 'null' operand is not supported on this GPU

s_add_u32 null, null, null
// NOSICIVI: error: 'null' operand is not supported on this GPU
// GFX10: s_add_u32 null, null, null ; encoding: [0x7d,0x7d,0x7d,0x80]
// NOVI: error: 'null' operand is not supported on this GPU

s_not_b64 s[2:3], null
// NOSICIVI: error: 'null' operand is not supported on this GPU
// GFX10: s_not_b64 s[2:3], null ; encoding: [0x7d,0x08,0x82,0xbe]
// NOVI: error: 'null' operand is not supported on this GPU
