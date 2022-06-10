// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1030 -show-encoding %s | FileCheck --check-prefix=GFX10 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -show-encoding %s | FileCheck --check-prefix=GFX11 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX11-ERR %s

// On GFX11+, EXECZ and VCCZ are no longer allowed to be used as sources to SALU and VALU instructions.
// The inline constants are removed. VCCZ and EXECZ still exist and can be use for conditional branches.
// LDS_DIRECT and POPS_EXITING_WAVE_ID are also no longer allowed.

//---------------------------------------------------------------------------//
// EXECZ
//---------------------------------------------------------------------------//

s_cbranch_execz 0x100
// GFX10: encoding: [0x00,0x01,0x88,0xbf]
// GFX11: encoding: [0x00,0x01,0xa5,0xbf]

s_add_i32 s0, execz, s2
// GFX10: encoding: [0xfc,0x02,0x00,0x81]
// GFX11-ERR: error: execz and vccz are not supported on this GPU

s_add_i32 s0, src_execz, s2
// GFX10: encoding: [0xfc,0x02,0x00,0x81]
// GFX11-ERR: error: execz and vccz are not supported on this GPU

s_add_i32 s0, s1, execz
// GFX10: encoding: [0x01,0xfc,0x00,0x81]
// GFX11-ERR: error: execz and vccz are not supported on this GPU

s_add_i32 s0, s1, src_execz
// GFX10: encoding: [0x01,0xfc,0x00,0x81]
// GFX11-ERR: error: execz and vccz are not supported on this GPU

v_add_f64 v[0:1], execz, v[2:3]
// GFX10: encoding: [0x00,0x00,0x64,0xd5,0xfc,0x04,0x02,0x00]
// GFX11-ERR: error: execz and vccz are not supported on this GPU

v_add_f64 v[0:1], src_execz, v[2:3]
// GFX10: encoding: [0x00,0x00,0x64,0xd5,0xfc,0x04,0x02,0x00]
// GFX11-ERR: error: execz and vccz are not supported on this GPU

v_add_f64 v[0:1], v[1:2], execz
// GFX10: encoding: [0x00,0x00,0x64,0xd5,0x01,0xf9,0x01,0x00]
// GFX11-ERR: error: execz and vccz are not supported on this GPU

v_add_f64 v[0:1], v[1:2], src_execz
// GFX10: encoding: [0x00,0x00,0x64,0xd5,0x01,0xf9,0x01,0x00]
// GFX11-ERR: error: execz and vccz are not supported on this GPU

//---------------------------------------------------------------------------//
// VCCZ
//---------------------------------------------------------------------------//

s_cbranch_vccz 0x100
// GFX10: encoding: [0x00,0x01,0x86,0xbf]
// GFX11: encoding: [0x00,0x01,0xa3,0xbf]

s_add_i32 s0, vccz, s2
// GFX10: encoding: [0xfb,0x02,0x00,0x81]
// GFX11-ERR: error: execz and vccz are not supported on this GPU

s_add_i32 s0, src_vccz, s2
// GFX10: encoding: [0xfb,0x02,0x00,0x81]
// GFX11-ERR: error: execz and vccz are not supported on this GPU

s_add_i32 s0, s1, vccz
// GFX10: encoding: [0x01,0xfb,0x00,0x81]
// GFX11-ERR: error: execz and vccz are not supported on this GPU

s_add_i32 s0, s1, src_vccz
// GFX10: encoding: [0x01,0xfb,0x00,0x81]
// GFX11-ERR: error: execz and vccz are not supported on this GPU

v_add_f64 v[0:1], vccz, v[2:3]
// GFX10: encoding: [0x00,0x00,0x64,0xd5,0xfb,0x04,0x02,0x00]
// GFX11-ERR: error: execz and vccz are not supported on this GPU

v_add_f64 v[0:1], src_vccz, v[2:3]
// GFX10: encoding: [0x00,0x00,0x64,0xd5,0xfb,0x04,0x02,0x00]
// GFX11-ERR: error: execz and vccz are not supported on this GPU

v_add_f64 v[0:1], v[1:2], vccz
// GFX10: encoding: [0x00,0x00,0x64,0xd5,0x01,0xf7,0x01,0x00]
// GFX11-ERR: error: execz and vccz are not supported on this GPU

v_add_f64 v[0:1], v[1:2], src_vccz
// GFX10: encoding: [0x00,0x00,0x64,0xd5,0x01,0xf7,0x01,0x00]
// GFX11-ERR: error: execz and vccz are not supported on this GPU

//---------------------------------------------------------------------------//
// LDS_DIRECT
//---------------------------------------------------------------------------//

v_readfirstlane_b32 s0, lds_direct
// GFX10: encoding: [0xfe,0x04,0x00,0x7e]
// GFX11-ERR: error: lds_direct is not supported on this GPU

v_readfirstlane_b32 s0, src_lds_direct
// GFX10: encoding: [0xfe,0x04,0x00,0x7e]
// GFX11-ERR: error: lds_direct is not supported on this GPU

v_mov_b32 v0, lds_direct
// GFX10: encoding: [0xfe,0x02,0x00,0x7e]
// GFX11-ERR: error: lds_direct is not supported on this GPU

v_mov_b32 v0, src_lds_direct
// GFX10: encoding: [0xfe,0x02,0x00,0x7e]
// GFX11-ERR: error: lds_direct is not supported on this GPU
