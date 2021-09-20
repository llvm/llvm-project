// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -show-encoding %s | FileCheck --check-prefix=GFX11 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX11-ERR %s

// On GFX11+, EXECZ and VCCZ are no longer allowed to be used as sources to SALU and VALU instructions.
// The inline constants are removed. VCCZ and EXECZ still exist and can be use for conditional branches.

s_cbranch_execz 0x100
// GFX11: encoding: [0x00,0x01,0xa5,0xbf]

s_cbranch_vccz 0x100
// GFX11: encoding: [0x00,0x01,0xa3,0xbf]

s_add_i32 s0, s1, s2
// GFX11: encoding: [0x01,0x02,0x00,0x81]

s_add_i32 s0, execz, s2
// GFX11-ERR: error: execz and vccz are not supported on this GPU

s_add_i32 s0, vccz, s2
// GFX11-ERR: error: execz and vccz are not supported on this GPU

s_add_i32 s0, src_execz, s2
// GFX11-ERR: error: execz and vccz are not supported on this GPU

s_add_i32 s0, src_vccz, s2
// GFX11-ERR: error: execz and vccz are not supported on this GPU

s_add_i32 s0, s1, execz
// GFX11-ERR: error: execz and vccz are not supported on this GPU

s_add_i32 s0, s1, vccz
// GFX11-ERR: error: execz and vccz are not supported on this GPU

s_add_i32 s0, s1, src_execz
// GFX11-ERR: error: execz and vccz are not supported on this GPU

s_add_i32 s0, s1, src_vccz
// GFX11-ERR: error: execz and vccz are not supported on this GPU

v_add_f64 v[0:1], v[1:2], v[2:3]
// GFX11: encoding: [0x00,0x00,0x27,0xd7,0x01,0x05,0x02,0x00]

v_add_f64 v[0:1], execz, v[2:3]
// GFX11-ERR: error: execz and vccz are not supported on this GPU

v_add_f64 v[0:1], vccz, v[2:3]
// GFX11-ERR: error: execz and vccz are not supported on this GPU

v_add_f64 v[0:1], src_execz, v[2:3]
// GFX11-ERR: error: execz and vccz are not supported on this GPU

v_add_f64 v[0:1], src_vccz, v[2:3]
// GFX11-ERR: error: execz and vccz are not supported on this GPU

v_add_f64 v[0:1], v[1:2], execz
// GFX11-ERR: error: execz and vccz are not supported on this GPU

v_add_f64 v[0:1], v[1:2], vccz
// GFX11-ERR: error: execz and vccz are not supported on this GPU

v_add_f64 v[0:1], v[1:2], src_execz
// GFX11-ERR: error: execz and vccz are not supported on this GPU

v_add_f64 v[0:1], v[1:2], src_vccz
// GFX11-ERR: error: execz and vccz are not supported on this GPU
