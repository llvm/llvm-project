; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GFX9 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefix=GFX10 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck -check-prefix=GFX11 %s

define <4 x float> @rebroadcast_v4f32(ptr addrspace(1) %arg0) {
; GFX9-LABEL: rebroadcast_v4f32:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dwordx4 v[0:3], v[0:1], off
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_mov_b32_e32 v0, v1
; GFX9-NEXT:  v_mov_b32_e32 v2, v1
; GFX9-NEXT:  v_mov_b32_e32 v3, v1
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: rebroadcast_v4f32:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dwordx4 v[0:3], v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_mov_b32_e32 v0, v1
; GFX10-NEXT:  v_mov_b32_e32 v2, v1
; GFX10-NEXT:  v_mov_b32_e32 v3, v1
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: rebroadcast_v4f32:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b128 v[0:3], v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_mov_b32_e32 v0, v1
; GFX11-NEXT:  v_mov_b32_e32 v2, v1
; GFX11-NEXT:  v_mov_b32_e32 v3, v1
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <4 x float>, ptr addrspace(1) %arg0
  %val1 = shufflevector <4 x float> %val0, <4 x float> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ret <4 x float> %val1
}
