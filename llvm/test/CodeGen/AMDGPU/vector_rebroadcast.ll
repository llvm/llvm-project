; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefix=GFX9 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 < %s | FileCheck -check-prefix=GFX10 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 < %s | FileCheck -check-prefix=GFX11 %s

define <2 x i8> @shuffle_v2i8_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v2i8_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_ushort v0, v[0:1], off
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_lshrrev_b16_e32 v0, 8, v0
; GFX9-NEXT:  v_mov_b32_e32 v1, v0
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v2i8_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_ushort v0, v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_lshrrev_b16 v0, 8, v0
; GFX10-NEXT:  v_mov_b32_e32 v1, v0
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v2i8_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_u16 v0, v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_lshrrev_b16 v0, 8, v0
; GFX11-NEXT:  s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:  v_mov_b32_e32 v1, v0
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <2 x i8>, ptr addrspace(1) %arg0
  %val1 = shufflevector <2 x i8> %val0, <2 x i8> poison, <2 x i32> <i32 1, i32 1>
  ret <2 x i8> %val1
}

define <4 x i8> @shuffle_v4i8_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v4i8_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v0, v[0:1], off
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_lshrrev_b32_e32 v0, 8, v0
; GFX9-NEXT:  v_mov_b32_e32 v1, v0
; GFX9-NEXT:  v_mov_b32_e32 v2, v0
; GFX9-NEXT:  v_mov_b32_e32 v3, v0
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v4i8_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v0, v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_lshrrev_b32_e32 v0, 8, v0
; GFX10-NEXT:  v_mov_b32_e32 v1, v0
; GFX10-NEXT:  v_mov_b32_e32 v2, v0
; GFX10-NEXT:  v_mov_b32_e32 v3, v0
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v4i8_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v0, v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_lshrrev_b32_e32 v0, 8, v0
; GFX11-NEXT:  s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:  v_mov_b32_e32 v1, v0
; GFX11-NEXT:  v_mov_b32_e32 v2, v0
; GFX11-NEXT:  v_mov_b32_e32 v3, v0
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <4 x i8>, ptr addrspace(1) %arg0
  %val1 = shufflevector <4 x i8> %val0, <4 x i8> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i8> %val1
}

define <8 x i8> @shuffle_v8i8_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v8i8_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v0, v[0:1], off
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_lshrrev_b32_e32 v0, 8, v0
; GFX9-NEXT:  v_mov_b32_e32 v1, v0
; GFX9-NEXT:  v_mov_b32_e32 v2, v0
; GFX9-NEXT:  v_mov_b32_e32 v3, v0
; GFX9-NEXT:  v_mov_b32_e32 v4, v0
; GFX9-NEXT:  v_mov_b32_e32 v5, v0
; GFX9-NEXT:  v_mov_b32_e32 v6, v0
; GFX9-NEXT:  v_mov_b32_e32 v7, v0
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v8i8_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v0, v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_lshrrev_b32_e32 v0, 8, v0
; GFX10-NEXT:  v_mov_b32_e32 v1, v0
; GFX10-NEXT:  v_mov_b32_e32 v2, v0
; GFX10-NEXT:  v_mov_b32_e32 v3, v0
; GFX10-NEXT:  v_mov_b32_e32 v4, v0
; GFX10-NEXT:  v_mov_b32_e32 v5, v0
; GFX10-NEXT:  v_mov_b32_e32 v6, v0
; GFX10-NEXT:  v_mov_b32_e32 v7, v0
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v8i8_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v0, v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_lshrrev_b32_e32 v0, 8, v0
; GFX11-NEXT:  s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:  v_mov_b32_e32 v1, v0
; GFX11-NEXT:  v_mov_b32_e32 v2, v0
; GFX11-NEXT:  v_mov_b32_e32 v3, v0
; GFX11-NEXT:  v_mov_b32_e32 v4, v0
; GFX11-NEXT:  v_mov_b32_e32 v5, v0
; GFX11-NEXT:  v_mov_b32_e32 v6, v0
; GFX11-NEXT:  v_mov_b32_e32 v7, v0
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <8 x i8>, ptr addrspace(1) %arg0
  %val1 = shufflevector <8 x i8> %val0, <8 x i8> poison, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <8 x i8> %val1
}

define <16 x i8> @shuffle_v16i8_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v16i8_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v0, v[0:1], off
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_lshrrev_b32_e32 v0, 8, v0
; GFX9-NEXT:  v_mov_b32_e32 v1, v0
; GFX9-NEXT:  v_mov_b32_e32 v2, v0
; GFX9-NEXT:  v_mov_b32_e32 v3, v0
; GFX9-NEXT:  v_mov_b32_e32 v4, v0
; GFX9-NEXT:  v_mov_b32_e32 v5, v0
; GFX9-NEXT:  v_mov_b32_e32 v6, v0
; GFX9-NEXT:  v_mov_b32_e32 v7, v0
; GFX9-NEXT:  v_mov_b32_e32 v8, v0
; GFX9-NEXT:  v_mov_b32_e32 v9, v0
; GFX9-NEXT:  v_mov_b32_e32 v10, v0
; GFX9-NEXT:  v_mov_b32_e32 v11, v0
; GFX9-NEXT:  v_mov_b32_e32 v12, v0
; GFX9-NEXT:  v_mov_b32_e32 v13, v0
; GFX9-NEXT:  v_mov_b32_e32 v14, v0
; GFX9-NEXT:  v_mov_b32_e32 v15, v0
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v16i8_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v0, v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_lshrrev_b32_e32 v0, 8, v0
; GFX10-NEXT:  v_mov_b32_e32 v1, v0
; GFX10-NEXT:  v_mov_b32_e32 v2, v0
; GFX10-NEXT:  v_mov_b32_e32 v3, v0
; GFX10-NEXT:  v_mov_b32_e32 v4, v0
; GFX10-NEXT:  v_mov_b32_e32 v5, v0
; GFX10-NEXT:  v_mov_b32_e32 v6, v0
; GFX10-NEXT:  v_mov_b32_e32 v7, v0
; GFX10-NEXT:  v_mov_b32_e32 v8, v0
; GFX10-NEXT:  v_mov_b32_e32 v9, v0
; GFX10-NEXT:  v_mov_b32_e32 v10, v0
; GFX10-NEXT:  v_mov_b32_e32 v11, v0
; GFX10-NEXT:  v_mov_b32_e32 v12, v0
; GFX10-NEXT:  v_mov_b32_e32 v13, v0
; GFX10-NEXT:  v_mov_b32_e32 v14, v0
; GFX10-NEXT:  v_mov_b32_e32 v15, v0
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v16i8_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v0, v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_lshrrev_b32_e32 v0, 8, v0
; GFX11-NEXT:  s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:  v_mov_b32_e32 v1, v0
; GFX11-NEXT:  v_mov_b32_e32 v2, v0
; GFX11-NEXT:  v_mov_b32_e32 v3, v0
; GFX11-NEXT:  v_mov_b32_e32 v4, v0
; GFX11-NEXT:  v_mov_b32_e32 v5, v0
; GFX11-NEXT:  v_mov_b32_e32 v6, v0
; GFX11-NEXT:  v_mov_b32_e32 v7, v0
; GFX11-NEXT:  v_mov_b32_e32 v8, v0
; GFX11-NEXT:  v_mov_b32_e32 v9, v0
; GFX11-NEXT:  v_mov_b32_e32 v10, v0
; GFX11-NEXT:  v_mov_b32_e32 v11, v0
; GFX11-NEXT:  v_mov_b32_e32 v12, v0
; GFX11-NEXT:  v_mov_b32_e32 v13, v0
; GFX11-NEXT:  v_mov_b32_e32 v14, v0
; GFX11-NEXT:  v_mov_b32_e32 v15, v0
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <16 x i8>, ptr addrspace(1) %arg0
  %val1 = shufflevector <16 x i8> %val0, <16 x i8> poison, <16 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <16 x i8> %val1
}

define <32 x i8> @shuffle_v32i8_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v32i8_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v0, v[0:1], off
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_lshrrev_b32_e32 v0, 8, v0
; GFX9-NEXT:  v_mov_b32_e32 v1, v0
; GFX9-NEXT:  v_mov_b32_e32 v2, v0
; GFX9-NEXT:  v_mov_b32_e32 v3, v0
; GFX9-NEXT:  v_mov_b32_e32 v4, v0
; GFX9-NEXT:  v_mov_b32_e32 v5, v0
; GFX9-NEXT:  v_mov_b32_e32 v6, v0
; GFX9-NEXT:  v_mov_b32_e32 v7, v0
; GFX9-NEXT:  v_mov_b32_e32 v8, v0
; GFX9-NEXT:  v_mov_b32_e32 v9, v0
; GFX9-NEXT:  v_mov_b32_e32 v10, v0
; GFX9-NEXT:  v_mov_b32_e32 v11, v0
; GFX9-NEXT:  v_mov_b32_e32 v12, v0
; GFX9-NEXT:  v_mov_b32_e32 v13, v0
; GFX9-NEXT:  v_mov_b32_e32 v14, v0
; GFX9-NEXT:  v_mov_b32_e32 v15, v0
; GFX9-NEXT:  v_mov_b32_e32 v16, v0
; GFX9-NEXT:  v_mov_b32_e32 v17, v0
; GFX9-NEXT:  v_mov_b32_e32 v18, v0
; GFX9-NEXT:  v_mov_b32_e32 v19, v0
; GFX9-NEXT:  v_mov_b32_e32 v20, v0
; GFX9-NEXT:  v_mov_b32_e32 v21, v0
; GFX9-NEXT:  v_mov_b32_e32 v22, v0
; GFX9-NEXT:  v_mov_b32_e32 v23, v0
; GFX9-NEXT:  v_mov_b32_e32 v24, v0
; GFX9-NEXT:  v_mov_b32_e32 v25, v0
; GFX9-NEXT:  v_mov_b32_e32 v26, v0
; GFX9-NEXT:  v_mov_b32_e32 v27, v0
; GFX9-NEXT:  v_mov_b32_e32 v28, v0
; GFX9-NEXT:  v_mov_b32_e32 v29, v0
; GFX9-NEXT:  v_mov_b32_e32 v30, v0
; GFX9-NEXT:  v_mov_b32_e32 v31, v0
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v32i8_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v0, v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_lshrrev_b32_e32 v0, 8, v0
; GFX10-NEXT:  v_mov_b32_e32 v1, v0
; GFX10-NEXT:  v_mov_b32_e32 v2, v0
; GFX10-NEXT:  v_mov_b32_e32 v3, v0
; GFX10-NEXT:  v_mov_b32_e32 v4, v0
; GFX10-NEXT:  v_mov_b32_e32 v5, v0
; GFX10-NEXT:  v_mov_b32_e32 v6, v0
; GFX10-NEXT:  v_mov_b32_e32 v7, v0
; GFX10-NEXT:  v_mov_b32_e32 v8, v0
; GFX10-NEXT:  v_mov_b32_e32 v9, v0
; GFX10-NEXT:  v_mov_b32_e32 v10, v0
; GFX10-NEXT:  v_mov_b32_e32 v11, v0
; GFX10-NEXT:  v_mov_b32_e32 v12, v0
; GFX10-NEXT:  v_mov_b32_e32 v13, v0
; GFX10-NEXT:  v_mov_b32_e32 v14, v0
; GFX10-NEXT:  v_mov_b32_e32 v15, v0
; GFX10-NEXT:  v_mov_b32_e32 v16, v0
; GFX10-NEXT:  v_mov_b32_e32 v17, v0
; GFX10-NEXT:  v_mov_b32_e32 v18, v0
; GFX10-NEXT:  v_mov_b32_e32 v19, v0
; GFX10-NEXT:  v_mov_b32_e32 v20, v0
; GFX10-NEXT:  v_mov_b32_e32 v21, v0
; GFX10-NEXT:  v_mov_b32_e32 v22, v0
; GFX10-NEXT:  v_mov_b32_e32 v23, v0
; GFX10-NEXT:  v_mov_b32_e32 v24, v0
; GFX10-NEXT:  v_mov_b32_e32 v25, v0
; GFX10-NEXT:  v_mov_b32_e32 v26, v0
; GFX10-NEXT:  v_mov_b32_e32 v27, v0
; GFX10-NEXT:  v_mov_b32_e32 v28, v0
; GFX10-NEXT:  v_mov_b32_e32 v29, v0
; GFX10-NEXT:  v_mov_b32_e32 v30, v0
; GFX10-NEXT:  v_mov_b32_e32 v31, v0
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v32i8_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v0, v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_lshrrev_b32_e32 v0, 8, v0
; GFX11-NEXT:  s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:  v_mov_b32_e32 v1, v0
; GFX11-NEXT:  v_mov_b32_e32 v2, v0
; GFX11-NEXT:  v_mov_b32_e32 v3, v0
; GFX11-NEXT:  v_mov_b32_e32 v4, v0
; GFX11-NEXT:  v_mov_b32_e32 v5, v0
; GFX11-NEXT:  v_mov_b32_e32 v6, v0
; GFX11-NEXT:  v_mov_b32_e32 v7, v0
; GFX11-NEXT:  v_mov_b32_e32 v8, v0
; GFX11-NEXT:  v_mov_b32_e32 v9, v0
; GFX11-NEXT:  v_mov_b32_e32 v10, v0
; GFX11-NEXT:  v_mov_b32_e32 v11, v0
; GFX11-NEXT:  v_mov_b32_e32 v12, v0
; GFX11-NEXT:  v_mov_b32_e32 v13, v0
; GFX11-NEXT:  v_mov_b32_e32 v14, v0
; GFX11-NEXT:  v_mov_b32_e32 v15, v0
; GFX11-NEXT:  v_mov_b32_e32 v16, v0
; GFX11-NEXT:  v_mov_b32_e32 v17, v0
; GFX11-NEXT:  v_mov_b32_e32 v18, v0
; GFX11-NEXT:  v_mov_b32_e32 v19, v0
; GFX11-NEXT:  v_mov_b32_e32 v20, v0
; GFX11-NEXT:  v_mov_b32_e32 v21, v0
; GFX11-NEXT:  v_mov_b32_e32 v22, v0
; GFX11-NEXT:  v_mov_b32_e32 v23, v0
; GFX11-NEXT:  v_mov_b32_e32 v24, v0
; GFX11-NEXT:  v_mov_b32_e32 v25, v0
; GFX11-NEXT:  v_mov_b32_e32 v26, v0
; GFX11-NEXT:  v_mov_b32_e32 v27, v0
; GFX11-NEXT:  v_mov_b32_e32 v28, v0
; GFX11-NEXT:  v_mov_b32_e32 v29, v0
; GFX11-NEXT:  v_mov_b32_e32 v30, v0
; GFX11-NEXT:  v_mov_b32_e32 v31, v0
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <32 x i8>, ptr addrspace(1) %arg0
  %val1 = shufflevector <32 x i8> %val0, <32 x i8> poison, <32 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <32 x i8> %val1
}

define <2 x i16> @shuffle_v2i16_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v2i16_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v0, v[0:1], off
; GFX9-NEXT:  s_mov_b32 s4, 0x7060302
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_perm_b32 v0, v0, v0, s4
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v2i16_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v0, v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v2i16_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v0, v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <2 x i16>, ptr addrspace(1) %arg0
  %val1 = shufflevector <2 x i16> %val0, <2 x i16> poison, <2 x i32> <i32 1, i32 1>
  ret <2 x i16> %val1
}

define <4 x i16> @shuffle_v4i16_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v4i16_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v0, v[0:1], off
; GFX9-NEXT:  s_mov_b32 s4, 0x7060302
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_perm_b32 v0, v0, v0, s4
; GFX9-NEXT:  v_mov_b32_e32 v1, v0
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v4i16_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v0, v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX10-NEXT:  v_mov_b32_e32 v1, v0
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v4i16_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v0, v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX11-NEXT:  s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:  v_mov_b32_e32 v1, v0
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <4 x i16>, ptr addrspace(1) %arg0
  %val1 = shufflevector <4 x i16> %val0, <4 x i16> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i16> %val1
}

define <8 x i16> @shuffle_v8i16_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v8i16_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v0, v[0:1], off
; GFX9-NEXT:  s_mov_b32 s4, 0x7060302
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_perm_b32 v0, v0, v0, s4
; GFX9-NEXT:  v_mov_b32_e32 v1, v0
; GFX9-NEXT:  v_mov_b32_e32 v2, v0
; GFX9-NEXT:  v_mov_b32_e32 v3, v0
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v8i16_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v0, v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX10-NEXT:  v_mov_b32_e32 v1, v0
; GFX10-NEXT:  v_mov_b32_e32 v2, v0
; GFX10-NEXT:  v_mov_b32_e32 v3, v0
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v8i16_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v0, v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX11-NEXT:  s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:  v_mov_b32_e32 v1, v0
; GFX11-NEXT:  v_mov_b32_e32 v2, v0
; GFX11-NEXT:  v_mov_b32_e32 v3, v0
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <8 x i16>, ptr addrspace(1) %arg0
  %val1 = shufflevector <8 x i16> %val0, <8 x i16> poison, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <8 x i16> %val1
}

define <16 x i16> @shuffle_v16i16_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v16i16_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v0, v[0:1], off
; GFX9-NEXT:  s_mov_b32 s4, 0x7060302
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_perm_b32 v0, v0, v0, s4
; GFX9-NEXT:  v_mov_b32_e32 v1, v0
; GFX9-NEXT:  v_mov_b32_e32 v2, v0
; GFX9-NEXT:  v_mov_b32_e32 v3, v0
; GFX9-NEXT:  v_mov_b32_e32 v4, v0
; GFX9-NEXT:  v_mov_b32_e32 v5, v0
; GFX9-NEXT:  v_mov_b32_e32 v6, v0
; GFX9-NEXT:  v_mov_b32_e32 v7, v0
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v16i16_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v0, v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX10-NEXT:  v_mov_b32_e32 v1, v0
; GFX10-NEXT:  v_mov_b32_e32 v2, v0
; GFX10-NEXT:  v_mov_b32_e32 v3, v0
; GFX10-NEXT:  v_mov_b32_e32 v4, v0
; GFX10-NEXT:  v_mov_b32_e32 v5, v0
; GFX10-NEXT:  v_mov_b32_e32 v6, v0
; GFX10-NEXT:  v_mov_b32_e32 v7, v0
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v16i16_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v0, v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX11-NEXT:  s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:  v_mov_b32_e32 v1, v0
; GFX11-NEXT:  v_mov_b32_e32 v2, v0
; GFX11-NEXT:  v_mov_b32_e32 v3, v0
; GFX11-NEXT:  v_mov_b32_e32 v4, v0
; GFX11-NEXT:  v_mov_b32_e32 v5, v0
; GFX11-NEXT:  v_mov_b32_e32 v6, v0
; GFX11-NEXT:  v_mov_b32_e32 v7, v0
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <16 x i16>, ptr addrspace(1) %arg0
  %val1 = shufflevector <16 x i16> %val0, <16 x i16> poison, <16 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <16 x i16> %val1
}

define <32 x i16> @shuffle_v32i16_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v32i16_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v0, v[0:1], off
; GFX9-NEXT:  s_mov_b32 s4, 0x7060302
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_perm_b32 v0, v0, v0, s4
; GFX9-NEXT:  v_mov_b32_e32 v1, v0
; GFX9-NEXT:  v_mov_b32_e32 v2, v0
; GFX9-NEXT:  v_mov_b32_e32 v3, v0
; GFX9-NEXT:  v_mov_b32_e32 v4, v0
; GFX9-NEXT:  v_mov_b32_e32 v5, v0
; GFX9-NEXT:  v_mov_b32_e32 v6, v0
; GFX9-NEXT:  v_mov_b32_e32 v7, v0
; GFX9-NEXT:  v_mov_b32_e32 v8, v0
; GFX9-NEXT:  v_mov_b32_e32 v9, v0
; GFX9-NEXT:  v_mov_b32_e32 v10, v0
; GFX9-NEXT:  v_mov_b32_e32 v11, v0
; GFX9-NEXT:  v_mov_b32_e32 v12, v0
; GFX9-NEXT:  v_mov_b32_e32 v13, v0
; GFX9-NEXT:  v_mov_b32_e32 v14, v0
; GFX9-NEXT:  v_mov_b32_e32 v15, v0
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v32i16_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v0, v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX10-NEXT:  v_mov_b32_e32 v1, v0
; GFX10-NEXT:  v_mov_b32_e32 v2, v0
; GFX10-NEXT:  v_mov_b32_e32 v3, v0
; GFX10-NEXT:  v_mov_b32_e32 v4, v0
; GFX10-NEXT:  v_mov_b32_e32 v5, v0
; GFX10-NEXT:  v_mov_b32_e32 v6, v0
; GFX10-NEXT:  v_mov_b32_e32 v7, v0
; GFX10-NEXT:  v_mov_b32_e32 v8, v0
; GFX10-NEXT:  v_mov_b32_e32 v9, v0
; GFX10-NEXT:  v_mov_b32_e32 v10, v0
; GFX10-NEXT:  v_mov_b32_e32 v11, v0
; GFX10-NEXT:  v_mov_b32_e32 v12, v0
; GFX10-NEXT:  v_mov_b32_e32 v13, v0
; GFX10-NEXT:  v_mov_b32_e32 v14, v0
; GFX10-NEXT:  v_mov_b32_e32 v15, v0
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v32i16_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v0, v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX11-NEXT:  s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:  v_mov_b32_e32 v1, v0
; GFX11-NEXT:  v_mov_b32_e32 v2, v0
; GFX11-NEXT:  v_mov_b32_e32 v3, v0
; GFX11-NEXT:  v_mov_b32_e32 v4, v0
; GFX11-NEXT:  v_mov_b32_e32 v5, v0
; GFX11-NEXT:  v_mov_b32_e32 v6, v0
; GFX11-NEXT:  v_mov_b32_e32 v7, v0
; GFX11-NEXT:  v_mov_b32_e32 v8, v0
; GFX11-NEXT:  v_mov_b32_e32 v9, v0
; GFX11-NEXT:  v_mov_b32_e32 v10, v0
; GFX11-NEXT:  v_mov_b32_e32 v11, v0
; GFX11-NEXT:  v_mov_b32_e32 v12, v0
; GFX11-NEXT:  v_mov_b32_e32 v13, v0
; GFX11-NEXT:  v_mov_b32_e32 v14, v0
; GFX11-NEXT:  v_mov_b32_e32 v15, v0
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <32 x i16>, ptr addrspace(1) %arg0
  %val1 = shufflevector <32 x i16> %val0, <32 x i16> poison, <32 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <32 x i16> %val1
}

define <2 x i32> @shuffle_v2i32_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v2i32_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v0, v[0:1], off offset:4
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_mov_b32_e32 v1, v0
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v2i32_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v0, v[0:1], off offset:4
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_mov_b32_e32 v1, v0
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v2i32_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v0, v[0:1], off offset:4
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_mov_b32_e32 v1, v0
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <2 x i32>, ptr addrspace(1) %arg0
  %val1 = shufflevector <2 x i32> %val0, <2 x i32> poison, <2 x i32> <i32 1, i32 1>
  ret <2 x i32> %val1
}

define <4 x i32> @shuffle_v4i32_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v4i32_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v0, v[0:1], off offset:4
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_mov_b32_e32 v1, v0
; GFX9-NEXT:  v_mov_b32_e32 v2, v0
; GFX9-NEXT:  v_mov_b32_e32 v3, v0
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v4i32_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v0, v[0:1], off offset:4
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_mov_b32_e32 v1, v0
; GFX10-NEXT:  v_mov_b32_e32 v2, v0
; GFX10-NEXT:  v_mov_b32_e32 v3, v0
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v4i32_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v0, v[0:1], off offset:4
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_mov_b32_e32 v1, v0
; GFX11-NEXT:  v_mov_b32_e32 v2, v0
; GFX11-NEXT:  v_mov_b32_e32 v3, v0
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <4 x i32>, ptr addrspace(1) %arg0
  %val1 = shufflevector <4 x i32> %val0, <4 x i32> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %val1
}

define <8 x i32> @shuffle_v8i32_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v8i32_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v0, v[0:1], off offset:4
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_mov_b32_e32 v1, v0
; GFX9-NEXT:  v_mov_b32_e32 v2, v0
; GFX9-NEXT:  v_mov_b32_e32 v3, v0
; GFX9-NEXT:  v_mov_b32_e32 v4, v0
; GFX9-NEXT:  v_mov_b32_e32 v5, v0
; GFX9-NEXT:  v_mov_b32_e32 v6, v0
; GFX9-NEXT:  v_mov_b32_e32 v7, v0
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v8i32_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v0, v[0:1], off offset:4
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_mov_b32_e32 v1, v0
; GFX10-NEXT:  v_mov_b32_e32 v2, v0
; GFX10-NEXT:  v_mov_b32_e32 v3, v0
; GFX10-NEXT:  v_mov_b32_e32 v4, v0
; GFX10-NEXT:  v_mov_b32_e32 v5, v0
; GFX10-NEXT:  v_mov_b32_e32 v6, v0
; GFX10-NEXT:  v_mov_b32_e32 v7, v0
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v8i32_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v0, v[0:1], off offset:4
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_mov_b32_e32 v1, v0
; GFX11-NEXT:  v_mov_b32_e32 v2, v0
; GFX11-NEXT:  v_mov_b32_e32 v3, v0
; GFX11-NEXT:  v_mov_b32_e32 v4, v0
; GFX11-NEXT:  v_mov_b32_e32 v5, v0
; GFX11-NEXT:  v_mov_b32_e32 v6, v0
; GFX11-NEXT:  v_mov_b32_e32 v7, v0
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <8 x i32>, ptr addrspace(1) %arg0
  %val1 = shufflevector <8 x i32> %val0, <8 x i32> poison, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <8 x i32> %val1
}

define <16 x i32> @shuffle_v16i32_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v16i32_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v0, v[0:1], off offset:4
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_mov_b32_e32 v1, v0
; GFX9-NEXT:  v_mov_b32_e32 v2, v0
; GFX9-NEXT:  v_mov_b32_e32 v3, v0
; GFX9-NEXT:  v_mov_b32_e32 v4, v0
; GFX9-NEXT:  v_mov_b32_e32 v5, v0
; GFX9-NEXT:  v_mov_b32_e32 v6, v0
; GFX9-NEXT:  v_mov_b32_e32 v7, v0
; GFX9-NEXT:  v_mov_b32_e32 v8, v0
; GFX9-NEXT:  v_mov_b32_e32 v9, v0
; GFX9-NEXT:  v_mov_b32_e32 v10, v0
; GFX9-NEXT:  v_mov_b32_e32 v11, v0
; GFX9-NEXT:  v_mov_b32_e32 v12, v0
; GFX9-NEXT:  v_mov_b32_e32 v13, v0
; GFX9-NEXT:  v_mov_b32_e32 v14, v0
; GFX9-NEXT:  v_mov_b32_e32 v15, v0
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v16i32_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v0, v[0:1], off offset:4
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_mov_b32_e32 v1, v0
; GFX10-NEXT:  v_mov_b32_e32 v2, v0
; GFX10-NEXT:  v_mov_b32_e32 v3, v0
; GFX10-NEXT:  v_mov_b32_e32 v4, v0
; GFX10-NEXT:  v_mov_b32_e32 v5, v0
; GFX10-NEXT:  v_mov_b32_e32 v6, v0
; GFX10-NEXT:  v_mov_b32_e32 v7, v0
; GFX10-NEXT:  v_mov_b32_e32 v8, v0
; GFX10-NEXT:  v_mov_b32_e32 v9, v0
; GFX10-NEXT:  v_mov_b32_e32 v10, v0
; GFX10-NEXT:  v_mov_b32_e32 v11, v0
; GFX10-NEXT:  v_mov_b32_e32 v12, v0
; GFX10-NEXT:  v_mov_b32_e32 v13, v0
; GFX10-NEXT:  v_mov_b32_e32 v14, v0
; GFX10-NEXT:  v_mov_b32_e32 v15, v0
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v16i32_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v0, v[0:1], off offset:4
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_mov_b32_e32 v1, v0
; GFX11-NEXT:  v_mov_b32_e32 v2, v0
; GFX11-NEXT:  v_mov_b32_e32 v3, v0
; GFX11-NEXT:  v_mov_b32_e32 v4, v0
; GFX11-NEXT:  v_mov_b32_e32 v5, v0
; GFX11-NEXT:  v_mov_b32_e32 v6, v0
; GFX11-NEXT:  v_mov_b32_e32 v7, v0
; GFX11-NEXT:  v_mov_b32_e32 v8, v0
; GFX11-NEXT:  v_mov_b32_e32 v9, v0
; GFX11-NEXT:  v_mov_b32_e32 v10, v0
; GFX11-NEXT:  v_mov_b32_e32 v11, v0
; GFX11-NEXT:  v_mov_b32_e32 v12, v0
; GFX11-NEXT:  v_mov_b32_e32 v13, v0
; GFX11-NEXT:  v_mov_b32_e32 v14, v0
; GFX11-NEXT:  v_mov_b32_e32 v15, v0
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <16 x i32>, ptr addrspace(1) %arg0
  %val1 = shufflevector <16 x i32> %val0, <16 x i32> poison, <16 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <16 x i32> %val1
}

define <32 x i32> @shuffle_v32i32_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v32i32_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v0, v[0:1], off offset:4
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_mov_b32_e32 v1, v0
; GFX9-NEXT:  v_mov_b32_e32 v2, v0
; GFX9-NEXT:  v_mov_b32_e32 v3, v0
; GFX9-NEXT:  v_mov_b32_e32 v4, v0
; GFX9-NEXT:  v_mov_b32_e32 v5, v0
; GFX9-NEXT:  v_mov_b32_e32 v6, v0
; GFX9-NEXT:  v_mov_b32_e32 v7, v0
; GFX9-NEXT:  v_mov_b32_e32 v8, v0
; GFX9-NEXT:  v_mov_b32_e32 v9, v0
; GFX9-NEXT:  v_mov_b32_e32 v10, v0
; GFX9-NEXT:  v_mov_b32_e32 v11, v0
; GFX9-NEXT:  v_mov_b32_e32 v12, v0
; GFX9-NEXT:  v_mov_b32_e32 v13, v0
; GFX9-NEXT:  v_mov_b32_e32 v14, v0
; GFX9-NEXT:  v_mov_b32_e32 v15, v0
; GFX9-NEXT:  v_mov_b32_e32 v16, v0
; GFX9-NEXT:  v_mov_b32_e32 v17, v0
; GFX9-NEXT:  v_mov_b32_e32 v18, v0
; GFX9-NEXT:  v_mov_b32_e32 v19, v0
; GFX9-NEXT:  v_mov_b32_e32 v20, v0
; GFX9-NEXT:  v_mov_b32_e32 v21, v0
; GFX9-NEXT:  v_mov_b32_e32 v22, v0
; GFX9-NEXT:  v_mov_b32_e32 v23, v0
; GFX9-NEXT:  v_mov_b32_e32 v24, v0
; GFX9-NEXT:  v_mov_b32_e32 v25, v0
; GFX9-NEXT:  v_mov_b32_e32 v26, v0
; GFX9-NEXT:  v_mov_b32_e32 v27, v0
; GFX9-NEXT:  v_mov_b32_e32 v28, v0
; GFX9-NEXT:  v_mov_b32_e32 v29, v0
; GFX9-NEXT:  v_mov_b32_e32 v30, v0
; GFX9-NEXT:  v_mov_b32_e32 v31, v0
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v32i32_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v0, v[0:1], off offset:4
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_mov_b32_e32 v1, v0
; GFX10-NEXT:  v_mov_b32_e32 v2, v0
; GFX10-NEXT:  v_mov_b32_e32 v3, v0
; GFX10-NEXT:  v_mov_b32_e32 v4, v0
; GFX10-NEXT:  v_mov_b32_e32 v5, v0
; GFX10-NEXT:  v_mov_b32_e32 v6, v0
; GFX10-NEXT:  v_mov_b32_e32 v7, v0
; GFX10-NEXT:  v_mov_b32_e32 v8, v0
; GFX10-NEXT:  v_mov_b32_e32 v9, v0
; GFX10-NEXT:  v_mov_b32_e32 v10, v0
; GFX10-NEXT:  v_mov_b32_e32 v11, v0
; GFX10-NEXT:  v_mov_b32_e32 v12, v0
; GFX10-NEXT:  v_mov_b32_e32 v13, v0
; GFX10-NEXT:  v_mov_b32_e32 v14, v0
; GFX10-NEXT:  v_mov_b32_e32 v15, v0
; GFX10-NEXT:  v_mov_b32_e32 v16, v0
; GFX10-NEXT:  v_mov_b32_e32 v17, v0
; GFX10-NEXT:  v_mov_b32_e32 v18, v0
; GFX10-NEXT:  v_mov_b32_e32 v19, v0
; GFX10-NEXT:  v_mov_b32_e32 v20, v0
; GFX10-NEXT:  v_mov_b32_e32 v21, v0
; GFX10-NEXT:  v_mov_b32_e32 v22, v0
; GFX10-NEXT:  v_mov_b32_e32 v23, v0
; GFX10-NEXT:  v_mov_b32_e32 v24, v0
; GFX10-NEXT:  v_mov_b32_e32 v25, v0
; GFX10-NEXT:  v_mov_b32_e32 v26, v0
; GFX10-NEXT:  v_mov_b32_e32 v27, v0
; GFX10-NEXT:  v_mov_b32_e32 v28, v0
; GFX10-NEXT:  v_mov_b32_e32 v29, v0
; GFX10-NEXT:  v_mov_b32_e32 v30, v0
; GFX10-NEXT:  v_mov_b32_e32 v31, v0
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v32i32_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v0, v[0:1], off offset:4
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_mov_b32_e32 v1, v0
; GFX11-NEXT:  v_mov_b32_e32 v2, v0
; GFX11-NEXT:  v_mov_b32_e32 v3, v0
; GFX11-NEXT:  v_mov_b32_e32 v4, v0
; GFX11-NEXT:  v_mov_b32_e32 v5, v0
; GFX11-NEXT:  v_mov_b32_e32 v6, v0
; GFX11-NEXT:  v_mov_b32_e32 v7, v0
; GFX11-NEXT:  v_mov_b32_e32 v8, v0
; GFX11-NEXT:  v_mov_b32_e32 v9, v0
; GFX11-NEXT:  v_mov_b32_e32 v10, v0
; GFX11-NEXT:  v_mov_b32_e32 v11, v0
; GFX11-NEXT:  v_mov_b32_e32 v12, v0
; GFX11-NEXT:  v_mov_b32_e32 v13, v0
; GFX11-NEXT:  v_mov_b32_e32 v14, v0
; GFX11-NEXT:  v_mov_b32_e32 v15, v0
; GFX11-NEXT:  v_mov_b32_e32 v16, v0
; GFX11-NEXT:  v_mov_b32_e32 v17, v0
; GFX11-NEXT:  v_mov_b32_e32 v18, v0
; GFX11-NEXT:  v_mov_b32_e32 v19, v0
; GFX11-NEXT:  v_mov_b32_e32 v20, v0
; GFX11-NEXT:  v_mov_b32_e32 v21, v0
; GFX11-NEXT:  v_mov_b32_e32 v22, v0
; GFX11-NEXT:  v_mov_b32_e32 v23, v0
; GFX11-NEXT:  v_mov_b32_e32 v24, v0
; GFX11-NEXT:  v_mov_b32_e32 v25, v0
; GFX11-NEXT:  v_mov_b32_e32 v26, v0
; GFX11-NEXT:  v_mov_b32_e32 v27, v0
; GFX11-NEXT:  v_mov_b32_e32 v28, v0
; GFX11-NEXT:  v_mov_b32_e32 v29, v0
; GFX11-NEXT:  v_mov_b32_e32 v30, v0
; GFX11-NEXT:  v_mov_b32_e32 v31, v0
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <32 x i32>, ptr addrspace(1) %arg0
  %val1 = shufflevector <32 x i32> %val0, <32 x i32> poison, <32 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <32 x i32> %val1
}

define <2 x bfloat> @shuffle_v2bf16_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v2bf16_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v0, v[0:1], off
; GFX9-NEXT:  s_mov_b32 s4, 0x7060302
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_perm_b32 v0, v0, v0, s4
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v2bf16_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v0, v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v2bf16_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v0, v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <2 x bfloat>, ptr addrspace(1) %arg0
  %val1 = shufflevector <2 x bfloat> %val0, <2 x bfloat> poison, <2 x i32> <i32 1, i32 1>
  ret <2 x bfloat> %val1
}

define <3 x bfloat> @shuffle_v3bf16_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v3bf16_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v1, v[0:1], off
; GFX9-NEXT:  s_mov_b32 s4, 0x7060302
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_perm_b32 v0, v1, v1, s4
; GFX9-NEXT:  v_alignbit_b32 v1, s4, v1, 16
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v3bf16_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v1, v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_perm_b32 v0, v1, v1, 0x7060302
; GFX10-NEXT:  v_alignbit_b32 v1, s4, v1, 16
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v3bf16_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v1, v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_perm_b32 v0, v1, v1, 0x7060302
; GFX11-NEXT:  v_alignbit_b32 v1, s0, v1, 16
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <3 x bfloat>, ptr addrspace(1) %arg0
  %val1 = shufflevector <3 x bfloat> %val0, <3 x bfloat> poison, <3 x i32> <i32 1, i32 1, i32 1>
  ret <3 x bfloat> %val1
}

define <4 x bfloat> @shuffle_v4bf16_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v4bf16_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v0, v[0:1], off
; GFX9-NEXT:  s_mov_b32 s4, 0x7060302
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_perm_b32 v0, v0, v0, s4
; GFX9-NEXT:  v_mov_b32_e32 v1, v0
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v4bf16_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v0, v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX10-NEXT:  v_mov_b32_e32 v1, v0
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v4bf16_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v0, v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX11-NEXT:  s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:  v_mov_b32_e32 v1, v0
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <4 x bfloat>, ptr addrspace(1) %arg0
  %val1 = shufflevector <4 x bfloat> %val0, <4 x bfloat> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ret <4 x bfloat> %val1
}

define <6 x bfloat> @shuffle_v6bf16_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v6bf16_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v0, v[0:1], off
; GFX9-NEXT:  s_mov_b32 s4, 0x7060302
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_perm_b32 v0, v0, v0, s4
; GFX9-NEXT:  v_mov_b32_e32 v1, v0
; GFX9-NEXT:  v_mov_b32_e32 v2, v0
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v6bf16_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v0, v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX10-NEXT:  v_mov_b32_e32 v1, v0
; GFX10-NEXT:  v_mov_b32_e32 v2, v0
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v6bf16_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v0, v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX11-NEXT:  s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:  v_mov_b32_e32 v1, v0
; GFX11-NEXT:  v_mov_b32_e32 v2, v0
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <6 x bfloat>, ptr addrspace(1) %arg0
  %val1 = shufflevector <6 x bfloat> %val0, <6 x bfloat> poison, <6 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <6 x bfloat> %val1
}

define <8 x bfloat> @shuffle_v8bf16_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v8bf16_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v0, v[0:1], off
; GFX9-NEXT:  s_mov_b32 s4, 0x7060302
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_perm_b32 v0, v0, v0, s4
; GFX9-NEXT:  v_mov_b32_e32 v1, v0
; GFX9-NEXT:  v_mov_b32_e32 v2, v0
; GFX9-NEXT:  v_mov_b32_e32 v3, v0
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v8bf16_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v0, v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX10-NEXT:  v_mov_b32_e32 v1, v0
; GFX10-NEXT:  v_mov_b32_e32 v2, v0
; GFX10-NEXT:  v_mov_b32_e32 v3, v0
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v8bf16_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v0, v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX11-NEXT:  s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:  v_mov_b32_e32 v1, v0
; GFX11-NEXT:  v_mov_b32_e32 v2, v0
; GFX11-NEXT:  v_mov_b32_e32 v3, v0
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <8 x bfloat>, ptr addrspace(1) %arg0
  %val1 = shufflevector <8 x bfloat> %val0, <8 x bfloat> poison, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <8 x bfloat> %val1
}

define <16 x bfloat> @shuffle_v16bf16_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v16bf16_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v0, v[0:1], off
; GFX9-NEXT:  s_mov_b32 s4, 0x7060302
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_perm_b32 v0, v0, v0, s4
; GFX9-NEXT:  v_mov_b32_e32 v1, v0
; GFX9-NEXT:  v_mov_b32_e32 v2, v0
; GFX9-NEXT:  v_mov_b32_e32 v3, v0
; GFX9-NEXT:  v_mov_b32_e32 v4, v0
; GFX9-NEXT:  v_mov_b32_e32 v5, v0
; GFX9-NEXT:  v_mov_b32_e32 v6, v0
; GFX9-NEXT:  v_mov_b32_e32 v7, v0
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v16bf16_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v0, v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX10-NEXT:  v_mov_b32_e32 v1, v0
; GFX10-NEXT:  v_mov_b32_e32 v2, v0
; GFX10-NEXT:  v_mov_b32_e32 v3, v0
; GFX10-NEXT:  v_mov_b32_e32 v4, v0
; GFX10-NEXT:  v_mov_b32_e32 v5, v0
; GFX10-NEXT:  v_mov_b32_e32 v6, v0
; GFX10-NEXT:  v_mov_b32_e32 v7, v0
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v16bf16_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v0, v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX11-NEXT:  s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:  v_mov_b32_e32 v1, v0
; GFX11-NEXT:  v_mov_b32_e32 v2, v0
; GFX11-NEXT:  v_mov_b32_e32 v3, v0
; GFX11-NEXT:  v_mov_b32_e32 v4, v0
; GFX11-NEXT:  v_mov_b32_e32 v5, v0
; GFX11-NEXT:  v_mov_b32_e32 v6, v0
; GFX11-NEXT:  v_mov_b32_e32 v7, v0
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <16 x bfloat>, ptr addrspace(1) %arg0
  %val1 = shufflevector <16 x bfloat> %val0, <16 x bfloat> poison, <16 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <16 x bfloat> %val1
}

define <32 x bfloat> @shuffle_v32bf16_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v32bf16_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v0, v[0:1], off
; GFX9-NEXT:  s_mov_b32 s4, 0x7060302
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_perm_b32 v0, v0, v0, s4
; GFX9-NEXT:  v_mov_b32_e32 v1, v0
; GFX9-NEXT:  v_mov_b32_e32 v2, v0
; GFX9-NEXT:  v_mov_b32_e32 v3, v0
; GFX9-NEXT:  v_mov_b32_e32 v4, v0
; GFX9-NEXT:  v_mov_b32_e32 v5, v0
; GFX9-NEXT:  v_mov_b32_e32 v6, v0
; GFX9-NEXT:  v_mov_b32_e32 v7, v0
; GFX9-NEXT:  v_mov_b32_e32 v8, v0
; GFX9-NEXT:  v_mov_b32_e32 v9, v0
; GFX9-NEXT:  v_mov_b32_e32 v10, v0
; GFX9-NEXT:  v_mov_b32_e32 v11, v0
; GFX9-NEXT:  v_mov_b32_e32 v12, v0
; GFX9-NEXT:  v_mov_b32_e32 v13, v0
; GFX9-NEXT:  v_mov_b32_e32 v14, v0
; GFX9-NEXT:  v_mov_b32_e32 v15, v0
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v32bf16_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v0, v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX10-NEXT:  v_mov_b32_e32 v1, v0
; GFX10-NEXT:  v_mov_b32_e32 v2, v0
; GFX10-NEXT:  v_mov_b32_e32 v3, v0
; GFX10-NEXT:  v_mov_b32_e32 v4, v0
; GFX10-NEXT:  v_mov_b32_e32 v5, v0
; GFX10-NEXT:  v_mov_b32_e32 v6, v0
; GFX10-NEXT:  v_mov_b32_e32 v7, v0
; GFX10-NEXT:  v_mov_b32_e32 v8, v0
; GFX10-NEXT:  v_mov_b32_e32 v9, v0
; GFX10-NEXT:  v_mov_b32_e32 v10, v0
; GFX10-NEXT:  v_mov_b32_e32 v11, v0
; GFX10-NEXT:  v_mov_b32_e32 v12, v0
; GFX10-NEXT:  v_mov_b32_e32 v13, v0
; GFX10-NEXT:  v_mov_b32_e32 v14, v0
; GFX10-NEXT:  v_mov_b32_e32 v15, v0
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v32bf16_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v0, v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX11-NEXT:  s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:  v_mov_b32_e32 v1, v0
; GFX11-NEXT:  v_mov_b32_e32 v2, v0
; GFX11-NEXT:  v_mov_b32_e32 v3, v0
; GFX11-NEXT:  v_mov_b32_e32 v4, v0
; GFX11-NEXT:  v_mov_b32_e32 v5, v0
; GFX11-NEXT:  v_mov_b32_e32 v6, v0
; GFX11-NEXT:  v_mov_b32_e32 v7, v0
; GFX11-NEXT:  v_mov_b32_e32 v8, v0
; GFX11-NEXT:  v_mov_b32_e32 v9, v0
; GFX11-NEXT:  v_mov_b32_e32 v10, v0
; GFX11-NEXT:  v_mov_b32_e32 v11, v0
; GFX11-NEXT:  v_mov_b32_e32 v12, v0
; GFX11-NEXT:  v_mov_b32_e32 v13, v0
; GFX11-NEXT:  v_mov_b32_e32 v14, v0
; GFX11-NEXT:  v_mov_b32_e32 v15, v0
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <32 x bfloat>, ptr addrspace(1) %arg0
  %val1 = shufflevector <32 x bfloat> %val0, <32 x bfloat> poison, <32 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <32 x bfloat> %val1
}

define <2 x half> @shuffle_v2f16_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v2f16_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v0, v[0:1], off
; GFX9-NEXT:  s_mov_b32 s4, 0x7060302
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_perm_b32 v0, v0, v0, s4
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v2f16_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v0, v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v2f16_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v0, v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <2 x half>, ptr addrspace(1) %arg0
  %val1 = shufflevector <2 x half> %val0, <2 x half> poison, <2 x i32> <i32 1, i32 1>
  ret <2 x half> %val1
}

define <3 x half> @shuffle_v3f16_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v3f16_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v1, v[0:1], off
; GFX9-NEXT:  s_mov_b32 s4, 0x7060302
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_perm_b32 v0, v1, v1, s4
; GFX9-NEXT:  v_alignbit_b32 v1, s4, v1, 16
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v3f16_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v1, v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_perm_b32 v0, v1, v1, 0x7060302
; GFX10-NEXT:  v_alignbit_b32 v1, s4, v1, 16
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v3f16_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v1, v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_perm_b32 v0, v1, v1, 0x7060302
; GFX11-NEXT:  v_alignbit_b32 v1, s0, v1, 16
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <3 x half>, ptr addrspace(1) %arg0
  %val1 = shufflevector <3 x half> %val0, <3 x half> poison, <3 x i32> <i32 1, i32 1, i32 1>
  ret <3 x half> %val1
}

define <4 x half> @shuffle_v4f16_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v4f16_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v0, v[0:1], off
; GFX9-NEXT:  s_mov_b32 s4, 0x7060302
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_perm_b32 v0, v0, v0, s4
; GFX9-NEXT:  v_mov_b32_e32 v1, v0
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v4f16_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v0, v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX10-NEXT:  v_mov_b32_e32 v1, v0
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v4f16_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v0, v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX11-NEXT:  s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:  v_mov_b32_e32 v1, v0
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <4 x half>, ptr addrspace(1) %arg0
  %val1 = shufflevector <4 x half> %val0, <4 x half> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ret <4 x half> %val1
}

define <6 x half> @shuffle_v6f16_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v6f16_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v0, v[0:1], off
; GFX9-NEXT:  s_mov_b32 s4, 0x7060302
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_perm_b32 v0, v0, v0, s4
; GFX9-NEXT:  v_mov_b32_e32 v1, v0
; GFX9-NEXT:  v_mov_b32_e32 v2, v0
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v6f16_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v0, v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX10-NEXT:  v_mov_b32_e32 v1, v0
; GFX10-NEXT:  v_mov_b32_e32 v2, v0
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v6f16_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v0, v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX11-NEXT:  s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:  v_mov_b32_e32 v1, v0
; GFX11-NEXT:  v_mov_b32_e32 v2, v0
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <6 x half>, ptr addrspace(1) %arg0
  %val1 = shufflevector <6 x half> %val0, <6 x half> poison, <6 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <6 x half> %val1
}

define <8 x half> @shuffle_v8f16_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v8f16_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v0, v[0:1], off
; GFX9-NEXT:  s_mov_b32 s4, 0x7060302
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_perm_b32 v0, v0, v0, s4
; GFX9-NEXT:  v_mov_b32_e32 v1, v0
; GFX9-NEXT:  v_mov_b32_e32 v2, v0
; GFX9-NEXT:  v_mov_b32_e32 v3, v0
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v8f16_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v0, v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX10-NEXT:  v_mov_b32_e32 v1, v0
; GFX10-NEXT:  v_mov_b32_e32 v2, v0
; GFX10-NEXT:  v_mov_b32_e32 v3, v0
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v8f16_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v0, v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX11-NEXT:  s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:  v_mov_b32_e32 v1, v0
; GFX11-NEXT:  v_mov_b32_e32 v2, v0
; GFX11-NEXT:  v_mov_b32_e32 v3, v0
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <8 x half>, ptr addrspace(1) %arg0
  %val1 = shufflevector <8 x half> %val0, <8 x half> poison, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <8 x half> %val1
}

define <16 x half> @shuffle_v16f16_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v16f16_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v0, v[0:1], off
; GFX9-NEXT:  s_mov_b32 s4, 0x7060302
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_perm_b32 v0, v0, v0, s4
; GFX9-NEXT:  v_mov_b32_e32 v1, v0
; GFX9-NEXT:  v_mov_b32_e32 v2, v0
; GFX9-NEXT:  v_mov_b32_e32 v3, v0
; GFX9-NEXT:  v_mov_b32_e32 v4, v0
; GFX9-NEXT:  v_mov_b32_e32 v5, v0
; GFX9-NEXT:  v_mov_b32_e32 v6, v0
; GFX9-NEXT:  v_mov_b32_e32 v7, v0
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v16f16_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v0, v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX10-NEXT:  v_mov_b32_e32 v1, v0
; GFX10-NEXT:  v_mov_b32_e32 v2, v0
; GFX10-NEXT:  v_mov_b32_e32 v3, v0
; GFX10-NEXT:  v_mov_b32_e32 v4, v0
; GFX10-NEXT:  v_mov_b32_e32 v5, v0
; GFX10-NEXT:  v_mov_b32_e32 v6, v0
; GFX10-NEXT:  v_mov_b32_e32 v7, v0
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v16f16_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v0, v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX11-NEXT:  s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:  v_mov_b32_e32 v1, v0
; GFX11-NEXT:  v_mov_b32_e32 v2, v0
; GFX11-NEXT:  v_mov_b32_e32 v3, v0
; GFX11-NEXT:  v_mov_b32_e32 v4, v0
; GFX11-NEXT:  v_mov_b32_e32 v5, v0
; GFX11-NEXT:  v_mov_b32_e32 v6, v0
; GFX11-NEXT:  v_mov_b32_e32 v7, v0
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <16 x half>, ptr addrspace(1) %arg0
  %val1 = shufflevector <16 x half> %val0, <16 x half> poison, <16 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <16 x half> %val1
}

define <32 x half> @shuffle_v32f16_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v32f16_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dword v0, v[0:1], off
; GFX9-NEXT:  s_mov_b32 s4, 0x7060302
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_perm_b32 v0, v0, v0, s4
; GFX9-NEXT:  v_mov_b32_e32 v1, v0
; GFX9-NEXT:  v_mov_b32_e32 v2, v0
; GFX9-NEXT:  v_mov_b32_e32 v3, v0
; GFX9-NEXT:  v_mov_b32_e32 v4, v0
; GFX9-NEXT:  v_mov_b32_e32 v5, v0
; GFX9-NEXT:  v_mov_b32_e32 v6, v0
; GFX9-NEXT:  v_mov_b32_e32 v7, v0
; GFX9-NEXT:  v_mov_b32_e32 v8, v0
; GFX9-NEXT:  v_mov_b32_e32 v9, v0
; GFX9-NEXT:  v_mov_b32_e32 v10, v0
; GFX9-NEXT:  v_mov_b32_e32 v11, v0
; GFX9-NEXT:  v_mov_b32_e32 v12, v0
; GFX9-NEXT:  v_mov_b32_e32 v13, v0
; GFX9-NEXT:  v_mov_b32_e32 v14, v0
; GFX9-NEXT:  v_mov_b32_e32 v15, v0
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v32f16_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dword v0, v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX10-NEXT:  v_mov_b32_e32 v1, v0
; GFX10-NEXT:  v_mov_b32_e32 v2, v0
; GFX10-NEXT:  v_mov_b32_e32 v3, v0
; GFX10-NEXT:  v_mov_b32_e32 v4, v0
; GFX10-NEXT:  v_mov_b32_e32 v5, v0
; GFX10-NEXT:  v_mov_b32_e32 v6, v0
; GFX10-NEXT:  v_mov_b32_e32 v7, v0
; GFX10-NEXT:  v_mov_b32_e32 v8, v0
; GFX10-NEXT:  v_mov_b32_e32 v9, v0
; GFX10-NEXT:  v_mov_b32_e32 v10, v0
; GFX10-NEXT:  v_mov_b32_e32 v11, v0
; GFX10-NEXT:  v_mov_b32_e32 v12, v0
; GFX10-NEXT:  v_mov_b32_e32 v13, v0
; GFX10-NEXT:  v_mov_b32_e32 v14, v0
; GFX10-NEXT:  v_mov_b32_e32 v15, v0
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v32f16_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b32 v0, v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_perm_b32 v0, v0, v0, 0x7060302
; GFX11-NEXT:  s_delay_alu instid0(VALU_DEP_1)
; GFX11-NEXT:  v_mov_b32_e32 v1, v0
; GFX11-NEXT:  v_mov_b32_e32 v2, v0
; GFX11-NEXT:  v_mov_b32_e32 v3, v0
; GFX11-NEXT:  v_mov_b32_e32 v4, v0
; GFX11-NEXT:  v_mov_b32_e32 v5, v0
; GFX11-NEXT:  v_mov_b32_e32 v6, v0
; GFX11-NEXT:  v_mov_b32_e32 v7, v0
; GFX11-NEXT:  v_mov_b32_e32 v8, v0
; GFX11-NEXT:  v_mov_b32_e32 v9, v0
; GFX11-NEXT:  v_mov_b32_e32 v10, v0
; GFX11-NEXT:  v_mov_b32_e32 v11, v0
; GFX11-NEXT:  v_mov_b32_e32 v12, v0
; GFX11-NEXT:  v_mov_b32_e32 v13, v0
; GFX11-NEXT:  v_mov_b32_e32 v14, v0
; GFX11-NEXT:  v_mov_b32_e32 v15, v0
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <32 x half>, ptr addrspace(1) %arg0
  %val1 = shufflevector <32 x half> %val0, <32 x half> poison, <32 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <32 x half> %val1
}

define <2 x float> @shuffle_v2f32_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v2f32_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dwordx2 v[0:1], v[0:1], off
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_mov_b32_e32 v0, v1
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v2f32_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dwordx2 v[0:1], v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_mov_b32_e32 v0, v1
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v2f32_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b64 v[0:1], v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_mov_b32_e32 v0, v1
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <2 x float>, ptr addrspace(1) %arg0
  %val1 = shufflevector <2 x float> %val0, <2 x float> poison, <2 x i32> <i32 1, i32 1>
  ret <2 x float> %val1
}

define <3 x float> @shuffle_v3f32_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v3f32_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dwordx3 v[0:2], v[0:1], off
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_mov_b32_e32 v0, v1
; GFX9-NEXT:  v_mov_b32_e32 v2, v1
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v3f32_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dwordx3 v[0:2], v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_mov_b32_e32 v0, v1
; GFX10-NEXT:  v_mov_b32_e32 v2, v1
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v3f32_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b96 v[0:2], v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_mov_b32_e32 v0, v1
; GFX11-NEXT:  v_mov_b32_e32 v2, v1
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <3 x float>, ptr addrspace(1) %arg0
  %val1 = shufflevector <3 x float> %val0, <3 x float> poison, <3 x i32> <i32 1, i32 1, i32 1>
  ret <3 x float> %val1
}

define <4 x float> @shuffle_v4f32_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v4f32_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dwordx4 v[0:3], v[0:1], off
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_mov_b32_e32 v0, v1
; GFX9-NEXT:  v_mov_b32_e32 v2, v1
; GFX9-NEXT:  v_mov_b32_e32 v3, v1
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v4f32_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dwordx4 v[0:3], v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_mov_b32_e32 v0, v1
; GFX10-NEXT:  v_mov_b32_e32 v2, v1
; GFX10-NEXT:  v_mov_b32_e32 v3, v1
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v4f32_rebroadcast:
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
  %val1 = shufflevector <4 x float> %val0, <4 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  ret <4 x float> %val1
}

define <6 x float> @shuffle_v6f32_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v6f32_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dwordx4 v[0:3], v[0:1], off
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_mov_b32_e32 v0, v1
; GFX9-NEXT:  v_mov_b32_e32 v2, v1
; GFX9-NEXT:  v_mov_b32_e32 v3, v1
; GFX9-NEXT:  v_mov_b32_e32 v4, v1
; GFX9-NEXT:  v_mov_b32_e32 v5, v1
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v6f32_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dwordx4 v[0:3], v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_mov_b32_e32 v0, v1
; GFX10-NEXT:  v_mov_b32_e32 v2, v1
; GFX10-NEXT:  v_mov_b32_e32 v3, v1
; GFX10-NEXT:  v_mov_b32_e32 v4, v1
; GFX10-NEXT:  v_mov_b32_e32 v5, v1
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v6f32_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b128 v[0:3], v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_mov_b32_e32 v0, v1
; GFX11-NEXT:  v_mov_b32_e32 v2, v1
; GFX11-NEXT:  v_mov_b32_e32 v3, v1
; GFX11-NEXT:  v_mov_b32_e32 v4, v1
; GFX11-NEXT:  v_mov_b32_e32 v5, v1
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <6 x float>, ptr addrspace(1) %arg0
  %val1 = shufflevector <6 x float> %val0, <6 x float> poison, <6 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <6 x float> %val1
}

define <8 x float> @shuffle_v8f32_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v8f32_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dwordx4 v[0:3], v[0:1], off
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_mov_b32_e32 v0, v1
; GFX9-NEXT:  v_mov_b32_e32 v2, v1
; GFX9-NEXT:  v_mov_b32_e32 v3, v1
; GFX9-NEXT:  v_mov_b32_e32 v4, v1
; GFX9-NEXT:  v_mov_b32_e32 v5, v1
; GFX9-NEXT:  v_mov_b32_e32 v6, v1
; GFX9-NEXT:  v_mov_b32_e32 v7, v1
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v8f32_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dwordx4 v[0:3], v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_mov_b32_e32 v0, v1
; GFX10-NEXT:  v_mov_b32_e32 v2, v1
; GFX10-NEXT:  v_mov_b32_e32 v3, v1
; GFX10-NEXT:  v_mov_b32_e32 v4, v1
; GFX10-NEXT:  v_mov_b32_e32 v5, v1
; GFX10-NEXT:  v_mov_b32_e32 v6, v1
; GFX10-NEXT:  v_mov_b32_e32 v7, v1
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v8f32_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b128 v[0:3], v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_mov_b32_e32 v0, v1
; GFX11-NEXT:  v_mov_b32_e32 v2, v1
; GFX11-NEXT:  v_mov_b32_e32 v3, v1
; GFX11-NEXT:  v_mov_b32_e32 v4, v1
; GFX11-NEXT:  v_mov_b32_e32 v5, v1
; GFX11-NEXT:  v_mov_b32_e32 v6, v1
; GFX11-NEXT:  v_mov_b32_e32 v7, v1
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <8 x float>, ptr addrspace(1) %arg0
  %val1 = shufflevector <8 x float> %val0, <8 x float> poison, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <8 x float> %val1
}

define <16 x float> @shuffle_v16f32_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v16f32_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dwordx4 v[0:3], v[0:1], off
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_mov_b32_e32 v0, v1
; GFX9-NEXT:  v_mov_b32_e32 v2, v1
; GFX9-NEXT:  v_mov_b32_e32 v3, v1
; GFX9-NEXT:  v_mov_b32_e32 v4, v1
; GFX9-NEXT:  v_mov_b32_e32 v5, v1
; GFX9-NEXT:  v_mov_b32_e32 v6, v1
; GFX9-NEXT:  v_mov_b32_e32 v7, v1
; GFX9-NEXT:  v_mov_b32_e32 v8, v1
; GFX9-NEXT:  v_mov_b32_e32 v9, v1
; GFX9-NEXT:  v_mov_b32_e32 v10, v1
; GFX9-NEXT:  v_mov_b32_e32 v11, v1
; GFX9-NEXT:  v_mov_b32_e32 v12, v1
; GFX9-NEXT:  v_mov_b32_e32 v13, v1
; GFX9-NEXT:  v_mov_b32_e32 v14, v1
; GFX9-NEXT:  v_mov_b32_e32 v15, v1
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v16f32_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dwordx4 v[0:3], v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_mov_b32_e32 v0, v1
; GFX10-NEXT:  v_mov_b32_e32 v2, v1
; GFX10-NEXT:  v_mov_b32_e32 v3, v1
; GFX10-NEXT:  v_mov_b32_e32 v4, v1
; GFX10-NEXT:  v_mov_b32_e32 v5, v1
; GFX10-NEXT:  v_mov_b32_e32 v6, v1
; GFX10-NEXT:  v_mov_b32_e32 v7, v1
; GFX10-NEXT:  v_mov_b32_e32 v8, v1
; GFX10-NEXT:  v_mov_b32_e32 v9, v1
; GFX10-NEXT:  v_mov_b32_e32 v10, v1
; GFX10-NEXT:  v_mov_b32_e32 v11, v1
; GFX10-NEXT:  v_mov_b32_e32 v12, v1
; GFX10-NEXT:  v_mov_b32_e32 v13, v1
; GFX10-NEXT:  v_mov_b32_e32 v14, v1
; GFX10-NEXT:  v_mov_b32_e32 v15, v1
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v16f32_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b128 v[0:3], v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_mov_b32_e32 v0, v1
; GFX11-NEXT:  v_mov_b32_e32 v2, v1
; GFX11-NEXT:  v_mov_b32_e32 v3, v1
; GFX11-NEXT:  v_mov_b32_e32 v4, v1
; GFX11-NEXT:  v_mov_b32_e32 v5, v1
; GFX11-NEXT:  v_mov_b32_e32 v6, v1
; GFX11-NEXT:  v_mov_b32_e32 v7, v1
; GFX11-NEXT:  v_mov_b32_e32 v8, v1
; GFX11-NEXT:  v_mov_b32_e32 v9, v1
; GFX11-NEXT:  v_mov_b32_e32 v10, v1
; GFX11-NEXT:  v_mov_b32_e32 v11, v1
; GFX11-NEXT:  v_mov_b32_e32 v12, v1
; GFX11-NEXT:  v_mov_b32_e32 v13, v1
; GFX11-NEXT:  v_mov_b32_e32 v14, v1
; GFX11-NEXT:  v_mov_b32_e32 v15, v1
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <16 x float>, ptr addrspace(1) %arg0
  %val1 = shufflevector <16 x float> %val0, <16 x float> poison, <16 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <16 x float> %val1
}

define <32 x float> @shuffle_v32f32_rebroadcast(ptr addrspace(1) %arg0) {
; GFX9-LABEL: shuffle_v32f32_rebroadcast:
; GFX9:       ; %bb.0: ; %entry
; GFX9-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:  global_load_dwordx4 v[0:3], v[0:1], off
; GFX9-NEXT:  s_waitcnt vmcnt(0)
; GFX9-NEXT:  v_mov_b32_e32 v0, v1
; GFX9-NEXT:  v_mov_b32_e32 v2, v1
; GFX9-NEXT:  v_mov_b32_e32 v3, v1
; GFX9-NEXT:  v_mov_b32_e32 v4, v1
; GFX9-NEXT:  v_mov_b32_e32 v5, v1
; GFX9-NEXT:  v_mov_b32_e32 v6, v1
; GFX9-NEXT:  v_mov_b32_e32 v7, v1
; GFX9-NEXT:  v_mov_b32_e32 v8, v1
; GFX9-NEXT:  v_mov_b32_e32 v9, v1
; GFX9-NEXT:  v_mov_b32_e32 v10, v1
; GFX9-NEXT:  v_mov_b32_e32 v11, v1
; GFX9-NEXT:  v_mov_b32_e32 v12, v1
; GFX9-NEXT:  v_mov_b32_e32 v13, v1
; GFX9-NEXT:  v_mov_b32_e32 v14, v1
; GFX9-NEXT:  v_mov_b32_e32 v15, v1
; GFX9-NEXT:  v_mov_b32_e32 v16, v1
; GFX9-NEXT:  v_mov_b32_e32 v17, v1
; GFX9-NEXT:  v_mov_b32_e32 v18, v1
; GFX9-NEXT:  v_mov_b32_e32 v19, v1
; GFX9-NEXT:  v_mov_b32_e32 v20, v1
; GFX9-NEXT:  v_mov_b32_e32 v21, v1
; GFX9-NEXT:  v_mov_b32_e32 v22, v1
; GFX9-NEXT:  v_mov_b32_e32 v23, v1
; GFX9-NEXT:  v_mov_b32_e32 v24, v1
; GFX9-NEXT:  v_mov_b32_e32 v25, v1
; GFX9-NEXT:  v_mov_b32_e32 v26, v1
; GFX9-NEXT:  v_mov_b32_e32 v27, v1
; GFX9-NEXT:  v_mov_b32_e32 v28, v1
; GFX9-NEXT:  v_mov_b32_e32 v29, v1
; GFX9-NEXT:  v_mov_b32_e32 v30, v1
; GFX9-NEXT:  v_mov_b32_e32 v31, v1
; GFX9-NEXT:  s_setpc_b64 s[30:31]
;
; GFX10-LABEL: shuffle_v32f32_rebroadcast:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX10-NEXT:  global_load_dwordx4 v[0:3], v[0:1], off
; GFX10-NEXT:  s_waitcnt vmcnt(0)
; GFX10-NEXT:  v_mov_b32_e32 v0, v1
; GFX10-NEXT:  v_mov_b32_e32 v2, v1
; GFX10-NEXT:  v_mov_b32_e32 v3, v1
; GFX10-NEXT:  v_mov_b32_e32 v4, v1
; GFX10-NEXT:  v_mov_b32_e32 v5, v1
; GFX10-NEXT:  v_mov_b32_e32 v6, v1
; GFX10-NEXT:  v_mov_b32_e32 v7, v1
; GFX10-NEXT:  v_mov_b32_e32 v8, v1
; GFX10-NEXT:  v_mov_b32_e32 v9, v1
; GFX10-NEXT:  v_mov_b32_e32 v10, v1
; GFX10-NEXT:  v_mov_b32_e32 v11, v1
; GFX10-NEXT:  v_mov_b32_e32 v12, v1
; GFX10-NEXT:  v_mov_b32_e32 v13, v1
; GFX10-NEXT:  v_mov_b32_e32 v14, v1
; GFX10-NEXT:  v_mov_b32_e32 v15, v1
; GFX10-NEXT:  v_mov_b32_e32 v16, v1
; GFX10-NEXT:  v_mov_b32_e32 v17, v1
; GFX10-NEXT:  v_mov_b32_e32 v18, v1
; GFX10-NEXT:  v_mov_b32_e32 v19, v1
; GFX10-NEXT:  v_mov_b32_e32 v20, v1
; GFX10-NEXT:  v_mov_b32_e32 v21, v1
; GFX10-NEXT:  v_mov_b32_e32 v22, v1
; GFX10-NEXT:  v_mov_b32_e32 v23, v1
; GFX10-NEXT:  v_mov_b32_e32 v24, v1
; GFX10-NEXT:  v_mov_b32_e32 v25, v1
; GFX10-NEXT:  v_mov_b32_e32 v26, v1
; GFX10-NEXT:  v_mov_b32_e32 v27, v1
; GFX10-NEXT:  v_mov_b32_e32 v28, v1
; GFX10-NEXT:  v_mov_b32_e32 v29, v1
; GFX10-NEXT:  v_mov_b32_e32 v30, v1
; GFX10-NEXT:  v_mov_b32_e32 v31, v1
; GFX10-NEXT:  s_setpc_b64 s[30:31]
;
; GFX11-LABEL: shuffle_v32f32_rebroadcast:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:  s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:  global_load_b128 v[0:3], v[0:1], off
; GFX11-NEXT:  s_waitcnt vmcnt(0)
; GFX11-NEXT:  v_mov_b32_e32 v0, v1
; GFX11-NEXT:  v_mov_b32_e32 v2, v1
; GFX11-NEXT:  v_mov_b32_e32 v3, v1
; GFX11-NEXT:  v_mov_b32_e32 v4, v1
; GFX11-NEXT:  v_mov_b32_e32 v5, v1
; GFX11-NEXT:  v_mov_b32_e32 v6, v1
; GFX11-NEXT:  v_mov_b32_e32 v7, v1
; GFX11-NEXT:  v_mov_b32_e32 v8, v1
; GFX11-NEXT:  v_mov_b32_e32 v9, v1
; GFX11-NEXT:  v_mov_b32_e32 v10, v1
; GFX11-NEXT:  v_mov_b32_e32 v11, v1
; GFX11-NEXT:  v_mov_b32_e32 v12, v1
; GFX11-NEXT:  v_mov_b32_e32 v13, v1
; GFX11-NEXT:  v_mov_b32_e32 v14, v1
; GFX11-NEXT:  v_mov_b32_e32 v15, v1
; GFX11-NEXT:  v_mov_b32_e32 v16, v1
; GFX11-NEXT:  v_mov_b32_e32 v17, v1
; GFX11-NEXT:  v_mov_b32_e32 v18, v1
; GFX11-NEXT:  v_mov_b32_e32 v19, v1
; GFX11-NEXT:  v_mov_b32_e32 v20, v1
; GFX11-NEXT:  v_mov_b32_e32 v21, v1
; GFX11-NEXT:  v_mov_b32_e32 v22, v1
; GFX11-NEXT:  v_mov_b32_e32 v23, v1
; GFX11-NEXT:  v_mov_b32_e32 v24, v1
; GFX11-NEXT:  v_mov_b32_e32 v25, v1
; GFX11-NEXT:  v_mov_b32_e32 v26, v1
; GFX11-NEXT:  v_mov_b32_e32 v27, v1
; GFX11-NEXT:  v_mov_b32_e32 v28, v1
; GFX11-NEXT:  v_mov_b32_e32 v29, v1
; GFX11-NEXT:  v_mov_b32_e32 v30, v1
; GFX11-NEXT:  v_mov_b32_e32 v31, v1
; GFX11-NEXT:  s_setpc_b64 s[30:31]
entry:
  %val0 = load <32 x float>, ptr addrspace(1) %arg0
  %val1 = shufflevector <32 x float> %val0, <32 x float> poison, <32 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  ret <32 x float> %val1
}
