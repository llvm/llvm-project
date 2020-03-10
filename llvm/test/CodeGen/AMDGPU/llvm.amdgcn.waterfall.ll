; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,VI,PRE_GFX10,GCN-64 %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9_UP,PRE_GFX10,GCN-64 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX10,GFX9_UP,GCN-32 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX10,GFX9_UP,GCN-64 %s

; GCN-LABEL: {{^}}test_waterfall_readlane:
; GCN: {{^}}BB0_1:
; GCN: v_readfirstlane_b32 [[VAL1:s[0-9]+]], [[VAL2:v[0-9]+]]
; GCN-64: v_cmp_eq_u32_e64 [[EXEC:s[[0-9]+:[0-9]+]]], [[VAL1]], [[VAL2]]
; GCN-32: v_cmp_eq_u32_e64 [[EXEC:s[0-9]+]], [[VAL1]], [[VAL2]]
; GCN-64: s_and_saveexec_b64 [[EXEC]], [[EXEC]]
; GCN-32: s_and_saveexec_b32 [[EXEC]], [[EXEC]]
; GCN: v_readlane_b32 [[RLVAL:s[0-9]+]], v1, [[VAL1]]
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[RLVAL]]
; GCN: v_or_b32_e32 [[ACCUM:v[0-9]+]], [[ACCUM]], [[VVAL]]
; GCN-64: s_xor_b64 exec, exec, [[EXEC]]
; GCN-32: s_xor_b32 exec_lo, exec_lo, [[EXEC]]
; GCN: s_cbranch_execnz BB0_1
; GCN-64: s_mov_b64 exec, s[{{[0-9]+:[0-9]+}}]
; GCN-32: s_mov_b32 exec_lo, s{{[0-9]+}}
; VI: flat_store_dword v[{{[0-9]+:[0-9]+}}], [[ACCUM]]
; GFX9_UP: global_store_dword v[{{[0-9]+:[0-9]+}}], [[ACCUM]], off
define amdgpu_ps void @test_waterfall_readlane(i32 addrspace(1)* inreg %out, <2 x i32> addrspace(1)* inreg %in, i32 %tid, i32 %val) #1 {
  %gep.in = getelementptr <2 x i32>, <2 x i32> addrspace(1)* %in, i32 %tid
  %args = load <2 x i32>, <2 x i32> addrspace(1)* %gep.in
  %value = extractelement <2 x i32> %args, i32 0
  %lane = extractelement <2 x i32> %args, i32 1
  %wf_token = call i32 @llvm.amdgcn.waterfall.begin.i32(i32 %lane)
  %readlane = call i32 @llvm.amdgcn.waterfall.readfirstlane.i32.i32(i32 %wf_token, i32 %lane)
  %readlane1 = call i32 @llvm.amdgcn.readlane(i32 %val, i32 %readlane)
  %readlane2 = call i32 @llvm.amdgcn.waterfall.end.i32(i32 %wf_token, i32 %readlane1)
  ; This store instruction should be outside the waterfall loop and the value
  ; being stored generated incrementally in the loop itself
  store i32 %readlane2, i32 addrspace(1)* %out, align 4

  ret void
}

; GCN-LABEL: {{^}}test_waterfall_non_uniform_img:
; GCN: v_mov_b32_e32 v[[DSTSTART:[0-9]+]], 0
; GCN: v_mov_b32_e32 v{{[0-9]+}}, 0
; GCN: v_mov_b32_e32 v{{[0-9]+}}, 0
; GCN: v_mov_b32_e32 v[[DSTEND:[0-9]+]], 0
; GCN-64: s_mov_b64 [[EXEC:s[[0-9]+:[0-9]+]]], exec
; GCN-32: s_mov_b32 [[EXEC:s[0-9]+]], exec_lo
; GCN: {{^}}BB1_1:
; GCN: v_readfirstlane_b32 s[[FIRSTVAL:[0-9]+]], v0
; GCN-64: v_cmp_eq_u32_e64 [[EXEC2:s[[0-9]+:[0-9]+]]], s[[FIRSTVAL]], v0
; GCN-64: s_and_saveexec_b64 [[EXEC3:s[[0-9]+:[0-9]+]]], [[EXEC2]]
; GCN-32: v_cmp_eq_u32_e64 [[EXEC2:s[0-9]+]], s[[FIRSTVAL]], v0
; GCN-32: s_and_saveexec_b32 [[EXEC3:s[0-9]+]], [[EXEC2]]
; GCN: s_lshl_b64 s{{\[}}[[FIRSTVALX32:[0-9]+]]:{{[0-9]+}}], s{{\[}}[[FIRSTVAL]]:{{[0-9]+}}], 5
; GCN: s_load_dwordx8 [[PTR:s\[[0-9]+:[0-9]+\]]], s{{\[}}[[FIRSTVALX32]]:{{[0-9]+}}], 0x0
; GCN: s_waitcnt lgkmcnt(0)
; GCN: image_sample v{{\[}}[[VALSTART:[0-9]+]]:[[VALEND:[0-9]+]]{{\]}}, v{{[0-9]+}}, [[PTR]], s[{{[0-9]+:[0-9]+}}] dmask:0xf
; GCN: v_or_b32_e32 v[[DSTSTART]], v[[DSTSTART]], v[[VALSTART]]
; GCN: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_or_b32_e32 v[[DSTEND]], v[[DSTEND]], v[[VALEND]]
; GCN-64: s_xor_b64 exec, exec, [[EXEC3]]
; GCN-32: s_xor_b32 exec_lo, exec_lo, [[EXEC3]]
; GCN: s_cbranch_execnz BB1_1
; GCN-64: s_mov_b64 exec, [[EXEC]]
; GCN-32: s_mov_b32 exec_lo, [[EXEC]]
define amdgpu_ps <4 x float> @test_waterfall_non_uniform_img(<8 x i32> addrspace(4)* inreg %in, i32 %index, float %s,
                                                             <4 x i32> inreg %samp) #1 {
  %wf_token = call i32 @llvm.amdgcn.waterfall.begin.i32(i32 %index)
  %s_idx = call i32 @llvm.amdgcn.waterfall.readfirstlane.i32.i32(i32 %wf_token, i32 %index)
  %ptr = getelementptr <8 x i32>, <8 x i32> addrspace(4)* %in, i32 %s_idx
  %rsrc = load <8 x i32>, <8 x i32> addrspace(4) * %ptr, align 32
  %r = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %r1 = call <4 x float> @llvm.amdgcn.waterfall.end.v4f32(i32 %wf_token, <4 x float> %r)

  ret <4 x float> %r1
}

; GCN-LABEL: {{^}}test_waterfall_non_uniform_img_single_read:
; VI: flat_load_dwordx4 v{{\[}}[[RSRCSTART:[0-9]+]]:{{[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; VI: flat_load_dwordx4 v[{{[0-9]+:}}[[RSRCEND:[0-9]+]]{{\]}}, v[{{[0-9]+:[0-9]+}}]
; GFX9_UP-DAG: global_load_dwordx4 v{{\[}}[[RSRCSTART:[0-9]+]]:{{[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX9_UP-DAG: global_load_dwordx4 v[{{[0-9]+:}}[[RSRCEND:[0-9]+]]{{\]}}, v[{{[0-9]+:[0-9]+}}], off offset:16
; GCN-DAG: v_mov_b32_e32 v[[DSTSTART:[0-9]+]], 0
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0
; GCN-DAG: v_mov_b32_e32 v[[DSTEND:[0-9]+]], 0
; GCN-64: s_mov_b64 [[EXEC:s[[0-9]+:[0-9]+]]], exec
; GCN-32: s_mov_b32 [[EXEC:s[0-9]+]], exec_lo
; GCN: {{^}}BB2_1:
; GCN: v_readfirstlane_b32 s[[FIRSTVAL:[0-9]+]], [[INDEX:v[0-9]+]]
; GCN-64-DAG: v_cmp_eq_u32_e64 [[EXEC2:s[[0-9]+:[0-9]+]]], s[[FIRSTVAL]], [[INDEX]]
; GCN-32-DAG: v_cmp_eq_u32_e64 [[EXEC2:s[0-9]+]], s[[FIRSTVAL]], [[INDEX]]
; GCN-DAG: v_readfirstlane_b32 s[[FIRSTRSRC:[0-9]+]], v[[RSRCSTART]]
; GCN-DAG: v_readfirstlane_b32 s[[ENDRSRC:[0-9]+]], v[[RSRCEND]]
; GCN-DAG: v_readfirstlane_b32 s{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_readfirstlane_b32 s{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_readfirstlane_b32 s{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_readfirstlane_b32 s{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_readfirstlane_b32 s{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_readfirstlane_b32 s{{[0-9]+}}, v{{[0-9]+}}
; GCN-64: s_and_saveexec_b64 [[EXEC3:s[[0-9]+:[0-9]+]]], [[EXEC2]]
; GCN-32: s_and_saveexec_b32 [[EXEC3:s[0-9]+]], [[EXEC2]]
; GCN: image_sample v{{\[}}[[VALSTART:[0-9]+]]:[[VALEND:[0-9]+]]{{\]}}, v{{[0-9]+}}, s{{\[}}[[FIRSTRSRC]]:[[ENDRSRC]]{{\]}}, s[{{[0-9]+:[0-9]+}}] dmask:0xf
; GCN-DAG: v_or_b32_e32 v[[DSTSTART]], v[[DSTSTART]], v[[VALSTART]]
; GCN-DAG: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_or_b32_e32 v[[DSTEND]], v[[DSTEND]], v[[VALEND]]
; GCN-64: s_xor_b64 exec, exec, [[EXEC3]]
; GCN-32: s_xor_b32 exec_lo, exec_lo, [[EXEC3]]
; GCN: s_cbranch_execnz BB2_1
; GCN-64: s_mov_b64 exec, [[EXEC]]
; GCN-32: s_mov_b32 exec_lo, [[EXEC]]
define amdgpu_ps <4 x float> @test_waterfall_non_uniform_img_single_read(<8 x i32> addrspace(4)* inreg %in, i32 %index, float %s,
                                                             <4 x i32> inreg %samp) #1 {
  %ptr = getelementptr <8 x i32>, <8 x i32> addrspace(4)* %in, i32 %index
  %rsrc = load <8 x i32>, <8 x i32> addrspace(4) * %ptr, align 32
  %wf_token = call i32 @llvm.amdgcn.waterfall.begin.i32(i32 %index)
  %s_rsrc = call <8 x i32> @llvm.amdgcn.waterfall.readfirstlane.v8i32.v8i32(i32 %wf_token, <8 x i32> %rsrc)
  %r = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %s, <8 x i32> %s_rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %r1 = call <4 x float> @llvm.amdgcn.waterfall.end.v4f32(i32 %wf_token, <4 x float> %r)

  ret <4 x float> %r1
}

; GCN-LABEL: {{^}}test_multiple_groups:
; GCN: {{^}}BB3_1:
; GCN: v_readfirstlane_b32 [[VAL1:s[0-9]+]], [[VAL2:v[0-9]+]]
; GCN-64: v_cmp_eq_u32_e64 [[EXEC:s[[0-9]+:[0-9]+]]], [[VAL1]], [[VAL2]]
; GCN-64: s_and_saveexec_b64 [[EXEC]], [[EXEC]]
; GCN-32: v_cmp_eq_u32_e64 [[EXEC:s[0-9]+]], [[VAL1]], [[VAL2]]
; GCN-32: s_and_saveexec_b32 [[EXEC]], [[EXEC]]
; GCN: v_readlane_b32 [[RLVAL:s[0-9]+]], v{{[0-9]+}}, [[VAL1]]
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[RLVAL]]
; GCN: v_or_b32_e32 [[ACCUM:v[0-9]+]], [[ACCUM]], [[VVAL]]
; GCN-64: s_xor_b64 exec, exec, [[EXEC]]
; GCN-32: s_xor_b32 exec_lo, exec_lo, [[EXEC]]
; GCN: s_cbranch_execnz BB3_1
; GCN-64: s_mov_b64 exec, s[{{[0-9]+:[0-9]+}}]
; GCN-32: s_mov_b32 exec_lo, s{{[0-9]+}}
; VI: flat_store_dword v[{{[0-9]+:[0-9]+}}], [[ACCUM]]
; GFX9_UP: global_store_dword v[{{[0-9]+:[0-9]+}}], [[ACCUM]], off
; GCN: {{^}}BB3_3:
; GCN: v_readfirstlane_b32 [[VAL2_1:s[0-9]+]], [[VAL2_2:v[0-9]+]]
; GCN-64: v_cmp_eq_u32_e64 [[EXEC2:s[[0-9]+:[0-9]+]]], [[VAL2_1]], [[VAL2_2]]
; GCN-64: s_and_saveexec_b64 [[EXEC2]], [[EXEC2]]
; GCN-32: v_cmp_eq_u32_e64 [[EXEC2:s[0-9]+]], [[VAL2_1]], [[VAL2_2]]
; GCN-32: s_and_saveexec_b32 [[EXEC2]], [[EXEC2]]
; GCN: v_readlane_b32 [[RLVAL2:s[0-9]+]], v{{[0-9]+}}, [[VAL2_1]]
; GCN: v_mov_b32_e32 [[VVAL2:v[0-9]+]], [[RLVAL2]]
; GCN: v_or_b32_e32 [[ACCUM2:v[0-9]+]], [[ACCUM2]], [[VVAL2]]
; GCN-64: s_xor_b64 exec, exec, [[EXEC2]]
; GCN-32: s_xor_b32 exec_lo, exec_lo, [[EXEC2]]
; GCN: s_cbranch_execnz BB3_3
; GCN-64: s_mov_b64 exec, s[{{[0-9]+:[0-9]+}}]
; GCN-32: s_mov_b32 exec_lo, s{{[0-9]+}}
; VI: flat_store_dword v[{{[0-9]+:[0-9]+}}], [[ACCUM2]]
; GFX9_UP: global_store_dword v[{{[0-9]+:[0-9]+}}], [[ACCUM2]], off

define amdgpu_ps void @test_multiple_groups(i32 addrspace(1)* inreg %out1, i32 addrspace(1)* inreg %out2,
                                            i32 %idx1, i32 %idx2, i32 %val) #1 {
  %wf_token = call i32 @llvm.amdgcn.waterfall.begin.i32(i32 %idx1)
  %readlane1 = call i32 @llvm.amdgcn.waterfall.readfirstlane.i32.i32(i32 %wf_token, i32 %idx1)
  %readlane1.1 = call i32 @llvm.amdgcn.readlane(i32 %val, i32 %readlane1)
  %readlane1.2 = call i32 @llvm.amdgcn.waterfall.end.i32(i32 %wf_token, i32 %readlane1.1)
  ; This store instruction should be outside the waterfall loop and the value
  ; being stored generated incrementally in the loop itself
  store i32 %readlane1.2, i32 addrspace(1)* %out1, align 4

  %wf_token2 = call i32 @llvm.amdgcn.waterfall.begin.i32(i32 %idx2)
  %readlane2 = call i32 @llvm.amdgcn.waterfall.readfirstlane.i32.i32(i32 %wf_token2, i32 %idx2)
  %readlane2.1 = call i32 @llvm.amdgcn.readlane(i32 %val, i32 %readlane2)
  %readlane2.2 = call i32 @llvm.amdgcn.waterfall.end.i32(i32 %wf_token2, i32 %readlane2.1)
  store i32 %readlane2.2, i32 addrspace(1)* %out2, align 4

  ret void
}


; GCN-LABEL: {{^}}test_waterfall_non_uniform_img_multi_rl:
; GCN: v_mov_b32_e32 v[[DSTSTART:[0-9]+]], 0
; GCN: v_mov_b32_e32 v{{[0-9]+}}, 0
; GCN: v_mov_b32_e32 v{{[0-9]+}}, 0
; GCN: v_mov_b32_e32 v[[DSTEND:[0-9]+]], 0
; GCN-64: s_mov_b64 [[EXEC:s[[0-9]+:[0-9]+]]], exec
; GCN-32: s_mov_b32 [[EXEC:s[0-9]+]], exec
; GCN: {{^}}BB4_1:
; GCN: v_readfirstlane_b32 s[[FIRSTVAL:[0-9]+]], [[IDX:v[0-9]+]]
; GCN-64-DAG: v_cmp_eq_u32_e64 [[EXEC2:s[[0-9]+:[0-9]+]]], s[[FIRSTVAL]], [[IDX]]
; GCN-32-DAG: v_cmp_eq_u32_e64 [[EXEC2:s[0-9]+]], s[[FIRSTVAL]], [[IDX]]
; GCN-DAG: v_readfirstlane_b32 s{{[0-9]+}}, v{{[0-9]+}}
; GCN-64: s_and_saveexec_b64 [[EXEC3:s[[0-9]+:[0-9]+]]], [[EXEC2]]
; GCN-32: s_and_saveexec_b32 [[EXEC3:s[0-9]+]], [[EXEC2]]
; GCN: s_load_dwordx8 [[PTR:s\[[0-9]+:[0-9]+\]]], s{{\[}}[[FIRSTVAL]]:{{[0-9]+}}], 0x0
; GCN: s_load_dwordx4 [[PTR2:s\[[0-9]+:[0-9]+\]]], s[{{[0-9]+:[0-9]+}}], 0x0 
; GCN: s_waitcnt lgkmcnt(0)
; GCN: image_sample v{{\[}}[[VALSTART:[0-9]+]]:[[VALEND:[0-9]+]]{{\]}}, v{{[0-9]+}}, [[PTR]], [[PTR2]] dmask:0xf
; GCN: v_or_b32_e32 v[[DSTSTART]], v[[DSTSTART]], v[[VALSTART]]
; GCN: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_or_b32_e32 v[[DSTEND]], v[[DSTEND]], v[[VALEND]]
; GCN-64: s_xor_b64 exec, exec, [[EXEC3]]
; GCN-32: s_xor_b32 exec_lo, exec_lo, [[EXEC3]]
; GCN: s_cbranch_execnz BB4_1
; GCN-64: s_mov_b64 exec, [[EXEC]]
; GCN-32: s_mov_b32 exec_lo, [[EXEC]]
define amdgpu_ps <4 x float> @test_waterfall_non_uniform_img_multi_rl(<8 x i32> addrspace(4)* inreg %in,
                                                                      <4 x i32> addrspace(4)* inreg %samp_in,
                                                                      i32 %index, float %s, i32 %val) #1 {
  %wf_token = call i32 @llvm.amdgcn.waterfall.begin.i32(i32 %index)
  %s_idx = call i32 @llvm.amdgcn.waterfall.readfirstlane.i32.i32(i32 %wf_token, i32 %index)
  %s_idx2 = call i32 @llvm.amdgcn.waterfall.readfirstlane.i32.i32(i32 %wf_token, i32 %val)
  %ptr = getelementptr <8 x i32>, <8 x i32> addrspace(4)* %in, i32 %s_idx
  %ptr2 = getelementptr <4 x i32>, <4 x i32> addrspace(4)* %samp_in, i32 %s_idx2
  %rsrc = load <8 x i32>, <8 x i32> addrspace(4) * %ptr, align 32
  %samp = load <4 x i32>, <4 x i32> addrspace(4) * %ptr2, align 32
  %r = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %r1 = call <4 x float> @llvm.amdgcn.waterfall.end.v4f32(i32 %wf_token, <4 x float> %r)

  ret <4 x float> %r1
}

; GCN-LABEL: {{^}}test_waterfall_non_uni_img_2_idx:
; GCN: v_mov_b32_e32 v[[DSTSTART:[0-9]+]], 0
; GCN: v_mov_b32_e32 v{{[0-9]+}}, 0
; GCN: v_mov_b32_e32 v{{[0-9]+}}, 0
; GCN-DAG: v_mov_b32_e32 v[[DSTEND:[0-9]+]], 0
; GCN-64-DAG: s_mov_b64 [[EXEC:s[[0-9]+:[0-9]+]]], exec
; GCN--32-DAG: s_mov_b32 [[EXEC:s[0-9]+]], exec
; GCN: {{^}}BB5_1:
; GCN: v_readfirstlane_b32 s[[FIRSTVAL1:[0-9]+]], [[IDX1:v[0-9]+]]
; GCN: v_readfirstlane_b32 s[[FIRSTVAL2:[0-9]+]], [[IDX2:v[0-9]+]]
; GCN-64: v_cmp_eq_u32_e64 [[EXEC2:s[[0-9]+:[0-9]+]]], s[[FIRSTVAL1]], [[IDX1]]
; GCN-64: v_cmp_eq_u32_e64 [[EXEC3:s[[0-9]+:[0-9]+]]], s[[FIRSTVAL2]], [[IDX2]]
; GCN-64: s_and_b64 [[CIDX:s\[[0-9]+:[0-9]+\]]], [[EXEC2]], [[EXEC3]]
; GCN-64: s_and_saveexec_b64 [[EXEC4:s[[0-9]+:[0-9]+]]], [[CIDX]]
; GCN-32: v_cmp_eq_u32_e64 [[EXEC2:s[0-9]+]], s[[FIRSTVAL1]], [[IDX1]]
; GCN-32: v_cmp_eq_u32_e64 [[EXEC3:s[0-9]+]], s[[FIRSTVAL2]], [[IDX2]]
; GCN-32: s_and_b32 [[CIDX:s[0-9]+]], [[EXEC2]], [[EXEC3]]
; GCN-32: s_and_saveexec_b32 [[EXEC4:s[0-9]+]], [[CIDX]]
; GCN: s_load_dwordx8 [[PTR:s\[[0-9]+:[0-9]+\]]], s[{{[0-9]+:[0-9]+}}], 0x0
; GCN: s_load_dwordx4 [[PTR2:s\[[0-9]+:[0-9]+\]]], s[{{[0-9]+:[0-9]+}}], 0x0 
; GCN: s_waitcnt lgkmcnt(0)
; GCN: image_sample v{{\[}}[[VALSTART:[0-9]+]]:[[VALEND:[0-9]+]]{{\]}}, v{{[0-9]+}}, [[PTR]], [[PTR2]] dmask:0xf
; GCN: v_or_b32_e32 v[[DSTSTART]], v[[DSTSTART]], v[[VALSTART]]
; GCN: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_or_b32_e32 v[[DSTEND]], v[[DSTEND]], v[[VALEND]]
; GCN-64: s_xor_b64 exec, exec, [[EXEC4]]
; GCN-32: s_xor_b32 exec_lo, exec_lo, [[EXEC4]]
; GCN: s_cbranch_execnz BB5_1
; GCN-64: s_mov_b64 exec, [[EXEC]]
; GCN-32: s_mov_b32 exec_lo, [[EXEC]]
define amdgpu_ps <4 x float> @test_waterfall_non_uni_img_2_idx(<8 x i32> addrspace(4)* inreg %in,
                                                               <4 x i32> addrspace(4)* inreg %samp_in,
                                                               i32 %index1, i32 %index2, float %s) #1 {
  %t_idx = insertelement <2 x i32> undef, i32 %index1, i32 0
  %combined_idx = insertelement <2 x i32> %t_idx, i32 %index2, i32 1
  %wf_token = call i32 @llvm.amdgcn.waterfall.begin.v2i32(<2 x i32> %combined_idx)
  %s_c_idx = call <2 x i32> @llvm.amdgcn.waterfall.readfirstlane.v2i32.v2i32(i32 %wf_token, <2 x i32> %combined_idx)
  %s_idx1 = extractelement <2 x i32> %s_c_idx, i32 0
  %s_idx2 = extractelement <2 x i32> %s_c_idx, i32 1
  %ptr = getelementptr <8 x i32>, <8 x i32> addrspace(4)* %in, i32 %s_idx1
  %ptr2 = getelementptr <4 x i32>, <4 x i32> addrspace(4)* %samp_in, i32 %s_idx2
  %rsrc = load <8 x i32>, <8 x i32> addrspace(4) * %ptr, align 32
  %samp = load <4 x i32>, <4 x i32> addrspace(4) * %ptr2, align 32
  %r = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %r1 = call <4 x float> @llvm.amdgcn.waterfall.end.v4f32(i32 %wf_token, <4 x float> %r)

  ret <4 x float> %r1
}

; GCN-LABEL: {{^}}test_waterfall_non_uniform_img_single_store:
; VI: flat_load_dwordx4 v{{\[}}[[RSRCSTART:[0-9]+]]:{{[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; VI: flat_load_dwordx4 v[{{[0-9]+:}}[[RSRCEND:[0-9]+]]{{\]}}, v[{{[0-9]+:[0-9]+}}]
; GFX9_UP-DAG: global_load_dwordx4 v{{\[}}[[RSRCSTART:[0-9]+]]:{{[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX9_UP-DAG: global_load_dwordx4 v[{{[0-9]+:}}[[RSRCEND:[0-9]+]]{{\]}}, v[{{[0-9]+:[0-9]+}}], off offset:16
; GCN-64-DAG: s_mov_b64 [[EXEC:s[[0-9]+:[0-9]+]]], exec
; GCN-32-DAG: s_mov_b32 [[EXEC:s[0-9]+]], exec
; GCN: {{^}}BB6_1:
; GCN: v_readfirstlane_b32 s[[FIRSTVAL:[0-9]+]], [[INDEX:v[0-9]+]]
; GCN-64-DAG: v_cmp_eq_u32_e64 [[EXEC2:s[[0-9]+:[0-9]+]]], s[[FIRSTVAL]], [[INDEX]]
; GCN-32-DAG: v_cmp_eq_u32_e64 [[EXEC2:s[0-9]+]], s[[FIRSTVAL]], [[INDEX]]
; GCN-DAG: v_readfirstlane_b32 s[[FIRSTRSRC:[0-9]+]], v[[RSRCSTART]]
; GCN-DAG: v_readfirstlane_b32 s[[ENDRSRC:[0-9]+]], v[[RSRCEND]]
; GCN-DAG: v_readfirstlane_b32 s{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_readfirstlane_b32 s{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_readfirstlane_b32 s{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_readfirstlane_b32 s{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_readfirstlane_b32 s{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_readfirstlane_b32 s{{[0-9]+}}, v{{[0-9]+}}
; GCN-64: s_and_saveexec_b64 [[EXEC3:s[[0-9]+:[0-9]+]]], [[EXEC2]]
; GCN-32: s_and_saveexec_b32 [[EXEC3:s[0-9]+]], [[EXEC2]]
; PRE_GFX10: image_store v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, s{{\[}}[[FIRSTRSRC]]:[[ENDRSRC]]{{\]}} dmask:0xf unorm
; GFX-10: image_store v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, s{{\[}}[[FIRSTRSRC]]:[[ENDRSRC]]{{\]}} dmask:0xf dim:SQ_RSRC_IMG_1D unorm
; GCN-64: s_xor_b64 exec, exec, [[EXEC3]]
; GCN-32: s_xor_b32 exec_lo, exec_lo, [[EXEC3]]
; GCN: s_cbranch_execnz BB6_1
define amdgpu_ps void @test_waterfall_non_uniform_img_single_store(<8 x i32> addrspace(4)* inreg %in, i32 %index, i32 %s,
                                                                   <4 x float> %data) #1 {
  %ptr = getelementptr <8 x i32>, <8 x i32> addrspace(4)* %in, i32 %index
  %rsrc = load <8 x i32>, <8 x i32> addrspace(4) * %ptr, align 32
  %wf_token = call i32 @llvm.amdgcn.waterfall.begin.i32(i32 %index)
  %s_rsrc = call <8 x i32> @llvm.amdgcn.waterfall.readfirstlane.v8i32.v8i32(i32 %wf_token, <8 x i32> %rsrc)
  %s_rsrc_use = call <8 x i32> @llvm.amdgcn.waterfall.last.use.v8i32(i32 %wf_token, <8 x i32> %s_rsrc)
  call void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float> %data, i32 15, i32 %s, <8 x i32> %s_rsrc_use, i32 0, i32 0)

  ret void
}

; GCN-LABEL: {{^}}test_remove_waterfall_last_use:
; GCN: s_load_dwordx8 s{{\[}}[[RSRCSTART:[0-9]+]]:[[RSRCEND:[0-9]+]]{{\]}}, s[{{[0-9]+:[0-9]+}}]
; PRE_GFX10-DAG: image_store v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, s{{\[}}[[RSRCSTART]]:[[RSRCEND]]{{\]}} dmask:0xf unorm
; GFX10-DAG: image_store v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, s{{\[}}[[RSRCSTART]]:[[RSRCEND]]{{\]}} dmask:0xf dim:SQ_RSRC_IMG_1D unorm
define amdgpu_ps void @test_remove_waterfall_last_use(<8 x i32> addrspace(4)* inreg %in, i32 %index, i32 %s,
                                                      <4 x float> %data) #1 {
  %rsrc = load <8 x i32>, <8 x i32> addrspace(4) * %in, align 32
  %wf_token = call i32 @llvm.amdgcn.waterfall.begin.i32(i32 %index)
  %s_rsrc = call <8 x i32> @llvm.amdgcn.waterfall.readfirstlane.v8i32.v8i32(i32 %wf_token, <8 x i32> %rsrc)
  %s_rsrc_use = call <8 x i32> @llvm.amdgcn.waterfall.last.use.v8i32(i32 %wf_token, <8 x i32> %s_rsrc)
  call void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float> %data, i32 15, i32 %s, <8 x i32> %s_rsrc_use, i32 0, i32 0)

  ret void
}

; GCN-LABEL: {{^}}test_remove_waterfall_multi_rl:
; GCN: s_load_dwordx8 s{{\[}}[[RSRCSTART:[0-9]+]]:[[RSRCEND:[0-9]+]]{{\]}}, s[{{[0-9]+:[0-9]+}}]
; GCN: s_load_dwordx4 s{{\[}}[[SAMPSTART:[0-9]+]]:[[SAMPEND:[0-9]+]]{{\]}}, s[{{[0-9]+:[0-9]+}}]
; GCN-DAG: image_sample v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, s{{\[}}[[RSRCSTART]]:[[RSRCEND]]{{\]}}, s{{\[}}[[SAMPSTART]]:[[SAMPEND]]{{\]}} dmask:0xf
define amdgpu_ps <4 x float> @test_remove_waterfall_multi_rl(<8 x i32> addrspace(4)* inreg %in,
                                                             <4 x i32> addrspace(4)* inreg %samp_in,
                                                             i32 %index, float %s, i32 inreg %val1, i32 inreg %val2) #1 {
  %wf_token = call i32 @llvm.amdgcn.waterfall.begin.i32(i32 %index)
  %s_idx = call i32 @llvm.amdgcn.waterfall.readfirstlane.i32.i32(i32 %wf_token, i32 %val1)
  %s_idx2 = call i32 @llvm.amdgcn.waterfall.readfirstlane.i32.i32(i32 %wf_token, i32 %val2)
  %ptr = getelementptr <8 x i32>, <8 x i32> addrspace(4)* %in, i32 %s_idx
  %ptr2 = getelementptr <4 x i32>, <4 x i32> addrspace(4)* %samp_in, i32 %s_idx2
  %rsrc = load <8 x i32>, <8 x i32> addrspace(4) * %ptr, align 32
  %samp = load <4 x i32>, <4 x i32> addrspace(4) * %ptr2, align 32
  %r = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %r1 = call <4 x float> @llvm.amdgcn.waterfall.end.v4f32(i32 %wf_token, <4 x float> %r)

  ret <4 x float> %r1
}

; GCN-LABEL: {{^}}test_keep_waterfall_multi_rl:
; GCN: {{^}}BB9_1:
; GCN: v_readfirstlane_b32 s[[FIRSTVAL:[0-9]+]], v0
; GCN: s_add_u32 s[[NONUSTART:[0-9]+]], s0, s[[FIRSTVAL]]
; GCN: s_addc_u32 s[[NONUEND:[0-9]+]], s1, s{{[0-9]+}}
; GCN-DAG: s_load_dwordx8 s{{\[}}[[RSRCSTART:[0-9]+]]:[[RSRCEND:[0-9]+]]{{\]}}, s{{\[}}[[NONUSTART]]:[[NONUEND]]{{\]}}
; GCN-DAG: s_add_u32 s[[UNISTART:[0-9]+]], s2, s{{[0-9]+}}
; GCN-DAG: s_addc_u32 s[[UNIEND:[0-9]+]], s3, s{{[0-9]+}}
; GCN: s_load_dwordx4 s{{\[}}[[SAMPSTART:[0-9]+]]:[[SAMPEND:[0-9]+]]{{\]}}, s{{\[}}[[UNISTART]]:[[UNIEND]]{{\]}}
; GCN-DAG: image_sample v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, s{{\[}}[[RSRCSTART]]:[[RSRCEND]]{{\]}}, s{{\[}}[[SAMPSTART]]:[[SAMPEND]]{{\]}} dmask:0xf
; GCN: s_cbranch_execnz BB9_1
define amdgpu_ps <4 x float> @test_keep_waterfall_multi_rl(<8 x i32> addrspace(4)* inreg %in,
                                                           <4 x i32> addrspace(4)* inreg %samp_in,
                                                           i32 %index, float %s, i32 inreg %val) #1 {
  %wf_token = call i32 @llvm.amdgcn.waterfall.begin.i32(i32 %index)
  %s_idx = call i32 @llvm.amdgcn.waterfall.readfirstlane.i32.i32(i32 %wf_token, i32 %index)
  %s_idx2 = call i32 @llvm.amdgcn.waterfall.readfirstlane.i32.i32(i32 %wf_token, i32 %val)
  %ptr = getelementptr <8 x i32>, <8 x i32> addrspace(4)* %in, i32 %s_idx
  %ptr2 = getelementptr <4 x i32>, <4 x i32> addrspace(4)* %samp_in, i32 %s_idx2
  %rsrc = load <8 x i32>, <8 x i32> addrspace(4) * %ptr, align 32
  %samp = load <4 x i32>, <4 x i32> addrspace(4) * %ptr2, align 32
  %r = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %r1 = call <4 x float> @llvm.amdgcn.waterfall.end.v4f32(i32 %wf_token, <4 x float> %r)

  ret <4 x float> %r1
}


declare i32 @llvm.amdgcn.waterfall.begin.i32(i32) #6
declare i32 @llvm.amdgcn.waterfall.begin.v2i32(<2 x i32>) #6
declare i32 @llvm.amdgcn.waterfall.readfirstlane.i32.i32(i32, i32) #6
declare <2 x i32> @llvm.amdgcn.waterfall.readfirstlane.v2i32.v2i32(i32, <2 x i32>) #6
declare <8 x i32> @llvm.amdgcn.waterfall.readfirstlane.v8i32.v8i32(i32, <8 x i32>) #6
declare i16 @llvm.amdgcn.waterfall.end.i16(i32, i16) #6
declare i32 @llvm.amdgcn.waterfall.end.i32(i32, i32) #6
declare <4 x float> @llvm.amdgcn.waterfall.end.v4f32(i32, <4 x float>) #6
declare <8 x i32> @llvm.amdgcn.waterfall.end.v8i32(i32, <8 x i32>) #6
declare <8 x i32> @llvm.amdgcn.waterfall.last.use.v8i32(i32, <8 x i32>) #6
declare i32 @llvm.amdgcn.readlane(i32, i32) #0
declare <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32)
declare void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float>, i32, i32, <8 x i32>, i32, i32)
declare i64 @llvm.amdgcn.s.getpc() #3
declare float @llvm.amdgcn.buffer.load.ushort(<4 x i32>, i32, i32, i1, i1) #4
declare float @llvm.amdgcn.buffer.load.f32(<4 x i32>, i32, i32, i1, i1) #4
declare <4 x float> @llvm.amdgcn.buffer.load.v4f32(<4 x i32>, i32, i32, i1, i1) #4
declare i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32>, i32, i1) #2
declare <4 x i32> @llvm.amdgcn.s.buffer.load.v4i32(<4 x i32>, i32, i1) #2
declare void @llvm.amdgcn.buffer.store.short(float, <4 x i32>, i32, i32, i1, i1) #5
declare void @llvm.amdgcn.buffer.store.f32(float, <4 x i32>, i32, i32, i1, i1) #5
declare void @llvm.amdgcn.buffer.store.v4f32(<4 x float>, <4 x i32>, i32, i32, i1, i1) #5

attributes #0 = { nounwind readnone convergent }
attributes #1 = { nounwind }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind readnone speculatable }
attributes #4 = { nounwind readonly }
attributes #5 = { nounwind writeonly }
attributes #6 = { nounwind }
