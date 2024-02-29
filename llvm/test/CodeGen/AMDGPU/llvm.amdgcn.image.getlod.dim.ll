; RUN: llc < %s -mtriple=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck --check-prefixes=GCN,PRE-GFX10,PRE-GFX12 %s
; RUN: llc < %s -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs | FileCheck --check-prefixes=GCN,PRE-GFX10,PRE-GFX12 %s
; RUN: llc < %s -mtriple=amdgcn -mcpu=gfx1010 -verify-machineinstrs | FileCheck --check-prefixes=GCN,GFX10,PRE-GFX12 %s
; RUN: llc < %s -mtriple=amdgcn -mcpu=gfx1100 -verify-machineinstrs | FileCheck --check-prefixes=GCN,GFX10,PRE-GFX12 %s
; RUN: llc < %s -mtriple=amdgcn -mcpu=gfx1200 -verify-machineinstrs | FileCheck --check-prefixes=GCN,GFX12 %s

; GCN-LABEL: {{^}}getlod_1d:
; PRE-GFX10: image_get_lod v[0:3], v0, s[0:7], s[8:11] dmask:0xf{{$}}
; GFX10: image_get_lod v[0:3], v0, s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D
; GFX12: image_get_lod v[0:3], v0, s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_1D
; PRE-GFX12: s_waitcnt vmcnt(0)
; GFX12: s_wait_samplecnt 0x0
define amdgpu_ps <4 x float> @getlod_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.getlod.1d.v4f32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %r
}

; GCN-LABEL: {{^}}getlod_2d:
; PRE-GFX10: image_get_lod v[0:1], v[0:1], s[0:7], s[8:11] dmask:0x3{{$}}
; GFX10: image_get_lod v[0:1], v[0:1], s[0:7], s[8:11] dmask:0x3 dim:SQ_RSRC_IMG_2D
; GFX12: image_get_lod v[0:1], [v0, v1], s[0:7], s[8:11] dmask:0x3 dim:SQ_RSRC_IMG_2D
; PRE-GFX12: s_waitcnt vmcnt(0)
; GFX12: s_wait_samplecnt 0x0
define amdgpu_ps <2 x float> @getlod_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %t) {
main_body:
  %r = call <2 x float> @llvm.amdgcn.image.getlod.2d.v2f32.f32(i32 3, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <2 x float> %r
}

; GCN-LABEL: {{^}}adjust_writemask_getlod_none_enabled:
; GCN-NOT: image
define amdgpu_ps <4 x float> @adjust_writemask_getlod_none_enabled(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %t) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.getlod.2d.v4f32.f32(i32 0, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %r
}

declare <4 x float> @llvm.amdgcn.image.getlod.1d.v4f32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #0
declare <4 x float> @llvm.amdgcn.image.getlod.2d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #0
declare <2 x float> @llvm.amdgcn.image.getlod.2d.v2f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #0

attributes #0 = { nounwind readnone }
