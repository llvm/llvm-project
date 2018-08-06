; RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck --check-prefix=GCN %s
; RUN: llc < %s -march=amdgcn -mcpu=gfx900 -verify-machineinstrs | FileCheck --check-prefix=GCN %s

; GCN-LABEL: {{^}}getlod_1d:
; GCN: image_get_lod v[0:3], v0, s[0:7], s[8:11] dmask:0xf{{$}}
; GCN: s_waitcnt vmcnt(0)
define amdgpu_ps <4 x float> @getlod_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s) {
main_body:
  %r = call <4 x float> @llvm.amdgcn.image.getlod.1d.v4f32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %r
}

; GCN-LABEL: {{^}}getlod_2d:
; GCN: image_get_lod v[0:1], v[0:1], s[0:7], s[8:11] dmask:0x3{{$}}
; GCN: s_waitcnt vmcnt(0)
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
