; RUN: llc < %s -mtriple=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck -check-prefixes=GCN,UNPACKED %s
; RUN: llc < %s -mtriple=amdgcn -mcpu=gfx810 -verify-machineinstrs | FileCheck --check-prefix=GCN %s
; RUN: llc < %s -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs | FileCheck -check-prefixes=GCN,GFX9 %s
; RUN: llc < %s -mtriple=amdgcn -mcpu=gfx9-generic --amdhsa-code-object-version=6 -verify-machineinstrs | FileCheck -check-prefixes=GCN,GFX9 %s
; RUN: llc < %s -mtriple=amdgcn -mcpu=gfx1010 -verify-machineinstrs | FileCheck -check-prefixes=GCN,GFX10 %s
; RUN: llc < %s -mtriple=amdgcn -mcpu=gfx10.1-generic --amdhsa-code-object-version=6  -verify-machineinstrs | FileCheck -check-prefixes=GCN,GFX10 %s
; RUN: llc < %s -mtriple=amdgcn -mcpu=gfx1100 -verify-machineinstrs | FileCheck -check-prefixes=GCN,GFX10 %s
; RUN: llc < %s -mtriple=amdgcn -mcpu=gfx11-generic --amdhsa-code-object-version=6 -verify-machineinstrs | FileCheck -check-prefixes=GCN,GFX10 %s
; RUN: llc < %s -mtriple=amdgcn -mcpu=gfx1200 -verify-machineinstrs | FileCheck -check-prefixes=GCN,GFX12 %s

; GCN-LABEL: {{^}}image_gather4_b_2d_v4f16:
; UNPACKED: image_gather4_b v[0:3], v[0:2], s[0:7], s[8:11] dmask:0x4 d16{{$}}
; PACKED: image_gather4_b v[0:1], v[0:2], s[0:7], s[8:11] dmask:0x4 d16{{$}}
; GFX810: image_gather4_b v[0:3], v[0:2], s[0:7], s[8:11] dmask:0x4 d16{{$}}
; GFX9: image_gather4_b v[0:3], v[0:2], s[0:7], s[8:11] dmask:0x4 d16{{$}}
; GFX10: image_gather4_b v[0:1], v[0:2], s[0:7], s[8:11] dmask:0x4 dim:SQ_RSRC_IMG_2D d16{{$}}
; GFX12: image_gather4_b v[0:1], [v0, v1, v2], s[0:7], s[8:11] dmask:0x4 dim:SQ_RSRC_IMG_2D d16{{$}}
define amdgpu_ps <2 x float> @image_gather4_b_2d_v4f16(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %bias, float %s, float %t) {
main_body:
  %tex = call <4 x half> @llvm.amdgcn.image.gather4.b.2d.v4f16.f32.f32(i32 4, float %bias, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 false, i32 0, i32 0)
  %r = bitcast <4 x half> %tex to <2 x float>
  ret <2 x float> %r
}

declare <4 x half> @llvm.amdgcn.image.gather4.b.2d.v4f16.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
attributes #2 = { nounwind readnone }
