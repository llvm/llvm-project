; RUN: llc < %s -mtriple=amdgpu8.02 | FileCheck -check-prefixes=GCN,UNPACKED %s
; RUN: llc < %s -mtriple=amdgpu8.10 | FileCheck --check-prefix=GCN %s
; RUN: llc < %s -mtriple=amdgpu9.00 | FileCheck -check-prefixes=GCN,GFX9 %s
; RUN: llc < %s -mtriple=amdgpu9 --amdhsa-code-object-version=6 | FileCheck -check-prefixes=GCN,GFX9 %s
; RUN: llc < %s -mtriple=amdgpu10.10 | FileCheck -check-prefixes=GCN,GFX10 %s
; RUN: llc < %s -mtriple=amdgpu10.1 --amdhsa-code-object-version=6  | FileCheck -check-prefixes=GCN,GFX10 %s
; RUN: llc < %s -mtriple=amdgpu11.00 | FileCheck -check-prefixes=GCN,GFX10 %s
; RUN: llc < %s -mtriple=amdgpu11 --amdhsa-code-object-version=6 | FileCheck -check-prefixes=GCN,GFX10 %s
; RUN: llc < %s -mtriple=amdgpu12.00 | FileCheck -check-prefixes=GCN,GFX12PLUS %s
; RUN: llc < %s -mtriple=amdgpu13.10 | FileCheck -check-prefixes=GCN,GFX12PLUS %s

; GCN-LABEL: {{^}}image_gather4_b_2d_v4f16:
; UNPACKED: image_gather4_b v[0:3], v[0:2], s[0:7], s[8:11] dmask:0x4 d16{{$}}
; PACKED: image_gather4_b v[0:1], v[0:2], s[0:7], s[8:11] dmask:0x4 d16{{$}}
; GFX810: image_gather4_b v[0:3], v[0:2], s[0:7], s[8:11] dmask:0x4 d16{{$}}
; GFX9: image_gather4_b v[0:3], v[0:2], s[0:7], s[8:11] dmask:0x4 d16{{$}}
; GFX10: image_gather4_b v[0:1], v[0:2], s[0:7], s[8:11] dmask:0x4 dim:SQ_RSRC_IMG_2D d16{{$}}
; GFX12PLUS: image_gather4_b v[0:1], [v0, v1, v2], s[0:7], s[8:11] dmask:0x4 dim:SQ_RSRC_IMG_2D d16{{$}}
define amdgpu_ps <2 x float> @image_gather4_b_2d_v4f16(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %bias, float %s, float %t) {
main_body:
  %tex = call <4 x half> @llvm.amdgcn.image.gather4.b.2d.v4f16.f32.f32(i32 4, float %bias, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 false, i32 0, i32 0)
  %r = bitcast <4 x half> %tex to <2 x float>
  ret <2 x float> %r
}

; GCN-LABEL: {{^}}image_gather4_b_2d_v4f16_tfe:
; UNPACKED: image_gather4_b v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], s[0:7], s[8:11] dmask:0x4 tfe d16{{$}}
; GFX9: image_gather4_b v[0:4], v[{{[0-9]+:[0-9]+}}], s[0:7], s[8:11] dmask:0x4 tfe d16{{$}}
; GFX10: image_gather4_b v[0:2], v[{{[0-9]+:[0-9]+}}], s[0:7], s[8:11] dmask:0x4 dim:SQ_RSRC_IMG_2D tfe d16{{$}}
; GFX12PLUS: image_gather4_b v[0:2], [v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}], s[0:7], s[8:11] dmask:0x4 dim:SQ_RSRC_IMG_2D tfe d16{{$}}
define amdgpu_ps <4 x half> @image_gather4_b_2d_v4f16_tfe(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %bias, float %s, float %t, ptr addrspace(1) %out) {
main_body:
  %r = call { <4 x half>, i32 } @llvm.amdgcn.image.gather4.b.2d.sl_v4f16i32s.f32.f32(i32 4, float %bias, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 false, i32 1, i32 0)
  %tex = extractvalue { <4 x half>, i32 } %r, 0
  %tfe = extractvalue { <4 x half>, i32 } %r, 1
  store i32 %tfe, ptr addrspace(1) %out
  ret <4 x half> %tex
}

declare <4 x half> @llvm.amdgcn.image.gather4.b.2d.v4f16.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare { <4 x half>, i32 } @llvm.amdgcn.image.gather4.b.2d.sl_v4f16i32s.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
attributes #2 = { nounwind readnone }
