; RUN: llc -march=amdgcn -mcpu=gfx1030 -verify-machineinstrs -show-mc-encoding < %s | FileCheck -check-prefixes=GCN,GFX10 %s

; GCN-LABEL: {{^}}load_2dmsaa:
; GFX10: image_msaa_load v[0:3], v[0:2], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm ;
define amdgpu_ps <4 x float> @load_2dmsaa(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %fragid) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.msaa.load.x.2dmsaa.v4f32.i32(i32 15, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_2dmsaa_both:
; GFX10: image_msaa_load v[0:4], v[{{[0-9]+:[0-9]+}}], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm tfe lwe ;
define amdgpu_ps <4 x float> @load_2dmsaa_both(<8 x i32> inreg %rsrc, ptr addrspace(1) inreg %out, i32 %s, i32 %t, i32 %fragid) {
main_body:
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.x.2dmsaa.v4f32i32.i32(i32 15, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 3, i32 0)
  %v.vec = extractvalue {<4 x float>, i32} %v, 0
  %v.err = extractvalue {<4 x float>, i32} %v, 1
  store i32 %v.err, ptr addrspace(1) %out, align 4
  ret <4 x float> %v.vec
}

; GCN-LABEL: {{^}}load_2darraymsaa:
; GFX10: image_msaa_load v[0:3], v[0:3], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA_ARRAY unorm ;
define amdgpu_ps <4 x float> @load_2darraymsaa(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %slice, i32 %fragid) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.msaa.load.x.2darraymsaa.v4f32.i32(i32 15, i32 %s, i32 %t, i32 %slice, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_2darraymsaa_tfe:
; GFX10: image_msaa_load v[0:4], v[{{[0-9]+:[0-9]+}}], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA_ARRAY unorm tfe ;
define amdgpu_ps <4 x float> @load_2darraymsaa_tfe(<8 x i32> inreg %rsrc, ptr addrspace(1) inreg %out, i32 %s, i32 %t, i32 %slice, i32 %fragid) {
main_body:
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.x.2darraymsaa.v4f32i32.i32(i32 15, i32 %s, i32 %t, i32 %slice, i32 %fragid, <8 x i32> %rsrc, i32 1, i32 0)
  %v.vec = extractvalue {<4 x float>, i32} %v, 0
  %v.err = extractvalue {<4 x float>, i32} %v, 1
  store i32 %v.err, ptr addrspace(1) %out, align 4
  ret <4 x float> %v.vec
}

; GCN-LABEL: {{^}}load_2dmsaa_tfe_V4_dmask3:
; GFX10: image_msaa_load v[0:3], v[{{[0-9]+:[0-9]+}}], s[0:7] dmask:0x7 dim:SQ_RSRC_IMG_2D_MSAA unorm tfe ;
define amdgpu_ps <4 x float> @load_2dmsaa_tfe_V4_dmask3(<8 x i32> inreg %rsrc, ptr addrspace(1) inreg %out, i32 %s, i32 %t, i32 %fragid) {
main_body:
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.x.2dmsaa.v4f32i32.i32(i32 7, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 1, i32 0)
  %v.vec = extractvalue {<4 x float>, i32} %v, 0
  %v.err = extractvalue {<4 x float>, i32} %v, 1
  store i32 %v.err, ptr addrspace(1) %out, align 4
  ret <4 x float> %v.vec
}

; GCN-LABEL: {{^}}load_2dmsaa_tfe_V4_dmask2:
; GFX10: image_msaa_load v[0:2], v[{{[0-9]+:[0-9]+}}], s[0:7] dmask:0x6 dim:SQ_RSRC_IMG_2D_MSAA unorm tfe ;
define amdgpu_ps <4 x float> @load_2dmsaa_tfe_V4_dmask2(<8 x i32> inreg %rsrc, ptr addrspace(1) inreg %out, i32 %s, i32 %t, i32 %fragid) {
main_body:
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.x.2dmsaa.v4f32i32.i32(i32 6, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 1, i32 0)
  %v.vec = extractvalue {<4 x float>, i32} %v, 0
  %v.err = extractvalue {<4 x float>, i32} %v, 1
  store i32 %v.err, ptr addrspace(1) %out, align 4
  ret <4 x float> %v.vec
}

; GCN-LABEL: {{^}}load_2dmsaa_tfe_V4_dmask1:
; GFX10: image_msaa_load v[0:1], [v4, v3, v2], s[0:7] dmask:0x8 dim:SQ_RSRC_IMG_2D_MSAA unorm tfe ;
define amdgpu_ps <4 x float> @load_2dmsaa_tfe_V4_dmask1(<8 x i32> inreg %rsrc, ptr addrspace(1) inreg %out, i32 %s, i32 %t, i32 %fragid) {
main_body:
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.x.2dmsaa.v4f32i32.i32(i32 8, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 1, i32 0)
  %v.vec = extractvalue {<4 x float>, i32} %v, 0
  %v.err = extractvalue {<4 x float>, i32} %v, 1
  store i32 %v.err, ptr addrspace(1) %out, align 4
  ret <4 x float> %v.vec
}

; GCN-LABEL: {{^}}load_2dmsaa_tfe_V2_dmask1:
; GFX10: image_msaa_load v[0:1], [v4, v3, v2], s[0:7] dmask:0x8 dim:SQ_RSRC_IMG_2D_MSAA unorm tfe ;
define amdgpu_ps <2 x float> @load_2dmsaa_tfe_V2_dmask1(<8 x i32> inreg %rsrc, ptr addrspace(1) inreg %out, i32 %s, i32 %t, i32 %fragid) {
main_body:
  %v = call {<2 x float>,i32} @llvm.amdgcn.image.msaa.load.x.2dmsaa.v2f32i32.i32(i32 8, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 1, i32 0)
  %v.vec = extractvalue {<2 x float>, i32} %v, 0
  %v.err = extractvalue {<2 x float>, i32} %v, 1
  store i32 %v.err, ptr addrspace(1) %out, align 4
  ret <2 x float> %v.vec
}

; GCN-LABEL: {{^}}load_2dmsaa_V1:
; GFX10: image_msaa_load v0, v[{{[0-9]+:[0-9]+}}], s[0:7] dmask:0x8 dim:SQ_RSRC_IMG_2D_MSAA unorm ;
define amdgpu_ps float @load_2dmsaa_V1(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %fragid) {
main_body:
  %v = call float @llvm.amdgcn.image.msaa.load.x.2dmsaa.f32.i32(i32 8, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
  ret float %v
}

; GCN-LABEL: {{^}}load_2dmsaa_V2:
; GFX10: image_msaa_load v[0:1], v[{{[0-9]+:[0-9]+}}], s[0:7] dmask:0x9 dim:SQ_RSRC_IMG_2D_MSAA unorm ;
define amdgpu_ps <2 x float> @load_2dmsaa_V2(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %fragid) {
main_body:
  %v = call <2 x float> @llvm.amdgcn.image.msaa.load.x.2dmsaa.v2f32.i32(i32 9, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
  ret <2 x float> %v
}

; GCN-LABEL: {{^}}load_2dmsaa_glc:
; GFX10: image_msaa_load v[0:3], v[{{[0-9]+:[0-9]+}}], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm glc ;
define amdgpu_ps <4 x float> @load_2dmsaa_glc(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %fragid) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.msaa.load.x.2dmsaa.v4f32.i32(i32 15, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 1)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_2dmsaa_slc:
; GFX10: image_msaa_load v[0:3], v[{{[0-9]+:[0-9]+}}], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm slc ;
define amdgpu_ps <4 x float> @load_2dmsaa_slc(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %fragid) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.msaa.load.x.2dmsaa.v4f32.i32(i32 15, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 2)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_2dmsaa_glc_slc:
; GFX10: image_msaa_load v[0:3], v[{{[0-9]+:[0-9]+}}], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm glc slc ;
define amdgpu_ps <4 x float> @load_2dmsaa_glc_slc(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %fragid) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.msaa.load.x.2dmsaa.v4f32.i32(i32 15, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 3)
  ret <4 x float> %v
}

declare <4 x float> @llvm.amdgcn.image.msaa.load.x.2dmsaa.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.x.2dmsaa.v4f32i32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.msaa.load.x.2darraymsaa.v4f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.x.2darraymsaa.v4f32i32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

declare float @llvm.amdgcn.image.msaa.load.x.2dmsaa.f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare <2 x float> @llvm.amdgcn.image.msaa.load.x.2dmsaa.v2f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare {<2 x float>,i32} @llvm.amdgcn.image.msaa.load.x.2dmsaa.v2f32i32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
