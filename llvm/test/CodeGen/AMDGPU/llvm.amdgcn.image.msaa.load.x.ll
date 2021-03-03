; RUN: llc -march=amdgcn -mcpu=gfx1030 -verify-machineinstrs -show-mc-encoding < %s | FileCheck -check-prefixes=GCN,GFX10 %s

; GCN-LABEL: {{^}}load_1d:
; GFX10: image_msaa_load v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D unorm ;
define amdgpu_ps <4 x float> @load_1d(<8 x i32> inreg %rsrc, i32 %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.msaa.load.x.1d.v4f32.i32(i32 15, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_1d_tfe:
; GFX10: image_msaa_load v[0:4], v{{[0-9]+}}, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D unorm tfe ;
define amdgpu_ps <4 x float> @load_1d_tfe(<8 x i32> inreg %rsrc, i32 addrspace(1)* inreg %out, i32 %s) {
main_body:
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.x.1d.v4f32i32.i32(i32 15, i32 %s, <8 x i32> %rsrc, i32 1, i32 0)
  %v.vec = extractvalue {<4 x float>, i32} %v, 0
  %v.err = extractvalue {<4 x float>, i32} %v, 1
  store i32 %v.err, i32 addrspace(1)* %out, align 4
  ret <4 x float> %v.vec
}

; GCN-LABEL: {{^}}load_1d_lwe:
; GFX10: image_msaa_load v[0:4], v{{[0-9]+}}, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D unorm lwe ;
define amdgpu_ps <4 x float> @load_1d_lwe(<8 x i32> inreg %rsrc, i32 addrspace(1)* inreg %out, i32 %s) {
main_body:
  %v = call {<4 x float>, i32} @llvm.amdgcn.image.msaa.load.x.1d.v4f32i32.i32(i32 15, i32 %s, <8 x i32> %rsrc, i32 2, i32 0)
  %v.vec = extractvalue {<4 x float>, i32} %v, 0
  %v.err = extractvalue {<4 x float>, i32} %v, 1
  store i32 %v.err, i32 addrspace(1)* %out, align 4
  ret <4 x float> %v.vec
}

; GCN-LABEL: {{^}}load_2d:
; GFX10: image_msaa_load v[0:3], v[0:1], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D unorm ;
define amdgpu_ps <4 x float> @load_2d(<8 x i32> inreg %rsrc, i32 %s, i32 %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.msaa.load.x.2d.v4f32.i32(i32 15, i32 %s, i32 %t, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_2d_tfe:
; GFX10: image_msaa_load v[0:4], v[{{[0-9]+:[0-9]+}}], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D unorm tfe ;
define amdgpu_ps <4 x float> @load_2d_tfe(<8 x i32> inreg %rsrc, i32 addrspace(1)* inreg %out, i32 %s, i32 %t) {
main_body:
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.x.2d.v4f32i32.i32(i32 15, i32 %s, i32 %t, <8 x i32> %rsrc, i32 1, i32 0)
  %v.vec = extractvalue {<4 x float>, i32} %v, 0
  %v.err = extractvalue {<4 x float>, i32} %v, 1
  store i32 %v.err, i32 addrspace(1)* %out, align 4
  ret <4 x float> %v.vec
}

; GCN-LABEL: {{^}}load_3d:
; GFX10: image_msaa_load v[0:3], v[0:2], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_3D unorm ;
define amdgpu_ps <4 x float> @load_3d(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %r) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.msaa.load.x.3d.v4f32.i32(i32 15, i32 %s, i32 %t, i32 %r, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_3d_tfe_lwe:
; GFX10: image_msaa_load v[0:4], v[{{[0-9]+:[0-9]+}}], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_3D unorm tfe lwe ;
define amdgpu_ps <4 x float> @load_3d_tfe_lwe(<8 x i32> inreg %rsrc, i32 addrspace(1)* inreg %out, i32 %s, i32 %t, i32 %r) {
main_body:
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.x.3d.v4f32i32.i32(i32 15, i32 %s, i32 %t, i32 %r, <8 x i32> %rsrc, i32 3, i32 0)
  %v.vec = extractvalue {<4 x float>, i32} %v, 0
  %v.err = extractvalue {<4 x float>, i32} %v, 1
  store i32 %v.err, i32 addrspace(1)* %out, align 4
  ret <4 x float> %v.vec
}

; GCN-LABEL: {{^}}load_1darray:
; GFX10: image_msaa_load v[0:3], v[0:1], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D_ARRAY unorm ;
define amdgpu_ps <4 x float> @load_1darray(<8 x i32> inreg %rsrc, i32 %s, i32 %slice) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.msaa.load.x.1darray.v4f32.i32(i32 15, i32 %s, i32 %slice, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_1darray_tfe:
; GFX10: image_msaa_load v[0:4], v[{{[0-9]+:[0-9]+}}], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D_ARRAY unorm tfe ;
define amdgpu_ps <4 x float> @load_1darray_tfe(<8 x i32> inreg %rsrc, i32 addrspace(1)* inreg %out, i32 %s, i32 %slice) {
main_body:
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.x.1darray.v4f32i32.i32(i32 15, i32 %s, i32 %slice, <8 x i32> %rsrc, i32 1, i32 0)
  %v.vec = extractvalue {<4 x float>, i32} %v, 0
  %v.err = extractvalue {<4 x float>, i32} %v, 1
  store i32 %v.err, i32 addrspace(1)* %out, align 4
  ret <4 x float> %v.vec
}

; GCN-LABEL: {{^}}load_2darray:
; GFX10: image_msaa_load v[0:3], v[0:2], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY unorm ;
define amdgpu_ps <4 x float> @load_2darray(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %slice) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.msaa.load.x.2darray.v4f32.i32(i32 15, i32 %s, i32 %t, i32 %slice, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_2darray_lwe:
; GFX10: image_msaa_load v[0:4], v[{{[0-9]+:[0-9]+}}], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY unorm lwe ;
define amdgpu_ps <4 x float> @load_2darray_lwe(<8 x i32> inreg %rsrc, i32 addrspace(1)* inreg %out, i32 %s, i32 %t, i32 %slice) {
main_body:
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.x.2darray.v4f32i32.i32(i32 15, i32 %s, i32 %t, i32 %slice, <8 x i32> %rsrc, i32 2, i32 0)
  %v.vec = extractvalue {<4 x float>, i32} %v, 0
  %v.err = extractvalue {<4 x float>, i32} %v, 1
  store i32 %v.err, i32 addrspace(1)* %out, align 4
  ret <4 x float> %v.vec
}

; GCN-LABEL: {{^}}load_2dmsaa:
; GFX10: image_msaa_load v[0:3], v[0:2], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm ;
define amdgpu_ps <4 x float> @load_2dmsaa(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %fragid) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.msaa.load.x.2dmsaa.v4f32.i32(i32 15, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_2dmsaa_both:
; GFX10: image_msaa_load v[0:4], v[{{[0-9]+:[0-9]+}}], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA unorm tfe lwe ;
define amdgpu_ps <4 x float> @load_2dmsaa_both(<8 x i32> inreg %rsrc, i32 addrspace(1)* inreg %out, i32 %s, i32 %t, i32 %fragid) {
main_body:
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.x.2dmsaa.v4f32i32.i32(i32 15, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 3, i32 0)
  %v.vec = extractvalue {<4 x float>, i32} %v, 0
  %v.err = extractvalue {<4 x float>, i32} %v, 1
  store i32 %v.err, i32 addrspace(1)* %out, align 4
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
define amdgpu_ps <4 x float> @load_2darraymsaa_tfe(<8 x i32> inreg %rsrc, i32 addrspace(1)* inreg %out, i32 %s, i32 %t, i32 %slice, i32 %fragid) {
main_body:
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.x.2darraymsaa.v4f32i32.i32(i32 15, i32 %s, i32 %t, i32 %slice, i32 %fragid, <8 x i32> %rsrc, i32 1, i32 0)
  %v.vec = extractvalue {<4 x float>, i32} %v, 0
  %v.err = extractvalue {<4 x float>, i32} %v, 1
  store i32 %v.err, i32 addrspace(1)* %out, align 4
  ret <4 x float> %v.vec
}

; GCN-LABEL: {{^}}load_1d_tfe_V4_dmask3:
; GFX10: image_msaa_load v[0:3], v{{[0-9]+}}, s[0:7] dmask:0x7 dim:SQ_RSRC_IMG_1D unorm tfe ;
define amdgpu_ps <4 x float> @load_1d_tfe_V4_dmask3(<8 x i32> inreg %rsrc, i32 addrspace(1)* inreg %out, i32 %s) {
main_body:
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.x.1d.v4f32i32.i32(i32 7, i32 %s, <8 x i32> %rsrc, i32 1, i32 0)
  %v.vec = extractvalue {<4 x float>, i32} %v, 0
  %v.err = extractvalue {<4 x float>, i32} %v, 1
  store i32 %v.err, i32 addrspace(1)* %out, align 4
  ret <4 x float> %v.vec
}

; GCN-LABEL: {{^}}load_1d_tfe_V4_dmask2:
; GFX10: image_msaa_load v[0:2], v{{[0-9]+}}, s[0:7] dmask:0x6 dim:SQ_RSRC_IMG_1D unorm tfe ;
define amdgpu_ps <4 x float> @load_1d_tfe_V4_dmask2(<8 x i32> inreg %rsrc, i32 addrspace(1)* inreg %out, i32 %s) {
main_body:
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.x.1d.v4f32i32.i32(i32 6, i32 %s, <8 x i32> %rsrc, i32 1, i32 0)
  %v.vec = extractvalue {<4 x float>, i32} %v, 0
  %v.err = extractvalue {<4 x float>, i32} %v, 1
  store i32 %v.err, i32 addrspace(1)* %out, align 4
  ret <4 x float> %v.vec
}

; GCN-LABEL: {{^}}load_1d_tfe_V4_dmask1:
; GFX10: image_msaa_load v[0:1], v{{[0-9]+}}, s[0:7] dmask:0x8 dim:SQ_RSRC_IMG_1D unorm tfe ;
define amdgpu_ps <4 x float> @load_1d_tfe_V4_dmask1(<8 x i32> inreg %rsrc, i32 addrspace(1)* inreg %out, i32 %s) {
main_body:
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.x.1d.v4f32i32.i32(i32 8, i32 %s, <8 x i32> %rsrc, i32 1, i32 0)
  %v.vec = extractvalue {<4 x float>, i32} %v, 0
  %v.err = extractvalue {<4 x float>, i32} %v, 1
  store i32 %v.err, i32 addrspace(1)* %out, align 4
  ret <4 x float> %v.vec
}

; GCN-LABEL: {{^}}load_1d_tfe_V2_dmask1:
; GFX10: image_msaa_load v[0:1], v{{[0-9]+}}, s[0:7] dmask:0x8 dim:SQ_RSRC_IMG_1D unorm tfe ;
define amdgpu_ps <2 x float> @load_1d_tfe_V2_dmask1(<8 x i32> inreg %rsrc, i32 addrspace(1)* inreg %out, i32 %s) {
main_body:
  %v = call {<2 x float>,i32} @llvm.amdgcn.image.msaa.load.x.1d.v2f32i32.i32(i32 8, i32 %s, <8 x i32> %rsrc, i32 1, i32 0)
  %v.vec = extractvalue {<2 x float>, i32} %v, 0
  %v.err = extractvalue {<2 x float>, i32} %v, 1
  store i32 %v.err, i32 addrspace(1)* %out, align 4
  ret <2 x float> %v.vec
}

; GCN-LABEL: {{^}}load_1d_V1:
; GFX10: image_msaa_load v0, v0, s[0:7] dmask:0x8 dim:SQ_RSRC_IMG_1D unorm ;
define amdgpu_ps float @load_1d_V1(<8 x i32> inreg %rsrc, i32 %s) {
main_body:
  %v = call float @llvm.amdgcn.image.msaa.load.x.1d.f32.i32(i32 8, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  ret float %v
}

; GCN-LABEL: {{^}}load_1d_V2:
; GFX10: image_msaa_load v[0:1], v0, s[0:7] dmask:0x9 dim:SQ_RSRC_IMG_1D unorm ;
define amdgpu_ps <2 x float> @load_1d_V2(<8 x i32> inreg %rsrc, i32 %s) {
main_body:
  %v = call <2 x float> @llvm.amdgcn.image.msaa.load.x.1d.v2f32.i32(i32 9, i32 %s, <8 x i32> %rsrc, i32 0, i32 0)
  ret <2 x float> %v
}

; GCN-LABEL: {{^}}load_1d_glc:
; GFX10: image_msaa_load v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D unorm glc ;
define amdgpu_ps <4 x float> @load_1d_glc(<8 x i32> inreg %rsrc, i32 %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.msaa.load.x.1d.v4f32.i32(i32 15, i32 %s, <8 x i32> %rsrc, i32 0, i32 1)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_1d_slc:
; GFX10: image_msaa_load v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D unorm slc ;
define amdgpu_ps <4 x float> @load_1d_slc(<8 x i32> inreg %rsrc, i32 %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.msaa.load.x.1d.v4f32.i32(i32 15, i32 %s, <8 x i32> %rsrc, i32 0, i32 2)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_1d_glc_slc:
; GFX10: image_msaa_load v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D unorm glc slc ;
define amdgpu_ps <4 x float> @load_1d_glc_slc(<8 x i32> inreg %rsrc, i32 %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.msaa.load.x.1d.v4f32.i32(i32 15, i32 %s, <8 x i32> %rsrc, i32 0, i32 3)
  ret <4 x float> %v
}

declare <4 x float> @llvm.amdgcn.image.msaa.load.x.1d.v4f32.i32(i32, i32, <8 x i32>, i32, i32) #1
declare {float,i32} @llvm.amdgcn.image.msaa.load.x.1d.f32i32.i32(i32, i32, <8 x i32>, i32, i32) #1
declare {<2 x float>,i32} @llvm.amdgcn.image.msaa.load.x.1d.v2f32i32.i32(i32, i32, <8 x i32>, i32, i32) #1
declare {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.x.1d.v4f32i32.i32(i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.msaa.load.x.2d.v4f32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1
declare {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.x.2d.v4f32i32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.msaa.load.x.3d.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.x.3d.v4f32i32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.msaa.load.x.1darray.v4f32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1
declare {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.x.1darray.v4f32i32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.msaa.load.x.2darray.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.x.2darray.v4f32i32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.msaa.load.x.2dmsaa.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.x.2dmsaa.v4f32i32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.msaa.load.x.2darraymsaa.v4f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.x.2darraymsaa.v4f32i32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

declare float @llvm.amdgcn.image.msaa.load.x.1d.f32.i32(i32, i32, <8 x i32>, i32, i32) #1
declare float @llvm.amdgcn.image.msaa.load.x.2d.f32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1
declare <2 x float> @llvm.amdgcn.image.msaa.load.x.1d.v2f32.i32(i32, i32, <8 x i32>, i32, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
