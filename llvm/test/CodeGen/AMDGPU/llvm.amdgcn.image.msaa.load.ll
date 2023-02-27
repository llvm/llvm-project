; RUN: llc -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs -show-mc-encoding < %s | FileCheck -check-prefixes=GCN,GFX11 %s
; RUN: llc -march=amdgcn -mcpu=gfx1200 -verify-machineinstrs -show-mc-encoding < %s | FileCheck -check-prefixes=GCN,GFX12 %s

; GCN-LABEL: {{^}}load_2dmsaa:
; GFX11: image_msaa_load v[0:3], v[0:2], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA unorm ;
; GFX12: image_msaa_load v[0:3], [v0, v1, v2], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA ;
define amdgpu_ps <4 x float> @load_2dmsaa(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %fragid) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 1, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_2dmsaa_both:
; GFX11: image_msaa_load v[0:4], v[{{[0-9]+:[0-9]+}}], s[0:7] dmask:0x2 dim:SQ_RSRC_IMG_2D_MSAA unorm tfe lwe ;
; GFX12: image_msaa_load v[0:4], [v0, v1, v2], s[0:7] dmask:0x2 dim:SQ_RSRC_IMG_2D_MSAA tfe ;
define amdgpu_ps <4 x float> @load_2dmsaa_both(<8 x i32> inreg %rsrc, ptr addrspace(1) inreg %out, i32 %s, i32 %t, i32 %fragid) {
main_body:
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32i32.i32(i32 2, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 3, i32 0)
  %v.vec = extractvalue {<4 x float>, i32} %v, 0
  %v.err = extractvalue {<4 x float>, i32} %v, 1
  store i32 %v.err, ptr addrspace(1) %out, align 4
  ret <4 x float> %v.vec
}

; GCN-LABEL: {{^}}load_2darraymsaa:
; GFX11: image_msaa_load v[0:3], v[0:3], s[0:7] dmask:0x4 dim:SQ_RSRC_IMG_2D_MSAA_ARRAY unorm ;
; GFX12: image_msaa_load v[0:3], [v0, v1, v2, v3], s[0:7] dmask:0x4 dim:SQ_RSRC_IMG_2D_MSAA_ARRAY ;
define amdgpu_ps <4 x float> @load_2darraymsaa(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %slice, i32 %fragid) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.msaa.load.2darraymsaa.v4f32.i32(i32 4, i32 %s, i32 %t, i32 %slice, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_2darraymsaa_tfe:
; GFX11: image_msaa_load v[0:4], v[{{[0-9]+:[0-9]+}}], s[0:7] dmask:0x8 dim:SQ_RSRC_IMG_2D_MSAA_ARRAY unorm tfe ;
; GFX12: image_msaa_load v[0:4], [v0, v1, v2, v3], s[0:7] dmask:0x8 dim:SQ_RSRC_IMG_2D_MSAA_ARRAY tfe ;
define amdgpu_ps <4 x float> @load_2darraymsaa_tfe(<8 x i32> inreg %rsrc, ptr addrspace(1) inreg %out, i32 %s, i32 %t, i32 %slice, i32 %fragid) {
main_body:
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.2darraymsaa.v4f32i32.i32(i32 8, i32 %s, i32 %t, i32 %slice, i32 %fragid, <8 x i32> %rsrc, i32 1, i32 0)
  %v.vec = extractvalue {<4 x float>, i32} %v, 0
  %v.err = extractvalue {<4 x float>, i32} %v, 1
  store i32 %v.err, ptr addrspace(1) %out, align 4
  ret <4 x float> %v.vec
}

; GCN-LABEL: {{^}}load_2dmsaa_glc:
; GFX11: image_msaa_load v[0:3], v[{{[0-9]+:[0-9]+}}], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA unorm glc ;
; GFX12: image_msaa_load v[0:3], [v0, v1, v2], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA th:TH_LOAD_NT ;
define amdgpu_ps <4 x float> @load_2dmsaa_glc(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %fragid) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 1, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 1)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_2dmsaa_slc:
; GFX11: image_msaa_load v[0:3], v[{{[0-9]+:[0-9]+}}], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA unorm slc ;
; GFX12: image_msaa_load v[0:3], [v0, v1, v2], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA th:TH_LOAD_HT ;
define amdgpu_ps <4 x float> @load_2dmsaa_slc(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %fragid) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 1, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 2)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_2dmsaa_glc_slc:
; GFX11: image_msaa_load v[0:3], v[{{[0-9]+:[0-9]+}}], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA unorm glc slc ;
; GFX12: image_msaa_load v[0:3], [v0, v1, v2], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA th:TH_LOAD_LU ;
define amdgpu_ps <4 x float> @load_2dmsaa_glc_slc(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %fragid) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32 1, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 3)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_2dmsaa_d16:
; GFX11: image_msaa_load v[0:1], v[0:2], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA unorm d16 ;
; GFX12: image_msaa_load v[0:1], [v0, v1, v2], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA d16 ;
define amdgpu_ps <4 x half> @load_2dmsaa_d16(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %fragid) {
main_body:
  %v = call <4 x half> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f16.i32(i32 1, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x half> %v
}

; GCN-LABEL: {{^}}load_2dmsaa_tfe_d16:
; GFX11: image_msaa_load v[0:2], v[{{[0-9]+:[0-9]+}}], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA unorm tfe d16 ;
; GFX12: image_msaa_load v[0:2], [v0, v1, v2], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA tfe d16 ;
define amdgpu_ps <4 x half> @load_2dmsaa_tfe_d16(<8 x i32> inreg %rsrc, ptr addrspace(1) inreg %out, i32 %s, i32 %t, i32 %fragid) {
main_body:
  %v = call {<4 x half>,i32} @llvm.amdgcn.image.msaa.load.2dmsaa.v4f16i32.i32(i32 1, i32 %s, i32 %t, i32 %fragid, <8 x i32> %rsrc, i32 1, i32 0)
  %v.vec = extractvalue {<4 x half>, i32} %v, 0
  %v.err = extractvalue {<4 x half>, i32} %v, 1
  store i32 %v.err, ptr addrspace(1) %out, align 4
  ret <4 x half> %v.vec
}

; GCN-LABEL: {{^}}load_2darraymsaa_d16:
; GFX11: image_msaa_load v[0:1], v[0:3], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA_ARRAY unorm d16 ;
; GFX12: image_msaa_load v[0:1], [v0, v1, v2, v3], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA_ARRAY d16 ;
define amdgpu_ps <4 x half> @load_2darraymsaa_d16(<8 x i32> inreg %rsrc, i32 %s, i32 %t, i32 %slice, i32 %fragid) {
main_body:
  %v = call <4 x half> @llvm.amdgcn.image.msaa.load.2darraymsaa.v4f16.i32(i32 1, i32 %s, i32 %t, i32 %slice, i32 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x half> %v
}

; GCN-LABEL: {{^}}load_2darraymsaa_tfe_d16:
; GFX11: image_msaa_load v[0:2], v[0:3], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA_ARRAY unorm tfe d16 ;
; GFX12: image_msaa_load v[0:2], [v0, v1, v2, v3], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA_ARRAY tfe d16 ;
define amdgpu_ps <4 x half> @load_2darraymsaa_tfe_d16(<8 x i32> inreg %rsrc, ptr addrspace(1) inreg %out, i32 %s, i32 %t, i32 %slice, i32 %fragid) {
main_body:
  %v = call {<4 x half>,i32} @llvm.amdgcn.image.msaa.load.2darraymsaa.v4f16i32.i32(i32 1, i32 %s, i32 %t, i32 %slice, i32 %fragid, <8 x i32> %rsrc, i32 1, i32 0)
  %v.vec = extractvalue {<4 x half>, i32} %v, 0
  %v.err = extractvalue {<4 x half>, i32} %v, 1
  store i32 %v.err, ptr addrspace(1) %out, align 4
  ret <4 x half> %v.vec
}

; GCN-LABEL: {{^}}load_2dmsaa_a16:
; GFX11: image_msaa_load v[0:3], v[1:2], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA unorm a16 ;
; GFX12: image_msaa_load v[0:3], [v0, v2], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA a16 ;
define amdgpu_ps <4 x float> @load_2dmsaa_a16(<8 x i32> inreg %rsrc, i16 %s, i16 %t, i16 %fragid) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i16(i32 1, i16 %s, i16 %t, i16 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}load_2darraymsaa_a16:
; GFX11: image_msaa_load v[0:3], v[1:2], s[0:7] dmask:0x4 dim:SQ_RSRC_IMG_2D_MSAA_ARRAY unorm a16 ;
; GFX12: image_msaa_load v[0:3], [v0, v2], s[0:7] dmask:0x4 dim:SQ_RSRC_IMG_2D_MSAA_ARRAY a16 ;
define amdgpu_ps <4 x float> @load_2darraymsaa_a16(<8 x i32> inreg %rsrc, i16 %s, i16 %t, i16 %slice, i16 %fragid) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.msaa.load.2darraymsaa.v4f32.i16(i32 4, i16 %s, i16 %t, i16 %slice, i16 %fragid, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

declare <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32i32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.msaa.load.2darraymsaa.v4f32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare {<4 x float>,i32} @llvm.amdgcn.image.msaa.load.2darraymsaa.v4f32i32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

declare <4 x half> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f16.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare {<4 x half>,i32} @llvm.amdgcn.image.msaa.load.2dmsaa.v4f16i32.i32(i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare <4 x half> @llvm.amdgcn.image.msaa.load.2darraymsaa.v4f16.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1
declare {<4 x half>,i32} @llvm.amdgcn.image.msaa.load.2darraymsaa.v4f16i32.i32(i32, i32, i32, i32, i32, <8 x i32>, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.msaa.load.2dmsaa.v4f32.i16(i32, i16, i16, i16, <8 x i32>, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.msaa.load.2darraymsaa.v4f32.i16(i32, i16, i16, i16, i16, <8 x i32>, i32, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
