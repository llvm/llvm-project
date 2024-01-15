; RUN: llc -march=amdgcn -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX90A,SDAG %s
; RUN: llc -march=amdgcn -mcpu=gfx90a -early-live-intervals -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX90A,SDAG %s
; RUN: llc -global-isel -march=amdgcn -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX90A,GISEL %s

; GFX90A-LABEL: {{^}}sample_1d:
; GFX90A-NOT: s_wqm_b64
; GFX90A:     image_sample v[{{[0-9:]+}}], v{{[0-9]+}}, s[{{[0-9:]+}}], s[{{[0-9:]+}}] dmask:0xf
define amdgpu_ps <4 x float> @sample_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GFX90A-LABEL: {{^}}sample_1d_lwe:
; GFX90A-NOT: s_wqm_b64
; GFX90A:     image_sample v[{{[0-9:]+}}], v{{[0-9]+}}, s[{{[0-9:]+}}], s[{{[0-9:]+}}] dmask:0xf lwe
define amdgpu_ps <4 x float> @sample_1d_lwe(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, ptr addrspace(1) inreg %out, float %s) {
main_body:
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.sample.1d.v4f32i32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 2, i32 0)
  %v.vec = extractvalue {<4 x float>, i32} %v, 0
  %v.err = extractvalue {<4 x float>, i32} %v, 1
  store i32 %v.err, ptr addrspace(1) %out, align 4
  ret <4 x float> %v.vec
}

; GFX90A-LABEL: {{^}}sample_2d:
; GFX90A-NOT: s_wqm_b64
; GFX90A:     image_sample v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}], s[{{[0-9:]+}}] dmask:0xf
define amdgpu_ps <4 x float> @sample_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32 15, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GFX90A-LABEL: {{^}}sample_3d:
; GFX90A-NOT: s_wqm_b64
; GFX90A:     image_sample v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}], s[{{[0-9:]+}}] dmask:0xf
define amdgpu_ps <4 x float> @sample_3d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %t, float %r) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.3d.v4f32.f32(i32 15, float %s, float %t, float %r, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GFX90A-LABEL: {{^}}sample_cube:
; GFX90A-NOT: s_wqm_b64
; GFX90A:     image_sample v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}], s[{{[0-9:]+}}] dmask:0xf da
define amdgpu_ps <4 x float> @sample_cube(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %t, float %face) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.cube.v4f32.f32(i32 15, float %s, float %t, float %face, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GFX90A-LABEL: {{^}}sample_1darray:
; GFX90A-NOT: s_wqm_b64
; GFX90A:     image_sample v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}], s[{{[0-9:]+}}] dmask:0xf da
define amdgpu_ps <4 x float> @sample_1darray(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %slice) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.1darray.v4f32.f32(i32 15, float %s, float %slice, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GFX90A-LABEL: {{^}}sample_1d_unorm:
; GFX90A-NOT: s_wqm_b64
; GFX90A:     image_sample v[{{[0-9:]+}}], v{{[0-9]+}}, s[{{[0-9:]+}}], s[{{[0-9:]+}}] dmask:0xf unorm
define amdgpu_ps <4 x float> @sample_1d_unorm(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 1, i32 0, i32 0)
  ret <4 x float> %v
}

; Address register must be even aligned.

; GFX90A-LABEL: {{^}}sample_1d_addr_align:
; GFX90A: v_mov_b32_e32 [[VADDR:v[0-9]?[02468]]], v1
; SDAG:   image_sample v{{[0-9]+}}, [[VADDR]], s[{{[0-9:]+}}], s[{{[0-9:]+}}] dmask:0x1
; GISEL:  image_sample v[{{[0-9:]+}}], [[VADDR]], s[{{[0-9:]+}}], s[{{[0-9:]+}}] dmask:0xf
define amdgpu_ps float @sample_1d_addr_align(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, <2 x float> %s) {
main_body:
  %s1 = extractelement <2 x float> %s, i32 1
  %v = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %s1, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %v1 = extractelement <4 x float> %v, i32 0
  ret float %v1
}

declare <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32)
declare {<4 x float>,i32} @llvm.amdgcn.image.sample.1d.v4f32i32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32)
declare <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32)
declare <4 x float> @llvm.amdgcn.image.sample.3d.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32)
declare <4 x float> @llvm.amdgcn.image.sample.cube.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32)
declare <4 x float> @llvm.amdgcn.image.sample.1darray.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32)
