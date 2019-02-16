; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s

; GCN-LABEL: {{^}}gather4_2d:
; GCN: image_gather4 v[0:3], v0, s[0:7], s[8:11] dmask:0x1 a16{{$}}
define amdgpu_ps <4 x float> @gather4_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %s, half %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.2d.v4f32.f16(i32 1, half %s, half %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_cube:
; GCN: image_gather4 v[0:3], v[0:1], s[0:7], s[8:11] dmask:0x1 a16 da{{$}}
define amdgpu_ps <4 x float> @gather4_cube(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %s, half %t, half %face) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.cube.v4f32.f16(i32 1, half %s, half %t, half %face, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_2darray:
; GCN: image_gather4 v[0:3], v[0:1], s[0:7], s[8:11] dmask:0x1 a16 da{{$}}
define amdgpu_ps <4 x float> @gather4_2darray(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %s, half %t, half %slice) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.2darray.v4f32.f16(i32 1, half %s, half %t, half %slice, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_c_2d:
; GCN: image_gather4_c v[0:3], v[0:1], s[0:7], s[8:11] dmask:0x1 a16{{$}}
define amdgpu_ps <4 x float> @gather4_c_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, half %s, half %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.c.2d.v4f32.f32(i32 1, float %zcompare, half %s, half %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_cl_2d:
; GCN: image_gather4_cl v[0:3], v[0:1], s[0:7], s[8:11] dmask:0x1 a16{{$}}
define amdgpu_ps <4 x float> @gather4_cl_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %s, half %t, half %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.cl.2d.v4f32.f16(i32 1, half %s, half %t, half %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_c_cl_2d:
; GCN: image_gather4_c_cl v[0:3], v[0:3], s[0:7], s[8:11] dmask:0x1 a16{{$}}
define amdgpu_ps <4 x float> @gather4_c_cl_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, half %s, half %t, half %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.c.cl.2d.v4f32.f32(i32 1, float %zcompare, half %s, half %t, half %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_b_2d:
; GCN: image_gather4_b v[0:3], v[0:1], s[0:7], s[8:11] dmask:0x1 a16{{$}}
define amdgpu_ps <4 x float> @gather4_b_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %bias, half %s, half %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.b.2d.v4f32.f32.f16(i32 1, float %bias, half %s, half %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_c_b_2d:
; GCN: image_gather4_c_b v[0:3], v[0:3], s[0:7], s[8:11] dmask:0x1 a16{{$}}
define amdgpu_ps <4 x float> @gather4_c_b_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %bias, float %zcompare, half %s, half %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.c.b.2d.v4f32.f32.f16(i32 1, float %bias, float %zcompare, half %s, half %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_b_cl_2d:
; GCN: image_gather4_b_cl v[0:3], v[0:3], s[0:7], s[8:11] dmask:0x1 a16{{$}}
define amdgpu_ps <4 x float> @gather4_b_cl_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %bias, half %s, half %t, half %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.b.cl.2d.v4f32.f32.f16(i32 1, float %bias, half %s, half %t, half %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_c_b_cl_2d:
; GCN: image_gather4_c_b_cl v[0:3], v[0:3], s[0:7], s[8:11] dmask:0x1 a16{{$}}
define amdgpu_ps <4 x float> @gather4_c_b_cl_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %bias, float %zcompare, half %s, half %t, half %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.c.b.cl.2d.v4f32.f32.f16(i32 1, float %bias, float %zcompare, half %s, half %t, half %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_l_2d:
; GCN: image_gather4_l v[0:3], v[0:1], s[0:7], s[8:11] dmask:0x1 a16{{$}}
define amdgpu_ps <4 x float> @gather4_l_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %s, half %t, half %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.l.2d.v4f32.f16(i32 1, half %s, half %t, half %lod, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_c_l_2d:
; GCN: image_gather4_c_l v[0:3], v[0:3], s[0:7], s[8:11] dmask:0x1 a16{{$}}
define amdgpu_ps <4 x float> @gather4_c_l_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, half %s, half %t, half %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.c.l.2d.v4f32.f32(i32 1, float %zcompare, half %s, half %t, half %lod, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_lz_2d:
; GCN: image_gather4_lz v[0:3], v0, s[0:7], s[8:11] dmask:0x1 a16{{$}}
define amdgpu_ps <4 x float> @gather4_lz_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %s, half %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.lz.2d.v4f32.f16(i32 1, half %s, half %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_c_lz_2d:
; GCN: image_gather4_c_lz v[0:3], v[0:1], s[0:7], s[8:11] dmask:0x1 a16{{$}}
define amdgpu_ps <4 x float> @gather4_c_lz_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, half %s, half %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.c.lz.2d.v4f32.f32(i32 1, float %zcompare, half %s, half %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

declare <4 x float> @llvm.amdgcn.image.gather4.2d.v4f32.f16(i32, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.gather4.cube.v4f32.f16(i32, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.gather4.2darray.v4f32.f16(i32, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.gather4.c.2d.v4f32.f32(i32, float, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.gather4.cl.2d.v4f32.f16(i32, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.gather4.c.cl.2d.v4f32.f32(i32, float, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.gather4.b.2d.v4f32.f32.f16(i32, float, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.gather4.c.b.2d.v4f32.f32.f16(i32, float, float, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.gather4.b.cl.2d.v4f32.f32.f16(i32, float, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.gather4.c.b.cl.2d.v4f32.f32.f16(i32, float, float, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.gather4.l.2d.v4f32.f16(i32, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.gather4.c.l.2d.v4f32.f32(i32, float, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.gather4.lz.2d.v4f32.f16(i32, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.gather4.c.lz.2d.v4f32.f32(i32, float, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
attributes #2 = { nounwind readnone }
