; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s
; GCN: image_sample v[0:3], v0, s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f16(i32 15, half %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_2d:
; GCN: image_sample v[0:3], v0, s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %s, half %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f16(i32 15, half %s, half %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_3d:
; GCN: image_sample v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_3d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %s, half %t, half %r) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.3d.v4f32.f16(i32 15, half %s, half %t, half %r, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_cube:
; GCN: image_sample v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf a16 da{{$}}
define amdgpu_ps <4 x float> @sample_cube(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %s, half %t, half %face) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.cube.v4f32.f16(i32 15, half %s, half %t, half %face, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_1darray:
; GCN: image_sample v[0:3], v0, s[0:7], s[8:11] dmask:0xf a16 da{{$}}
define amdgpu_ps <4 x float> @sample_1darray(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %s, half %slice) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.1darray.v4f32.f16(i32 15, half %s, half %slice, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_2darray:
; GCN: image_sample v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf a16 da{{$}}
define amdgpu_ps <4 x float> @sample_2darray(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %s, half %t, half %slice) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.2darray.v4f32.f16(i32 15, half %s, half %t, half %slice, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_1d:
; GCN: image_sample_c v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_c_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, half %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.1d.v4f32.f16(i32 15, float %zcompare, half %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_2d:
; GCN: image_sample_c v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_c_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, half %s, half %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.2d.v4f32.f16(i32 15, float %zcompare, half %s, half %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_cl_1d:
; GCN: image_sample_cl v[0:3], v0, s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_cl_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %s, half %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.cl.1d.v4f32.f16(i32 15, half %s, half %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_cl_2d:
; GCN: image_sample_cl v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_cl_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %s, half %t, half %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.cl.2d.v4f32.f16(i32 15, half %s, half %t, half %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_cl_1d:
; GCN: image_sample_c_cl v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_c_cl_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, half %s, half %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.cl.1d.v4f32.f16(i32 15, float %zcompare, half %s, half %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_cl_2d:
; GCN: image_sample_c_cl v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_c_cl_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, half %s, half %t, half %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.cl.2d.v4f32.f16(i32 15, float %zcompare, half %s, half %t, half %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_b_1d:
; GCN: image_sample_b v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_b_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %bias, half %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.b.1d.v4f32.f32.f16(i32 15, float %bias, half %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_b_2d:
; GCN: image_sample_b v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_b_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %bias, half %s, half %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.b.2d.v4f32.f32.f16(i32 15, float %bias, half %s, half %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_b_1d:
; GCN: image_sample_c_b v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_c_b_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %bias, float %zcompare, half %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.b.1d.v4f32.f32.f16(i32 15, float %bias, float %zcompare, half %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_b_2d:
; GCN: image_sample_c_b v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_c_b_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %bias, float %zcompare, half %s, half %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.b.2d.v4f32.f32.f16(i32 15, float %bias, float %zcompare, half %s, half %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_b_cl_1d:
; GCN: image_sample_b_cl v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_b_cl_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %bias, half %s, half %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.b.cl.1d.v4f32.f32.f16(i32 15, float %bias, half %s, half %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_b_cl_2d:
; GCN: image_sample_b_cl v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_b_cl_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %bias, half %s, half %t, half %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.b.cl.2d.v4f32.f32.f16(i32 15, float %bias, half %s, half %t, half %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_b_cl_1d:
; GCN: image_sample_c_b_cl v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_c_b_cl_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %bias, float %zcompare, half %s, half %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.b.cl.1d.v4f32.f32.f16(i32 15, float %bias, float %zcompare, half %s, half %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_b_cl_2d:
; GCN: image_sample_c_b_cl v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_c_b_cl_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %bias, float %zcompare, half %s, half %t, half %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.b.cl.2d.v4f32.f32.f16(i32 15, float %bias, float %zcompare, half %s, half %t, half %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_d_1d:
; GCN: image_sample_d v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_d_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %dsdh, half %dsdv, half %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.d.1d.v4f32.f16.f16(i32 15, half %dsdh, half %dsdv, half %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_d_2d:
; GCN: image_sample_d v[0:3], v[1:4], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_d_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %dsdh, half %dtdh, half %dsdv, half %dtdv, half %s, half %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.d.2d.v4f32.f16.f16(i32 15, half %dsdh, half %dtdh, half %dsdv, half %dtdv, half %s, half %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABAL: {{^}}sample_d_3d:
; GCN: image_sample_d v[0:3], v[2:9], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_d_3d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %dsdh, half %dtdh, half %drdh, half %dsdv, half %dtdv, half %drdv, half %s, half %t, half %r) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.d.3d.v4f32.f16.f16(i32 15, half %dsdh, half %dtdh, half %drdh, half %dsdv, half %dtdv, half %drdv, half %s, half %t, half %r, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_d_1d:
; GCN: image_sample_c_d v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_c_d_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, half %dsdh, half %dsdv, half %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.d.1d.v4f32.f32.f16(i32 15, float %zcompare, half %dsdh, half %dsdv, half %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_d_2d:
; GCN: image_sample_c_d v[0:3], v[1:4], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_c_d_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, half %dsdh, half %dtdh, half %dsdv, half %dtdv, half %s, half %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.d.2d.v4f32.f32.f16(i32 15, float %zcompare, half %dsdh, half %dtdh, half %dsdv, half %dtdv, half %s, half %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_d_cl_1d:
; GCN: image_sample_d_cl v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_d_cl_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %dsdh, half %dsdv, half %s, half %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.d.cl.1d.v4f32.f16.f16(i32 15, half %dsdh, half %dsdv, half %s, half %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_d_cl_2d:
; GCN: image_sample_d_cl v[0:3], v[2:5], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_d_cl_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %dsdh, half %dtdh, half %dsdv, half %dtdv, half %s, half %t, half %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.d.cl.2d.v4f32.f16.f16(i32 15, half %dsdh, half %dtdh, half %dsdv, half %dtdv, half %s, half %t, half %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_d_cl_1d:
; GCN: image_sample_c_d_cl v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_c_d_cl_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, half %dsdh, half %dsdv, half %s, half %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.d.cl.1d.v4f32.f32.f16(i32 15, float %zcompare, half %dsdh, half %dsdv, half %s, half %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_d_cl_2d:
; GCN: image_sample_c_d_cl v[0:3], v[2:9], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_c_d_cl_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, half %dsdh, half %dtdh, half %dsdv, half %dtdv, half %s, half %t, half %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.d.cl.2d.v4f32.f32.f16(i32 15, float %zcompare, half %dsdh, half %dtdh, half %dsdv, half %dtdv, half %s, half %t, half %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_cd_1d:
; GCN: image_sample_cd v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_cd_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %dsdh, half %dsdv, half %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.cd.1d.v4f32.f16.f16(i32 15, half %dsdh, half %dsdv, half %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_cd_2d:
; GCN: image_sample_cd v[0:3], v[1:4], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_cd_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %dsdh, half %dtdh, half %dsdv, half %dtdv, half %s, half %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.cd.2d.v4f32.f16.f16(i32 15, half %dsdh, half %dtdh, half %dsdv, half %dtdv, half %s, half %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_cd_1d:
; GCN: image_sample_c_cd v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_c_cd_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, half %dsdh, half %dsdv, half %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.cd.1d.v4f32.f32.f16(i32 15, float %zcompare, half %dsdh, half %dsdv, half %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_cd_2d:
; GCN: image_sample_c_cd v[0:3], v[1:4], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_c_cd_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, half %dsdh, half %dtdh, half %dsdv, half %dtdv, half %s, half %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.cd.2d.v4f32.f32.f16(i32 15, float %zcompare, half %dsdh, half %dtdh, half %dsdv, half %dtdv, half %s, half %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_cd_cl_1d:
; GCN: image_sample_cd_cl v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_cd_cl_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %dsdh, half %dsdv, half %s, half %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.cd.cl.1d.v4f32.f16.f16(i32 15, half %dsdh, half %dsdv, half %s, half %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_cd_cl_2d:
; GCN: image_sample_cd_cl v[0:3], v[2:5], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_cd_cl_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %dsdh, half %dtdh, half %dsdv, half %dtdv, half %s, half %t, half %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.cd.cl.2d.v4f32.f16.f16(i32 15, half %dsdh, half %dtdh, half %dsdv, half %dtdv, half %s, half %t, half %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_cd_cl_1d:
; GCN: image_sample_c_cd_cl v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_c_cd_cl_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, half %dsdh, half %dsdv, half %s, half %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.cd.cl.1d.v4f32.f32.f16(i32 15, float %zcompare, half %dsdh, half %dsdv, half %s, half %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_cd_cl_2d:
; GCN: image_sample_c_cd_cl v[0:3], v[2:9], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_c_cd_cl_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, half %dsdh, half %dtdh, half %dsdv, half %dtdv, half %s, half %t, half %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.cd.cl.2d.v4f32.f32.f16(i32 15, float %zcompare, half %dsdh, half %dtdh, half %dsdv, half %dtdv, half %s, half %t, half %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_l_1d:
; GCN: image_sample_l v[0:3], v0, s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_l_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %s, half %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.l.1d.v4f32.f16(i32 15, half %s, half %lod, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_l_2d:
; GCN: image_sample_l v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_l_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %s, half %t, half %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.l.2d.v4f32.f16(i32 15, half %s, half %t, half %lod, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_l_1d:
; GCN: image_sample_c_l v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_c_l_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, half %s, half %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.l.1d.v4f32.f16(i32 15, float %zcompare, half %s, half %lod, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_l_2d:
; GCN: image_sample_c_l v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_c_l_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, half %s, half %t, half %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.l.2d.v4f32.f16(i32 15, float %zcompare, half %s, half %t, half %lod, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_lz_1d:
; GCN: image_sample_lz v[0:3], v0, s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_lz_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.lz.1d.v4f32.f16(i32 15, half %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_lz_2d:
; GCN: image_sample_lz v[0:3], v0, s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_lz_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, half %s, half %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.lz.2d.v4f32.f16(i32 15, half %s, half %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_lz_1d:
; GCN: image_sample_c_lz v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_c_lz_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, half %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.lz.1d.v4f32.f16(i32 15, float %zcompare, half %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_lz_2d:
; GCN: image_sample_c_lz v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf a16{{$}}
define amdgpu_ps <4 x float> @sample_c_lz_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, half %s, half %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.lz.2d.v4f32.f16(i32 15, float %zcompare, half %s, half %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_d_o_2darray_V1:
; GCN: image_sample_c_d_o v0, v[2:9], s[0:7], s[8:11] dmask:0x4 a16 da{{$}}
define amdgpu_ps float @sample_c_d_o_2darray_V1(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, half %dsdh, half %dtdh, half %dsdv, half %dtdv, half %s, half %t, half %slice) {
main_body:
  %v = call float @llvm.amdgcn.image.sample.c.d.o.2darray.f32.f16.f16(i32 4, i32 %offset, float %zcompare, half %dsdh, half %dtdh, half %dsdv, half %dtdv, half %s, half %t, half %slice, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret float %v
}

; GCN-LABEL: {{^}}sample_c_d_o_2darray_V2:
; GCN: image_sample_c_d_o v[0:1], v[2:9], s[0:7], s[8:11] dmask:0x6 a16 da{{$}}
define amdgpu_ps <2 x float> @sample_c_d_o_2darray_V2(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, half %dsdh, half %dtdh, half %dsdv, half %dtdv, half %s, half %t, half %slice) {
main_body:
  %v = call <2 x float> @llvm.amdgcn.image.sample.c.d.o.2darray.v2f32.f32.f16(i32 6, i32 %offset, float %zcompare, half %dsdh, half %dtdh, half %dsdv, half %dtdv, half %s, half %t, half %slice, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <2 x float> %v
}

declare <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f16(i32, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <8 x float> @llvm.amdgcn.image.sample.1d.v8f32.f16(i32, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f16(i32, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.3d.v4f32.f16(i32, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.cube.v4f32.f16(i32, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.1darray.v4f32.f16(i32, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.2darray.v4f32.f16(i32, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.sample.c.1d.v4f32.f16(i32, float, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.2d.v4f32.f16(i32, float, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.cl.1d.v4f32.f16(i32, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.cl.2d.v4f32.f16(i32, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.cl.1d.v4f32.f16(i32, float, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.cl.2d.v4f32.f16(i32, float, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.sample.b.1d.v4f32.f32.f16(i32, float, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.b.2d.v4f32.f32.f16(i32, float, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.b.1d.v4f32.f32.f16(i32, float, float, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.b.2d.v4f32.f32.f16(i32, float, float, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.b.cl.1d.v4f32.f32.f16(i32, float, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.b.cl.2d.v4f32.f32.f16(i32, float, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.b.cl.1d.v4f32.f32.f16(i32, float, float, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.b.cl.2d.v4f32.f32.f16(i32, float, float, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.sample.d.1d.v4f32.f16.f16(i32, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.d.2d.v4f32.f16.f16(i32, half, half, half, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.d.3d.v4f32.f16.f16(i32, half, half, half, half, half, half, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.d.1d.v4f32.f32.f16(i32, float, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.d.2d.v4f32.f32.f16(i32, float, half, half, half, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.d.cl.1d.v4f32.f16.f16(i32, half, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.d.cl.2d.v4f32.f16.f16(i32, half, half, half, half, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.d.cl.1d.v4f32.f32.f16(i32, float, half, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.d.cl.2d.v4f32.f32.f16(i32, float, half, half, half, half, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.sample.cd.1d.v4f32.f16.f16(i32, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.cd.2d.v4f32.f16.f16(i32, half, half, half, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.cd.1d.v4f32.f32.f16(i32, float, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.cd.2d.v4f32.f32.f16(i32, float, half, half, half, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.cd.cl.1d.v4f32.f16.f16(i32, half, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.cd.cl.2d.v4f32.f16.f16(i32, half, half, half, half, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.cd.cl.1d.v4f32.f32.f16(i32, float, half, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.cd.cl.2d.v4f32.f32.f16(i32, float, half, half, half, half, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.sample.l.1d.v4f32.f16(i32, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.l.2d.v4f32.f16(i32, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.l.1d.v4f32.f16(i32, float, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.l.2d.v4f32.f16(i32, float, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.sample.lz.1d.v4f32.f16(i32, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.lz.2d.v4f32.f16(i32, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.lz.1d.v4f32.f16(i32, float, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.lz.2d.v4f32.f16(i32, float, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare float @llvm.amdgcn.image.sample.c.d.o.2darray.f32.f16.f16(i32, i32, float, half, half, half, half, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <2 x float> @llvm.amdgcn.image.sample.c.d.o.2darray.v2f32.f32.f16(i32, i32, float, half, half, half, half, half, half, half, <8 x i32>, <4 x i32>, i1, i32, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
attributes #2 = { nounwind readnone }
