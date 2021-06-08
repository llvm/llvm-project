; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s

; GCN-LABEL: {{^}}sample_o_1d:
; GCN: image_sample_o v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_o_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.o.1d.v4f32.f32(i32 15, i32 %offset, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_o_2d:
; GCN: image_sample_o v[0:3], v[0:2], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.o.2d.v4f32.f32(i32 15, i32 %offset, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_o_1d:
; GCN: image_sample_c_o v[0:3], v[0:2], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_c_o_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.o.1d.v4f32.f32(i32 15, i32 %offset, float %zcompare, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_o_2d:
; GCN: image_sample_c_o v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_c_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.o.2d.v4f32.f32(i32 15, i32 %offset, float %zcompare, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_cl_o_1d:
; GCN: image_sample_cl_o v[0:3], v[0:2], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_cl_o_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %s, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.cl.o.1d.v4f32.f32(i32 15, i32 %offset, float %s, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_cl_o_2d:
; GCN: image_sample_cl_o v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_cl_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %s, float %t, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.cl.o.2d.v4f32.f32(i32 15, i32 %offset, float %s, float %t, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_cl_o_1d:
; GCN: image_sample_c_cl_o v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_c_cl_o_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, float %s, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.cl.o.1d.v4f32.f32(i32 15, i32 %offset, float %zcompare, float %s, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_cl_o_2d:
; GCN: image_sample_c_cl_o v[0:3], v[0:4], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_c_cl_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, float %s, float %t, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.cl.o.2d.v4f32.f32(i32 15, i32 %offset, float %zcompare, float %s, float %t, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_b_o_1d:
; GCN: image_sample_b_o v[0:3], v[0:2], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_b_o_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %bias, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.b.o.1d.v4f32.f32.f32(i32 15, i32 %offset, float %bias, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_b_o_2d:
; GCN: image_sample_b_o v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_b_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %bias, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.b.o.2d.v4f32.f32.f32(i32 15, i32 %offset, float %bias, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_b_o_1d:
; GCN: image_sample_c_b_o v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_c_b_o_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %bias, float %zcompare, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.b.o.1d.v4f32.f32.f32(i32 15, i32 %offset, float %bias, float %zcompare, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_b_o_2d:
; GCN: image_sample_c_b_o v[0:3], v[0:4], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_c_b_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %bias, float %zcompare, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.b.o.2d.v4f32.f32.f32(i32 15, i32 %offset, float %bias, float %zcompare, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_b_cl_o_1d:
; GCN: image_sample_b_cl_o v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_b_cl_o_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %bias, float %s, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.b.cl.o.1d.v4f32.f32.f32(i32 15, i32 %offset, float %bias, float %s, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_b_cl_o_2d:
; GCN: image_sample_b_cl_o v[0:3], v[0:4], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_b_cl_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %bias, float %s, float %t, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.b.cl.o.2d.v4f32.f32.f32(i32 15, i32 %offset, float %bias, float %s, float %t, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_b_cl_o_1d:
; GCN: image_sample_c_b_cl_o v[0:3], v[0:4], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_c_b_cl_o_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %bias, float %zcompare, float %s, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.b.cl.o.1d.v4f32.f32.f32(i32 15, i32 %offset, float %bias, float %zcompare, float %s, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_b_cl_o_2d:
; GCN: image_sample_c_b_cl_o v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_c_b_cl_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %bias, float %zcompare, float %s, float %t, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.b.cl.o.2d.v4f32.f32.f32(i32 15, i32 %offset, float %bias, float %zcompare, float %s, float %t, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_d_o_1d:
; GCN: image_sample_d_o v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_d_o_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %dsdh, float %dsdv, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.d.o.1d.v4f32.f32.f32(i32 15, i32 %offset, float %dsdh, float %dsdv, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_d_o_2d:
; GCN: image_sample_d_o v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_d_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.d.o.2d.v4f32.f32.f32(i32 15, i32 %offset, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_d_o_1d:
; GCN: image_sample_c_d_o v[0:3], v[0:4], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_c_d_o_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, float %dsdh, float %dsdv, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.d.o.1d.v4f32.f32.f32(i32 15, i32 %offset, float %zcompare, float %dsdh, float %dsdv, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_d_o_2d:
; GCN: image_sample_c_d_o v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_c_d_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.d.o.2d.v4f32.f32.f32(i32 15, i32 %offset, float %zcompare, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_d_cl_o_1d:
; GCN: image_sample_d_cl_o v[0:3], v[0:4], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_d_cl_o_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %dsdh, float %dsdv, float %s, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.d.cl.o.1d.v4f32.f32.f32(i32 15, i32 %offset, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_d_cl_o_2d:
; GCN: image_sample_d_cl_o v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_d_cl_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.d.cl.o.2d.v4f32.f32.f32(i32 15, i32 %offset, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_d_cl_o_1d:
; GCN: image_sample_c_d_cl_o v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_c_d_cl_o_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, float %dsdh, float %dsdv, float %s, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.d.cl.o.1d.v4f32.f32.f32(i32 15, i32 %offset, float %zcompare, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_d_cl_o_2d:
; GCN: image_sample_c_d_cl_o v[0:3], v[0:15], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_c_d_cl_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.d.cl.o.2d.v4f32.f32.f32(i32 15, i32 %offset, float %zcompare, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_cd_o_1d:
; GCN: image_sample_cd_o v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_cd_o_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %dsdh, float %dsdv, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.cd.o.1d.v4f32.f32.f32(i32 15, i32 %offset, float %dsdh, float %dsdv, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_cd_o_2d:
; GCN: image_sample_cd_o v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_cd_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.cd.o.2d.v4f32.f32.f32(i32 15, i32 %offset, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_cd_o_1d:
; GCN: image_sample_c_cd_o v[0:3], v[0:4], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_c_cd_o_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, float %dsdh, float %dsdv, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.cd.o.1d.v4f32.f32.f32(i32 15, i32 %offset, float %zcompare, float %dsdh, float %dsdv, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_cd_o_2d:
; GCN: image_sample_c_cd_o v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_c_cd_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.cd.o.2d.v4f32.f32.f32(i32 15, i32 %offset, float %zcompare, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_cd_cl_o_1d:
; GCN: image_sample_cd_cl_o v[0:3], v[0:4], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_cd_cl_o_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %dsdh, float %dsdv, float %s, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.cd.cl.o.1d.v4f32.f32.f32(i32 15, i32 %offset, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_cd_cl_o_2d:
; GCN: image_sample_cd_cl_o v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_cd_cl_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.cd.cl.o.2d.v4f32.f32.f32(i32 15, i32 %offset, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_cd_cl_o_1d:
; GCN: image_sample_c_cd_cl_o v[0:3], v[0:7], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_c_cd_cl_o_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, float %dsdh, float %dsdv, float %s, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.cd.cl.o.1d.v4f32.f32.f32(i32 15, i32 %offset, float %zcompare, float %dsdh, float %dsdv, float %s, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_cd_cl_o_2d:
; GCN: image_sample_c_cd_cl_o v[0:3], v[0:15], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_c_cd_cl_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, float %clamp) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.cd.cl.o.2d.v4f32.f32.f32(i32 15, i32 %offset, float %zcompare, float %dsdh, float %dtdh, float %dsdv, float %dtdv, float %s, float %t, float %clamp, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_l_o_1d:
; GCN: image_sample_l_o v[0:3], v[0:2], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_l_o_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %s, float %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.l.o.1d.v4f32.f32(i32 15, i32 %offset, float %s, float %lod, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_l_o_2d:
; GCN: image_sample_l_o v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_l_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %s, float %t, float %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.l.o.2d.v4f32.f32(i32 15, i32 %offset, float %s, float %t, float %lod, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_l_o_1d:
; GCN: image_sample_c_l_o v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_c_l_o_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, float %s, float %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.l.o.1d.v4f32.f32(i32 15, i32 %offset, float %zcompare, float %s, float %lod, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_l_o_2d:
; GCN: image_sample_c_l_o v[0:3], v[0:4], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_c_l_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, float %s, float %t, float %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.l.o.2d.v4f32.f32(i32 15, i32 %offset, float %zcompare, float %s, float %t, float %lod, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_lz_o_1d:
; GCN: image_sample_lz_o v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_lz_o_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.lz.o.1d.v4f32.f32(i32 15, i32 %offset, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_lz_o_2d:
; GCN: image_sample_lz_o v[0:3], v[0:2], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_lz_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.lz.o.2d.v4f32.f32(i32 15, i32 %offset, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_lz_o_1d:
; GCN: image_sample_c_lz_o v[0:3], v[0:2], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_c_lz_o_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.lz.o.1d.v4f32.f32(i32 15, i32 %offset, float %zcompare, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_lz_o_2d:
; GCN: image_sample_c_lz_o v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_c_lz_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.lz.o.2d.v4f32.f32(i32 15, i32 %offset, float %zcompare, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

declare <4 x float> @llvm.amdgcn.image.sample.o.1d.v4f32.f32(i32, i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.o.2d.v4f32.f32(i32, i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.o.1d.v4f32.f32(i32, i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.o.2d.v4f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.cl.o.1d.v4f32.f32(i32, i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.cl.o.2d.v4f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.cl.o.1d.v4f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.cl.o.2d.v4f32.f32(i32, i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.sample.b.o.1d.v4f32.f32.f32(i32, i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.b.o.2d.v4f32.f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.b.o.1d.v4f32.f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.b.o.2d.v4f32.f32.f32(i32, i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.b.cl.o.1d.v4f32.f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.b.cl.o.2d.v4f32.f32.f32(i32, i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.b.cl.o.1d.v4f32.f32.f32(i32, i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.b.cl.o.2d.v4f32.f32.f32(i32, i32, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.sample.d.o.1d.v4f32.f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.d.o.2d.v4f32.f32.f32(i32, i32, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.d.o.1d.v4f32.f32.f32(i32, i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.d.o.2d.v4f32.f32.f32(i32, i32, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.d.cl.o.1d.v4f32.f32.f32(i32, i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.d.cl.o.2d.v4f32.f32.f32(i32, i32, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.d.cl.o.1d.v4f32.f32.f32(i32, i32, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.d.cl.o.2d.v4f32.f32.f32(i32, i32, float, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.sample.cd.o.1d.v4f32.f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.cd.o.2d.v4f32.f32.f32(i32, i32, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.cd.o.1d.v4f32.f32.f32(i32, i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.cd.o.2d.v4f32.f32.f32(i32, i32, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.cd.cl.o.1d.v4f32.f32.f32(i32, i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.cd.cl.o.2d.v4f32.f32.f32(i32, i32, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.cd.cl.o.1d.v4f32.f32.f32(i32, i32, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.cd.cl.o.2d.v4f32.f32.f32(i32, i32, float, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.sample.l.o.1d.v4f32.f32(i32, i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.l.o.2d.v4f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.l.o.1d.v4f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.l.o.2d.v4f32.f32(i32, i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.lz.o.1d.v4f32.f32(i32, i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.lz.o.2d.v4f32.f32(i32, i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.lz.o.1d.v4f32.f32(i32, i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.lz.o.2d.v4f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
attributes #2 = { nounwind readnone }
