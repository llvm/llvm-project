; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s


; GCN-LABEL: {{^}}sample_l_1d:
; GCN: image_sample_lz v[0:3], v0, s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_l_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.l.1d.v4f32.f32(i32 15, float %s, float 0.0, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_l_2d:
; GCN: image_sample_lz v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_l_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %t, float %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.l.2d.v4f32.f32(i32 15, float %s, float %t, float -0.0, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_l_1d:
; GCN: image_sample_c_lz v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_c_l_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %s, float %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.l.1d.v4f32.f32(i32 15, float %zcompare, float %s, float -2.0, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_l_2d:
; GCN: image_sample_c_lz v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_c_l_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %s, float %t, float %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.l.2d.v4f32.f32(i32 15, float %zcompare, float %s, float %t, float 0.0, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_l_o_1d:
; GCN: image_sample_lz_o v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_l_o_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %s, float %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.l.o.1d.v4f32.f32(i32 15, i32 %offset, float %s, float 0.0, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_l_o_2d:
; GCN: image_sample_lz_o v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_l_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %s, float %t, float %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.l.o.2d.v4f32.f32(i32 15, i32 %offset, float %s, float %t, float 0.0, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_l_o_1d:
; GCN: image_sample_c_lz_o v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_c_l_o_1d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, float %s, float %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.l.o.1d.v4f32.f32(i32 15, i32 %offset, float %zcompare, float %s, float 0.0, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_c_l_o_2d:
; GCN: image_sample_c_lz_o v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @sample_c_l_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, float %s, float %t, float %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.c.l.o.2d.v4f32.f32(i32 15, i32 %offset, float %zcompare, float %s, float %t, float 0.0, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_l_2d:
; GCN: image_gather4_lz v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @gather4_l_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %t, float %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.l.2d.v4f32.f32(i32 15, float %s, float %t, float 0.0, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_c_l_2d:
; GCN: image_gather4_c_lz v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @gather4_c_l_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %s, float %t, float %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.c.l.2d.v4f32.f32(i32 15, float %zcompare, float %s, float %t, float 0.0, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_l_o_2d:
; GCN: image_gather4_lz_o v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @gather4_l_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %s, float %t, float %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.l.o.2d.v4f32.f32(i32 15, i32 %offset, float %s, float %t, float 0.0, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}gather4_c_l_o_2d:
; GCN: image_gather4_c_lz_o v[0:3], v[0:3], s[0:7], s[8:11] dmask:0xf{{$}}
define amdgpu_ps <4 x float> @gather4_c_l_o_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, i32 %offset, float %zcompare, float %s, float %t, float %lod) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.c.l.o.2d.v4f32.f32(i32 15, i32 %offset, float %zcompare, float %s, float %t, float 0.0, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

declare <4 x float> @llvm.amdgcn.image.sample.l.1d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.l.2d.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.l.1d.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.l.2d.v4f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.l.o.1d.v4f32.f32(i32, i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.l.o.2d.v4f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.l.o.1d.v4f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.c.l.o.2d.v4f32.f32(i32, i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare <4 x float> @llvm.amdgcn.image.gather4.l.2d.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.gather4.c.l.2d.v4f32.f32(i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.gather4.l.o.2d.v4f32.f32(i32, i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.gather4.c.l.o.2d.v4f32.f32(i32, i32, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
