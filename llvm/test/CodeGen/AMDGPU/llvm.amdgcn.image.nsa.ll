; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=-nsa-encoding -verify-machineinstrs -show-mc-encoding < %s | FileCheck -check-prefixes=GCN,NONSA %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs -show-mc-encoding < %s | FileCheck -check-prefixes=GCN,NSA %s

; GCN-LABEL: {{^}}sample_2d:
;
; TODO: use NSA here
; GCN: v_mov_b32_e32 v2, v0
;
; GCN: image_sample v[0:3], v[1:2],
define amdgpu_ps <4 x float> @sample_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %t, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32 15, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_3d:
; NONSA: v_mov_b32_e32 v3, v0
; NONSA: image_sample v[0:3], v[1:4],
; NSA: image_sample v[0:3], [v1, v2, v0],
define amdgpu_ps <4 x float> @sample_3d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %r, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.3d.v4f32.f32(i32 15, float %s, float %t, float %r, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_d_3d:
; NSA: image_sample_d v[0:3], [v3, v8, v7, v5, v4, v6, v0, v2, v1],
define amdgpu_ps <4 x float> @sample_d_3d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %r, float %t, float %dsdh, float %dtdv, float %dsdv, float %drdv, float %drdh, float %dtdh) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.d.3d.v4f32.f32(i32 15, float %dsdh, float %dtdh, float %drdh, float %dsdv, float %dtdv, float %drdv, float %s, float %t, float %r, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_contig_nsa:
; GCN: image_sample_c_l v0, v[0:7],
; NSA: image_sample v1, [v6, v7, v5],
define amdgpu_ps <2 x float> @sample_contig_nsa(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %s1, float %t1, float %r1, float %lod, float %r2, float %s2, float %t2) {
main_body:
  %v1 = call float @llvm.amdgcn.image.sample.c.l.3d.f32.f32(i32 1, float %zcompare, float %s1, float %t1, float %r1, float %lod, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %v2 = call float @llvm.amdgcn.image.sample.3d.f32.f32(i32 1, float %s2, float %t2, float %r2, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %r.0 = insertelement <2 x float> undef, float %v1, i32 0
  %r = insertelement <2 x float> %r.0, float %v2, i32 1
  ret <2 x float> %r
}

; GCN-LABEL: {{^}}sample_nsa_nsa:
; NSA: image_sample_c_l v0, [v1, v2, v3, v4, v0],
; NSA: image_sample v1, [v6, v7, v5],
define amdgpu_ps <2 x float> @sample_nsa_nsa(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %lod, float %zcompare, float %s1, float %t1, float %r1, float %r2, float %s2, float %t2) {
main_body:
  %v1 = call float @llvm.amdgcn.image.sample.c.l.3d.f32.f32(i32 1, float %zcompare, float %s1, float %t1, float %r1, float %lod, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %v2 = call float @llvm.amdgcn.image.sample.3d.f32.f32(i32 1, float %s2, float %t2, float %r2, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %r.0 = insertelement <2 x float> undef, float %v1, i32 0
  %r = insertelement <2 x float> %r.0, float %v2, i32 1
  ret <2 x float> %r
}

; GCN-LABEL: {{^}}sample_nsa_contig:
; NSA: image_sample_c_l v0, [v1, v2, v3, v4, v0],
; NSA: image_sample v1, v[5:7],
define amdgpu_ps <2 x float> @sample_nsa_contig(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %lod, float %zcompare, float %s1, float %t1, float %r1, float %s2, float %t2, float %r2) {
main_body:
  %v1 = call float @llvm.amdgcn.image.sample.c.l.3d.f32.f32(i32 1, float %zcompare, float %s1, float %t1, float %r1, float %lod, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %v2 = call float @llvm.amdgcn.image.sample.3d.f32.f32(i32 1, float %s2, float %t2, float %r2, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %r.0 = insertelement <2 x float> undef, float %v1, i32 0
  %r = insertelement <2 x float> %r.0, float %v2, i32 1
  ret <2 x float> %r
}

; GCN-LABEL: {{^}}sample_contig_contig:
; GCN: image_sample_c_l v0, v[0:7],
; NSA: image_sample v1, v[5:7],
; NONSA: image_sample v1, v[5:8],
define amdgpu_ps <2 x float> @sample_contig_contig(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %s1, float %t1, float %r1, float %lod, float %s2, float %t2, float %r2) {
main_body:
  %v1 = call float @llvm.amdgcn.image.sample.c.l.3d.f32.f32(i32 1, float %zcompare, float %s1, float %t1, float %r1, float %lod, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %v2 = call float @llvm.amdgcn.image.sample.3d.f32.f32(i32 1, float %s2, float %t2, float %r2, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %r.0 = insertelement <2 x float> undef, float %v1, i32 0
  %r = insertelement <2 x float> %r.0, float %v2, i32 1
  ret <2 x float> %r
}


declare <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.3d.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.d.3d.v4f32.f32(i32, float, float, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare float @llvm.amdgcn.image.sample.3d.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare float @llvm.amdgcn.image.sample.c.l.3d.f32.f32(i32, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

attributes #1 = { nounwind readonly }
