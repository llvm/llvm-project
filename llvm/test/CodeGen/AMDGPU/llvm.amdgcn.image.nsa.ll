; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=-nsa-encoding -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,NONSA,GFX10-NONSA %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -amdgpu-nsa-threshold=32 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,NONSA,GFX10-NONSA %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -amdgpu-nsa-threshold=2 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,NSA,NSA-T2 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,NSA,NSA-T3,GFX1010-NSA %s
; RUN: llc -march=amdgcn -mcpu=gfx1030 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,NSA,NSA-T3,GFX1030-NSA %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -mattr=-nsa-encoding -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,NONSA,GFX11-NONSA %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -amdgpu-nsa-threshold=32 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,NONSA,GFX11-NONSA %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -amdgpu-nsa-threshold=2 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,NSA,NSA-T2 %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,NSA,NSA-T3,GFX11-NSA %s

; Default NSA threshold is 3 addresses
; GCN-LABEL: {{^}}sample_2d:
; NONSA: v_mov_b32_e32 v2, v0
; NONSA: image_sample v[0:3], v[1:2],
; NSA-T2: image_sample v[0:3], [v1, v0],
; NSA-T3: v_mov_b32_e32 v2, v0
; NSA-T3: image_sample v[0:3], v[1:2],
define amdgpu_ps <4 x float> @sample_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %t, float %s) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32 15, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_3d:
; NONSA: v_mov_b32_e32 v3, v0
; NONSA: image_sample v[0:3], v[1:3],
; NSA: image_sample v[0:3], [v1, v2, v0],
define amdgpu_ps <4 x float> @sample_3d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %r, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.3d.v4f32.f32(i32 15, float %s, float %t, float %r, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_d_3d:
; GFX1010-NSA: image_sample_d v[0:3], v[7:15],
; GFX1030-NSA: image_sample_d v[0:3], [v3, v8, v7, v5, v4, v6, v0, v2, v1],
; GFX11-NSA: image_sample_d v[0:3], v[7:15],
define amdgpu_ps <4 x float> @sample_d_3d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %r, float %t, float %dsdh, float %dtdv, float %dsdv, float %drdv, float %drdh, float %dtdh) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.d.3d.v4f32.f32(i32 15, float %dsdh, float %dtdh, float %drdh, float %dsdv, float %dtdv, float %drdv, float %s, float %t, float %r, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}

; GCN-LABEL: {{^}}sample_contig_nsa:
; GFX10-NONSA: image_sample_c_l v5, v[0:4],
; GFX11-NONSA: image_sample_c_l v0, v[0:4],
; GFX1010-NSA: image_sample_c_l v8, v[0:4],
; GFX1010-NSA: image_sample v9, [v6, v7, v5],
; GFX1030-NSA: image_sample_c_l v0, v[0:4],
; GFX1030-NSA: image_sample v1, [v6, v7, v5],
; GFX11-NSA: image_sample_c_l v0, v[0:4],
; GFX11-NSA: image_sample v1, [v6, v7, v5],
define amdgpu_ps <2 x float> @sample_contig_nsa(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %s1, float %t1, float %r1, float %lod, float %r2, float %s2, float %t2) {
main_body:
  %v1 = call float @llvm.amdgcn.image.sample.c.l.3d.f32.f32(i32 1, float %zcompare, float %s1, float %t1, float %r1, float %lod, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %v2 = call float @llvm.amdgcn.image.sample.3d.f32.f32(i32 1, float %s2, float %t2, float %r2, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %r.0 = insertelement <2 x float> undef, float %v1, i32 0
  %r = insertelement <2 x float> %r.0, float %v2, i32 1
  ret <2 x float> %r
}

; GCN-LABEL: {{^}}sample_nsa_nsa:
; GFX1010-NSA: image_sample_c_l v8, [v1, v2, v3, v4, v0],
; GFX1010-NSA: image_sample v9, [v6, v7, v5],
; GFX1030-NSA: image_sample_c_l v0, [v1, v2, v3, v4, v0],
; GFX1030-NSA: image_sample v1, [v6, v7, v5],
; GFX11-NSA: image_sample_c_l v0, [v1, v2, v3, v4, v0],
; GFX11-NSA: image_sample v1, [v6, v7, v5],
define amdgpu_ps <2 x float> @sample_nsa_nsa(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %lod, float %zcompare, float %s1, float %t1, float %r1, float %r2, float %s2, float %t2) {
main_body:
  %v1 = call float @llvm.amdgcn.image.sample.c.l.3d.f32.f32(i32 1, float %zcompare, float %s1, float %t1, float %r1, float %lod, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %v2 = call float @llvm.amdgcn.image.sample.3d.f32.f32(i32 1, float %s2, float %t2, float %r2, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %r.0 = insertelement <2 x float> undef, float %v1, i32 0
  %r = insertelement <2 x float> %r.0, float %v2, i32 1
  ret <2 x float> %r
}

; GCN-LABEL: {{^}}sample_nsa_contig:
; GFX1010-NSA: image_sample_c_l v8, [v1, v2, v3, v4, v0],
; GFX1010-NSA: image_sample v9, v[5:7],
; GFX1030-NSA: image_sample_c_l v0, [v1, v2, v3, v4, v0],
; GFX1030-NSA: image_sample v1, v[5:7],
; GFX11-NSA: image_sample_c_l v0, [v1, v2, v3, v4, v0],
; GFX11-NSA: image_sample v1, v[5:7],
define amdgpu_ps <2 x float> @sample_nsa_contig(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %lod, float %zcompare, float %s1, float %t1, float %r1, float %s2, float %t2, float %r2) {
main_body:
  %v1 = call float @llvm.amdgcn.image.sample.c.l.3d.f32.f32(i32 1, float %zcompare, float %s1, float %t1, float %r1, float %lod, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %v2 = call float @llvm.amdgcn.image.sample.3d.f32.f32(i32 1, float %s2, float %t2, float %r2, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %r.0 = insertelement <2 x float> undef, float %v1, i32 0
  %r = insertelement <2 x float> %r.0, float %v2, i32 1
  ret <2 x float> %r
}

; GCN-LABEL: {{^}}sample_contig_contig:
; GFX1010-NSA: image_sample_c_l v8, v[0:4],
; GFX1010-NSA: image_sample v9, v[5:7],
; GFX1030-NSA: image_sample_c_l v0, v[0:4],
; GFX1030-NSA: image_sample v1, v[5:7],
; GFX11-NSA: image_sample_c_l v0, v[0:4],
; GFX11-NSA: image_sample v1, v[5:7],
; GFX10-NONSA: image_sample_c_l v8, v[0:4],
; GFX10-NONSA: image_sample v9, v[5:7],
; GFX11-NONSA: image_sample_c_l v0, v[0:4],
; GFX11-NONSA: image_sample v1, v[5:7],
define amdgpu_ps <2 x float> @sample_contig_contig(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %s1, float %t1, float %r1, float %lod, float %s2, float %t2, float %r2) {
main_body:
  %v1 = call float @llvm.amdgcn.image.sample.c.l.3d.f32.f32(i32 1, float %zcompare, float %s1, float %t1, float %r1, float %lod, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %v2 = call float @llvm.amdgcn.image.sample.3d.f32.f32(i32 1, float %s2, float %t2, float %r2, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %r.0 = insertelement <2 x float> undef, float %v1, i32 0
  %r = insertelement <2 x float> %r.0, float %v2, i32 1
  ret <2 x float> %r
}

; Test that undef inputs with NSA are handled safely; these tests used to crash.

; GCN-LABEL: {{^}}sample_undef_undef_undef_undef:
; GCN: image_sample_c_b v0, v[0:3], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_1D_ARRAY
define amdgpu_ps float @sample_undef_undef_undef_undef(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp) {
  %r = call float @llvm.amdgcn.image.sample.c.b.1darray.f32.f32.f32(i32 1, float undef, float undef, float undef, float undef, <8 x i32> %rsrc, <4 x i32> %samp, i1 false, i32 0, i32 0)
  ret float %r
}

; GCN-LABEL: {{^}}sample_undef_undef_undef_def:
; NONSA: v_mov_b32_e32 v3, v0
; NONSA: image_sample_c_b v0, v[0:3], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_1D_ARRAY
; NSA: image_sample_c_b v0, [v0, v0, v0, v0], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_1D_ARRAY
define amdgpu_ps float @sample_undef_undef_undef_def(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %layer) {
  %r = call float @llvm.amdgcn.image.sample.c.b.1darray.f32.f32.f32(i32 1, float undef, float undef, float undef, float %layer, <8 x i32> %rsrc, <4 x i32> %samp, i1 false, i32 0, i32 0)
  ret float %r
}

; GCN-LABEL: {{^}}sample_undef_undef_undef_def_rnd:
; GCN: v_rndne_f32_e32 v3, v0
; GCN: image_sample_c_b v0, v[0:3], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_1D_ARRAY
define amdgpu_ps float @sample_undef_undef_undef_def_rnd(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %layer) {
  %layer_rnd = call float @llvm.rint.f32(float %layer)
  %r = call float @llvm.amdgcn.image.sample.c.b.1darray.f32.f32.f32(i32 1, float undef, float undef, float undef, float %layer_rnd, <8 x i32> %rsrc, <4 x i32> %samp, i1 false, i32 0, i32 0)
  ret float %r
}

; GCN-LABEL: {{^}}sample_def_undef_undef_undef:
; GCN: v_add_f32_e32 v0, 1.0, v0
; GCN: image_sample_c_b v0, v[0:3], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_1D_ARRAY
define amdgpu_ps float @sample_def_undef_undef_undef(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %z0) {
  ; The NSA reassign pass is conservative (quite reasonably!) when one of the operands
  ; comes directly from a function argument (via COPY). To test that NSA can be
  ; eliminated in the presence of undef, just add an arbitrary intermediate
  ; computation.
  %c0 = fadd float %z0, 1.0
  %r = call float @llvm.amdgcn.image.sample.c.b.1darray.f32.f32.f32(i32 1, float %c0, float undef, float undef, float undef, <8 x i32> %rsrc, <4 x i32> %samp, i1 false, i32 0, i32 0)
  ret float %r
}

declare <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.3d.v4f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare <4 x float> @llvm.amdgcn.image.sample.d.3d.v4f32.f32(i32, float, float, float, float, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare float @llvm.amdgcn.image.sample.3d.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1
declare float @llvm.amdgcn.image.sample.c.l.3d.f32.f32(i32, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

declare float @llvm.rint.f32(float) #2
declare float @llvm.amdgcn.image.sample.c.b.1darray.f32.f32.f32(i32 immarg, float, float, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg) #1

attributes #1 = { nounwind readonly }
attributes #2 = { nounwind readnone speculatable willreturn }
