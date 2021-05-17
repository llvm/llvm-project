; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=-xnack -verify-machineinstrs -enable-misched=0 < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}sample_contig_nsa:
; GCN-DAG: image_sample_c_l v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}],
; GCN-DAG: image_sample v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}],
define amdgpu_ps <2 x float> @sample_contig_nsa(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %s1, float %t1, float %r1, float %lod, float %r2, float %s2, float %t2) {
main_body:
  %zcompare.1 = fadd float %zcompare, 1.0
  %s1.1 = fadd float %s1, 1.0
  %t1.1 = fadd float %t1, 1.0
  %r1.1 = fadd float %r1, 1.0
  %s2.1 = fadd float %s2, 1.0
  %t2.1 = fadd float %t2, 1.0
  %r2.1 = fadd float %r2, 1.0
  %lod.1 = fadd float %lod, 1.0
  %v1 = call float @llvm.amdgcn.image.sample.c.l.3d.f32.f32(i32 1, float %zcompare.1, float %s1.1, float %t1.1, float %r1.1, float %lod.1, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %v2 = call float @llvm.amdgcn.image.sample.3d.f32.f32(i32 1, float %s2.1, float %t2.1, float %r2.1, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %r.0 = insertelement <2 x float> undef, float %v1, i32 0
  %r = insertelement <2 x float> %r.0, float %v2, i32 1
  ret <2 x float> %r
}

; GCN-LABEL: {{^}}sample_contig_nsa_10vgprs:
; GCN-DAG: image_sample_c_l v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}],
; GCN-DAG: image_sample v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}],
define amdgpu_ps <2 x float> @sample_contig_nsa_10vgprs(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %s1, float %t1, float %r1, float %lod, float %r2, float %s2, float %t2) #0 {
main_body:
  %zcompare.1 = fadd float %zcompare, 1.0
  %s1.1 = fadd float %s1, 1.0
  %t1.1 = fadd float %t1, 1.0
  %r1.1 = fadd float %r1, 1.0
  %s2.1 = fadd float %s2, 1.0
  %t2.1 = fadd float %t2, 1.0
  %r2.1 = fadd float %r2, 1.0
  %lod.1 = fadd float %lod, 1.0
  %v1 = call float @llvm.amdgcn.image.sample.c.l.3d.f32.f32(i32 1, float %zcompare.1, float %s1.1, float %t1.1, float %r1.1, float %lod.1, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %v2 = call float @llvm.amdgcn.image.sample.3d.f32.f32(i32 1, float %s2.1, float %t2.1, float %r2.1, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %r.0 = insertelement <2 x float> undef, float %v1, i32 0
  %r = insertelement <2 x float> %r.0, float %v2, i32 1
  ret <2 x float> %r
}

; GCN-LABEL: {{^}}sample_contig_nsa_conflict:
; GCN-DAG: image_sample v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}],
; GCN-DAG: image_sample v{{[0-9]+}}, [{{v[0-9]+, v[0-9]+, v[0-9]+}}],
define amdgpu_ps <2 x float> @sample_contig_nsa_conflict(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %s1, float %t1, float %r1, float %lod, float %r2, float %s2, float %t2) {
main_body:
  %zcompare.1 = fadd float %zcompare, 1.0
  %s1.1 = fadd float %s1, 1.0
  %t1.1 = fadd float %t1, 1.0
  %r1.1 = fadd float %r1, 1.0
  %s2.1 = fadd float %s2, 1.0
  %t2.1 = fadd float %t2, 1.0
  %r2.1 = fadd float %r2, 1.0
  %lod.1 = fadd float %lod, 1.0
  %v2 = call float @llvm.amdgcn.image.sample.3d.f32.f32(i32 1, float %s2.1, float %t2.1, float %r2.1, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %v1 = call float @llvm.amdgcn.image.sample.3d.f32.f32(i32 1, float %t2.1, float %s2.1, float %r2.1, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %r.0 = insertelement <2 x float> undef, float %v1, i32 0
  %r = insertelement <2 x float> %r.0, float %v2, i32 1
  ret <2 x float> %r
}

; GCN-LABEL: {{^}}sample_contig_nsa_same_addr:
; GCN-DAG: image_sample v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}],
; GCN-DAG: image_sample v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}],
define amdgpu_ps <2 x float> @sample_contig_nsa_same_addr(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %s1, float %t1, float %r1, float %lod, float %r2, float %s2, float %t2) {
main_body:
  %zcompare.1 = fadd float %zcompare, 1.0
  %s1.1 = fadd float %s1, 1.0
  %t1.1 = fadd float %t1, 1.0
  %r1.1 = fadd float %r1, 1.0
  %s2.1 = fadd float %s2, 1.0
  %t2.1 = fadd float %t2, 1.0
  %r2.1 = fadd float %r2, 1.0
  %lod.1 = fadd float %lod, 1.0
  %v2 = call float @llvm.amdgcn.image.sample.3d.f32.f32(i32 1, float %s2.1, float %t2.1, float %r2.1, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 1)
  %v1 = call float @llvm.amdgcn.image.sample.3d.f32.f32(i32 1, float %s2.1, float %t2.1, float %r2.1, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %r.0 = insertelement <2 x float> undef, float %v1, i32 0
  %r = insertelement <2 x float> %r.0, float %v2, i32 1
  ret <2 x float> %r
}

; GCN-LABEL: {{^}}sample_contig_nsa_same_reg:
; GCN-DAG: image_sample v{{[0-9]+}}, [{{v[0-9]+, v[0-9]+, v[0-9]+}}],
define amdgpu_ps float @sample_contig_nsa_same_reg(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %zcompare, float %s1, float %t1, float %r1, float %lod, float %r2, float %s2, float %t2) {
main_body:
  %zcompare.1 = fadd float %zcompare, 1.0
  %s1.1 = fadd float %s1, 1.0
  %t1.1 = fadd float %t1, 1.0
  %r1.1 = fadd float %r1, 1.0
  %s2.1 = fadd float %s2, 1.0
  %t2.1 = fadd float %t2, 1.0
  %r2.1 = fadd float %r2, 1.0
  %lod.1 = fadd float %lod, 1.0
  %v = call float @llvm.amdgcn.image.sample.3d.f32.f32(i32 1, float %t2.1, float %t2.1, float %r2.1, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret float %v
}

declare float @llvm.amdgcn.image.sample.3d.f32.f32(i32, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32)
declare float @llvm.amdgcn.image.sample.c.l.3d.f32.f32(i32, float, float, float, float, float, <8 x i32>, <4 x i32>, i1, i32, i32)

attributes #0 = {"amdgpu-num-vgpr"="10"}
