; RUN: llc -march=amdgcn -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; Check that write mask is 0xf.

; GCN-LABEL: {{^}}sample_2d_vectorized_use:
; GCN: image_sample v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}], s[{{[0-9:]+}}] dmask:0xf
define amdgpu_ps <4 x float> @sample_2d_vectorized_use(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %t, <4 x float> %a) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32 15, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  %r = fadd <4 x float> %v, %a
  ret <4 x float> %r
}

declare <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32)
