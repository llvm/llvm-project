; RUN: opt -mtriple=amdgcn--amdpal -S -instcombine <%s | FileCheck --check-prefixes=GCN %s

; Check that sin/cos is not folded to tan on amdgcn.

; GCN-LABEL: define amdgpu_ps float @llpc.shader.FS.main
; GCN: call float @llvm.sin.f32
; GCN: call float @llvm.cos.f32

declare float @llvm.sin.f32(float) #0
declare float @llvm.cos.f32(float) #0

define amdgpu_ps float @llpc.shader.FS.main(float %arg) {
.entry:
  %tmp32 = call float @llvm.sin.f32(float %arg)
  %tmp33 = call float @llvm.cos.f32(float %arg)
  %tmp34 = fdiv reassoc nnan nsz arcp contract float 1.000000e+00, %tmp33
  %tmp35 = fmul reassoc nnan nsz arcp contract float %tmp32, %tmp34
  ret float %tmp35
}

attributes #0 = { nounwind readnone speculatable willreturn }
