; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

target triple = "nvptx-nvidia-cuda"

; Function Attrs: nounwind
; CHECK: .entry foo
define ptx_kernel void @foo(ptr nocapture %a) #0 {
  %val = load float, ptr %a
  %tan = tail call fastcc float @__nv_fast_tanf(float %val)
  store float %tan, ptr %a
  ret void
}

; Function Attrs: nounwind readnone
declare float @llvm.nvvm.sin.approx.ftz.f(float) #1

; Function Attrs: nounwind readnone
declare float @llvm.nvvm.cos.approx.ftz.f(float) #1

; Function Attrs: nounwind readnone
declare float @llvm.nvvm.div.approx.ftz.f(float, float) #1

; Function Attrs: alwaysinline inlinehint nounwind readnone
; CHECK: .func (.param .b32 func_retval0) __nv_fast_tanf
define internal fastcc float @__nv_fast_tanf(float %a) #2 {
entry:
  %0 = tail call float @llvm.nvvm.sin.approx.ftz.f(float %a)
  %1 = tail call float @llvm.nvvm.cos.approx.ftz.f(float %a)
  %2 = tail call float @llvm.nvvm.div.approx.ftz.f(float %0, float %1)
  ret float %2
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { alwaysinline inlinehint nounwind readnone }
