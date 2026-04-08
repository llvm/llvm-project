; RUN: llc < %s -mcpu=sm_20 -verify-machineinstrs | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mcpu=sm_20 -verify-machineinstrs | %ptxas-verify %}

target triple = "nvptx64-unknown-nvcl"

declare { float, float, float, float } @llvm.nvvm.tex.1d.v4f32.s32(i64, i64, i32)

; CHECK: .entry foo
define ptx_kernel void @foo(i64 %img, i64 %sampler, ptr %red, i32 %idx) {
; CHECK: tex.1d.v4.f32.s32 {%r[[RED:[0-9]+]], %r[[GREEN:[0-9]+]], %r[[BLUE:[0-9]+]], %r[[ALPHA:[0-9]+]]}, [foo_param_0, foo_param_1, {%r{{[0-9]+}}}]
  %val = tail call { float, float, float, float } @llvm.nvvm.tex.1d.v4f32.s32(i64 %img, i64 %sampler, i32 %idx)
  %ret = extractvalue { float, float, float, float } %val, 0
; CHECK: st.b32 [%rd{{[0-9]+}}], %r[[RED]]
  store float %ret, ptr %red
  ret void
}

!nvvm.annotations = !{!2, !3}
!2 = !{ptr @foo, !"rdoimage", i32 0}
!3 = !{ptr @foo, !"sampler", i32 1}
