; RUN: llc < %s -mtriple=nvptx64-nvidia-nvcl -mcpu=sm_60 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64-nvidia-nvcl -mcpu=sm_60 | %ptxas-verify %}

target triple = "nvptx-unknown-nvcl"

define ptx_kernel void @foo(i64 %img, i64 %sampler, ptr align 32 %v1, ptr %v2) {
; The parameter alignment is determined by the align attribute (default 1).
; CHECK-LABEL: .entry foo(
; CHECK: .param .u64 .ptr .align 32 foo_param_2
; CHECK: .param .u64 .ptr .align 1 foo_param_3
  ret void
}

!nvvm.annotations = !{!2, !3}
!2 = !{ptr @foo, !"rdoimage", i32 0}
!3 = !{ptr @foo, !"sampler", i32 1}
