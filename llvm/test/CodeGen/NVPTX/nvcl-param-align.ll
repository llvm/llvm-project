; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

target triple = "nvptx-unknown-nvcl"

define void @foo(i64 %img, i64 %sampler, ptr align 32 %v1, ptr %v2) {
; The parameter alignment is determined by the align attribute (default 1).
; CHECK-LABEL: .entry foo(
; CHECK: .param .u64 .ptr .align 32 foo_param_2
; CHECK: .param .u64 .ptr .align 1 foo_param_3
  ret void
}

!nvvm.annotations = !{!1, !2, !3}
!1 = !{ptr @foo, !"kernel", i32 1}
!2 = !{ptr @foo, !"rdoimage", i32 0}
!3 = !{ptr @foo, !"sampler", i32 1}
