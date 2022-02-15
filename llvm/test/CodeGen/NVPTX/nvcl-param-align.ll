; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s

target triple = "nvptx-unknown-nvcl"

define void @foo(i64 %img, i64 %sampler, <5 x float>* align 32 %v1, i32* %v2) {
; The parameter alignment is determined by the align attribute (default 1).
; CHECK-LABEL: .entry foo(
; CHECK: .param .u32 .ptr .align 32 foo_param_2
; CHECK: .param .u32 .ptr .align 1 foo_param_3
  ret void
}

!nvvm.annotations = !{!1, !2, !3}
!1 = !{void (i64, i64, <5 x float>*, i32*)* @foo, !"kernel", i32 1}
!2 = !{void (i64, i64, <5 x float>*, i32*)* @foo, !"rdoimage", i32 0}
!3 = !{void (i64, i64, <5 x float>*, i32*)* @foo, !"sampler", i32 1}
