; RUN: llc -mtriple=arm64_32-apple-ios9.0 -o - %s | FileCheck %s

declare void @callee([8 x i64], ptr, ptr)

; Make sure we don't accidentally store X0 or XZR, which might well
; clobber other arguments or data.
define void @test_stack_ptr_32bits(ptr %in) {
; CHECK-LABEL: test_stack_ptr_32bits:
; CHECK-DAG: stp wzr, w0, [sp]

  call void @callee([8 x i64] undef, ptr null, ptr %in)
  ret void
}
