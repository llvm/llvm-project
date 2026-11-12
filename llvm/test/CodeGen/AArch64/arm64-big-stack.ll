; RUN: llc < %s | FileCheck %s
target triple = "arm64-apple-macosx10"

; Check that big stacks are generated correctly.
; Currently, this is done by a sequence of sub instructions,
; which can encode immediate with a 12 bits mask an optionally
; shift left (up to 12). I.e., 16773120 is the biggest value.
; <rdar://12513931>
; CHECK-LABEL: foo:
; CHECK: sub sp, sp, #4095, lsl #12
; CHECK: sub sp, sp, #4095, lsl #12
; CHECK: sub sp, sp, #2, lsl #12
define void @foo() nounwind ssp {
entry:
  %buffer = alloca [33554432 x i8], align 1
  call void @doit(ptr %buffer) nounwind
  ret void
}

declare void @doit(ptr)
