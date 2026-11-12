; RUN: llc < %s -mtriple=arm64-eabi | FileCheck %s

define i64 @foo(i32 %a, i32 %b) nounwind readnone ssp {
entry:
; CHECK-LABEL: foo:
; CHECK: add w0, w1, w0
; CHECK: ret
  %add = add i32 %b, %a
  %conv = zext i32 %add to i64
  ret i64 %conv
}
