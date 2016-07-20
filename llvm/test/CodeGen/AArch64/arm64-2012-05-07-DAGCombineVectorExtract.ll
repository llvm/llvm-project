; RUN: llc < %s -mtriple=arm64-eabi | FileCheck %s

define i32 @foo(<4 x i32> %a, i32 %n) nounwind {
; CHECK-LABEL: foo:
; CHECK: fmov w0, s0
; CHECK-NEXT: ret
  %b = bitcast <4 x i32> %a to i128
  %c = trunc i128 %b to i32
  ret i32 %c
}

define i64 @bar(<2 x i64> %a, i64 %n) nounwind {
; CHECK-LABEL: bar:
; CHECK: fmov x0, d0
; CHECK-NEXT: ret
  %b = bitcast <2 x i64> %a to i128
  %c = trunc i128 %b to i64
  ret i64 %c
}

