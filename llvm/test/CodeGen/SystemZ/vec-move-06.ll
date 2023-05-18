; Test vector builds using VLVGP.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test the basic v2i64 usage.
define <2 x i64> @f1(i64 %a, i64 %b) {
; CHECK-LABEL: f1:
; CHECK: vlvgp %v24, %r2, %r3
; CHECK: br %r14
  %veca = insertelement <2 x i64> undef, i64 %a, i32 0
  %vecb = insertelement <2 x i64> %veca, i64 %b, i32 1
  ret <2 x i64> %vecb
}
