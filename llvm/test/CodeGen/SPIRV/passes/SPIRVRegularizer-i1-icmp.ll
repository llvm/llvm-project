; RUN: opt -S -passes=spirv-regularizer -mtriple=spirv64-unknown-unknown < %s | FileCheck %s

; Verify that i1 ICMP comparisons are lowered to logical operations,
; matching the legacy pass behavior in runLowerI1Comparisons.

define i1 @ult_i1(i1 %p, i1 %q) {
; CHECK-LABEL: define i1 @ult_i1(
; CHECK-SAME: i1 [[P:%.*]], i1 [[Q:%.*]]) {
; CHECK:    [[NOT:%.*]] = xor i1 [[P]], true
; CHECK:    [[R:%.*]] = and i1 [[Q]], [[NOT]]
; CHECK:    ret i1 [[R]]
  %r = icmp ult i1 %p, %q
  ret i1 %r
}

define i1 @ugt_i1(i1 %p, i1 %q) {
; CHECK-LABEL: define i1 @ugt_i1(
; CHECK-SAME: i1 [[P:%.*]], i1 [[Q:%.*]]) {
; CHECK:    [[NOT:%.*]] = xor i1 [[Q]], true
; CHECK:    [[R:%.*]] = and i1 [[P]], [[NOT]]
; CHECK:    ret i1 [[R]]
  %r = icmp ugt i1 %p, %q
  ret i1 %r
}

define <4 x i1> @ult_v4i1(<4 x i1> %p, <4 x i1> %q) {
; CHECK-LABEL: define <4 x i1> @ult_v4i1(
; CHECK-SAME: <4 x i1> [[P:%.*]], <4 x i1> [[Q:%.*]]) {
; CHECK:    [[NOT:%.*]] = xor <4 x i1> [[P]], splat (i1 true)
; CHECK:    [[R:%.*]] = and <4 x i1> [[Q]], [[NOT]]
; CHECK:    ret <4 x i1> [[R]]
  %r = icmp ult <4 x i1> %p, %q
  ret <4 x i1> %r
}

define <4 x i1> @ugt_v4i1(<4 x i1> %p, <4 x i1> %q) {
; CHECK-LABEL: define <4 x i1> @ugt_v4i1(
; CHECK-SAME: <4 x i1> [[P:%.*]], <4 x i1> [[Q:%.*]]) {
; CHECK:    [[NOT:%.*]] = xor <4 x i1> [[Q]], splat (i1 true)
; CHECK:    [[R:%.*]] = and <4 x i1> [[P]], [[NOT]]
; CHECK:    ret <4 x i1> [[R]]
  %r = icmp ugt <4 x i1> %p, %q
  ret <4 x i1> %r
}
