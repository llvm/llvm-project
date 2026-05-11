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
