; RUN: llc -mtriple=hexagon -mattr=+hvxv69,+hvx-length128b < %s | FileCheck %s
; RUN: llc -mtriple=hexagon -mattr=+hvxv73,+hvx-length128b < %s | FileCheck %s

; Regression test for crash in TargetLowering::expandCttzElts.
;
; expandCttzElts() did not handle the empty StepVec returned by
; getLegalMaskAndStepVector() when the implied step-vector type
; required TypeSplitVector, causing a null SDValue dereference.

; CHECK-LABEL: ctz_v32i1:
; CHECK: .cfi_endproc
define i32 @ctz_v32i1(<32 x i1> %m) {
  %res = call i32 @llvm.experimental.cttz.elts.i32.v32i1(<32 x i1> %m, i1 false)
  ret i32 %res
}