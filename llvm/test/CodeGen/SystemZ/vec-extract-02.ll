; Verify ReplaceExtractVectorEltOfLoadWithNarrowedLoad fixes
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test a case where a vector extraction can be simplified to a scalar load.
; The index must be extended from i32 to i64.
define i32 @f1(ptr %ptr, i32 %index) {
; CHECK-LABEL: f1:
; CHECK: risbgn {{%r[0-5]}}, %r3, 60, 189, 2
; CHECK: l %r2,
; CHECK: br %r14
  %vec = load <4 x i32>, ptr %ptr
  %res = extractelement <4 x i32> %vec, i32 %index
  ret i32 %res
}
