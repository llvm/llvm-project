; RUN: llc -mtriple=powerpc-unknown-openbsd < %s | FileCheck %s

; BitPermutationSelector in PPCISelDAGToDAG.cpp was taking the wrong
; result of a load <pre-inc> after optimizing away a permutation.
; Here, the big end of i48 %3 was %1 but should be %0.

define i32 @hop(ptr %out, ptr %in) {
entry:
  %0 = getelementptr i8, ptr %in, i32 28
  %1 = load i32, ptr %0, align 4
  %2 = ptrtoint ptr %0 to i48
  %3 = shl i48 %2, 16
  store i48 %3, ptr %out, align 4
  ret i32 %1
}
; The stw should store POINTER, not VALUE.
; CHECK:        lwzu [[VALUE:[0-9]+]], 28([[POINTER:[0-9]+]])
; CHECK:        stw [[POINTER]], 0({{[0-9]+}})
