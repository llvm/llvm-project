; RUN: split-file %s %t
; RUN: not llvm-as < %t/float-mask.ll 2>&1 | FileCheck %s
; RUN: not llvm-as < %t/scalable-mismatch.ll 2>&1 | FileCheck %s
; RUN: not llvm-as < %t/scalable-length-change.ll 2>&1 | FileCheck %s
; CHECK: invalid shufflevector operands

;--- float-mask.ll
define <4 x i32> @f(<4 x i32> %a, <4 x i32> %b, <4 x float> %m) {
  %r = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x float> %m
  ret <4 x i32> %r
}
;--- scalable-mismatch.ll
define <vscale x 4 x i32> @g(<4 x i32> %a, <4 x i32> %b, <vscale x 4 x i8> %m) {
  %r = shufflevector <4 x i32> %a, <4 x i32> %b, <vscale x 4 x i8> %m
  ret <vscale x 4 x i32> %r
}
;--- scalable-length-change.ll
; A run-time mask on scalable vectors must have the same element count as the
; input vectors.
define <vscale x 8 x i32> @h(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 8 x i32> %m) {
  %r = shufflevector <vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 8 x i32> %m
  ret <vscale x 8 x i32> %r
}
