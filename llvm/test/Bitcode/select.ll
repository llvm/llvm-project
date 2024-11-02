; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder < %s

define <2 x i32> @main() {
  ret <2 x i32> select (<2 x i1> <i1 false, i1 undef>, <2 x i32> zeroinitializer, <2 x i32> <i32 0, i32 undef>)
}

; CHECK: define <2 x i32> @main() {
; CHECK:   ret <2 x i32> <i32 0, i32 undef>
; CHECK: }

define <2 x float> @f() {
  ret <2 x float> select (i1 ptrtoint (<2 x float> ()* @f to i1), <2 x float> <float 1.000000e+00, float 0.000000e+00>, <2 x float> zeroinitializer)
}

; CHECK: define <2 x float> @f() {
; CHECK:   ret <2 x float> select (i1 ptrtoint (ptr @f to i1), <2 x float> <float 1.000000e+00, float 0.000000e+00>, <2 x float> zeroinitializer)
; CHECK: }
