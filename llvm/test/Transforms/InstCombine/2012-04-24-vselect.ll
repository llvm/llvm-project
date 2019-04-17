; RUN: opt -instcombine -S < %s | FileCheck %s

; CHECK-LABEL: @foo(
; CHECK: <i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>

define <8 x i32> @foo() nounwind {
entry:
  %v1.i = select <8 x i1> <i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>,
    <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>,
    <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i32> %v1.i
}

