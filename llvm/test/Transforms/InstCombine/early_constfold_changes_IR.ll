; This run line verifies that we get the expected constant fold.
; RUN: opt < %s -instcombine -S | FileCheck %s

define i32 @foo(i32 %arg) #0 {
; CHECK-LABEL: @foo(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[AND:%.*]] = and i32 [[ARG:%.*]], 7
; CHECK-NEXT:    ret i32 [[AND]]
;
entry:
  %or = or i32 0, 7
  %and = and i32 %arg, %or
  ret i32 %and
}

