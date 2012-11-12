; RUN: opt < %s -instcombine -S | FileCheck %s
; PR6165

define i32 @f() {
entry:
  br label %BB1

BB1:                                              ; preds = %BB1, %entry
; CHECK: BB1:
  %x = phi i32 [ -29, %entry ], [ 0, %BB1 ]       ; <i32> [#uses=2]
  %rem = srem i32 %x, 2                           ; <i32> [#uses=1]
  %t = icmp eq i32 %rem, -1                       ; <i1> [#uses=1]
  br i1 %t, label %BB2, label %BB1
; CHECK-NOT: br i1 false

BB2:                                              ; preds = %BB1
; CHECK: BB2:
  ret i32 %x
}
