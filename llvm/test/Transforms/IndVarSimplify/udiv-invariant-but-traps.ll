; RUN: opt -indvars -S < %s | FileCheck %s

@b = common global i32 0, align 4

define i32 @foo(i32 %x, i1 %y) {
bb0:
  br label %bb1

bb1:
  br i1 %y, label %bb14, label %bb8

bb8:
  %i = phi i64 [ %i.next, %bb8 ], [ 0, %bb1 ]
  %i.next = add i64 %i, 1
  %div = udiv i32 1, %x
  %c = icmp eq i64 %i.next, 6
  br i1 %c, label %bb11, label %bb8

bb11:
  br i1 %y, label %bb1, label %bb13

bb13:
  store i32 %div, i32* @b, align 4
  br label %bb14

bb14:
  ret i32 0
}

; CHECK-LABEL: @foo(
; CHECK: bb8:
; CHECK: udiv
