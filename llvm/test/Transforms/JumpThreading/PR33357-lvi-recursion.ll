; RUN: opt -S -jump-threading -verify -o - %s | FileCheck %s
@a = external global i16, align 1

; CHECK-LABEL: f
; CHECK: bb6:
; CHECK: bb2:
; CHECK: bb3:
; CHECK-NOT: bb0:
; CHECK-NOT: bb1:
; CHECK-NOT: bb4:
; CHECK-NOT: bb5:
define void @f(i32 %p1) {
bb0:
  %0 = icmp eq i32 %p1, 0
  br i1 undef, label %bb6, label %bb1

bb1:
  br label %bb2

bb2:
  %1 = phi i1 [ %0, %bb1 ], [ %2, %bb4 ]
  %2 = and i1 %1, undef
  br i1 %2, label %bb3, label %bb4

bb3:
  store i16 undef, i16* @a, align 1
  br label %bb4

bb4:
  br i1 %0, label %bb2, label %bb5

bb5:
  unreachable

bb6:
  ret void
}
