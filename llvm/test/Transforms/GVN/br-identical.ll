; RUN: opt -gvn -S -o - %s | FileCheck %s

; If a branch has two identical successors, we cannot declare either dead.

define void @widget(i1 %p) {
entry:
  br label %bb2

bb2:
  %t1 = phi i64 [ 0, %entry ], [ %t5, %bb7 ]
  %t2 = add i64 %t1, 1
  %t3 = icmp ult i64 0, %t2
  br i1 %t3, label %bb3, label %bb4

bb3:
  %t4 = call i64 @f()
  br label %bb4

bb4:
  ; CHECK-NOT: phi {{.*}} undef
  %foo = phi i64 [ %t4, %bb3 ], [ 0, %bb2 ]
  br i1 %p, label %bb5, label %bb6

bb5:
  br i1 true, label %bb7, label %bb7

bb6:
  br i1 true, label %bb7, label %bb7

bb7:
  %t5 = add i64 %t1, 1
  br i1 %p, label %bb2, label %bb8

bb8:
  ret void
}

declare i64 @f()
