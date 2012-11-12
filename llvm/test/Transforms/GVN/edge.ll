; RUN: opt %s -gvn -S -o - | FileCheck %s

define i32 @f1(i32 %x) {
  ; CHECK: define i32 @f1(
bb0:
  %cmp = icmp eq i32 %x, 0
  br i1 %cmp, label %bb2, label %bb1
bb1:
  br label %bb2
bb2:
  %cond = phi i32 [ %x, %bb0 ], [ 0, %bb1 ]
  %foo = add i32 %cond, %x
  ret i32 %foo
  ; CHECK: bb2:
  ; CHECK: ret i32 %x
}

define i32 @f2(i32 %x) {
  ; CHECK: define i32 @f2(
bb0:
  %cmp = icmp ne i32 %x, 0
  br i1 %cmp, label %bb1, label %bb2
bb1:
  br label %bb2
bb2:
  %cond = phi i32 [ %x, %bb0 ], [ 0, %bb1 ]
  %foo = add i32 %cond, %x
  ret i32 %foo
  ; CHECK: bb2:
  ; CHECK: ret i32 %x
}

define i32 @f3(i32 %x) {
  ; CHECK: define i32 @f3(
bb0:
  switch i32 %x, label %bb1 [ i32 0, label %bb2]
bb1:
  br label %bb2
bb2:
  %cond = phi i32 [ %x, %bb0 ], [ 0, %bb1 ]
  %foo = add i32 %cond, %x
  ret i32 %foo
  ; CHECK: bb2:
  ; CHECK: ret i32 %x
}

declare void @g(i1)
define void @f4(i8 * %x)  {
; CHECK: define void @f4(
bb0:
  %y = icmp eq i8* null, %x
  br i1 %y, label %bb2, label %bb1
bb1:
  br label %bb2
bb2:
  %zed = icmp eq i8* null, %x
  call void @g(i1 %zed)
; CHECK: call void @g(i1 %y)
  ret void
}
