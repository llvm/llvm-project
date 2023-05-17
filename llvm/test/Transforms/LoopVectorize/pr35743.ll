; RUN: opt < %s -passes=loop-vectorize -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-ni:1"
target triple = "x86_64-unknown-linux-gnu"

; This cannot be correctly vectorized with type i1.
define i8 @test_01(i8 %c) #0 {

; CHECK-LABEL: @test_01(
; CHECK-NOT:   vector.body:
; CHECK-NOT:   zext i1 {{.*}} to i8

entry:
  br label %loop

exit:                                           ; preds = %loop
  ret i8 %accum.plus

loop:                                            ; preds = %loop, %entry
  %accum.phi = phi i8 [ %c, %entry ], [ %accum.plus, %loop ]
  %iv = phi i32 [ 1, %entry ], [ %iv.next, %loop ]
  %accum.and = and i8 %accum.phi, 1
  %accum.plus = add nuw nsw i8 %accum.and, 3
  %iv.next = add nuw nsw i32 %iv, 1
  %cond = icmp ugt i32 %iv, 191
  br i1 %cond, label %exit, label %loop
}

; TODO: This can be vectorized with type i1 because the result is not used.
define void @test_02(i8 %c) #0 {

; CHECK-LABEL: @test_02(
; CHECK-NOT:   vector.body:

entry:
  br label %loop

exit:                                           ; preds = %loop
  %lcssa = phi i8 [ %accum.plus, %loop ]
  ret void

loop:                                            ; preds = %loop, %entry
  %accum.phi = phi i8 [ %c, %entry ], [ %accum.plus, %loop ]
  %iv = phi i32 [ 1, %entry ], [ %iv.next, %loop ]
  %accum.and = and i8 %accum.phi, 1
  %accum.plus = add nuw nsw i8 %accum.and, 3
  %iv.next = add nuw nsw i32 %iv, 1
  %cond = icmp ugt i32 %iv, 191
  br i1 %cond, label %exit, label %loop
}

; This can be vectorized with type i1 because the result is truncated properly.
define i1 @test_03(i8 %c) #0 {

; CHECK-LABEL: @test_03(
; CHECK:   vector.body:
; CHECK:   zext i1 {{.*}} to i8

entry:
  br label %loop

exit:                                           ; preds = %loop
  %lcssa = phi i8 [ %accum.plus, %loop ]
  %trunc = trunc i8 %lcssa to i1
  ret i1 %trunc

loop:                                            ; preds = %loop, %entry
  %accum.phi = phi i8 [ %c, %entry ], [ %accum.plus, %loop ]
  %iv = phi i32 [ 1, %entry ], [ %iv.next, %loop ]
  %accum.and = and i8 %accum.phi, 1
  %accum.plus = add nuw nsw i8 %accum.and, 3
  %iv.next = add nuw nsw i32 %iv, 1
  %cond = icmp ugt i32 %iv, 191
  br i1 %cond, label %exit, label %loop
}

; This cannot be vectorized with type i1 because the result is truncated to a
; wrong type.
; TODO: It can also be vectorized with type i32 (or maybe i4?)
define i4 @test_04(i8 %c) #0 {

; CHECK-LABEL: @test_04(
; CHECK-NOT:   vector.body:
; CHECK-NOT:   zext i1 {{.*}} to i8

entry:
  br label %loop

exit:                                           ; preds = %loop
  %lcssa = phi i8 [ %accum.plus, %loop ]
  %trunc = trunc i8 %lcssa to i4
  ret i4 %trunc

loop:                                            ; preds = %loop, %entry
  %accum.phi = phi i8 [ %c, %entry ], [ %accum.plus, %loop ]
  %iv = phi i32 [ 1, %entry ], [ %iv.next, %loop ]
  %accum.and = and i8 %accum.phi, 1
  %accum.plus = add nuw nsw i8 %accum.and, 3
  %iv.next = add nuw nsw i32 %iv, 1
  %cond = icmp ugt i32 %iv, 191
  br i1 %cond, label %exit, label %loop
}
