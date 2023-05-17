; RUN: opt -passes=loop-vectorize %s -mtriple=arm64-apple-iphoneos -S | FileCheck %s

; CHECK-LABEL: define void @test(
; CHECK: vector.body

define void @test(ptr %dst, ptr %src) {
entry:
  %l = load i32, ptr %src
  br label %loop.ph

loop.ph:
  br label %loop

loop:
  %iv = phi i64 [ 0, %loop.ph ], [ %iv.next, %loop ]
  %l.cast = sext i32 %l to i64
  %dst.idx = getelementptr i64, ptr %dst, i64 %iv
  store i64 %l.cast, ptr %dst.idx
  %iv.next = add nuw nsw i64 %iv, 1
  %cmp9.us = icmp ult i64 %iv.next, 20
  br i1 %cmp9.us, label %loop, label %exit

exit:
  ret void
}
