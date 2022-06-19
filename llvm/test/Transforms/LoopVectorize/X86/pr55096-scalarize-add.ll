; RUN: opt -passes=loop-vectorize -force-vector-width=2 -force-vector-interleave=1 -S %s | FileCheck %s

; REQUIRES: asserts
; XFAIL: *

target triple = "x86_64-apple-macosx"

; CHECK: vector.body

define void @test_pr55096(i64 %c, ptr %p) {
entry:
  br label %loop.header

loop.header:
  %iv.1 = phi i64 [ 122, %entry ], [ %iv.1.next, %loop.latch ]
  %iv.2 = phi i16 [ 6229, %entry ], [ %iv.2.next, %loop.latch ]
  %iv.2.next = add i16 %iv.2, 2008
  %cmp = icmp ult i64 %iv.1, %c
  br i1 %cmp, label %loop.latch, label %loop.then

loop.then:
  %div = udiv i16 4943, %iv.2.next
  %gep = getelementptr inbounds i16, ptr %p, i16 %div
  store i16 0, ptr %gep, align 2
  br label %loop.latch

loop.latch:
  %iv.1.next = add nuw nsw i64 %iv.1, 1
  %exitcond.not = icmp eq i64 %iv.1.next, 462
  br i1 %exitcond.not, label %exit, label %loop.header

exit:
  ret void
}
