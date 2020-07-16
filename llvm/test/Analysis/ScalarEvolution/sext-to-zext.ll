; RUN: opt -analyze -enable-new-pm=0 -scalar-evolution < %s | FileCheck %s
; RUN: opt -disable-output "-passes=print<scalar-evolution>" < %s 2>&1 | FileCheck %s

define void @f(i1 %c) {
; CHECK-LABEL: Classifying expressions for: @f
entry:
  %start = select i1 %c, i32 100, i32 0
  %step =  select i1 %c, i32 -1,  i32 1
  br label %loop

loop:
  %iv = phi i32 [ %start, %entry ], [ %iv.dec, %loop ]
  %iv.tc = phi i32 [ 0, %entry ], [ %iv.tc.inc, %loop ]
  %iv.tc.inc = add i32 %iv.tc, 1
  %iv.dec = add nsw i32 %iv, %step
  %iv.sext = sext i32 %iv to i64
; CHECK:  %iv.sext = sext i32 %iv to i64
; CHECK-NEXT:  -->  {(sext i32 %start to i64),+,(sext i32 %step to i64)}<nsw><%loop>
  %be = icmp ne i32 %iv.tc.inc, 100
  br i1 %be, label %loop, label %leave

leave:
  ret void
}
