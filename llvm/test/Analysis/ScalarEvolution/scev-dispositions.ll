; RUN: opt -disable-output "-passes=print<scalar-evolution>" < %s 2>&1 | FileCheck %s

define void @single_loop(i32* %buf, i32 %start) {
; CHECK-LABEL: Classifying expressions for: @single_loop
 entry:
  %val = add i32 %start, 400
  br label %loop

 loop:
  %counter = phi i32 [ 0, %entry ], [ %counter.inc, %loop ]
  %idx = phi i32 [ %start, %entry ], [ %idx.inc, %loop ]

; CHECK:  %counter = phi i32 [ 0, %entry ], [ %counter.inc, %loop ]
; CHECK-NEXT:  -->  {{.*}} LoopDispositions: { %loop: Computable }
; CHECK:  %idx = phi i32 [ %start, %entry ], [ %idx.inc, %loop ]
; CHECK-NEXT:  -->  {{.*}} LoopDispositions: { %loop: Computable }
; CHECK:  %val2 = add i32 %start, 400
; CHECK-NEXT:  -->  {{.*}} LoopDispositions: { %loop: Invariant }
; CHECK:  %idx.inc = add nsw i32 %idx, 1
; CHECK-NEXT:  -->  {{.*}} LoopDispositions: { %loop: Computable }
; CHECK:  %val3 = load volatile i32, i32* %buf
; CHECK-NEXT:  -->  {{.*}} LoopDispositions: { %loop: Variant }

  %val2 = add i32 %start, 400
  %idx.inc = add nsw i32 %idx, 1
  %idx.inc.sext = sext i32 %idx.inc to i64
  %condition = icmp eq i32 %counter, 1
  %counter.inc = add i32 %counter, 1
  %val3 = load volatile i32, i32* %buf
  br i1 %condition, label %exit, label %loop

 exit:
  ret void
}


define void @nested_loop(double* %p, i64 %m) {
; CHECK-LABEL: Classifying expressions for: @nested_loop

; CHECK:  %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
; CHECK-NEXT:  -->  {{.*}} LoopDispositions: { %outer.loop: Computable, %bb: Invariant }
; CHECK:  %i = phi i64 [ 0, %outer.loop ], [ %i.next, %bb ]
; CHECK-NEXT:  -->  {{.*}} LoopDispositions: { %bb: Computable, %outer.loop: Variant }
; CHECK:  %j.add = add i64 %j, 100
; CHECK-NEXT:  -->  {{.*}} LoopDispositions: { %bb: Invariant, %outer.loop: Computable }
; CHECK:  %i.next = add i64 %i, 1
; CHECK-NEXT:  -->  {{.*}} LoopDispositions: { %bb: Computable, %outer.loop: Variant }
; CHECK:  %j.next = add i64 %j, 91
; CHECK-NEXT:  -->  {{.*}} LoopDispositions: { %outer.loop: Computable, %bb: Invariant }

entry:
  %k = icmp sgt i64 %m, 0
  br i1 %k, label %outer.loop, label %return

outer.loop:
  %j = phi i64 [ 0, %entry ], [ %j.next, %outer.latch ]
  br label %bb

bb:
  %i = phi i64 [ 0, %outer.loop ], [ %i.next, %bb ]
  %j.add = add i64 %j, 100
  %i.next = add i64 %i, 1
  %exitcond = icmp eq i64 %i.next, 91
  br i1 %exitcond, label %outer.latch, label %bb

outer.latch:
  %j.next = add i64 %j, 91
  %h = icmp eq i64 %j.next, %m
  br i1 %h, label %return, label %outer.loop

return:
  ret void
}
