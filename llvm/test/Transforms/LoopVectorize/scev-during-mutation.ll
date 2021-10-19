; RUN: opt -loop-vectorize -force-vector-width=2 -force-vector-interleave=2 -scev-verify-ir -S %s | FileCheck %s

; Make sure SCEV is not queried while the IR is temporarily invalid. The tests
; deliberately do not check for details of the vectorized IR, because that's
; not the focus of the test.

define void @pr49538() {
; CHECK-LABEL: @pr49538
; CHECK: vector.body:
;
entry:
  br label %loop.0

loop.0:
  %iv.0 = phi i16 [ -1, %entry ], [ %iv.0.next, %loop.0.latch ]
  br label %loop.1

loop.1:
  %iv.1 = phi i16 [ -1, %loop.0 ], [ %iv.1.next, %loop.1 ]
  %iv.1.next = add nsw i16 %iv.1, 1
  %i6 = icmp eq i16 %iv.1.next, %iv.0
  br i1 %i6, label %loop.0.latch, label %loop.1

loop.0.latch:
  %i8 = phi i16 [ 1, %loop.1 ]
  %iv.0.next = add nsw i16 %iv.0, 1
  %ec.0 = icmp eq i16 %iv.0.next, %i8
  br i1 %ec.0, label %exit, label %loop.0

exit:
  ret void
}

define void @pr49900(i32 %x, i64* %ptr) {
; CHECK-LABEL: @pr49900
; CHECK: vector.body{{.*}}:
; CHECK: vector.body{{.*}}:
;
entry:
  br label %loop.0

loop.0:                                              ; preds = %bb2, %bb
  %ec.0 = icmp slt i32 %x, 0
  br i1 %ec.0, label %loop.0, label %loop.1.ph

loop.1.ph:                                              ; preds = %bb2
  br label %loop.1

loop.1:                                             ; preds = %bb33, %bb5
  %iv.1 = phi i32 [ 0, %loop.1.ph ], [ %iv.3.next, %loop.1.latch ]
  br label %loop.2

loop.2:
  %iv.2 = phi i32 [ %iv.1, %loop.1 ], [ %iv.2.next, %loop.2 ]
  %tmp54 = add i32 %iv.2, 12
  %iv.2.next = add i32 %iv.2, 13
  %ext = zext i32 %iv.2.next to i64
  %tmp56 = add nuw nsw i64 %ext, 1
  %C6 = icmp sle i32 %tmp54, 65536
  br i1 %C6, label %loop.2, label %loop.3.ph

loop.3.ph:
  br label %loop.3

loop.3:
  %iv.3 = phi i32 [ %iv.2.next, %loop.3.ph ], [ %iv.3.next, %loop.3 ]
  %iv.3.next = add i32 %iv.3 , 13
  %C1 = icmp ult i32 %iv.3.next, 65536
  br i1 %C1, label %loop.3, label %loop.1.latch

loop.1.latch:
  %ec = icmp ne i32 %iv.1, 9999
  br i1 %ec, label %loop.1, label %exit

exit:
  ret void
}
