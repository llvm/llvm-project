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

; CHECK-LABEL: @pr52024(
; CHECK: vector.body:
;
define void @pr52024(i32* %dst, i16 %N) {
entry:
  br label %loop.1

loop.1:
  %iv.1 = phi i16 [ 1, %entry ], [ %iv.1.next, %loop.1.latch ]
  %iv.1.next = mul i16 %iv.1, 3
  %exitcond.1 = icmp uge i16 %iv.1.next, 99
  br i1 %exitcond.1, label %loop.1.latch, label %exit

loop.1.latch:
  %exitcond.2 = icmp eq i16 %iv.1.next, %N
  br i1 %exitcond.2, label %loop.2.ph, label %loop.1

loop.2.ph:
  %iv.1.next.lcssa = phi i16 [ %iv.1.next, %loop.1.latch ]
  %iv.1.next.ext = sext i16 %iv.1.next.lcssa to i64
  br label %loop.2.header

loop.2.header:
  %iv.1.rem = urem i64 100, %iv.1.next.ext
  %rem.trunc = trunc i64 %iv.1.rem to i16
  br label %loop.3

loop.3:
  %iv.3 = phi i32 [ 8, %loop.2.header ], [ %iv.3.next, %loop.3 ]
  %sub.phi = phi i16 [ 0, %loop.2.header ], [ %sub, %loop.3 ]
  %sub = sub i16 %sub.phi, %rem.trunc
  %sub.ext = zext i16 %sub to i32
  %gep.dst = getelementptr i32, i32* %dst, i32 %iv.3
  store i32 %sub.ext, i32* %gep.dst
  %iv.3.next= add nuw nsw i32 %iv.3, 1
  %exitcond.3 = icmp eq i32 %iv.3.next, 34
  br i1 %exitcond.3, label %loop.2.latch, label %loop.3

loop.2.latch:
  %sub.lcssa = phi i16 [ %sub, %loop.3 ]
  %exitcond = icmp uge i16 %sub.lcssa, 200
  br i1 %exitcond, label %exit, label %loop.2.header

exit:
  ret void
}
