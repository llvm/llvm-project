; RUN: opt -passes=loop-vectorize -pass-remarks-analysis=loop-vectorize -S < %s 2>&1 | FileCheck %s

; Make sure the unsafe user specified interleave count is ignored.

; CHECK: remark: <unknown>:0:0: Ignoring user-specified interleave count due to possibly unsafe dependencies in the loop.
; CHECK-LABEL: @loop_distance_4
define void @loop_distance_4(ptr %a, ptr %b) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 4, %entry ], [ %iv.next, %loop ]
  %0 = getelementptr i32, ptr %b, i64 %iv
  %arrayidx = getelementptr i8, ptr %0, i64 -16
  %1 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds nuw i32, ptr %a, i64 %iv
  %2 = load i32, ptr %arrayidx2, align 4
  %add = add nsw i32 %2, %1
  store i32 %add, ptr %0, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 64
  br i1 %exitcond.not, label %for.end, label %loop, !llvm.loop !1

for.end:
  ret void
}

!1 = !{!1, !2, !3}
!2 = !{!"llvm.loop.interleave.count", i32 4}
!3 = !{!"llvm.loop.vectorize.width", i32 4}
