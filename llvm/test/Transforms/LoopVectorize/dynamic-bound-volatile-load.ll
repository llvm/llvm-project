; RUN: opt < %s -passes=loop-vectorize -enable-vectorize-loads-as-bound -force-vector-width=4 -force-vector-interleave=1 -S | FileCheck %s
;
; Negative test: the bound is loaded via a volatile (non-simple) load. The detector
; rejects it (isSafeToHoistBoundLoad requires a simple load), so the loop will be rejected.
;
;   void foo(int *A, int *B, int *C, volatile int *Len) {
;     for (int i = 0; i < *Len; i++)
;       A[i] = B[i] + C[i];
;   }

define void @foo(ptr %a, ptr %b, ptr %c, ptr %len) {
; CHECK-LABEL: @foo(
; CHECK-NOT:   vector.body
; CHECK-NOT:   .bound.pre
entry:
  %len0 = load volatile i32, ptr %len, align 4
  %cmp0 = icmp sgt i32 %len0, 0
  br i1 %cmp0, label %loop, label %exit

exit:
  ret void

loop:
  %indvars.iv = phi i64 [ %indvars.iv.next, %loop ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i32, ptr %b, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds nuw i32, ptr %c, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %arrayidx4 = getelementptr inbounds nuw i32, ptr %a, i64 %indvars.iv
  store i32 %add, ptr %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %len.val = load volatile i32, ptr %len, align 4
  %ext = sext i32 %len.val to i64
  %ec = icmp slt i64 %indvars.iv.next, %ext
  br i1 %ec, label %loop, label %exit
}
