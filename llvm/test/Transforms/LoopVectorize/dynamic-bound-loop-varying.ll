; RUN: opt < %s -passes=loop-vectorize -enable-vectorize-loads-as-bound -force-vector-width=4 -force-vector-interleave=1 -S | FileCheck %s
;
; Negative test: the bound is Len[i], which depends on the
; induction variable hence not invariant.  The dependency chain traces back to a loop header
; PHI, so the loop will be rejected.
;
;   void foo(int *A, int *B, int *C, int *Len) {
;     for (int i = 0; i < Len[i]; i++)
;       A[i] = B[i] + C[i];
;   }

define void @foo(ptr %a, ptr %b, ptr %c, ptr %len) {
; CHECK-LABEL: @foo(
; CHECK-NOT:   vector.body
; CHECK-NOT:   .bound.pre
entry:
  %0 = load i32, ptr %len, align 4
  %cmp12 = icmp sgt i32 %0, 0
  br i1 %cmp12, label %loop, label %exit

exit:
  ret void

loop:
  %indvars.iv = phi i64 [ %indvars.iv.next, %loop ], [ 0, %entry ]
  %arrayidx2 = getelementptr inbounds nuw i32, ptr %b, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds nuw i32, ptr %c, i64 %indvars.iv
  %2 = load i32, ptr %arrayidx4, align 4
  %add = add nsw i32 %2, %1
  %arrayidx6 = getelementptr inbounds nuw i32, ptr %a, i64 %indvars.iv
  store i32 %add, ptr %arrayidx6, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx = getelementptr inbounds nuw i32, ptr %len, i64 %indvars.iv.next
  %3 = load i32, ptr %arrayidx, align 4
  %4 = sext i32 %3 to i64
  %ec = icmp slt i64 %indvars.iv.next, %4
  br i1 %ec, label %loop, label %exit
}
