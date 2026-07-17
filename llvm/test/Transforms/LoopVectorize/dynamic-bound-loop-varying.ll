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

define void @foo(ptr %A, ptr %B, ptr %C, ptr %Len) {
; CHECK-LABEL: @foo(
; CHECK-NOT:   vector.body
; CHECK-NOT:   .bound.pre
entry:
  %0 = load i32, ptr %Len, align 4
  %cmp12 = icmp sgt i32 %0, 0
  br i1 %cmp12, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx2 = getelementptr inbounds nuw i32, ptr %B, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds nuw i32, ptr %C, i64 %indvars.iv
  %2 = load i32, ptr %arrayidx4, align 4
  %add = add nsw i32 %2, %1
  %arrayidx6 = getelementptr inbounds nuw i32, ptr %A, i64 %indvars.iv
  store i32 %add, ptr %arrayidx6, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx = getelementptr inbounds nuw i32, ptr %Len, i64 %indvars.iv.next
  %3 = load i32, ptr %arrayidx, align 4
  %4 = sext i32 %3 to i64
  %cmp = icmp slt i64 %indvars.iv.next, %4
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}
