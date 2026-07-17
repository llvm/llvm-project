; RUN: opt < %s -passes=loop-vectorize -enable-vectorize-loads-as-bound -force-vector-width=4 -force-vector-interleave=1 -S | FileCheck %s
;
; Negative test: Loop bound loaded from an array element A[2], where A is also
; written inside the loop. The load and store use different indices, so
; pointer identity does not flag a conflict but then LoopAccessAnalysis 
; finds an unresolvable (same-base) dependence between the A[2] read
; and the A[i] write and refuses to vectorize
;
;   void foo(int *A, int *B, int *C, int *Len) {
;     for (int i = 0; i < A[2]; i++)
;       A[i] = B[i] + C[i];
;   }
;

define void @foo(ptr %A, ptr %B, ptr %C, ptr %Len) {
; CHECK-LABEL: @foo(
; CHECK-NOT:   vector.body
entry:
  %arrayidx = getelementptr inbounds nuw i32, ptr %A, i64 2
  %0 = load i32, ptr %arrayidx, align 4
  %cmp11 = icmp sgt i32 %0, 0
  br i1 %cmp11, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx1 = getelementptr inbounds nuw i32, ptr %B, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx1, align 4
  %arrayidx3 = getelementptr inbounds nuw i32, ptr %C, i64 %indvars.iv
  %2 = load i32, ptr %arrayidx3, align 4
  %add = add nsw i32 %2, %1
  %arrayidx5 = getelementptr inbounds nuw i32, ptr %A, i64 %indvars.iv
  store i32 %add, ptr %arrayidx5, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %3 = load i32, ptr %arrayidx, align 4
  %4 = sext i32 %3 to i64
  %cmp = icmp slt i64 %indvars.iv.next, %4
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}
