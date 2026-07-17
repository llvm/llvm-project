; RUN: opt < %s -passes=loop-vectorize -enable-vectorize-loads-as-bound -force-vector-width=4 -force-vector-interleave=1 -S | FileCheck %s
;
; Negative test: the upper bound is **PtrLen (an indirect load).
; LoopAccessAnalysis cannot compute SCEV bounds for the indirect 
; load (load i32, ptr %ptr) because %ptr itself is loaded inside the loop. 
; The loop will be rejected.
;
;   void foo(int *A, int *B, int *C, int **PtrLen) {
;     for (int i = 0; i < **PtrLen; i++)
;       A[i] = B[i] + C[i];
;   }

define void @foo(ptr %A, ptr %B, ptr %C, ptr %PtrLen) {
; CHECK-LABEL: @foo(
; CHECK-NOT:   vector.body
; CHECK-NOT:   .bound.pre
entry:
  %ptr0 = load ptr, ptr %PtrLen, align 8
  %val0 = load i32, ptr %ptr0, align 4
  %cmp9 = icmp sgt i32 %val0, 0
  br i1 %cmp9, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds nuw i32, ptr %B, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds nuw i32, ptr %C, i64 %indvars.iv
  %2 = load i32, ptr %arrayidx2, align 4
  %add = add nsw i32 %2, %1
  %arrayidx4 = getelementptr inbounds nuw i32, ptr %A, i64 %indvars.iv
  store i32 %add, ptr %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %ptr = load ptr, ptr %PtrLen, align 8
  %val = load i32, ptr %ptr, align 4
  %3 = sext i32 %val to i64
  %cmp = icmp slt i64 %indvars.iv.next, %3
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}
