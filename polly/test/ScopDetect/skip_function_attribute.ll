; RUN: opt %loadPolly -polly-print-detect -disable-output < %s | FileCheck %s
;
; Verify polly skips this function
;
; CHECK-NOT: Valid Region for Scop
;
;    void polly_skip_me(int *A, int N) {
;      for (int i = 0; i < N; i++)
;        A[i] = A[i] * A[i] + A[i];
;    }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @polly_skip_me(ptr %A, i32 %N) #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %cmp1 = icmp sgt i32 %N, 0
  br i1 %cmp1, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry.split
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.02 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i.02
  %tmp = load i32, ptr %arrayidx, align 4
  %mul = mul nsw i32 %tmp, %tmp
  %add = add nsw i32 %mul, %tmp
  %arrayidx3 = getelementptr inbounds i32, ptr %A, i32 %i.02
  store i32 %add, ptr %arrayidx3, align 4
  %inc = add nsw i32 %i.02, 1
  %cmp = icmp slt i32 %inc, %N
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry.split
  ret void
}

attributes #0 = { "polly.skip.fn" }
