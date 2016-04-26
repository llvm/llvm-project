; RUN: opt %loadPolly -pass-remarks-analysis="polly-scops" -polly-scops -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s --check-prefix=SCOP
;
; CHECK:      remark: <unknown>:0:0: SCoP begins here.
; CHECK-NEXT: remark: <unknown>:0:0: Use user assumption: [N] -> { : N >= 2 }
; CHECK-NEXT: remark: <unknown>:0:0: SCoP ends here.

; SCOP:      Context:
; SCOP-NEXT: [N, M] -> { : 2 <= N <= 2147483647 and -2147483648 <= M <= 2147483647 }
; SCOP:      Assumed Context:
; SCOP-NEXT: [N, M] -> { : }
; SCOP:      Invalid Context:
; SCOP-NEXT: [N, M] -> { : 1 = 0 }
;
;    int f(int *A, int N, int M) {
;      __builtin_assume(M > 0 && N > M);
;      for (int i = 0; i < N; i++)
;        A[i]++;
;      return M;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define i32 @f(i32* %A, i32 %N, i32 %M) {
entry:
  %cmp = icmp sgt i32 %M, 0
  %cmp1 = icmp sgt i32 %N, %M
  %and = and i1 %cmp, %cmp1
  call void @llvm.assume(i1 %and)
  %tmp1 = sext i32 %N to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %land.end
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %cmp2 = icmp slt i64 %indvars.iv, %tmp1
  br i1 %cmp2, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp2 = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %tmp2, 1
  store i32 %inc, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret i32 %M
}

declare void @llvm.assume(i1) #1

