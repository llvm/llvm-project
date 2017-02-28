; RUN: opt %loadPolly -pass-remarks-analysis="polly-scops" -polly-scops \
; RUN:    -polly-precise-inbounds -disable-output < %s 2>&1 | FileCheck %s
;
; CHECK:      remark: <unknown>:0:0: SCoP begins here.
; CHECK-NEXT: remark: <unknown>:0:0: Use user assumption: [i, N, M] -> {  : N <= i or (N > i and N >= 0) }
; CHECK-NEXT: remark: <unknown>:0:0: Inbounds assumption:    [i, N, M] -> {  : N <= i or (N > i and M <= 100) }
; CHECK-NEXT: remark: <unknown>:0:0: SCoP ends here.
;
;    void f(int *restrict A, int *restrict B, int i, int N, int M, int C[100][100]) {
;      for (; i < N; i++) {
;        __builtin_assume(N >= 0);
;        for (int j = 0; j != M; j++) {
;          __builtin_assume(N >= 0);
;          C[i][j] += A[i * M + j] + B[i + j];
;        }
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* noalias %A, i32* noalias %B, i32 %i, i32 %N, i32 %M, [100 x i32]* %C) {
entry:
  %tmp = zext i32 %M to i64
  %tmp6 = sext i32 %i to i64
  %tmp7 = sext i32 %N to i64
  %tmp8 = sext i32 %M to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc.15, %entry
  %indvars.iv3 = phi i64 [ %indvars.iv.next4, %for.inc.15 ], [ %tmp6, %entry ]
  %cmp = icmp slt i64 %indvars.iv3, %tmp7
  br i1 %cmp, label %for.body, label %for.end.17

for.body:                                         ; preds = %for.cond
  %cmp1 = icmp sgt i32 %N, -1
  call void @llvm.assume(i1 %cmp1)
  br label %for.cond.2

for.cond.2:                                       ; preds = %for.inc, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %for.body ]
  %cmp3 = icmp eq i64 %indvars.iv, %tmp
  br i1 %cmp3, label %for.end, label %for.body.4

for.body.4:                                       ; preds = %for.cond.2
  %tmp9 = mul nsw i64 %indvars.iv3, %tmp8
  %tmp10 = add nsw i64 %tmp9, %indvars.iv
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %tmp10
  %tmp11 = load i32, i32* %arrayidx, align 4
  %tmp12 = add nsw i64 %indvars.iv3, %indvars.iv
  %arrayidx8 = getelementptr inbounds i32, i32* %B, i64 %tmp12
  %tmp13 = load i32, i32* %arrayidx8, align 4
  %add9 = add nsw i32 %tmp11, %tmp13
  %arrayidx13 = getelementptr inbounds [100 x i32], [100 x i32]* %C, i64 %indvars.iv3, i64 %indvars.iv
  %tmp14 = load i32, i32* %arrayidx13, align 4
  %add14 = add nsw i32 %tmp14, %add9
  store i32 %add14, i32* %arrayidx13, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body.4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond.2

for.end:                                          ; preds = %for.cond.2
  br label %for.inc.15

for.inc.15:                                       ; preds = %for.end
  %indvars.iv.next4 = add nsw i64 %indvars.iv3, 1
  br label %for.cond

for.end.17:                                       ; preds = %for.cond
  ret void
}

declare void @llvm.assume(i1) #1

