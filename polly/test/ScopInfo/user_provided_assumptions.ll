; RUN: opt %loadNPMPolly -pass-remarks-analysis="polly-scops" '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s --check-prefix=SCOP
;
; CHECK:      remark: <unknown>:0:0: SCoP begins here.
; CHECK-NEXT: remark: <unknown>:0:0: Use user assumption: [M, N] -> {  : N <= 2147483647 - M }
; CHECK-NEXT: remark: <unknown>:0:0: Use user assumption: [M, N] -> {  : -2147483648 - M <= N <= 2147483647 - M }
; CHECK-NEXT: remark: <unknown>:0:0: Use user assumption: [M, N, Debug] -> {  : Debug = 0 and 0 < M <= 100 and -2147483648 - M <= N <= 2147483647 - M }
; CHECK-NEXT: remark: <unknown>:0:0: Use user assumption: [M, N, Debug] -> {  : Debug = 0 and 0 < M <= 100 and 0 < N <= 2147483647 - M }
; CHECK-NEXT: remark: <unknown>:0:0: SCoP ends here.

; SCOP:      Context:
; SCOP-NEXT: [N, M, Debug] -> { : Debug = 0 and N > 0 and 0 < M <= 2147483647 - N and M <= 100 }
; SCOP:      Assumed Context:
; SCOP-NEXT: [N, M, Debug] -> {  :  }
; SCOP:      Invalid Context:
; SCOP-NEXT: [N, M, Debug] -> {  : false }
;
;    #include <stdio.h>
;
;    void valid(int * restrict A, int * restrict B, int N, int M, int C[100][100], int Debug) {
;      __builtin_assume(M <= 2147483647 - N);
;      __builtin_assume(M >= -2147483648 - N);
;      __builtin_assume(Debug == 0 && M <= 100 && M >= 1 && N >= 1);
;      if (N + M == -1)
;        C[0][0] = 0;
;
;      for (int i = 0; i < N; i++) {
;        for (int j = 0; j != M; j++) {
;          C[i][j] += A[i * M + j] + B[i + j];
;        }
;
;        if (Debug)
;          printf("Printf!");
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@.str = private unnamed_addr constant [8 x i8] c"Printf!\00", align 1

define void @valid(ptr noalias %A, ptr noalias %B, i32 %N, i32 %M, ptr %C, i32 %Debug) {
entry:
  %sub = sub nsw i32 2147483647, %N
  %cmp = icmp sge i32 %sub, %M
  call void @llvm.assume(i1 %cmp)
  %conv = sext i32 %M to i64
  %conv1 = sext i32 %N to i64
  %sub2 = sub nsw i64 -2147483648, %conv1
  %cmp3 = icmp sge i64 %conv, %sub2
  call void @llvm.assume(i1 %cmp3)
  %cmp5 = icmp eq i32 %Debug, 0
  %cmp7 = icmp slt i32 %M, 101
  %or.cond = and i1 %cmp5, %cmp7
  %cmp10 = icmp sgt i32 %M, 0
  %or.cond1 = and i1 %or.cond, %cmp10
  %cmp12 = icmp sgt i32 %N, 0
  call void @llvm.assume(i1 %or.cond1)
  call void @llvm.assume(i1 %cmp12)
  %add = add nsw i32 %N, %M
  %cmp14 = icmp eq i32 %add, -1
  br label %entry.split

entry.split:
  br i1 %cmp14, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 0, ptr %C, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %M64 = sext i32 %M to i64
  %N64 = sext i32 %N to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc.36, %if.end
  %indvars.iv3 = phi i64 [ %indvars.iv.next4, %for.inc.36 ], [ 0, %if.end ]
  %cmp17 = icmp slt i64 %indvars.iv3, %N64
  br i1 %cmp17, label %for.cond.19, label %for.end.38

for.cond.19:                                      ; preds = %for.cond, %for.body.22
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body.22 ], [ 0, %for.cond ]
  %cmp20 = icmp eq i64 %indvars.iv, %M64
  br i1 %cmp20, label %for.end, label %for.body.22

for.body.22:                                      ; preds = %for.cond.19
  %tmp9 = mul nsw i64 %indvars.iv3, %M64
  %tmp10 = add nsw i64 %tmp9, %indvars.iv
  %arrayidx24 = getelementptr inbounds i32, ptr %A, i64 %tmp10
  %tmp11 = load i32, ptr %arrayidx24, align 4
  %tmp12 = add nuw nsw i64 %indvars.iv3, %indvars.iv
  %arrayidx27 = getelementptr inbounds i32, ptr %B, i64 %tmp12
  %tmp13 = load i32, ptr %arrayidx27, align 4
  %add28 = add nsw i32 %tmp11, %tmp13
  %arrayidx32 = getelementptr inbounds [100 x i32], ptr %C, i64 %indvars.iv3, i64 %indvars.iv
  %tmp14 = load i32, ptr %arrayidx32, align 4
  %add33 = add nsw i32 %tmp14, %add28
  store i32 %add33, ptr %arrayidx32, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond.19

for.end:                                          ; preds = %for.cond.19
  %tobool = icmp eq i32 %Debug, 0
  br i1 %tobool, label %for.inc.36, label %if.then.34

if.then.34:                                       ; preds = %for.end
  %call = call i32 (ptr, ...) @printf(ptr nonnull @.str)
  br label %for.inc.36

for.inc.36:                                       ; preds = %for.end, %if.then.34
  %indvars.iv.next4 = add nuw nsw i64 %indvars.iv3, 1
  br label %for.cond

for.end.38:                                       ; preds = %for.cond
  ret void
}

; Function Attrs: nounwind
declare void @llvm.assume(i1) #0

declare i32 @printf(ptr, ...)

attributes #0 = { nounwind }
