; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
;
; CHECK: Domain :=
; CHECK:   [N, P] -> { Stmt_if_end[i0] : 0 <= i0 < N and (i0 > P or i0 < P) };
;
;    void f(int *A, int N, int P, int Q) {
;      for (int i = 0; i < N; i++) {
;        if (i == P)
;          continue;
;        A[i]++;
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %A, i32 %N, i32 %P, i32 %Q) {
entry:
  %tmp = sext i32 %N to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %if.then ], [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp = icmp slt i64 %indvars.iv, %tmp
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp1 = trunc i64 %indvars.iv to i32
  %cmp1 = icmp eq i32 %tmp1, %P
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  br label %for.cond

if.end:                                           ; preds = %for.body
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp2 = load i32, ptr %arrayidx, align 4
  %inc = add nsw i32 %tmp2, 1
  store i32 %inc, ptr %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.end, %if.then
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
