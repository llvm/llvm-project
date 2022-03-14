; RUN: opt %loadPolly -polly-print-scops -polly-detect-keep-going -polly-allow-nonaffine -disable-output < %s | FileCheck %s
;
; TODO: FIXME: Investigate why "-polly-detect-keep-going" is needed to detect
;              this SCoP. That flag should not make a difference.
;
; CHECK:         Context:
; CHECK-NEXT:    [N] -> {  : -2147483648 <= N <= 2147483647 }
; CHECK-NEXT:    Assumed Context:
; CHECK-NEXT:    [N] -> {  :  }
; CHECK-NEXT:    Invalid Context:
; CHECK-NEXT:    [N] -> {  : N >= 514 }
;
; CHECK:         Statements {
; CHECK-NEXT:    	Stmt_if_end3
; CHECK-NEXT:            Domain :=
; CHECK-NEXT:                [N] -> { Stmt_if_end3[i0] : 0 <= i0 < N };
; CHECK-NEXT:            Schedule :=
; CHECK-NEXT:                [N] -> { Stmt_if_end3[i0] -> [i0] };
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:                [N] -> { Stmt_if_end3[i0] -> MemRef_A[i0] };
; CHECK-NEXT:            MustWriteAccess :=	[Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:                [N] -> { Stmt_if_end3[i0] -> MemRef_A[i0] };
; CHECK-NEXT:    }
;
;    int f();
;    void g(int *A, int N) {
;      for (int i = 0; i < N; i++) {
;        if (i > 512) {
;          int v = f();
;        S:
;          A[v]++;
;        }
;        A[i]++;
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @g(i32* %A, i32 %N) {
entry:
  %tmp = sext i32 %N to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %cmp = icmp slt i64 %indvars.iv, %tmp
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %cmp1 = icmp sgt i64 %indvars.iv, 512
  br i1 %cmp1, label %if.then, label %if.end3

if.then:                                          ; preds = %for.body
  %call = call i32 (...) @f()
  br label %S

S:                                                ; preds = %if.then
  %idxprom = sext i32 %call to i64
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom
  %tmp1 = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %tmp1, 1
  store i32 %inc, i32* %arrayidx, align 4
  br label %if.end3

if.end3:                                          ; preds = %if.end, %for.body
  %arrayidx5 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp2 = load i32, i32* %arrayidx5, align 4
  %inc6 = add nsw i32 %tmp2, 1
  store i32 %inc6, i32* %arrayidx5, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.end3, %if.then2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

declare i32 @f(...)
