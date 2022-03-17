; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
; CHECK:         Assumed Context:
; CHECK-NEXT:    [N] -> {  :  }
; CHECK-NEXT:    Invalid Context:
; CHECK-NEXT:    [N] -> {  : false }
; CHECK:         p0: %N
; CHECK:         Statements {
; CHECK-NEXT:    	Stmt_if_then
; CHECK-NEXT:            Domain :=
; CHECK-NEXT:                [N] -> { Stmt_if_then[i0] : (1 + i0) mod 2 = 0 and 0 < i0 < N }
; CHECK-NEXT:            Schedule :=
; CHECK-NEXT:                [N] -> { Stmt_if_then[i0] -> [i0] };
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:                [N] -> { Stmt_if_then[i0] -> MemRef_A[i0] };
; CHECK-NEXT:            MustWriteAccess :=	[Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:                [N] -> { Stmt_if_then[i0] -> MemRef_A[i0] };
; CHECK-NEXT:    }
;
;    void f(int *A, int N) {
;      for (int i = 0; i < N; i++) {
;        if (i & 1)
;          A[i]++;
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A, i32 %N) {
entry:
  %tmp = sext i32 %N to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %cmp = icmp slt i64 %indvars.iv, %tmp
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp1 = trunc i64 %indvars.iv to i32
  %and = and i32 %tmp1, 1
  %tobool = icmp eq i32 %and, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %for.body
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp2 = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %tmp2, 1
  store i32 %inc, i32* %arrayidx, align 4
  br label %if.end

if.end:                                           ; preds = %for.body, %if.then
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
