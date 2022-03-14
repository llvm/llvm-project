; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
; CHECK:         Context:
; CHECK-NEXT:    [N] -> {  : -2147483648 <= N <= 2147483647 }
; CHECK-NEXT:    Assumed Context:
; CHECK-NEXT:    [N] -> {  :  }
; CHECK-NEXT:    Invalid Context:
; CHECK-NEXT:    [N] -> {  : N >= 514 }
;
; CHECK:         Statements {
; CHECK-NEXT:    	Stmt_if_end
; CHECK-NEXT:            Domain :=
; CHECK-NEXT:                [N] -> { Stmt_if_end[i0] : 0 <= i0 < N };
; CHECK-NEXT:            Schedule :=
; CHECK-NEXT:                [N] -> { Stmt_if_end[i0] -> [i0] };
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:                [N] -> { Stmt_if_end[i0] -> MemRef_A[i0] };
; CHECK-NEXT:            MustWriteAccess :=	[Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:                [N] -> { Stmt_if_end[i0] -> MemRef_A[i0] };
; CHECK-NEXT:    }
;
;    void f();
;    void g(int *A, int N) {
;      for (int i = 0; i < N; i++) {
;        if (i > 512) {
;          f();
;          A[i]++;
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
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  call void (...) @f() #2
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp12 = load i32, i32* %arrayidx2, align 4
  %inc2 = add nsw i32 %tmp12, 1
  store i32 %inc2, i32* %arrayidx2, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp1 = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %tmp1, 1
  store i32 %inc, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

declare void @f(...)
