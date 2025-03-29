; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
;
; CHECK:         Assumed Context:
; CHECK-NEXT:    [N] -> {  :  }
; CHECK-NEXT:    Invalid Context:
; CHECK-NEXT:    [N] -> {  : false }
; CHECK:         p0: %N
; CHECK:         Statements {
; CHECK-NEXT:    	Stmt_for_body
; CHECK-NEXT:            Domain :=
; CHECK-NEXT:                [N] -> { Stmt_for_body[i0] : 0 <= i0 < N };
; CHECK-NEXT:            Schedule :=
; CHECK-NEXT:                [N] -> { Stmt_for_body[i0] -> [i0] };
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:                [N] -> { Stmt_for_body[i0] -> MemRef_A[1] : (1 + i0) mod 2 = 0; Stmt_for_body[i0] -> MemRef_A[0] : (i0) mod 2 = 0 }
; CHECK-NEXT:            MustWriteAccess :=	[Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:               [N] -> { Stmt_for_body[i0] -> MemRef_A[1] : (1 + i0) mod 2 = 0; Stmt_for_body[i0] -> MemRef_A[0] : (i0) mod 2 = 0 };
; CHECK-NEXT:    }
;
;    void f(int *A, int N) {
;      for (int i = 0; i < N; i++) {
;        A[i % 2]++;
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %A, i32 %N) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc1, %for.inc ]
  %cmp = icmp slt i32 %i.0, %N
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %i.t = trunc i32 %i.0 to i1
  %rem = zext i1 %i.t to i32
  %idxprom = sext i32 %rem to i64
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %idxprom
  %tmp = load i32, ptr %arrayidx, align 4
  %inc = add nsw i32 %tmp, 1
  store i32 %inc, ptr %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc1 = add nuw nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
