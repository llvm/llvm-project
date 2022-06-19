; RUN: opt %loadPolly -polly-invariant-load-hoisting=true -polly-print-scops -disable-output < %s | FileCheck %s
;
;    void f(int *A, unsigned N) {
;      for (unsigned i = 0; i < N; i++)
;        A[i / 3] = A[5 * N / 3];
;    }
;
; CHECK:         Invariant Accesses: {
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                [N] -> { Stmt_for_body[i0] -> MemRef_A[o0] : -2 + 5N <= 3o0 <= 5N };
; CHECK-NEXT:            Execution Context: [N] -> {  : 0 < N <= 1383505805528216371 }
; CHECK-NEXT:    }
;
; CHECK:         Assumed Context:
; CHECK-NEXT:    [N] -> {  :  }
; CHECK-NEXT:    Invalid Context:
; CHECK-NEXT:    [N] -> { : N >= 1383505805528216372 }

; CHECK:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:        [N] -> { Stmt_for_body[i0] -> MemRef_A[o0] : -2 + i0 <= 3o0 <= i0 };

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A, i64 %N) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, %N
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %mul = mul nsw i64 %N, 5
  %div2 = udiv i64 %mul, 3
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %div2
  %load = load i32, i32* %arrayidx2, align 4
  %div = udiv i64 %indvars.iv, 3
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %div
  store i32 %load, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
