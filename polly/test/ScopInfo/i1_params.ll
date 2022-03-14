; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
; Check that both a signed as well as an unsigned extended i1 parameter
; is represented correctly.
;
;    void f(signed i1 p0, unsigned i1 p1, int *A) {
;      for (int i = 0; i < 100; i++)
;        A[i + p0] = A[i + p1];
;    }
;
; CHECK:       Context:
; CHECK-NEXT:    [p1, p0] -> {  : -1 <= p1 <= 0 and -1 <= p0 <= 0 }
;
; CHECK:       ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:    [p1, p0] -> { Stmt_for_body[i0] -> MemRef_A[1 + i0] : p1 = -1; Stmt_for_body[i0] -> MemRef_A[i0] : p1 = 0 };
; CHECK-NEXT:  MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:    [p1, p0] -> { Stmt_for_body[i0] -> MemRef_A[p0 + i0] };
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i1 %p0, i1 %p1, i32* %A) {
entry:
  %tmp4 = sext i1 %p0 to i64
  %tmp = zext i1 %p1 to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 100
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp5 = add nsw i64 %indvars.iv, %tmp
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %tmp5
  %tmp6 = load i32, i32* %arrayidx, align 4
  %tmp7 = add nsw i64 %indvars.iv, %tmp4
  %arrayidx3 = getelementptr inbounds i32, i32* %A, i64 %tmp7
  store i32 %tmp6, i32* %arrayidx3, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
