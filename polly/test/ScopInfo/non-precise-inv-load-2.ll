; RUN: opt %loadPolly -polly-print-scops -polly-invariant-load-hoisting=true -disable-output < %s | FileCheck %s
;
;
; CHECK:       Invariant Accesses: {
; CHECK-NEXT:    ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:    [c] -> { Stmt_for_body[i0] -> MemRef_I[-1 + c] };
; CHECK-NEXT:    Execution Context: [c] -> {  : c > 0 }
; CHECK-NEXT:  }
; CHECK-NEXT:  Context:
; CHECK-NEXT:  [c] -> {  : -128 <= c <= 127 }
; CHECK-NEXT:  Assumed Context:
; CHECK-NEXT:  [c] -> {  :  }
; CHECK-NEXT:  Invalid Context:
; CHECK-NEXT:  [c] -> {  : c <= 0 }
;
;    void f(int *A, int *I, unsigned char c) {
;      for (int i = 0; i < 10; i++)
;        A[i] += I[c - (char)1];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %A, ptr %I, i8 zeroext %c) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 10
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %sub = add i8 %c, -1
  %conv = zext i8 %sub to i64
  %arrayidx = getelementptr inbounds i32, ptr %I, i64 %conv
  %tmp = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp1 = load i32, ptr %arrayidx2, align 4
  %add = add nsw i32 %tmp1, %tmp
  store i32 %add, ptr %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
