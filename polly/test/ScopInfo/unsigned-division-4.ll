; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
;    void f(int *A, unsigned char N) {
;      for (unsigned i = 0; i < (N / -128) + 3; i++)
;        A[i]++;
;    }
;
; CHECK:         Assumed Context:
; CHECK-NEXT:    [N] -> {  :  }
; CHECK-NEXT:    Invalid Context:
; CHECK-NEXT:    [N] -> {  : N < 0 }
;
; CHECK:       Domain :=
; CHECK-NEXT:    [N] -> { Stmt_for_body[i0] : i0 >= 0 and 128i0 <= 256 + N };
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A, i8 %N) {
entry:
  %tmp1 = udiv i8 %N, -128
  %tmp = add i8 %tmp1, 3
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %lftr.wideiv = trunc i64 %indvars.iv to i8
  %exitcond = icmp ne i8 %lftr.wideiv, %tmp
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp2 = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %tmp2, 1
  store i32 %inc, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
