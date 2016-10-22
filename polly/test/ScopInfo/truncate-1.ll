; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
;    void f(char *A, short N) {
;      for (char i = 0; i < (char)N; i++)
;        A[i]++;
;    }
;
; FIXME: We should the truncate precisely... or just make it a separate parameter.
; CHECK:       Assumed Context:
; CHECK-NEXT:  [N] -> {  :  }
; CHECK-NEXT:  Invalid Context:
; CHECK-NEXT:  [N] -> { : N <= -129 or N >= 128 }
;
; CHECK:       Domain :=
; CHECK-NEXT:    [N] -> { Stmt_for_body[i0] : 0 <= i0 < N };
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i8* %A, i16 signext %N) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i8 [ 0, %entry ], [ %inc4, %for.inc ]
  %conv = sext i8 %i.0 to i32
  %conv1 = zext i16 %N to i32
  %sext = shl i32 %conv1, 24
  %conv2 = ashr exact i32 %sext, 24
  %cmp = icmp slt i32 %conv, %conv2
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %idxprom = sext i8 %i.0 to i64
  %arrayidx = getelementptr inbounds i8, i8* %A, i64 %idxprom
  %tmp = load i8, i8* %arrayidx, align 1
  %inc = add i8 %tmp, 1
  store i8 %inc, i8* %arrayidx, align 1
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc4 = add nsw i8 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
