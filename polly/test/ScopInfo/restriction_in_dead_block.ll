; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
; Verify we do not generate an empty invalid context only because the wrap
; in the second conditional will always happen if the block is executed.
;
; CHECK:       Invalid Context:
; CHECK-NEXT:    [N] -> {  : N > 0 }
;
;    void f(char *A, char N) {
;      for (char i = 0; i < 10; i++) {
;        if (N > 0)
;          if (1 + 127 * N > 0)
;            A[i] = 1;
;        A[i] = 0;
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i8* %A, i8 signext %N) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i8 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i8 %indvars.iv, 10
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %cmp3 = icmp sgt i8 %N, 0
  br i1 %cmp3, label %if.then, label %if.end10

if.then:                                           ; preds = %for.body
  %mul = mul i8 %N, 127
  %add = add i8 1, %mul
  %cmp7 = icmp sgt i8 %add, 0
  br i1 %cmp7, label %if.then9, label %if.end10

if.then9:                                         ; preds = %if.end
  %arrayidx = getelementptr inbounds i8, i8* %A, i8 %indvars.iv
  store i8 1, i8* %arrayidx, align 1
  br label %if.end10

if.end10:                                         ; preds = %if.then9, %if.end
  %arrayidx12 = getelementptr inbounds i8, i8* %A, i8 %indvars.iv
  store i8 0, i8* %arrayidx12, align 1
  br label %for.inc

for.inc:                                          ; preds = %if.end10, %if.then
  %indvars.iv.next = add nuw nsw i8 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
