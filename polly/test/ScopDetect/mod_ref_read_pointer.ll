; RUN: opt %loadPolly -basicaa -polly-detect -analyze \
; RUN:  -polly-allow-modref-calls < %s | FileCheck %s -check-prefix=MODREF
; RUN: opt %loadPolly -basicaa -polly-detect -analyze \
; RUN:  < %s | FileCheck %s
;
; CHECK-NOT: Valid Region for Scop: for.cond => for.end
; MODREF: Valid Region for Scop: for.cond => for.end
;
;    #pragma readonly
;    int func(int *A);
;
;    void jd(int *A) {
;      for (int i = 0; i < 1024; i++)
;        A[i + 2] = func(A);
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare i32 @func(i32* %A) #1

define void @jd(i32* %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %call = call i32 @func(i32* %A)
  %tmp = add nsw i64 %indvars.iv, 2
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %tmp
  store i32 %call, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

attributes #1 = { nounwind readonly }
