; RUN: opt %loadNPMPolly -aa-pipeline=basic-aa -polly-allow-modref-calls '-passes=print<polly-detect>' -disable-output < %s 2>&1 | FileCheck %s -check-prefix=MODREF
; RUN: opt %loadNPMPolly -aa-pipeline=basic-aa                           '-passes=print<polly-detect>' -disable-output < %s 2>&1 | FileCheck %s
;
; CHECK-NOT: Valid Region for Scop: for.body => for.end
; MODREF: Valid Region for Scop: for.body => for.end
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

declare i32 @func(ptr %A) #1

define void @jd(ptr %A) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.inc ]
  %call = call i32 @func(ptr %A)
  %tmp = add nsw i64 %i, 2
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %tmp
  store i32 %call, ptr %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %i.next = add nuw nsw i64 %i, 1
  %exitcond = icmp ne i64 %i.next, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.inc
  ret void
}

attributes #1 = { nounwind readonly }
