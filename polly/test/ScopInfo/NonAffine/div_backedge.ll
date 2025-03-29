; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
;
;    void foo(float *A) {
;      for (long i = 1;; i++) {
;        A[i] += 1;
;        if (i / 7 == 4)
;          break;
;      }
;    }
;
; CHECK:  Domain :=
; CHECK:    { Stmt_for_body[i0] : 0 <= i0 <= 27 };
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(ptr %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i64 [ 1, %entry ], [ %inc, %for.inc ]
  br label %for.body

for.body:                                         ; preds = %for.cond
  %arrayidx0 = getelementptr inbounds float, ptr %A, i64 %i.0
  %tmp0 = load float, ptr %arrayidx0, align 4
  %add0 = fadd float %tmp0, 2.000000e+00
  store float %add0, ptr %arrayidx0, align 4
  %rem1 = sdiv i64 %i.0, 7
  %tobool = icmp eq i64 %rem1, 4
  br i1 %tobool, label %for.end, label %if.end

if.end:                                           ; preds = %for.body, %if.then
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %inc = add nuw nsw i64 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
