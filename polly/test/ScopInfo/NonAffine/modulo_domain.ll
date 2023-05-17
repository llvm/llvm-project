; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
; TODO: The new domain generation cannot handle modulo domain constraints,
;       hence modulo handling has been disabled completely. Once this is
;       resolved this test should work again. Until then we approximate the
;       whole loop body.
;
; CHECK:   Domain :=
; CHECK:       { Stmt_for_body[i0] : 0 <= i0 <= 15 };
;
;    void foo(float *A) {
;      for (long i = 0; i < 16; i++) {
;        A[i] += 1;
;        if (i % 2)
;          A[i] += 2;
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(ptr %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i64 %i.0, 16
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx0 = getelementptr inbounds float, ptr %A, i64 %i.0
  %tmp0 = load float, ptr %arrayidx0, align 4
  %add0 = fadd float %tmp0, 2.000000e+00
  store float %add0, ptr %arrayidx0, align 4
  %rem1 = srem i64 %i.0, 2
  %tobool = icmp eq i64 %rem1, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %for.body
  %arrayidx = getelementptr inbounds float, ptr %A, i64 %i.0
  %tmp = load float, ptr %arrayidx, align 4
  %add = fadd float %tmp, 2.000000e+00
  store float %add, ptr %arrayidx, align 4
  br label %if.end

if.end:                                           ; preds = %for.body, %if.then
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %inc = add nuw nsw i64 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
