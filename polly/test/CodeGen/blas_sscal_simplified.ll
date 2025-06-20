; RUN: opt %loadNPMPolly -passes=polly-codegen < %s
;
; Regression test for a bug in the runtime check generation.

; This was extracted from the blas testcase. It crashed in one
; part of the runtime check generation at some point. To be
; precise, we couldn't find a suitable block to put the RTC code in.
;
; int sscal(int n, float sa, float *sx) {
;   for(int i=0; i<n; i++, sx++)
;     *sx *= sa;
;   return 0;
; }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define i32 @sscal(i32 %n, float %sa, ptr %sx) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %cmp1 = icmp sgt i32 %n, 0
  br i1 %cmp1, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry.split
  %0 = zext i32 %n to i64
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %indvar = phi i64 [ 0, %for.body.lr.ph ], [ %indvar.next, %for.body ]
  %sx.addr.02 = getelementptr float, ptr %sx, i64 %indvar
  %tmp = load float, ptr %sx.addr.02, align 4
  %mul = fmul float %tmp, %sa
  store float %mul, ptr %sx.addr.02, align 4
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp ne i64 %indvar.next, %0
  br i1 %exitcond, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry.split
  ret i32 0
}
