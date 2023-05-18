; RUN: opt %loadPolly -S -polly-codegen < %s
;
; Excerpt from the test-suite's oggenc reduced using bugpoint.
;
; It features a SCEV value using %div44 for the inner loop (for.body.51 =>
; for.cond.60.preheader) that is computed within the body of the outer loop
; (for.cond.30.preheader => for.cond.60.preheader). CodeGenerator would add a
; computation of the SCEV to before the scop that references %div44, which is
; not available then.
;
; CHECK:      polly.split_new_and_old:
; CHECK-NEXT:   %div23.neg.polly.copy = sdiv i64 0, -4
;
target triple = "x86_64-unknown-linux-gnu"

define void @_vorbis_apply_window(ptr %d) {
entry:
  %0 = load ptr, ptr undef, align 8
  %div23.neg = sdiv i64 0, -4
  %sub24 = add i64 0, %div23.neg
  br label %for.cond.30.preheader

for.cond.30.preheader:                            ; preds = %for.body, %entry
  %sext = shl i64 %sub24, 32
  %conv48.74 = ashr exact i64 %sext, 32
  %cmp49.75 = icmp slt i64 %conv48.74, 0
  br i1 %cmp49.75, label %for.body.51.lr.ph, label %for.cond.60.preheader

for.body.51.lr.ph:                                ; preds = %for.cond.30.preheader
  %div44 = sdiv i64 0, 2
  %sub45 = add nsw i64 %div44, 4294967295
  %1 = trunc i64 %sub45 to i32
  %2 = sext i32 %1 to i64
  br label %for.body.51

for.cond.60.preheader:                            ; preds = %for.body.51, %for.cond.30.preheader
  ret void

for.body.51:                                      ; preds = %for.body.51, %for.body.51.lr.ph
  %indvars.iv86 = phi i64 [ %2, %for.body.51.lr.ph ], [ undef, %for.body.51 ]
  %arrayidx53 = getelementptr inbounds float, ptr %0, i64 %indvars.iv86
  %3 = load float, ptr %arrayidx53, align 4
  %mul56 = fmul float %3, undef
  store float %mul56, ptr %d, align 4
  br i1 false, label %for.body.51, label %for.cond.60.preheader
}
