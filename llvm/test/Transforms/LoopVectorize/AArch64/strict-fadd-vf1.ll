; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -force-ordered-reductions=true -force-vector-width=1 -S < %s -debug 2> %t.debug | FileCheck %s
; RUN: cat %t.debug | FileCheck %s --check-prefix=CHECK-DEBUG

target triple = "aarch64-unknown-linux-gnu"

; CHECK-DEBUG: LV: Not interleaving scalar ordered reductions.

define void @foo(ptr noalias nocapture %dst, ptr noalias nocapture readonly %src, i64 %M, i64 %N) {
; CHECK-LABEL: @foo(
; CHECK-NOT: vector.body

entry:
  br label %for.body.us

for.body.us:                                      ; preds = %entry, %for.cond3
  %i.023.us = phi i64 [ %inc8.us, %for.cond3 ], [ 0, %entry ]
  %arrayidx.us = getelementptr inbounds float, ptr %dst, i64 %i.023.us
  %mul.us = mul nsw i64 %i.023.us, %N
  br label %for.body3.us

for.body3.us:                                     ; preds = %for.body.us, %for.body3.us
  %0 = phi float [ 0.000000e+00, %for.body.us ], [ %add6.us, %for.body3.us ]
  %j.021.us = phi i64 [ 0, %for.body.us ], [ %inc.us, %for.body3.us ]
  %add.us = add nsw i64 %j.021.us, %mul.us
  %arrayidx4.us = getelementptr inbounds float, ptr %src, i64 %add.us
  %1 = load float, ptr %arrayidx4.us, align 4
  %add6.us = fadd float %1, %0
  %inc.us = add nuw nsw i64 %j.021.us, 1
  %exitcond.not = icmp eq i64 %inc.us, %N
  br i1 %exitcond.not, label %for.cond3, label %for.body3.us

for.cond3:                                        ; preds = %for.body3.us
  %add6.us.lcssa = phi float [ %add6.us, %for.body3.us ]
  store float %add6.us.lcssa, ptr %arrayidx.us, align 4
  %inc8.us = add nuw nsw i64 %i.023.us, 1
  %exitcond26.not = icmp eq i64 %inc8.us, %M
  br i1 %exitcond26.not, label %exit, label %for.body.us

exit:                                             ; preds = %for.cond3
  ret void
}
