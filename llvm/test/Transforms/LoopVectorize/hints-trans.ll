; RUN: opt -S -passes=loop-vectorize,instsimplify,simplifycfg -force-vector-interleave=1 -force-vector-width=4 -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck %s
; Note: -passes=instsimplify,simplifycfg -simplifycfg-require-and-preserve-domtree=1 remove the (now dead) original loop, making
; it easy to test that the llvm.loop.unroll.disable hint is still present.
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: norecurse nounwind uwtable
define void @foo(ptr nocapture %b) #0 {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  store i32 1, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 16
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !llvm.loop !0
}

; CHECK-LABEL: @foo
; CHECK: = !{!"llvm.loop.unroll.disable"}

attributes #0 = { norecurse nounwind uwtable }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
