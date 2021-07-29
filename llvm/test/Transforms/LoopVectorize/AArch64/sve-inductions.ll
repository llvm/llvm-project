; RUN: opt -loop-vectorize -scalable-vectorization=on -force-target-instruction-cost=1 -dce -instcombine < %s -S | FileCheck %s

target triple = "aarch64-linux-gnu"

; Test a case where the vectorised induction variable is used to
; generate a mask:
;   for (long long i = 0; i < n; i++) {
;     if (i & 0x1)
;       a[i] = b[i];
;   }

define void @cond_ind64(i32* noalias nocapture %a, i32* noalias nocapture readonly %b, i64 %n) #0 {
; CHECK-LABEL: @cond_ind64(
; CHECK:    vector.body:
; CHECK-NEXT: %[[INDEX:.*]] = phi i64 [ 0, %vector.ph ], [ %{{.*}}, %vector.body ]
; CHECK:      %[[STEPVEC:.*]] = call <vscale x 4 x i64> @llvm.experimental.stepvector.nxv4i64()
; CHECK-NEXT: %[[TMP1:.*]] = insertelement <vscale x 4 x i64> poison, i64 %[[INDEX]], i32 0
; CHECK-NEXT: %[[IDXSPLT:.*]] = shufflevector <vscale x 4 x i64> %[[TMP1]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT: %[[VECIND:.*]] = add <vscale x 4 x i64> %[[IDXSPLT]], %[[STEPVEC]]
; CHECK-NEXT: %[[MASK:.*]] = trunc <vscale x 4 x i64> %[[VECIND]] to <vscale x 4 x i1>
; CHECK:      %[[LOAD:.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>* %{{.*}}, i32 4, <vscale x 4 x i1> %[[MASK]], <vscale x 4 x i32> poison)
; CHECK:      call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> %[[LOAD]], <vscale x 4 x i32>* %{{.*}}, i32 4, <vscale x 4 x i1> %[[MASK]])
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %i.08 = phi i64 [ %inc, %for.inc ], [ 0, %entry ]
  %and = and i64 %i.08, 1
  %tobool.not = icmp eq i64 %and, 0
  br i1 %tobool.not, label %for.inc, label %if.then

if.then:                                          ; preds = %for.body
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %i.08
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %a, i64 %i.08
  store i32 %0, i32* %arrayidx1, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %inc = add nuw nsw i64 %i.08, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body, !llvm.loop !0

exit:                                 ; preds = %for.inc
  ret void
}

attributes #0 = { "target-features"="+sve" }

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
!3 = !{!"llvm.loop.vectorize.enable", i1 true}
!4 = !{!"llvm.loop.vectorize.width", i32 4}
!5 = !{!"llvm.loop.interleave.count", i32 1}
