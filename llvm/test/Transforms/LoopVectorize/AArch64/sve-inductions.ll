; RUN: opt -mtriple aarch64-linux-gnu -mattr=+sve -loop-vectorize -dce -instcombine < %s -S 2>%t | FileCheck %s

; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

; Test that we can add on the induction variable
;   for (long long i = 0; i < n; i++) {
;     a[i] = b[i] + i;
;   }
; with an unroll factor (interleave count) of 2.

define void @add_ind64_unrolled(i64* noalias nocapture %a, i64* noalias nocapture readonly %b, i64 %n) {
; CHECK-LABEL: @add_ind64_unrolled(
; CHECK-NEXT:  entry:
; CHECK: vector.body:
; CHECK-NEXT:  %[[INDEX:.*]] = phi i64 [ 0, %vector.ph ], [ %{{.*}}, %vector.body ]
; CHECK-NEXT:  %[[STEPVEC:.*]] = call <vscale x 2 x i64> @llvm.experimental.stepvector.nxv2i64()
; CHECK-NEXT:  %[[TMP1:.*]] = insertelement <vscale x 2 x i64> poison, i64 %[[INDEX]], i32 0
; CHECK-NEXT:  %[[IDXSPLT:.*]] = shufflevector <vscale x 2 x i64> %[[TMP1]], <vscale x 2 x i64> poison, <vscale x 2 x i32> zeroinitializer
; CHECK-NEXT:  %[[VECIND1:.*]] = add <vscale x 2 x i64> %[[IDXSPLT]], %[[STEPVEC]]
; CHECK-NEXT:  %[[VSCALE:.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:  %[[EC:.*]] = shl i64 %[[VSCALE]], 1
; CHECK-NEXT:  %[[TMP2:.*]] = insertelement <vscale x 2 x i64> poison, i64 %[[EC]], i32 0
; CHECK-NEXT:  %[[ECSPLT:.*]] = shufflevector <vscale x 2 x i64> %[[TMP2]], <vscale x 2 x i64> poison, <vscale x 2 x i32> zeroinitializer
; CHECK-NEXT:  %[[TMP3:.*]] = add <vscale x 2 x i64> %[[ECSPLT]], %[[STEPVEC]]
; CHECK-NEXT:  %[[VECIND2:.*]] = add <vscale x 2 x i64> %[[IDXSPLT]], %[[TMP3]]
; CHECK:       %[[LOAD1:.*]] = load <vscale x 2 x i64>
; CHECK:       %[[LOAD2:.*]] = load <vscale x 2 x i64>
; CHECK:       %[[STOREVAL1:.*]] = add nsw <vscale x 2 x i64> %[[LOAD1]], %[[VECIND1]]
; CHECK:       %[[STOREVAL2:.*]] = add nsw <vscale x 2 x i64> %[[LOAD2]], %[[VECIND2]]
; CHECK:       store <vscale x 2 x i64> %[[STOREVAL1]]
; CHECK:       store <vscale x 2 x i64> %[[STOREVAL2]]

entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.08 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i64, i64* %b, i64 %i.08
  %0 = load i64, i64* %arrayidx, align 8
  %add = add nsw i64 %0, %i.08
  %arrayidx1 = getelementptr inbounds i64, i64* %a, i64 %i.08
  store i64 %add, i64* %arrayidx1, align 8
  %inc = add nuw nsw i64 %i.08, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body, !llvm.loop !0

exit:                                 ; preds = %for.body
  ret void
}


; Same as above, except we test with a vectorisation factor of (1, scalable)

define void @add_ind64_unrolled_nxv1i64(i64* noalias nocapture %a, i64* noalias nocapture readonly %b, i64 %n) {
; CHECK-LABEL: @add_ind64_unrolled_nxv1i64(
; CHECK-NEXT:  entry:
; CHECK: vector.body:
; CHECK-NEXT:  %[[INDEX:.*]] = phi i64 [ 0, %vector.ph ], [ %{{.*}}, %vector.body ]
; CHECK-NEXT:  %[[STEPVEC:.*]] = call <vscale x 1 x i64> @llvm.experimental.stepvector.nxv1i64()
; CHECK-NEXT:  %[[TMP1:.*]] = insertelement <vscale x 1 x i64> poison, i64 %[[INDEX]], i32 0
; CHECK-NEXT:  %[[IDXSPLT:.*]] = shufflevector <vscale x 1 x i64> %[[TMP1]], <vscale x 1 x i64> poison, <vscale x 1 x i32> zeroinitializer
; CHECK-NEXT:  %[[VECIND1:.*]] = add <vscale x 1 x i64> %[[IDXSPLT]], %[[STEPVEC]]
; CHECK-NEXT:  %[[EC:.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:  %[[TMP2:.*]] = insertelement <vscale x 1 x i64> poison, i64 %[[EC]], i32 0
; CHECK-NEXT:  %[[ECSPLT:.*]] = shufflevector <vscale x 1 x i64> %[[TMP2]], <vscale x 1 x i64> poison, <vscale x 1 x i32> zeroinitializer
; CHECK-NEXT:  %[[TMP3:.*]] = add <vscale x 1 x i64> %[[ECSPLT]], %[[STEPVEC]]
; CHECK-NEXT:  %[[VECIND2:.*]] = add <vscale x 1 x i64> %[[IDXSPLT]], %[[TMP3]]
; CHECK:       %[[LOAD1:.*]] = load <vscale x 1 x i64>
; CHECK:       %[[LOAD2:.*]] = load <vscale x 1 x i64>
; CHECK:       %[[STOREVAL1:.*]] = add nsw <vscale x 1 x i64> %[[LOAD1]], %[[VECIND1]]
; CHECK:       %[[STOREVAL2:.*]] = add nsw <vscale x 1 x i64> %[[LOAD2]], %[[VECIND2]]
; CHECK:       store <vscale x 1 x i64> %[[STOREVAL1]]
; CHECK:       store <vscale x 1 x i64> %[[STOREVAL2]]

entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.08 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i64, i64* %b, i64 %i.08
  %0 = load i64, i64* %arrayidx, align 8
  %add = add nsw i64 %0, %i.08
  %arrayidx1 = getelementptr inbounds i64, i64* %a, i64 %i.08
  store i64 %add, i64* %arrayidx1, align 8
  %inc = add nuw nsw i64 %i.08, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body, !llvm.loop !9

exit:                                 ; preds = %for.body
  ret void
}


; Test that we can vectorize a separate induction variable (not used for the branch)
;   int r = 0;
;   for (long long i = 0; i < n; i++) {
;     a[i] = r;
;     r += 2;
;   }
; with an unroll factor (interleave count) of 1.


define void @add_unique_ind32(i32* noalias nocapture %a, i64 %n) {
; CHECK-LABEL: @add_unique_ind32(
; CHECK:    vector.ph:
; CHECK:      %[[STEPVEC:.*]] = call <vscale x 4 x i32> @llvm.experimental.stepvector.nxv4i32()
; CHECK-NEXT: %[[INDINIT:.*]] = shl <vscale x 4 x i32> %[[STEPVEC]], shufflevector (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> undef, i32 1, i32 0), <vscale x 4 x i32> undef, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT: %[[VSCALE:.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT: %[[INC:.*]] = shl i32 %[[VSCALE]], 3
; CHECK-NEXT: %[[TMP:.*]] = insertelement <vscale x 4 x i32> poison, i32 %[[INC]], i32 0
; CHECK-NEXT: %[[VECINC:.*]] = shufflevector <vscale x 4 x i32> %[[TMP]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK:    vector.body:
; CHECK:      %[[VECIND:.*]] = phi <vscale x 4 x i32> [ %[[INDINIT]], %vector.ph ], [ %[[VECINDNXT:.*]], %vector.body ]
; CHECK:      store <vscale x 4 x i32> %[[VECIND]]
; CHECK:      %[[VECINDNXT]] = add <vscale x 4 x i32> %[[VECIND]], %[[VECINC]]
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.08 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %r.07 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %i.08
  store i32 %r.07, i32* %arrayidx, align 4
  %add = add nuw nsw i32 %r.07, 2
  %inc = add nuw nsw i64 %i.08, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body, !llvm.loop !6

exit:                                 ; preds = %for.body
  ret void
}


; Test that we can vectorize a separate FP induction variable (not used for the branch)
;   float r = 0;
;   for (long long i = 0; i < n; i++) {
;     a[i] = r;
;     r += 2;
;   }
; with an unroll factor (interleave count) of 1.

define void @add_unique_indf32(float* noalias nocapture %a, i64 %n) {
; CHECK-LABEL: @add_unique_indf32(
; CHECK:    vector.ph:
; CHECK:      %[[STEPVEC:.*]] = call <vscale x 4 x i32> @llvm.experimental.stepvector.nxv4i32()
; CHECK-NEXT: %[[TMP1:.*]] = uitofp <vscale x 4 x i32> %[[STEPVEC]] to <vscale x 4 x float>
; CHECK-NEXT: %[[TMP2:.*]] = fmul <vscale x 4 x float> %[[TMP1]], shufflevector (<vscale x 4 x float> insertelement (<vscale x 4 x float> poison, float 2.000000e+00, i32 0), <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT: %[[INDINIT:.*]] = fadd <vscale x 4 x float> %[[TMP2]], shufflevector (<vscale x 4 x float> insertelement (<vscale x 4 x float> poison, float 0.000000e+00, i32 0), <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT: %[[VSCALE:.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT: %[[TMP3:.*]] = shl i32 %8, 2
; CHECK-NEXT: %[[TMP4:.*]] = sitofp i32 %[[TMP3]] to float
; CHECK-NEXT: %[[INC:.*]] = fmul float %[[TMP4]], 2.000000e+00
; CHECK-NEXT: %[[TMP5:.*]] = insertelement <vscale x 4 x float> poison, float %[[INC]], i32 0
; CHECK-NEXT: %[[VECINC:.*]] = shufflevector <vscale x 4 x float> %[[TMP5]], <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer
; CHECK:   vector.body:
; CHECK:     %[[VECIND:.*]] = phi <vscale x 4 x float> [ %[[INDINIT]], %vector.ph ], [ %[[VECINDNXT:.*]], %vector.body ]
; CHECK:     store <vscale x 4 x float> %[[VECIND]]
; CHECK:     %[[VECINDNXT]] = fadd <vscale x 4 x float> %[[VECIND]], %[[VECINC]]

entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.08 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %r.07 = phi float [ %add, %for.body ], [ 0.000000e+00, %entry ]
  %arrayidx = getelementptr inbounds float, float* %a, i64 %i.08
  store float %r.07, float* %arrayidx, align 4
  %add = fadd float %r.07, 2.000000e+00
  %inc = add nuw nsw i64 %i.08, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body, !llvm.loop !6

exit:                                 ; preds = %for.body
  ret void
}

; Test a case where the vectorised induction variable is used to
; generate a mask:
;   for (long long i = 0; i < n; i++) {
;     if (i & 0x1)
;       a[i] = b[i];
;   }

define void @cond_ind64(i32* noalias nocapture %a, i32* noalias nocapture readonly %b, i64 %n) {
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
  br i1 %exitcond.not, label %exit, label %for.body, !llvm.loop !6

exit:                                 ; preds = %for.inc
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.vectorize.width", i32 2}
!3 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
!4 = !{!"llvm.loop.interleave.count", i32 2}
!5 = !{!"llvm.loop.vectorize.enable", i1 true}
!6 = distinct !{!6, !1, !7, !3, !8, !5}
!7 = !{!"llvm.loop.vectorize.width", i32 4}
!8 = !{!"llvm.loop.interleave.count", i32 1}
!9 = distinct !{!9, !1, !10, !3, !4, !5}
!10 = !{!"llvm.loop.vectorize.width", i32 1}
