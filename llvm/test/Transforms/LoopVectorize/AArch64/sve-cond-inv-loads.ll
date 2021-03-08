; RUN: opt -loop-vectorize -dce -instcombine -mtriple aarch64-linux-gnu -mattr=+sve -S %s -o - | FileCheck %s

define void @cond_inv_load_i32i32i16(i32* noalias nocapture %a, i32* noalias nocapture readonly %cond, i16* noalias nocapture readonly %inv, i64 %n) {
; CHECK-LABEL: @cond_inv_load_i32i32i16
; CHECK:     vector.ph:
; CHECK:       %[[INVINS:.*]] = insertelement <vscale x 4 x i16*> poison, i16* %inv, i32 0
; CHECK:       %[[INVSPLAT:.*]] = shufflevector <vscale x 4 x i16*> %[[INVINS]], <vscale x 4 x i16*> poison, <vscale x 4 x i32> zeroinitializer
; CHECK:     vector.body:
; CHECK:       %[[GEPCOND:.*]] = getelementptr inbounds i32, i32* %cond, i64 %index
; CHECK-NEXT:  %[[GEPCOND2:.*]] = bitcast i32* %[[GEPCOND]] to <vscale x 4 x i32>*
; CHECK-NEXT:  %[[CONDVALS:.*]] = load <vscale x 4 x i32>, <vscale x 4 x i32>* %[[GEPCOND2]], align 4
; CHECK-NEXT:  %[[MASK:.*]] = icmp ne <vscale x 4 x i32> %[[CONDVALS]],
; CHECK-NEXT:  %[[GATHERLOAD:.*]] = call <vscale x 4 x i16> @llvm.masked.gather.nxv4i16.nxv4p0i16(<vscale x 4 x i16*> %[[INVSPLAT]], i32 2, <vscale x 4 x i1> %[[MASK]], <vscale x 4 x i16> undef)
; CHECK-NEXT:  %[[GATHERLOAD2:.*]] = sext <vscale x 4 x i16> %[[GATHERLOAD]] to <vscale x 4 x i32>
; CHECK:       call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> %[[GATHERLOAD2]]
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %i.07 = phi i64 [ %inc, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %cond, i64 %i.07
  %0 = load i32, i32* %arrayidx, align 4
  %tobool.not = icmp eq i32 %0, 0
  br i1 %tobool.not, label %for.inc, label %if.then

if.then:                                          ; preds = %for.body
  %1 = load i16, i16* %inv, align 2
  %conv = sext i16 %1 to i32
  %arrayidx1 = getelementptr inbounds i32, i32* %a, i64 %i.07
  store i32 %conv, i32* %arrayidx1, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %inc = add nuw nsw i64 %i.07, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body, !llvm.loop !0

exit:                        ; preds = %for.inc
  ret void
}

define void @cond_inv_load_f64f64f64(double* noalias nocapture %a, double* noalias nocapture readonly %cond, double* noalias nocapture readonly %inv, i64 %n) {
; CHECK-LABEL: @cond_inv_load_f64f64f64
; CHECK:     vector.ph:
; CHECK:       %[[INVINS:.*]] = insertelement <vscale x 4 x double*> poison, double* %inv, i32 0
; CHECK:       %[[INVSPLAT:.*]] = shufflevector <vscale x 4 x double*> %[[INVINS]], <vscale x 4 x double*> poison, <vscale x 4 x i32> zeroinitializer
; CHECK:     vector.body:
; CHECK:       %[[GEPCOND:.*]] = getelementptr inbounds double, double* %cond, i64 %index
; CHECK-NEXT:  %[[GEPCOND2:.*]] = bitcast double* %[[GEPCOND]] to <vscale x 4 x double>*
; CHECK-NEXT:  %[[CONDVALS:.*]] = load <vscale x 4 x double>, <vscale x 4 x double>* %[[GEPCOND2]], align 8
; CHECK-NEXT:  %[[MASK:.*]] = fcmp ogt <vscale x 4 x double> %[[CONDVALS]],
; CHECK-NEXT:  %[[GATHERLOAD:.*]] = call <vscale x 4 x double> @llvm.masked.gather.nxv4f64.nxv4p0f64(<vscale x 4 x double*> %[[INVSPLAT]], i32 8, <vscale x 4 x i1> %[[MASK]], <vscale x 4 x double> undef)
; CHECK:       call void @llvm.masked.store.nxv4f64.p0nxv4f64(<vscale x 4 x double> %[[GATHERLOAD]]
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.inc
  %i.08 = phi i64 [ %inc, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double, double* %cond, i64 %i.08
  %0 = load double, double* %arrayidx, align 8
  %cmp1 = fcmp ogt double %0, 4.000000e-01
  br i1 %cmp1, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %1 = load double, double* %inv, align 8
  %arrayidx2 = getelementptr inbounds double, double* %a, i64 %i.08
  store double %1, double* %arrayidx2, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %inc = add nuw nsw i64 %i.08, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body, !llvm.loop !0

exit:                        ; preds = %for.inc
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.vectorize.width", i32 4}
!3 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
!4 = !{!"llvm.loop.interleave.count", i32 1}
!5 = !{!"llvm.loop.vectorize.enable", i1 true}
