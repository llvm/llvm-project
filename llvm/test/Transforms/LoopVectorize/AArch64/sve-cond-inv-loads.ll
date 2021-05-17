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

define void @invariant_load_cond(i32* noalias nocapture %a, i32* nocapture readonly %b, i32* nocapture readonly %cond, i64 %n) {
; CHECK-LABEL: @invariant_load_cond
; CHECK: vector.body
; CHECK: %[[GEP:.*]] = getelementptr inbounds i32, i32* %b, i64 42
; CHECK-NEXT: %[[SPLATINS:.*]] = insertelement <vscale x 4 x i32*> poison, i32* %[[GEP]], i32 0
; CHECK-NEXT: %[[SPLAT:.*]] = shufflevector <vscale x 4 x i32*> %[[SPLATINS]], <vscale x 4 x i32*> poison, <vscale x 4 x i32> zeroinitializer
; CHECK: %[[LOAD:.*]] = load <vscale x 4 x i32>, <vscale x 4 x i32>*
; CHECK-NEXT: %[[ICMP:.*]] = icmp ne <vscale x 4 x i32> %[[LOAD]], shufflevector (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> poison, i32 0, i32 0), <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK: %[[MASKED_LOAD:.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>* %[[BITCAST:.*]], i32 4, <vscale x 4 x i1> %[[ICMP]], <vscale x 4 x i32> poison)
; CHECK-NEXT: %[[MASKED_GATHER:.*]] = call <vscale x 4 x i32> @llvm.masked.gather.nxv4i32.nxv4p0i32(<vscale x 4 x i32*> %[[SPLAT]], i32 4, <vscale x 4 x i1> %[[ICMP]], <vscale x 4 x i32> undef)
; CHECK-NEXT: %[[ADD:.*]] = add nsw <vscale x 4 x i32> %[[MASKED_GATHER]], %[[MASKED_LOAD]]
; CHECK: call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> %[[ADD]], <vscale x 4 x i32>* %[[BITCAST1:.*]], i32 4, <vscale x 4 x i1> %[[ICMP]])
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %arrayidx1 = getelementptr inbounds i32, i32* %b, i64 42
  %arrayidx2 = getelementptr inbounds i32, i32* %cond, i64 %iv
  %0 = load i32, i32* %arrayidx2, align 4
  %tobool.not = icmp eq i32 %0, 0
  br i1 %tobool.not, label %for.inc, label %if.then

if.then:
  %arrayidx3 = getelementptr inbounds i32, i32* %b, i64 %iv
  %1 = load i32, i32* %arrayidx3, align 4
  %2 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %2, %1
  %arrayidx4 = getelementptr inbounds i32, i32* %a, i64 %iv
  store i32 %add, i32* %arrayidx4, align 4
  br label %for.inc

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.vectorize.width", i32 4}
!3 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
!4 = !{!"llvm.loop.interleave.count", i32 1}
!5 = !{!"llvm.loop.vectorize.enable", i1 true}
