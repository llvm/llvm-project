; RUN: opt -mtriple aarch64-linux-gnu -mattr=+sve -loop-vectorize -dce -instcombine -S < %s | FileCheck %s

; Ensure that we can vectorize loops such as:
;   int *ptr = c;
;   for (long long i = 0; i < n; i++) {
;     int X1 = *ptr++;
;     int X2 = *ptr++;
;     a[i] = X1 + 1;
;     b[i] = X2 + 1;
;   }
; with scalable vectors, including unrolling. The test below makes sure
; that we can use gather instructions with the correct offsets, taking
; vscale into account.

define void @widen_ptr_phi_unrolled(i32* noalias nocapture %a, i32* noalias nocapture %b, i32* nocapture readonly %c, i64 %n) {
; CHECK-LABEL: @widen_ptr_phi_unrolled(
; CHECK:       vector.body:
; CHECK-NEXT:    [[POINTER_PHI:%.*]] = phi i32* [ %c, %vector.ph ], [ %[[PTR_IND:.*]], %vector.body ]
; CHECK:         [[TMP5:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP6:%.*]] = shl i64 [[TMP5]], 2
; CHECK-NEXT:    [[TMP7:%.*]] = shl i64 [[TMP5]], 4
; CHECK-NEXT:    [[TMP8:%.*]] = call <vscale x 4 x i64> @llvm.experimental.stepvector.nxv4i64()
; CHECK-NEXT:    [[VECTOR_GEP:%.*]] = shl <vscale x 4 x i64> [[TMP8]], shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> undef, i64 1, i32 0), <vscale x 4 x i64> undef, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[TMP9:%.*]] = getelementptr i32, i32* [[POINTER_PHI]], <vscale x 4 x i64> [[VECTOR_GEP]]
; CHECK-NEXT:    [[DOTSPLATINSERT2:%.*]] = insertelement <vscale x 4 x i64> poison, i64 [[TMP6]], i32 0
; CHECK-NEXT:    [[DOTSPLAT3:%.*]] = shufflevector <vscale x 4 x i64> [[DOTSPLATINSERT2]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP10:%.*]] = call <vscale x 4 x i64> @llvm.experimental.stepvector.nxv4i64()
; CHECK-NEXT:    [[TMP11:%.*]] = add <vscale x 4 x i64> [[DOTSPLAT3]], [[TMP10]]
; CHECK-NEXT:    [[VECTOR_GEP4:%.*]] = shl <vscale x 4 x i64> [[TMP11]], shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> undef, i64 1, i32 0), <vscale x 4 x i64> undef, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[TMP12:%.*]] = getelementptr i32, i32* [[POINTER_PHI]], <vscale x 4 x i64> [[VECTOR_GEP4]]
; CHECK-NEXT:    [[TMP13:%.*]] = getelementptr inbounds i32, <vscale x 4 x i32*> [[TMP9]], i64 1
; CHECK-NEXT:    [[TMP14:%.*]] = getelementptr inbounds i32, <vscale x 4 x i32*> [[TMP12]], i64 1
; CHECK-NEXT:    {{%.*}} = call <vscale x 4 x i32> @llvm.masked.gather.nxv4i32.nxv4p0i32(<vscale x 4 x i32*> [[TMP9]],
; CHECK-NEXT:    {{%.*}} = call <vscale x 4 x i32> @llvm.masked.gather.nxv4i32.nxv4p0i32(<vscale x 4 x i32*> [[TMP12]],
; CHECK-NEXT:    {{%.*}} = call <vscale x 4 x i32> @llvm.masked.gather.nxv4i32.nxv4p0i32(<vscale x 4 x i32*> [[TMP13]],
; CHECK-NEXT:    {{%.*}} = call <vscale x 4 x i32> @llvm.masked.gather.nxv4i32.nxv4p0i32(<vscale x 4 x i32*> [[TMP14]],
; CHECK:         [[PTR_IND]] = getelementptr i32, i32* [[POINTER_PHI]], i64 [[TMP7]]
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %ptr.014 = phi i32* [ %incdec.ptr1, %for.body ], [ %c, %entry ]
  %i.013 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %incdec.ptr = getelementptr inbounds i32, i32* %ptr.014, i64 1
  %0 = load i32, i32* %ptr.014, align 4
  %incdec.ptr1 = getelementptr inbounds i32, i32* %ptr.014, i64 2
  %1 = load i32, i32* %incdec.ptr, align 4
  %add = add nsw i32 %0, 1
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %i.013
  store i32 %add, i32* %arrayidx, align 4
  %add2 = add nsw i32 %1, 1
  %arrayidx3 = getelementptr inbounds i32, i32* %b, i64 %i.013
  store i32 %add2, i32* %arrayidx3, align 4
  %inc = add nuw nsw i64 %i.013, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %for.exit, label %for.body, !llvm.loop !0

for.exit:                                 ; preds = %for.body
  ret void
}


; Ensure we can vectorise loops without interleaving, e.g.:
;   int *D = dst;
;   int *S = src;
;   for (long long i = 0; i < n; i++) {
;     *D = *S * 2;
;     D++;
;     S++;
;   }
; This takes us down a different codepath to the test above, where
; here we treat the PHIs as being uniform.

define void @widen_2ptrs_phi_unrolled(i32* noalias nocapture %dst, i32* noalias nocapture readonly %src, i64 %n) #0 {
; CHECK-LABEL: @widen_2ptrs_phi_unrolled(
; CHECK:       vector.body:
; CHECK-NEXT:    %[[IDX:.*]] = phi i64 [ 0, %vector.ph ], [ %{{.*}}, %vector.body ]
; CHECK-NEXT:    %[[LGEP1:.*]] = getelementptr i32, i32* %src, i64 %[[IDX]]
; CHECK-NEXT:    %[[SGEP1:.*]] = getelementptr i32, i32* %dst, i64 %[[IDX]]
; CHECK-NEXT:    %[[LPTR1:.*]] = bitcast i32* %[[LGEP1]] to <vscale x 4 x i32>*
; CHECK-NEXT:    %{{.*}} = load <vscale x 4 x i32>, <vscale x 4 x i32>* %[[LPTR1]], align 4
; CHECK-NEXT:    %[[VSCALE1:.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT:    %[[TMP1:.*]] = shl i32 %[[VSCALE1]], 2
; CHECK-NEXT:    %[[TMP2:.*]] = sext i32 %[[TMP1]] to i64
; CHECK-NEXT:    %[[LGEP2:.*]] = getelementptr i32, i32* %[[LGEP1]], i64 %[[TMP2]]
; CHECK-NEXT:    %[[LPTR2:.*]] = bitcast i32* %[[LGEP2]] to <vscale x 4 x i32>*
; CHECK-NEXT:    %{{.*}} = load <vscale x 4 x i32>, <vscale x 4 x i32>* %[[LPTR2]], align 4
; CHECK:         %[[SPTR1:.*]] = bitcast i32* %[[SGEP1]] to <vscale x 4 x i32>*
; CHECK-NEXT:    store <vscale x 4 x i32> %{{.*}}, <vscale x 4 x i32>* %[[SPTR1]], align 4
; CHECK-NEXT:    %[[VSCALE2:.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT:    %[[TMP3:.*]] = shl i32 %[[VSCALE2]], 2
; CHECK-NEXT:    %[[TMP4:.*]] = sext i32 %[[TMP3]] to i64
; CHECK-NEXT:    %[[SGEP2:.*]] = getelementptr i32, i32* %[[SGEP1]], i64 %[[TMP4]]
; CHECK-NEXT:    %[[SPTR2:.*]] = bitcast i32* %[[SGEP2]] to <vscale x 4 x i32>*
; CHECK-NEXT:    store <vscale x 4 x i32> %{{.*}}, <vscale x 4 x i32>* %[[SPTR2]], align 4

entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.011 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %S.010 = phi i32* [ %incdec.ptr1, %for.body ], [ %src, %entry ]
  %D.09 = phi i32* [ %incdec.ptr, %for.body ], [ %dst, %entry ]
  %0 = load i32, i32* %S.010, align 4
  %mul = shl nsw i32 %0, 1
  store i32 %mul, i32* %D.09, align 4
  %incdec.ptr = getelementptr inbounds i32, i32* %D.09, i64 1
  %incdec.ptr1 = getelementptr inbounds i32, i32* %S.010, i64 1
  %inc = add nuw nsw i64 %i.011, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !0

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.vectorize.width", i32 4}
!3 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
!4 = !{!"llvm.loop.vectorize.enable", i1 true}
!5 = !{!"llvm.loop.interleave.count", i32 2}
