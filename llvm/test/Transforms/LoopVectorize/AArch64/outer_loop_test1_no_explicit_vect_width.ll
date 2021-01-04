; RUN: opt -S -loop-vectorize -enable-vplan-native-path -mtriple aarch64-gnu-linux < %s | FileCheck %s

; extern int arr[8][8];
; extern int arr2[8];
;
; void foo(int n)
; {
;   int i1, i2;
;
; #pragma clang loop vectorize(enable)
;   for (i1 = 0; i1 < 8; i1++) {
;     arr2[i1] = i1;
;     for (i2 = 0; i2 < 8; i2++)
;       arr[i2][i1] = i1 + n;
;   }
; }
;

; CHECK-LABEL: @foo_i32(
; CHECK-LABEL: vector.ph:
; CHECK: %[[SplatVal:.*]] = insertelement <4 x i32> poison, i32 %n, i32 0
; CHECK: %[[Splat:.*]] = shufflevector <4 x i32> %[[SplatVal]], <4 x i32> poison, <4 x i32> zeroinitializer

; CHECK-LABEL: vector.body:
; CHECK: %[[Ind:.*]] = phi i64 [ 0, %vector.ph ], [ %[[IndNext:.*]], %[[ForInc:.*]] ]
; CHECK: %[[VecInd:.*]] = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, %vector.ph ], [ %[[VecIndNext:.*]], %[[ForInc]] ]
; CHECK: %[[AAddr:.*]] = getelementptr inbounds [8 x i32], [8 x i32]* @arr2, i64 0, <4 x i64> %[[VecInd]]
; CHECK: %[[VecIndTr:.*]] = trunc <4 x i64> %[[VecInd]] to <4 x i32>
; CHECK: call void @llvm.masked.scatter.v4i32.v4p0i32(<4 x i32> %[[VecIndTr]], <4 x i32*> %[[AAddr]], i32 4, <4 x i1> <i1 true, i1 true, i1 true, i1 true>)
; CHECK: %[[VecIndTr2:.*]] = trunc <4 x i64> %[[VecInd]] to <4 x i32>
; CHECK: %[[StoreVal:.*]] = add nsw <4 x i32> %[[VecIndTr2]], %[[Splat]]
; CHECK: br label %[[InnerLoop:.+]]

; CHECK: [[InnerLoop]]:
; CHECK: %[[InnerPhi:.*]] = phi <4 x i64> [ %[[InnerPhiNext:.*]], %[[InnerLoop]] ], [ zeroinitializer, %vector.body ]
; CHECK: %[[AAddr2:.*]] = getelementptr inbounds [8 x [8 x i32]], [8 x [8 x i32]]* @arr, i64 0, <4 x i64> %[[InnerPhi]], <4 x i64> %[[VecInd]]
; CHECK: call void @llvm.masked.scatter.v4i32.v4p0i32(<4 x i32> %[[StoreVal]], <4 x i32*> %[[AAddr2]], i32 4, <4 x i1> <i1 true, i1 true, i1 true
; CHECK: %[[InnerPhiNext]] = add nuw nsw <4 x i64> %[[InnerPhi]], <i64 1, i64 1, i64 1, i64 1>
; CHECK: %[[VecCond:.*]] = icmp eq <4 x i64> %[[InnerPhiNext]], <i64 8, i64 8, i64 8, i64 8>
; CHECK: %[[InnerCond:.*]] = extractelement <4 x i1> %[[VecCond]], i32 0
; CHECK: br i1 %[[InnerCond]], label %[[ForInc]], label %[[InnerLoop]]

; CHECK: [[ForInc]]:
; CHECK: %[[IndNext]] = add i64 %[[Ind]], 4
; CHECK: %[[VecIndNext]] = add <4 x i64> %[[VecInd]], <i64 4, i64 4, i64 4, i64 4>
; CHECK: %[[Cmp:.*]] = icmp eq i64 %[[IndNext]], 8
; CHECK: br i1 %[[Cmp]], label %middle.block, label %vector.body

@arr2 = external global [8 x i32], align 16
@arr = external global [8 x [8 x i32]], align 16

@arrX = external global [8 x i64], align 16
@arrY = external global [8 x [8 x i64]], align 16

; Function Attrs: norecurse nounwind uwtable
define void @foo_i32(i32 %n) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.inc8, %entry
  %indvars.iv21 = phi i64 [ 0, %entry ], [ %indvars.iv.next22, %for.inc8 ]
  %arrayidx = getelementptr inbounds [8 x i32], [8 x i32]* @arr2, i64 0, i64 %indvars.iv21
  %0 = trunc i64 %indvars.iv21 to i32
  store i32 %0, i32* %arrayidx, align 4
  %1 = trunc i64 %indvars.iv21 to i32
  %add = add nsw i32 %1, %n
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.body
  %indvars.iv = phi i64 [ 0, %for.body ], [ %indvars.iv.next, %for.body3 ]
  %arrayidx7 = getelementptr inbounds [8 x [8 x i32]], [8 x [8 x i32]]* @arr, i64 0, i64 %indvars.iv, i64 %indvars.iv21
  store i32 %add, i32* %arrayidx7, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 8
  br i1 %exitcond, label %for.inc8, label %for.body3

for.inc8:                                         ; preds = %for.body3
  %indvars.iv.next22 = add nuw nsw i64 %indvars.iv21, 1
  %exitcond23 = icmp eq i64 %indvars.iv.next22, 8
  br i1 %exitcond23, label %for.end10, label %for.body, !llvm.loop !1

for.end10:                                        ; preds = %for.inc8
  ret void
}

; CHECK-LABEL: @foo_i64(
; CHECK-LABEL: vector.ph:
; CHECK: %[[SplatVal:.*]] = insertelement <2 x i64> poison, i64 %n, i32 0
; CHECK: %[[Splat:.*]] = shufflevector <2 x i64> %[[SplatVal]], <2 x i64> poison, <2 x i32> zeroinitializer

; CHECK-LABEL: vector.body:
; CHECK: %[[Ind:.*]] = phi i64 [ 0, %vector.ph ], [ %[[IndNext:.*]], %[[ForInc:.*]] ]
; CHECK: %[[VecInd:.*]] = phi <2 x i64> [ <i64 0, i64 1>, %vector.ph ], [ %[[VecIndNext:.*]], %[[ForInc]] ]
; CHECK: %[[AAddr:.*]] = getelementptr inbounds [8 x i64], [8 x i64]* @arrX, i64 0, <2 x i64> %[[VecInd]]
; CHECK: call void @llvm.masked.scatter.v2i64.v2p0i64(<2 x i64> %[[VecInd]], <2 x i64*> %[[AAddr]], i32 4, <2 x i1> <i1 true, i1 true>)
; CHECK: %[[StoreVal:.*]] = add nsw <2 x i64> %[[VecInd]], %[[Splat]]
; CHECK: br label %[[InnerLoop:.+]]

; CHECK: [[InnerLoop]]:
; CHECK: %[[InnerPhi:.*]] = phi <2 x i64> [ %[[InnerPhiNext:.*]], %[[InnerLoop]] ], [ zeroinitializer, %vector.body ]
; CHECK: %[[AAddr2:.*]] = getelementptr inbounds [8 x [8 x i64]], [8 x [8 x i64]]* @arrY, i64 0, <2 x i64> %[[InnerPhi]], <2 x i64> %[[VecInd]]
; CHECK: call void @llvm.masked.scatter.v2i64.v2p0i64(<2 x i64> %[[StoreVal]], <2 x i64*> %[[AAddr2]], i32 4, <2 x i1> <i1 true, i1 true>
; CHECK: %[[InnerPhiNext]] = add nuw nsw <2 x i64> %[[InnerPhi]], <i64 1, i64 1>
; CHECK: %[[VecCond:.*]] = icmp eq <2 x i64> %[[InnerPhiNext]], <i64 8, i64 8>
; CHECK: %[[InnerCond:.*]] = extractelement <2 x i1> %[[VecCond]], i32 0
; CHECK: br i1 %[[InnerCond]], label %[[ForInc]], label %[[InnerLoop]]

; CHECK: [[ForInc]]:
; CHECK: %[[IndNext]] = add i64 %[[Ind]], 2
; CHECK: %[[VecIndNext]] = add <2 x i64> %[[VecInd]], <i64 2, i64 2>
; CHECK: %[[Cmp:.*]] = icmp eq i64 %[[IndNext]], 8
; CHECK: br i1 %[[Cmp]], label %middle.block, label %vector.body
; Function Attrs: norecurse nounwind uwtable
define void @foo_i64(i64 %n) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.inc8, %entry
  %indvars.iv21 = phi i64 [ 0, %entry ], [ %indvars.iv.next22, %for.inc8 ]
  %arrayidx = getelementptr inbounds [8 x i64], [8 x i64]* @arrX, i64 0, i64 %indvars.iv21
  store i64 %indvars.iv21, i64* %arrayidx, align 4
  %add = add nsw i64 %indvars.iv21, %n
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.body
  %indvars.iv = phi i64 [ 0, %for.body ], [ %indvars.iv.next, %for.body3 ]
  %arrayidx7 = getelementptr inbounds [8 x [8 x i64]], [8 x [8 x i64]]* @arrY, i64 0, i64 %indvars.iv, i64 %indvars.iv21
  store i64 %add, i64* %arrayidx7, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 8
  br i1 %exitcond, label %for.inc8, label %for.body3

for.inc8:                                         ; preds = %for.body3
  %indvars.iv.next22 = add nuw nsw i64 %indvars.iv21, 1
  %exitcond23 = icmp eq i64 %indvars.iv.next22, 8
  br i1 %exitcond23, label %for.end10, label %for.body, !llvm.loop !1

for.end10:                                        ; preds = %for.inc8
  ret void
}


!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.vectorize.enable", i1 true}
