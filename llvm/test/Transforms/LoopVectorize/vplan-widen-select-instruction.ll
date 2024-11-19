; RUN: opt -passes=loop-vectorize -force-vector-width=4 -enable-vplan-native-path -S %s | FileCheck %s

; Test that VPlan native path is able to widen select instruction in the
; innermost loop under different conditions when outer loop is marked to be
; vectorized. These conditions include following:
; * Inner and outer loop invariant select condition
; * Select condition depending on outer loop iteration variable.
; * Select condidition depending on inner loop iteration variable.
; * Select conditition depending on both outer and inner loop iteration
;   variables.

define void @loop_invariant_select(ptr noalias nocapture %out, i1 %select, double %a, double %b) {
; CHECK-LABEL: @loop_invariant_select(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 false, label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <4 x double> poison, double [[A:%.*]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <4 x double> [[BROADCAST_SPLATINSERT]], <4 x double> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT2:%.*]] = insertelement <4 x double> poison, double [[B:%.*]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT3:%.*]] = shufflevector <4 x double> [[BROADCAST_SPLATINSERT2]], <4 x double> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[FOR1_LATCH4:%.*]] ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, [[VECTOR_PH]] ], [ [[VEC_IND_NEXT:%.*]], [[FOR1_LATCH4]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds double, ptr [[OUT:%.*]], <4 x i64> [[VEC_IND]]
; CHECK-NEXT:    br label [[FOR2_HEADER1:%.*]]
; CHECK:       for2.header1:
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <4 x i64> [ zeroinitializer, [[VECTOR_BODY]] ], [ [[TMP2:%.*]], [[FOR2_HEADER1]] ]
; CHECK-NEXT:    [[TMP1:%.*]] = select i1 [[SELECT:%.*]], <4 x double> [[BROADCAST_SPLAT]], <4 x double> [[BROADCAST_SPLAT3]]
; CHECK-NEXT:    call void @llvm.masked.scatter.v4f64.v4p0(<4 x double> [[TMP1]], <4 x ptr> [[TMP0]], i32 8, <4 x i1> splat (i1 true))
entry:
  br label %for1.header

for1.header:
  %indvar1 = phi i64 [ 0, %entry ], [ %indvar11, %for1.latch ]
  %ptr = getelementptr inbounds double, ptr %out, i64 %indvar1
  br label %for2.header

for2.header:
  %indvar2 = phi i64 [ 0, %for1.header ], [ %indvar21, %for2.header ]
  ; Select condition is loop invariant for both inner and outer loop.
  %select.b = select i1 %select, double %a, double %b
  store double %select.b, ptr %ptr, align 8
  %indvar21 = add nuw nsw i64 %indvar2, 1
  %for2.cond = icmp eq i64 %indvar21, 10000
  br i1 %for2.cond, label %for1.latch, label %for2.header

for1.latch:
  %indvar11 = add nuw nsw i64 %indvar1, 1
  %for1.cond = icmp eq i64 %indvar11, 1000
  br i1 %for1.cond, label %exit, label %for1.header, !llvm.loop !0

exit:
  ret void
}

define void @outer_loop_dependant_select(ptr noalias nocapture %out, double %a, double %b) {
; CHECK-LABEL: @outer_loop_dependant_select(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 false, label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <4 x double> poison, double [[A:%.*]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <4 x double> [[BROADCAST_SPLATINSERT]], <4 x double> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT2:%.*]] = insertelement <4 x double> poison, double [[B:%.*]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT3:%.*]] = shufflevector <4 x double> [[BROADCAST_SPLATINSERT2]], <4 x double> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[FOR1_LATCH4:%.*]] ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, [[VECTOR_PH]] ], [ [[VEC_IND_NEXT:%.*]], [[FOR1_LATCH4]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds double, ptr [[OUT:%.*]], <4 x i64> [[VEC_IND]]
; CHECK-NEXT:    br label [[FOR2_HEADER1:%.*]]
; CHECK:       for2.header1:
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <4 x i64> [ zeroinitializer, [[VECTOR_BODY]] ], [ [[TMP3:%.*]], [[FOR2_HEADER1]] ]
; CHECK-NEXT:    [[TMP1:%.*]] = trunc <4 x i64> [[VEC_IND]] to <4 x i1>
; CHECK-NEXT:    [[TMP2:%.*]] = select <4 x i1> [[TMP1]], <4 x double> [[BROADCAST_SPLAT]], <4 x double> [[BROADCAST_SPLAT3]]
; CHECK-NEXT:    call void @llvm.masked.scatter.v4f64.v4p0(<4 x double> [[TMP2]], <4 x ptr> [[TMP0]], i32 8, <4 x i1> splat (i1 true))
entry:
  br label %for1.header

for1.header:
  %indvar1 = phi i64 [ 0, %entry ], [ %indvar11, %for1.latch ]
  %ptr = getelementptr inbounds double, ptr %out, i64 %indvar1
  br label %for2.header

for2.header:
  %indvar2 = phi i64 [ 0, %for1.header ], [ %indvar21, %for2.header ]
  %select = trunc i64 %indvar1 to i1
  ; Select condition only depends on outer loop iteration variable.
  %select.b = select i1 %select, double %a, double %b
  store double %select.b, ptr %ptr, align 8
  %indvar21 = add nuw nsw i64 %indvar2, 1
  %for2.cond = icmp eq i64 %indvar21, 10000
  br i1 %for2.cond, label %for1.latch, label %for2.header

for1.latch:
  %indvar11 = add nuw nsw i64 %indvar1, 1
  %for1.cond = icmp eq i64 %indvar11, 1000
  br i1 %for1.cond, label %exit, label %for1.header, !llvm.loop !0

exit:
  ret void
}

define void @inner_loop_dependant_select(ptr noalias nocapture %out, double %a, double %b) {
; CHECK-LABEL: @inner_loop_dependant_select(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 false, label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <4 x double> poison, double [[A:%.*]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <4 x double> [[BROADCAST_SPLATINSERT]], <4 x double> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT2:%.*]] = insertelement <4 x double> poison, double [[B:%.*]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT3:%.*]] = shufflevector <4 x double> [[BROADCAST_SPLATINSERT2]], <4 x double> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[FOR1_LATCH4:%.*]] ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, [[VECTOR_PH]] ], [ [[VEC_IND_NEXT:%.*]], [[FOR1_LATCH4]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds double, ptr [[OUT:%.*]], <4 x i64> [[VEC_IND]]
; CHECK-NEXT:    br label [[FOR2_HEADER1:%.*]]
; CHECK:       for2.header1:
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <4 x i64> [ zeroinitializer, [[VECTOR_BODY]] ], [ [[TMP3:%.*]], [[FOR2_HEADER1]] ]
; CHECK-NEXT:    [[TMP1:%.*]] = trunc <4 x i64> [[VEC_PHI]] to <4 x i1>
; CHECK-NEXT:    [[TMP2:%.*]] = select <4 x i1> [[TMP1]], <4 x double> [[BROADCAST_SPLAT]], <4 x double> [[BROADCAST_SPLAT3]]
; CHECK-NEXT:    call void @llvm.masked.scatter.v4f64.v4p0(<4 x double> [[TMP2]], <4 x ptr> [[TMP0]], i32 8, <4 x i1> splat (i1 true))
entry:
  br label %for1.header

for1.header:
  %indvar1 = phi i64 [ 0, %entry ], [ %indvar11, %for1.latch ]
  %ptr = getelementptr inbounds double, ptr %out, i64 %indvar1
  br label %for2.header

for2.header:
  %indvar2 = phi i64 [ 0, %for1.header ], [ %indvar21, %for2.header ]
  %select = trunc i64 %indvar2 to i1
  ; Select condition only depends on inner loop iteration variable.
  %select.b = select i1 %select, double %a, double %b
  store double %select.b, ptr %ptr, align 8
  %indvar21 = add nuw nsw i64 %indvar2, 1
  %for2.cond = icmp eq i64 %indvar21, 10000
  br i1 %for2.cond, label %for1.latch, label %for2.header

for1.latch:
  %indvar11 = add nuw nsw i64 %indvar1, 1
  %for1.cond = icmp eq i64 %indvar11, 1000
  br i1 %for1.cond, label %exit, label %for1.header, !llvm.loop !0

exit:
  ret void
}

define void @outer_and_inner_loop_dependant_select(ptr noalias nocapture %out, double %a, double %b) {
; CHECK-LABEL: @outer_and_inner_loop_dependant_select(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 false, label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <4 x double> poison, double [[A:%.*]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <4 x double> [[BROADCAST_SPLATINSERT]], <4 x double> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT2:%.*]] = insertelement <4 x double> poison, double [[B:%.*]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT3:%.*]] = shufflevector <4 x double> [[BROADCAST_SPLATINSERT2]], <4 x double> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[FOR1_LATCH4:%.*]] ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, [[VECTOR_PH]] ], [ [[VEC_IND_NEXT:%.*]], [[FOR1_LATCH4]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds double, ptr [[OUT:%.*]], <4 x i64> [[VEC_IND]]
; CHECK-NEXT:    br label [[FOR2_HEADER1:%.*]]
; CHECK:       for2.header1:
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <4 x i64> [ zeroinitializer, [[VECTOR_BODY]] ], [ [[TMP4:%.*]], [[FOR2_HEADER1]] ]
; CHECK-NEXT:    [[TMP1:%.*]] = add nuw nsw <4 x i64> [[VEC_IND]], [[VEC_PHI]]
; CHECK-NEXT:    [[TMP2:%.*]] = trunc <4 x i64> [[TMP1]] to <4 x i1>
; CHECK-NEXT:    [[TMP3:%.*]] = select <4 x i1> [[TMP2]], <4 x double> [[BROADCAST_SPLAT]], <4 x double> [[BROADCAST_SPLAT3]]
; CHECK-NEXT:    call void @llvm.masked.scatter.v4f64.v4p0(<4 x double> [[TMP3]], <4 x ptr> [[TMP0]], i32 8, <4 x i1> splat (i1 true))
entry:
  br label %for1.header

for1.header:
  %indvar1 = phi i64 [ 0, %entry ], [ %indvar11, %for1.latch ]
  %ptr = getelementptr inbounds double, ptr %out, i64 %indvar1
  br label %for2.header

for2.header:
  %indvar2 = phi i64 [ 0, %for1.header ], [ %indvar21, %for2.header ]
  %sum = add nuw nsw i64 %indvar1, %indvar2
  %select = trunc i64 %sum to i1
  ; Select condition depends on both inner and outer loop iteration variables.
  %select.b = select i1 %select, double %a, double %b
  store double %select.b, ptr %ptr, align 8
  %indvar21 = add nuw nsw i64 %indvar2, 1
  %for2.cond = icmp eq i64 %indvar21, 10000
  br i1 %for2.cond, label %for1.latch, label %for2.header

for1.latch:
  %indvar11 = add nuw nsw i64 %indvar1, 1
  %for1.cond = icmp eq i64 %indvar11, 1000
  br i1 %for1.cond, label %exit, label %for1.header, !llvm.loop !0

exit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable", i1 true}
