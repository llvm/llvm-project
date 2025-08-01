; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --version 4
; RUN: opt -S < %s -passes=loop-vectorize -mtriple aarch64-linux-gnu -mattr=+sve 2>&1 | FileCheck %s

define void @clamped_tc_8(ptr nocapture %dst, i32 %n, i64 %val) vscale_range(1,16) {
; CHECK-LABEL: define void @clamped_tc_8(
; CHECK-SAME: ptr captures(none) [[DST:%.*]], i32 [[N:%.*]], i64 [[VAL:%.*]]) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 false, label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP0:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP1:%.*]] = mul nuw i64 [[TMP0]], 8
; CHECK-NEXT:    [[TMP4:%.*]] = sub i64 [[TMP1]], 1
; CHECK-NEXT:    [[N_RND_UP:%.*]] = add i64 8, [[TMP4]]
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N_RND_UP]], [[TMP1]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[N_RND_UP]], [[N_MOD_VF]]
; CHECK-NEXT:    [[TMP5:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP6:%.*]] = mul nuw i64 [[TMP5]], 8
; CHECK-NEXT:    [[ACTIVE_LANE_MASK_ENTRY:%.*]] = call <vscale x 8 x i1> @llvm.get.active.lane.mask.nxv8i1.i64(i64 0, i64 8)
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 8 x i64> poison, i64 [[VAL]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 8 x i64> [[BROADCAST_SPLATINSERT]], <vscale x 8 x i64> poison, <vscale x 8 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP8:%.*]] = call <vscale x 8 x i64> @llvm.stepvector.nxv8i64()
; CHECK-NEXT:    [[TMP7:%.*]] = mul <vscale x 8 x i64> [[TMP8]], splat (i64 1)
; CHECK-NEXT:    [[INDUCTION:%.*]] = add <vscale x 8 x i64> zeroinitializer, [[TMP7]]
; CHECK-NEXT:    [[TMP12:%.*]] = mul i64 1, [[TMP6]]
; CHECK-NEXT:    [[DOTSPLATINSERT:%.*]] = insertelement <vscale x 8 x i64> poison, i64 [[TMP12]], i64 0
; CHECK-NEXT:    [[DOTSPLAT:%.*]] = shufflevector <vscale x 8 x i64> [[DOTSPLATINSERT]], <vscale x 8 x i64> poison, <vscale x 8 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK:%.*]] = phi <vscale x 8 x i1> [ [[ACTIVE_LANE_MASK_ENTRY]], [[VECTOR_PH]] ], [ [[ACTIVE_LANE_MASK_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <vscale x 8 x i64> [ [[INDUCTION]], [[VECTOR_PH]] ], [ [[VEC_IND_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[NEXT_GEP:%.*]] = getelementptr i8, ptr [[DST]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP10:%.*]] = shl nuw nsw <vscale x 8 x i64> [[VEC_IND]], splat (i64 3)
; CHECK-NEXT:    [[TMP11:%.*]] = lshr <vscale x 8 x i64> [[BROADCAST_SPLAT]], [[TMP10]]
; CHECK-NEXT:    [[TMP14:%.*]] = trunc <vscale x 8 x i64> [[TMP11]] to <vscale x 8 x i8>
; CHECK-NEXT:    call void @llvm.masked.store.nxv8i8.p0(<vscale x 8 x i8> [[TMP14]], ptr [[NEXT_GEP]], i32 1, <vscale x 8 x i1> [[ACTIVE_LANE_MASK]])
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[TMP6]]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK_NEXT]] = call <vscale x 8 x i1> @llvm.get.active.lane.mask.nxv8i1.i64(i64 [[INDEX_NEXT]], i64 8)
; CHECK-NEXT:    [[VEC_IND_NEXT]] = add <vscale x 8 x i64> [[VEC_IND]], [[DOTSPLAT]]
; CHECK-NEXT:    br i1 true, label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    br label [[FOR_COND_CLEANUP:%.*]]
; CHECK:       scalar.ph:
; CHECK-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ]
; CHECK-NEXT:    [[BC_RESUME_VAL1:%.*]] = phi ptr [ [[DST]], [[ENTRY]] ]
; CHECK-NEXT:    br label [[FOR_BODY:%.*]]
; CHECK:       for.body:
; CHECK-NEXT:    [[INDVARS_IV:%.*]] = phi i64 [ [[BC_RESUME_VAL]], [[SCALAR_PH]] ], [ [[INDVARS_IV_NEXT:%.*]], [[FOR_BODY]] ]
; CHECK-NEXT:    [[P_OUT_TAIL_09:%.*]] = phi ptr [ [[BC_RESUME_VAL1]], [[SCALAR_PH]] ], [ [[INCDEC_PTR:%.*]], [[FOR_BODY]] ]
; CHECK-NEXT:    [[TMP19:%.*]] = shl nuw nsw i64 [[INDVARS_IV]], 3
; CHECK-NEXT:    [[SHR3:%.*]] = lshr i64 [[VAL]], [[TMP19]]
; CHECK-NEXT:    [[CONV4:%.*]] = trunc i64 [[SHR3]] to i8
; CHECK-NEXT:    store i8 [[CONV4]], ptr [[P_OUT_TAIL_09]], align 1
; CHECK-NEXT:    [[INCDEC_PTR]] = getelementptr inbounds i8, ptr [[P_OUT_TAIL_09]], i64 1
; CHECK-NEXT:    [[INDVARS_IV_NEXT]] = add nuw nsw i64 [[INDVARS_IV]], 1
; CHECK-NEXT:    [[EXITCOND_NOT:%.*]] = icmp eq i64 [[INDVARS_IV_NEXT]], 8
; CHECK-NEXT:    br i1 [[EXITCOND_NOT]], label [[FOR_COND_CLEANUP]], label [[FOR_BODY]], !llvm.loop [[LOOP3:![0-9]+]]
; CHECK:       for.cond.cleanup:
; CHECK-NEXT:    ret void
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %p_out_tail.09 = phi ptr [ %dst, %entry ], [ %incdec.ptr, %for.body ]
  %0 = shl nuw nsw i64 %indvars.iv, 3
  %shr3 = lshr i64 %val, %0
  %conv4 = trunc i64 %shr3 to i8
  store i8 %conv4, ptr %p_out_tail.09, align 1
  %incdec.ptr = getelementptr inbounds i8, ptr %p_out_tail.09, i64 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 8
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

define void @clamped_tc_max_8(ptr nocapture %dst, i32 %n, i64 %val) vscale_range(1,16) {
; CHECK-LABEL: define void @clamped_tc_max_8(
; CHECK-SAME: ptr captures(none) [[DST:%.*]], i32 [[N:%.*]], i64 [[VAL:%.*]]) #[[ATTR0]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[REM:%.*]] = and i32 [[N]], 63
; CHECK-NEXT:    [[CMP8_NOT:%.*]] = icmp eq i32 [[REM]], 0
; CHECK-NEXT:    br i1 [[CMP8_NOT]], label [[FOR_COND_CLEANUP:%.*]], label [[FOR_BODY_PREHEADER:%.*]]
; CHECK:       for.body.preheader:
; CHECK-NEXT:    [[ADD:%.*]] = add nuw nsw i32 [[REM]], 7
; CHECK-NEXT:    [[SHR:%.*]] = lshr i32 [[ADD]], 3
; CHECK-NEXT:    [[WIDE_TRIP_COUNT:%.*]] = zext i32 [[SHR]] to i64
; CHECK-NEXT:    br i1 false, label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP0:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP1:%.*]] = mul nuw i64 [[TMP0]], 8
; CHECK-NEXT:    [[TMP4:%.*]] = sub i64 [[TMP1]], 1
; CHECK-NEXT:    [[N_RND_UP:%.*]] = add i64 [[WIDE_TRIP_COUNT]], [[TMP4]]
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N_RND_UP]], [[TMP1]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[N_RND_UP]], [[N_MOD_VF]]
; CHECK-NEXT:    [[TMP5:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP6:%.*]] = mul nuw i64 [[TMP5]], 8
; CHECK-NEXT:    [[ACTIVE_LANE_MASK_ENTRY:%.*]] = call <vscale x 8 x i1> @llvm.get.active.lane.mask.nxv8i1.i64(i64 0, i64 [[WIDE_TRIP_COUNT]])
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 8 x i64> poison, i64 [[VAL]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 8 x i64> [[BROADCAST_SPLATINSERT]], <vscale x 8 x i64> poison, <vscale x 8 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP8:%.*]] = call <vscale x 8 x i64> @llvm.stepvector.nxv8i64()
; CHECK-NEXT:    [[TMP7:%.*]] = mul <vscale x 8 x i64> [[TMP8]], splat (i64 1)
; CHECK-NEXT:    [[INDUCTION:%.*]] = add <vscale x 8 x i64> zeroinitializer, [[TMP7]]
; CHECK-NEXT:    [[TMP12:%.*]] = mul i64 1, [[TMP6]]
; CHECK-NEXT:    [[DOTSPLATINSERT:%.*]] = insertelement <vscale x 8 x i64> poison, i64 [[TMP12]], i64 0
; CHECK-NEXT:    [[DOTSPLAT:%.*]] = shufflevector <vscale x 8 x i64> [[DOTSPLATINSERT]], <vscale x 8 x i64> poison, <vscale x 8 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK:%.*]] = phi <vscale x 8 x i1> [ [[ACTIVE_LANE_MASK_ENTRY]], [[VECTOR_PH]] ], [ [[ACTIVE_LANE_MASK_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <vscale x 8 x i64> [ [[INDUCTION]], [[VECTOR_PH]] ], [ [[VEC_IND_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[NEXT_GEP:%.*]] = getelementptr i8, ptr [[DST]], i64 [[INDEX]]
; CHECK-NEXT:    [[TMP10:%.*]] = shl nuw nsw <vscale x 8 x i64> [[VEC_IND]], splat (i64 3)
; CHECK-NEXT:    [[TMP11:%.*]] = lshr <vscale x 8 x i64> [[BROADCAST_SPLAT]], [[TMP10]]
; CHECK-NEXT:    [[TMP14:%.*]] = trunc <vscale x 8 x i64> [[TMP11]] to <vscale x 8 x i8>
; CHECK-NEXT:    call void @llvm.masked.store.nxv8i8.p0(<vscale x 8 x i8> [[TMP14]], ptr [[NEXT_GEP]], i32 1, <vscale x 8 x i1> [[ACTIVE_LANE_MASK]])
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[TMP6]]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK_NEXT]] = call <vscale x 8 x i1> @llvm.get.active.lane.mask.nxv8i1.i64(i64 [[INDEX_NEXT]], i64 [[WIDE_TRIP_COUNT]])
; CHECK-NEXT:    [[VEC_IND_NEXT]] = add <vscale x 8 x i64> [[VEC_IND]], [[DOTSPLAT]]
; CHECK-NEXT:    br i1 true, label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP4:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    br label [[FOR_COND_CLEANUP_LOOPEXIT:%.*]]
; CHECK:       scalar.ph:
; CHECK-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ 0, [[FOR_BODY_PREHEADER]] ]
; CHECK-NEXT:    [[BC_RESUME_VAL1:%.*]] = phi ptr [ [[DST]], [[FOR_BODY_PREHEADER]] ]
; CHECK-NEXT:    br label [[FOR_BODY:%.*]]
; CHECK:       for.body:
; CHECK-NEXT:    [[INDVARS_IV:%.*]] = phi i64 [ [[BC_RESUME_VAL]], [[SCALAR_PH]] ], [ [[INDVARS_IV_NEXT:%.*]], [[FOR_BODY]] ]
; CHECK-NEXT:    [[P_OUT_TAIL_09:%.*]] = phi ptr [ [[BC_RESUME_VAL1]], [[SCALAR_PH]] ], [ [[INCDEC_PTR:%.*]], [[FOR_BODY]] ]
; CHECK-NEXT:    [[TMP19:%.*]] = shl nuw nsw i64 [[INDVARS_IV]], 3
; CHECK-NEXT:    [[SHR3:%.*]] = lshr i64 [[VAL]], [[TMP19]]
; CHECK-NEXT:    [[CONV4:%.*]] = trunc i64 [[SHR3]] to i8
; CHECK-NEXT:    store i8 [[CONV4]], ptr [[P_OUT_TAIL_09]], align 1
; CHECK-NEXT:    [[INCDEC_PTR]] = getelementptr inbounds i8, ptr [[P_OUT_TAIL_09]], i64 1
; CHECK-NEXT:    [[INDVARS_IV_NEXT]] = add nuw nsw i64 [[INDVARS_IV]], 1
; CHECK-NEXT:    [[EXITCOND_NOT:%.*]] = icmp eq i64 [[INDVARS_IV_NEXT]], [[WIDE_TRIP_COUNT]]
; CHECK-NEXT:    br i1 [[EXITCOND_NOT]], label [[FOR_COND_CLEANUP_LOOPEXIT]], label [[FOR_BODY]], !llvm.loop [[LOOP5:![0-9]+]]
; CHECK:       for.cond.cleanup.loopexit:
; CHECK-NEXT:    br label [[FOR_COND_CLEANUP]]
; CHECK:       for.cond.cleanup:
; CHECK-NEXT:    ret void
;

entry:
  %rem = and i32 %n, 63
  %cmp8.not = icmp eq i32 %rem, 0
  br i1 %cmp8.not, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %add = add nuw nsw i32 %rem, 7
  %shr = lshr i32 %add, 3
  %wide.trip.count = zext i32 %shr to i64
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %p_out_tail.09 = phi ptr [ %dst, %for.body.preheader ], [ %incdec.ptr, %for.body ]
  %0 = shl nuw nsw i64 %indvars.iv, 3
  %shr3 = lshr i64 %val, %0
  %conv4 = trunc i64 %shr3 to i8
  store i8 %conv4, ptr %p_out_tail.09, align 1
  %incdec.ptr = getelementptr inbounds i8, ptr %p_out_tail.09, i64 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}
;.
; CHECK: [[LOOP0]] = distinct !{[[LOOP0]], [[META1:![0-9]+]], [[META2:![0-9]+]]}
; CHECK: [[META1]] = !{!"llvm.loop.isvectorized", i32 1}
; CHECK: [[META2]] = !{!"llvm.loop.unroll.runtime.disable"}
; CHECK: [[LOOP3]] = distinct !{[[LOOP3]], [[META2]], [[META1]]}
; CHECK: [[LOOP4]] = distinct !{[[LOOP4]], [[META1]], [[META2]]}
; CHECK: [[LOOP5]] = distinct !{[[LOOP5]], [[META2]], [[META1]]}
;.
