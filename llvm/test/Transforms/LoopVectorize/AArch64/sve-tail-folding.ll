; RUN: opt -S -hints-allow-reordering=false -loop-vectorize -prefer-predicate-over-epilogue=predicate-dont-vectorize < %s | FileCheck %s
; RUN: opt -S -hints-allow-reordering=false -loop-vectorize -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue < %s | FileCheck %s

; CHECK-NOT: vector.body:

target triple = "aarch64-unknown-linux-gnu"


define void @simple_memset(i32 %val, i32* %ptr, i64 %n) #0 {
; CHECK-LABEL: @simple_memset(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[UMAX:%.*]] = call i64 @llvm.umax.i64(i64 [[N:%.*]], i64 1)
; CHECK-NEXT:    br i1 false, label %scalar.ph, label %vector.ph
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP0:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP1:%.*]] = mul i64 [[TMP0]], 4
; CHECK-NEXT:    [[TMP2:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP3:%.*]] = mul i64 [[TMP2]], 4
; CHECK-NEXT:    [[TMP4:%.*]] = sub i64 [[TMP3]], 1
; CHECK-NEXT:    [[N_RND_UP:%.*]] = add i64 [[UMAX]], [[TMP4]]
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N_RND_UP]], [[TMP1]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[N_RND_UP]], [[N_MOD_VF]]
; CHECK-NEXT:    [[TRIP_COUNT_MINUS_1:%.*]] = sub i64 [[UMAX]], 1
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i64> poison, i64 [[TRIP_COUNT_MINUS_1]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 4 x i64> [[BROADCAST_SPLATINSERT]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT5:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[VAL:%.*]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT6:%.*]] = shufflevector <vscale x 4 x i32> [[BROADCAST_SPLATINSERT5]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    br label %vector.body
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX1:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT2:%.*]], %vector.body ]
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT3:%.*]] = insertelement <vscale x 4 x i64> poison, i64 [[INDEX1]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT4:%.*]] = shufflevector <vscale x 4 x i64> [[BROADCAST_SPLATINSERT3]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP5:%.*]] = call <vscale x 4 x i64> @llvm.experimental.stepvector.nxv4i64()
; CHECK-NEXT:    [[TMP6:%.*]] = add <vscale x 4 x i64> [[TMP5]], zeroinitializer
; CHECK-NEXT:    [[TMP7:%.*]] = mul <vscale x 4 x i64> [[TMP6]], shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 1, i32 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[INDUCTION:%.*]] = add <vscale x 4 x i64> [[BROADCAST_SPLAT4]], [[TMP7]]
; CHECK-NEXT:    [[TMP8:%.*]] = add i64 [[INDEX1]], 0
; CHECK-NEXT:    [[TMP9:%.*]] = icmp ule <vscale x 4 x i64> [[INDUCTION]], [[BROADCAST_SPLAT]]
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr i32, i32* [[PTR:%.*]], i64 [[TMP8]]
; CHECK-NEXT:    [[TMP11:%.*]] = getelementptr i32, i32* [[TMP10]], i32 0
; CHECK-NEXT:    [[TMP12:%.*]] = bitcast i32* [[TMP11]] to <vscale x 4 x i32>*
; CHECK-NEXT:    call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> [[BROADCAST_SPLAT6]], <vscale x 4 x i32>* [[TMP12]], i32 4, <vscale x 4 x i1> [[TMP9]])
; CHECK-NEXT:    [[TMP13:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP14:%.*]] = mul i64 [[TMP13]], 4
; CHECK-NEXT:    [[INDEX_NEXT2]] = add i64 [[INDEX1]], [[TMP14]]
; CHECK-NEXT:    [[TMP15:%.*]] = icmp eq i64 [[INDEX_NEXT2]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP15]], label %middle.block, label %vector.body
; CHECK:       middle.block:
; CHECK-NEXT:    br i1 true, label %while.end.loopexit, label %scalar.ph
;
entry:
  br label %while.body

while.body:                                       ; preds = %while.body, %entry
  %index = phi i64 [ %index.next, %while.body ], [ 0, %entry ]
  %gep = getelementptr i32, i32* %ptr, i64 %index
  store i32 %val, i32* %gep
  %index.next = add nsw i64 %index, 1
  %cmp10 = icmp ult i64 %index.next, %n
  br i1 %cmp10, label %while.body, label %while.end.loopexit, !llvm.loop !0

while.end.loopexit:                               ; preds = %while.body
  ret void
}


define void @simple_memcpy(i32* noalias %dst, i32* noalias %src, i64 %n) #0 {
; CHECK-LABEL: @simple_memcpy(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[UMAX:%.*]] = call i64 @llvm.umax.i64(i64 [[N:%.*]], i64 1)
; CHECK-NEXT:    br i1 false, label %scalar.ph, label %vector.ph
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP0:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP1:%.*]] = mul i64 [[TMP0]], 4
; CHECK-NEXT:    [[TMP2:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP3:%.*]] = mul i64 [[TMP2]], 4
; CHECK-NEXT:    [[TMP4:%.*]] = sub i64 [[TMP3]], 1
; CHECK-NEXT:    [[N_RND_UP:%.*]] = add i64 [[UMAX]], [[TMP4]]
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N_RND_UP]], [[TMP1]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[N_RND_UP]], [[N_MOD_VF]]
; CHECK-NEXT:    [[TRIP_COUNT_MINUS_1:%.*]] = sub i64 [[UMAX]], 1
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i64> poison, i64 [[TRIP_COUNT_MINUS_1]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 4 x i64> [[BROADCAST_SPLATINSERT]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    br label %vector.body
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX1:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT2:%.*]], %vector.body ]
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT3:%.*]] = insertelement <vscale x 4 x i64> poison, i64 [[INDEX1]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT4:%.*]] = shufflevector <vscale x 4 x i64> [[BROADCAST_SPLATINSERT3]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP5:%.*]] = call <vscale x 4 x i64> @llvm.experimental.stepvector.nxv4i64()
; CHECK-NEXT:    [[TMP6:%.*]] = add <vscale x 4 x i64> [[TMP5]], zeroinitializer
; CHECK-NEXT:    [[TMP7:%.*]] = mul <vscale x 4 x i64> [[TMP6]], shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 1, i32 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[INDUCTION:%.*]] = add <vscale x 4 x i64> [[BROADCAST_SPLAT4]], [[TMP7]]
; CHECK-NEXT:    [[TMP8:%.*]] = add i64 [[INDEX1]], 0
; CHECK-NEXT:    [[TMP9:%.*]] = icmp ule <vscale x 4 x i64> [[INDUCTION]], [[BROADCAST_SPLAT]]
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr i32, i32* [[SRC:%.*]], i64 [[TMP8]]
; CHECK-NEXT:    [[TMP11:%.*]] = getelementptr i32, i32* [[TMP10]], i32 0
; CHECK-NEXT:    [[TMP12:%.*]] = bitcast i32* [[TMP11]] to <vscale x 4 x i32>*
; CHECK-NEXT:    [[WIDE_MASKED_LOAD:%.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>* [[TMP12]], i32 4, <vscale x 4 x i1> [[TMP9]], <vscale x 4 x i32> poison)
; CHECK-NEXT:    [[TMP13:%.*]] = getelementptr i32, i32* [[DST:%.*]], i64 [[TMP8]]
; CHECK-NEXT:    [[TMP14:%.*]] = getelementptr i32, i32* [[TMP13]], i32 0
; CHECK-NEXT:    [[TMP15:%.*]] = bitcast i32* [[TMP14]] to <vscale x 4 x i32>*
; CHECK-NEXT:    call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> [[WIDE_MASKED_LOAD]], <vscale x 4 x i32>* [[TMP15]], i32 4, <vscale x 4 x i1> [[TMP9]])
; CHECK-NEXT:    [[TMP16:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP17:%.*]] = mul i64 [[TMP16]], 4
; CHECK-NEXT:    [[INDEX_NEXT2]] = add i64 [[INDEX1]], [[TMP17]]
; CHECK-NEXT:    [[TMP18:%.*]] = icmp eq i64 [[INDEX_NEXT2]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP18]], label %middle.block, label %vector.body, !llvm.loop [[LOOP4:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    br i1 true, label %while.end.loopexit, label %scalar.ph
;
entry:
  br label %while.body

while.body:                                       ; preds = %while.body, %entry
  %index = phi i64 [ %index.next, %while.body ], [ 0, %entry ]
  %gep1 = getelementptr i32, i32* %src, i64 %index
  %val = load i32, i32* %gep1
  %gep2 = getelementptr i32, i32* %dst, i64 %index
  store i32 %val, i32* %gep2
  %index.next = add nsw i64 %index, 1
  %cmp10 = icmp ult i64 %index.next, %n
  br i1 %cmp10, label %while.body, label %while.end.loopexit, !llvm.loop !0

while.end.loopexit:                               ; preds = %while.body
  ret void
}


define void @copy_stride4(i32* noalias %dst, i32* noalias %src, i64 %n) #0 {
; CHECK-LABEL: @copy_stride4(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[UMAX:%.*]] = call i64 @llvm.umax.i64(i64 [[N:%.*]], i64 4)
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[UMAX]], -1
; CHECK-NEXT:    [[TMP1:%.*]] = lshr i64 [[TMP0]], 2
; CHECK-NEXT:    [[TMP2:%.*]] = add nuw nsw i64 [[TMP1]], 1
; CHECK-NEXT:    br i1 false, label %scalar.ph, label %vector.ph
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP3:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP4:%.*]] = mul i64 [[TMP3]], 4
; CHECK-NEXT:    [[TMP5:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP6:%.*]] = mul i64 [[TMP5]], 4
; CHECK-NEXT:    [[TMP7:%.*]] = sub i64 [[TMP6]], 1
; CHECK-NEXT:    [[N_RND_UP:%.*]] = add i64 [[TMP2]], [[TMP7]]
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N_RND_UP]], [[TMP4]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[N_RND_UP]], [[N_MOD_VF]]
; CHECK-NEXT:    [[IND_END:%.*]] = mul i64 [[N_VEC]], 4
; CHECK-NEXT:    [[TRIP_COUNT_MINUS_1:%.*]] = sub i64 [[TMP2]], 1
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i64> poison, i64 [[TRIP_COUNT_MINUS_1]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 4 x i64> [[BROADCAST_SPLATINSERT]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP8:%.*]] = call <vscale x 4 x i64> @llvm.experimental.stepvector.nxv4i64()
; CHECK-NEXT:    [[TMP9:%.*]] = add <vscale x 4 x i64> [[TMP8]], zeroinitializer
; CHECK-NEXT:    [[TMP10:%.*]] = mul <vscale x 4 x i64> [[TMP9]], shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 4, i32 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[INDUCTION:%.*]] = add <vscale x 4 x i64> zeroinitializer, [[TMP10]]
; CHECK-NEXT:    [[TMP11:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP12:%.*]] = mul i64 [[TMP11]], 4
; CHECK-NEXT:    [[TMP13:%.*]] = mul i64 4, [[TMP12]]
; CHECK-NEXT:    [[DOTSPLATINSERT:%.*]] = insertelement <vscale x 4 x i64> poison, i64 [[TMP13]], i32 0
; CHECK-NEXT:    [[DOTSPLAT:%.*]] = shufflevector <vscale x 4 x i64> [[DOTSPLATINSERT]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    br label %vector.body
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX1:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT2:%.*]], %vector.body ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <vscale x 4 x i64> [ [[INDUCTION]], %vector.ph ], [ [[VEC_IND_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT3:%.*]] = insertelement <vscale x 4 x i64> poison, i64 [[INDEX1]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT4:%.*]] = shufflevector <vscale x 4 x i64> [[BROADCAST_SPLATINSERT3]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP14:%.*]] = call <vscale x 4 x i64> @llvm.experimental.stepvector.nxv4i64()
; CHECK-NEXT:    [[TMP15:%.*]] = add <vscale x 4 x i64> zeroinitializer, [[TMP14]]
; CHECK-NEXT:    [[VEC_IV:%.*]] = add <vscale x 4 x i64> [[BROADCAST_SPLAT4]], [[TMP15]]
; CHECK-NEXT:    [[TMP16:%.*]] = icmp ule <vscale x 4 x i64> [[VEC_IV]], [[BROADCAST_SPLAT]]
; CHECK-NEXT:    [[TMP17:%.*]] = getelementptr i32, i32* [[SRC:%.*]], <vscale x 4 x i64> [[VEC_IND]]
; CHECK-NEXT:    [[WIDE_MASKED_GATHER:%.*]] = call <vscale x 4 x i32> @llvm.masked.gather.nxv4i32.nxv4p0i32(<vscale x 4 x i32*> [[TMP17]], i32 4, <vscale x 4 x i1> [[TMP16]], <vscale x 4 x i32> undef)
; CHECK-NEXT:    [[TMP18:%.*]] = getelementptr i32, i32* [[DST:%.*]], <vscale x 4 x i64> [[VEC_IND]]
; CHECK-NEXT:    call void @llvm.masked.scatter.nxv4i32.nxv4p0i32(<vscale x 4 x i32> [[WIDE_MASKED_GATHER]], <vscale x 4 x i32*> [[TMP18]], i32 4, <vscale x 4 x i1> [[TMP16]])
; CHECK-NEXT:    [[TMP19:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP20:%.*]] = mul i64 [[TMP19]], 4
; CHECK-NEXT:    [[INDEX_NEXT2]] = add i64 [[INDEX1]], [[TMP20]]
; CHECK-NEXT:    [[VEC_IND_NEXT]] = add <vscale x 4 x i64> [[VEC_IND]], [[DOTSPLAT]]
; CHECK-NEXT:    [[TMP21:%.*]] = icmp eq i64 [[INDEX_NEXT2]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP21]], label %middle.block, label %vector.body
; CHECK:       middle.block:
; CHECK-NEXT:    br i1 true, label %while.end.loopexit, label %scalar.ph
;
entry:
  br label %while.body

while.body:                                       ; preds = %while.body, %entry
  %index = phi i64 [ %index.next, %while.body ], [ 0, %entry ]
  %gep1 = getelementptr i32, i32* %src, i64 %index
  %val = load i32, i32* %gep1
  %gep2 = getelementptr i32, i32* %dst, i64 %index
  store i32 %val, i32* %gep2
  %index.next = add nsw i64 %index, 4
  %cmp10 = icmp ult i64 %index.next, %n
  br i1 %cmp10, label %while.body, label %while.end.loopexit, !llvm.loop !0

while.end.loopexit:                               ; preds = %while.body
  ret void
}


define void @simple_gather_scatter(i32* noalias %dst, i32* noalias %src, i32* noalias %ind, i64 %n) #0 {
; CHECK-LABEL: @simple_gather_scatter(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[UMAX:%.*]] = call i64 @llvm.umax.i64(i64 [[N:%.*]], i64 1)
; CHECK-NEXT:    br i1 false, label %scalar.ph, label %vector.ph
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP0:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP1:%.*]] = mul i64 [[TMP0]], 4
; CHECK-NEXT:    [[TMP2:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP3:%.*]] = mul i64 [[TMP2]], 4
; CHECK-NEXT:    [[TMP4:%.*]] = sub i64 [[TMP3]], 1
; CHECK-NEXT:    [[N_RND_UP:%.*]] = add i64 [[UMAX]], [[TMP4]]
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N_RND_UP]], [[TMP1]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[N_RND_UP]], [[N_MOD_VF]]
; CHECK-NEXT:    [[TRIP_COUNT_MINUS_1:%.*]] = sub i64 [[UMAX]], 1
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i64> poison, i64 [[TRIP_COUNT_MINUS_1]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 4 x i64> [[BROADCAST_SPLATINSERT]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX1:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT2:%.*]], %vector.body ]
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT3:%.*]] = insertelement <vscale x 4 x i64> poison, i64 [[INDEX1]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT4:%.*]] = shufflevector <vscale x 4 x i64> [[BROADCAST_SPLATINSERT3]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP5:%.*]] = call <vscale x 4 x i64> @llvm.experimental.stepvector.nxv4i64()
; CHECK-NEXT:    [[TMP6:%.*]] = add <vscale x 4 x i64> [[TMP5]], zeroinitializer
; CHECK-NEXT:    [[TMP7:%.*]] = mul <vscale x 4 x i64> [[TMP6]], shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 1, i32 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[INDUCTION:%.*]] = add <vscale x 4 x i64> [[BROADCAST_SPLAT4]], [[TMP7]]
; CHECK-NEXT:    [[TMP8:%.*]] = add i64 [[INDEX1]], 0
; CHECK-NEXT:    [[TMP9:%.*]] = icmp ule <vscale x 4 x i64> [[INDUCTION]], [[BROADCAST_SPLAT]]
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr i32, i32* [[IND:%.*]], i64 [[TMP8]]
; CHECK-NEXT:    [[TMP11:%.*]] = getelementptr i32, i32* [[TMP10]], i32 0
; CHECK-NEXT:    [[TMP12:%.*]] = bitcast i32* [[TMP11]] to <vscale x 4 x i32>*
; CHECK-NEXT:    [[WIDE_MASKED_LOAD:%.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>* [[TMP12]], i32 4, <vscale x 4 x i1> [[TMP9]], <vscale x 4 x i32> poison)
; CHECK-NEXT:    [[TMP13:%.*]] = getelementptr i32, i32* [[SRC:%.*]], <vscale x 4 x i32> [[WIDE_MASKED_LOAD]]
; CHECK-NEXT:    [[WIDE_MASKED_GATHER:%.*]] = call <vscale x 4 x i32> @llvm.masked.gather.nxv4i32.nxv4p0i32(<vscale x 4 x i32*> [[TMP13]], i32 4, <vscale x 4 x i1> [[TMP9]], <vscale x 4 x i32> undef)
; CHECK-NEXT:    [[TMP14:%.*]] = getelementptr i32, i32* [[DST:%.*]], <vscale x 4 x i32> [[WIDE_MASKED_LOAD]]
; CHECK-NEXT:    call void @llvm.masked.scatter.nxv4i32.nxv4p0i32(<vscale x 4 x i32> [[WIDE_MASKED_GATHER]], <vscale x 4 x i32*> [[TMP14]], i32 4, <vscale x 4 x i1> [[TMP9]])
; CHECK-NEXT:    [[TMP15:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP16:%.*]] = mul i64 [[TMP15]], 4
; CHECK-NEXT:    [[INDEX_NEXT2]] = add i64 [[INDEX1]], [[TMP16]]
; CHECK-NEXT:    [[TMP17:%.*]] = icmp eq i64 [[INDEX_NEXT2]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP17]], label %middle.block, label %vector.body
; CHECK:       middle.block:
; CHECK-NEXT:    br i1 true, label %while.end.loopexit, label %scalar.ph
;
entry:
  br label %while.body

while.body:                                       ; preds = %while.body, %entry
  %index = phi i64 [ %index.next, %while.body ], [ 0, %entry ]
  %gep1 = getelementptr i32, i32* %ind, i64 %index
  %ind_val = load i32, i32* %gep1
  %gep2 = getelementptr i32, i32* %src, i32 %ind_val
  %val = load i32, i32* %gep2
  %gep3 = getelementptr i32, i32* %dst, i32 %ind_val
  store i32 %val, i32* %gep3
  %index.next = add nsw i64 %index, 1
  %cmp10 = icmp ult i64 %index.next, %n
  br i1 %cmp10, label %while.body, label %while.end.loopexit, !llvm.loop !0

while.end.loopexit:                               ; preds = %while.body
  ret void
}


; The original loop had an unconditional uniform load. Let's make sure
; we don't artificially create new predicated blocks for the load.
define void @uniform_load(i32* noalias %dst, i32* noalias readonly %src, i64 %n) #0 {
; CHECK-LABEL: @uniform_load(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 false, label %scalar.ph, label %vector.ph
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP0:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP1:%.*]] = mul i64 [[TMP0]], 4
; CHECK-NEXT:    [[TMP2:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP3:%.*]] = mul i64 [[TMP2]], 4
; CHECK-NEXT:    [[TMP4:%.*]] = sub i64 [[TMP3]], 1
; CHECK-NEXT:    [[N_RND_UP:%.*]] = add i64 [[N:%.*]], [[TMP4]]
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N_RND_UP]], [[TMP1]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[N_RND_UP]], [[N_MOD_VF]]
; CHECK-NEXT:    [[TRIP_COUNT_MINUS_1:%.*]] = sub i64 [[N]], 1
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i64> poison, i64 [[TRIP_COUNT_MINUS_1]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 4 x i64> [[BROADCAST_SPLATINSERT]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    br label %vector.body
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT1:%.*]] = insertelement <vscale x 4 x i64> poison, i64 [[INDEX]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT2:%.*]] = shufflevector <vscale x 4 x i64> [[BROADCAST_SPLATINSERT1]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP5:%.*]] = call <vscale x 4 x i64> @llvm.experimental.stepvector.nxv4i64()
; CHECK-NEXT:    [[TMP6:%.*]] = add <vscale x 4 x i64> [[TMP5]], zeroinitializer
; CHECK-NEXT:    [[TMP7:%.*]] = mul <vscale x 4 x i64> [[TMP6]], shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 1, i32 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[INDUCTION:%.*]] = add <vscale x 4 x i64> [[BROADCAST_SPLAT2]], [[TMP7]]
; CHECK-NEXT:    [[TMP8:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP9:%.*]] = icmp ule <vscale x 4 x i64> [[INDUCTION]], [[BROADCAST_SPLAT]]
; CHECK-NEXT:    [[TMP10:%.*]] = load i32, i32* [[SRC:%.*]], align 4
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT3:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[TMP10]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT4:%.*]] = shufflevector <vscale x 4 x i32> [[BROADCAST_SPLATINSERT3]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP11:%.*]] = getelementptr inbounds i32, i32* [[DST:%.*]], i64 [[TMP8]]
; CHECK-NEXT:    [[TMP12:%.*]] = getelementptr inbounds i32, i32* [[TMP11]], i32 0
; CHECK-NEXT:    [[TMP13:%.*]] = bitcast i32* [[TMP12]] to <vscale x 4 x i32>*
; CHECK-NEXT:    call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> [[BROADCAST_SPLAT4]], <vscale x 4 x i32>* [[TMP13]], i32 4, <vscale x 4 x i1> [[TMP9]])
; CHECK-NEXT:    [[TMP14:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP15:%.*]] = mul i64 [[TMP14]], 4
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[TMP15]]
; CHECK-NEXT:    [[TMP16:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP16]], label %middle.block, label %vector.body, !llvm.loop [[LOOP6:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    br i1 true, label %for.end, label %scalar.ph
;

entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %val = load i32, i32* %src, align 4
  %arrayidx = getelementptr inbounds i32, i32* %dst, i64 %indvars.iv
  store i32 %val, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:                                          ; preds = %for.body, %entry
  ret void
}


; The original loop had a conditional uniform load. In this case we actually
; do need to perform conditional loads and so we end up using a gather instead.
; However, we at least ensure the mask is the overlap of the loop predicate
; and the original condition.
define void @cond_uniform_load(i32* noalias %dst, i32* noalias readonly %src, i32* noalias readonly %cond, i64 %n) #0 {
; CHECK-LABEL: @cond_uniform_load(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 false, label %scalar.ph, label %vector.ph
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP0:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP1:%.*]] = mul i64 [[TMP0]], 4
; CHECK-NEXT:    [[TMP2:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP3:%.*]] = mul i64 [[TMP2]], 4
; CHECK-NEXT:    [[TMP4:%.*]] = sub i64 [[TMP3]], 1
; CHECK-NEXT:    [[N_RND_UP:%.*]] = add i64 [[N:%.*]], [[TMP4]]
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N_RND_UP]], [[TMP1]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[N_RND_UP]], [[N_MOD_VF]]
; CHECK-NEXT:    [[TRIP_COUNT_MINUS_1:%.*]] = sub i64 [[N]], 1
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i64> poison, i64 [[TRIP_COUNT_MINUS_1]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 4 x i64> [[BROADCAST_SPLATINSERT]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT5:%.*]] = insertelement <vscale x 4 x i32*> poison, i32* [[SRC:%.*]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT6:%.*]] = shufflevector <vscale x 4 x i32*> [[BROADCAST_SPLATINSERT5]], <vscale x 4 x i32*> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    br label %vector.body
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX1:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT2:%.*]], %vector.body ]
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT3:%.*]] = insertelement <vscale x 4 x i64> poison, i64 [[INDEX1]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT4:%.*]] = shufflevector <vscale x 4 x i64> [[BROADCAST_SPLATINSERT3]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP5:%.*]] = call <vscale x 4 x i64> @llvm.experimental.stepvector.nxv4i64()
; CHECK-NEXT:    [[TMP6:%.*]] = add <vscale x 4 x i64> [[TMP5]], zeroinitializer
; CHECK-NEXT:    [[TMP7:%.*]] = mul <vscale x 4 x i64> [[TMP6]], shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 1, i32 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[INDUCTION:%.*]] = add <vscale x 4 x i64> [[BROADCAST_SPLAT4]], [[TMP7]]
; CHECK-NEXT:    [[TMP8:%.*]] = add i64 [[INDEX1]], 0
; CHECK-NEXT:    [[TMP9:%.*]] = icmp ule <vscale x 4 x i64> [[INDUCTION]], [[BROADCAST_SPLAT]]
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr inbounds i32, i32* [[COND:%.*]], i64 [[TMP8]]
; CHECK-NEXT:    [[TMP11:%.*]] = getelementptr inbounds i32, i32* [[TMP10]], i32 0
; CHECK-NEXT:    [[TMP12:%.*]] = bitcast i32* [[TMP11]] to <vscale x 4 x i32>*
; CHECK-NEXT:    [[WIDE_MASKED_LOAD:%.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>* [[TMP12]], i32 4, <vscale x 4 x i1> [[TMP9]], <vscale x 4 x i32> poison)
; CHECK-NEXT:    [[TMP13:%.*]] = icmp eq <vscale x 4 x i32> [[WIDE_MASKED_LOAD]], zeroinitializer
; CHECK-NEXT:    [[TMP14:%.*]] = xor <vscale x 4 x i1> [[TMP13]], shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i32 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[TMP15:%.*]] = select <vscale x 4 x i1> [[TMP9]], <vscale x 4 x i1> [[TMP14]], <vscale x 4 x i1> zeroinitializer
; CHECK-NEXT:    [[WIDE_MASKED_GATHER:%.*]] = call <vscale x 4 x i32> @llvm.masked.gather.nxv4i32.nxv4p0i32(<vscale x 4 x i32*> [[BROADCAST_SPLAT6]], i32 4, <vscale x 4 x i1> [[TMP15]], <vscale x 4 x i32> undef)
; CHECK-NEXT:    [[TMP16:%.*]] = select <vscale x 4 x i1> [[TMP9]], <vscale x 4 x i1> [[TMP13]], <vscale x 4 x i1> zeroinitializer
; CHECK-NEXT:    [[PREDPHI:%.*]] = select <vscale x 4 x i1> [[TMP16]], <vscale x 4 x i32> zeroinitializer, <vscale x 4 x i32> [[WIDE_MASKED_GATHER]]
; CHECK-NEXT:    [[TMP17:%.*]] = getelementptr inbounds i32, i32* [[DST:%.*]], i64 [[TMP8]]
; CHECK-NEXT:    [[TMP18:%.*]] = or <vscale x 4 x i1> [[TMP15]], [[TMP16]]
; CHECK-NEXT:    [[TMP19:%.*]] = getelementptr inbounds i32, i32* [[TMP17]], i32 0
; CHECK-NEXT:    [[TMP20:%.*]] = bitcast i32* [[TMP19]] to <vscale x 4 x i32>*
; CHECK-NEXT:    call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> [[PREDPHI]], <vscale x 4 x i32>* [[TMP20]], i32 4, <vscale x 4 x i1> [[TMP18]])
; CHECK-NEXT:    [[TMP21:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP22:%.*]] = mul i64 [[TMP21]], 4
; CHECK-NEXT:    [[INDEX_NEXT2]] = add i64 [[INDEX1]], [[TMP22]]
; CHECK-NEXT:    [[TMP23:%.*]] = icmp eq i64 [[INDEX_NEXT2]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP23]], label %middle.block, label %vector.body
; CHECK:       middle.block:
; CHECK-NEXT:    br i1 true, label %for.end, label %scalar.ph
;

entry:
  br label %for.body

for.body:                                         ; preds = %entry, %if.end
  %index = phi i64 [ %index.next, %if.end ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %cond, i64 %index
  %0 = load i32, i32* %arrayidx, align 4
  %tobool.not = icmp eq i32 %0, 0
  br i1 %tobool.not, label %if.end, label %if.then

if.then:                                          ; preds = %for.body
  %1 = load i32, i32* %src, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  %val.0 = phi i32 [ %1, %if.then ], [ 0, %for.body ]
  %arrayidx1 = getelementptr inbounds i32, i32* %dst, i64 %index
  store i32 %val.0, i32* %arrayidx1, align 4
  %index.next = add nuw i64 %index, 1
  %exitcond.not = icmp eq i64 %index.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:                                          ; preds = %for.inc, %entry
  ret void
}


; The original loop had an unconditional uniform store. Let's make sure
; we don't artificially create new predicated blocks for the load.
define void @uniform_store(i32* noalias %dst, i32* noalias readonly %src, i64 %n) #0 {
; CHECK-LABEL: @uniform_store(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 false, label %scalar.ph, label %vector.ph
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP0:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP1:%.*]] = mul i64 [[TMP0]], 4
; CHECK-NEXT:    [[TMP2:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP3:%.*]] = mul i64 [[TMP2]], 4
; CHECK-NEXT:    [[TMP4:%.*]] = sub i64 [[TMP3]], 1
; CHECK-NEXT:    [[N_RND_UP:%.*]] = add i64 [[N:%.*]], [[TMP4]]
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N_RND_UP]], [[TMP1]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[N_RND_UP]], [[N_MOD_VF]]
; CHECK-NEXT:    [[TRIP_COUNT_MINUS_1:%.*]] = sub i64 [[N]], 1
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i64> poison, i64 [[TRIP_COUNT_MINUS_1]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 4 x i64> [[BROADCAST_SPLATINSERT]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT3:%.*]] = insertelement <vscale x 4 x i32*> poison, i32* [[DST:%.*]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT4:%.*]] = shufflevector <vscale x 4 x i32*> [[BROADCAST_SPLATINSERT3]], <vscale x 4 x i32*> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    br label %vector.body
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT1:%.*]] = insertelement <vscale x 4 x i64> poison, i64 [[INDEX]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT2:%.*]] = shufflevector <vscale x 4 x i64> [[BROADCAST_SPLATINSERT1]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP5:%.*]] = call <vscale x 4 x i64> @llvm.experimental.stepvector.nxv4i64()
; CHECK-NEXT:    [[TMP6:%.*]] = add <vscale x 4 x i64> [[TMP5]], zeroinitializer
; CHECK-NEXT:    [[TMP7:%.*]] = mul <vscale x 4 x i64> [[TMP6]], shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 1, i32 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[INDUCTION:%.*]] = add <vscale x 4 x i64> [[BROADCAST_SPLAT2]], [[TMP7]]
; CHECK-NEXT:    [[TMP8:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP9:%.*]] = icmp ule <vscale x 4 x i64> [[INDUCTION]], [[BROADCAST_SPLAT]]
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr inbounds i32, i32* [[SRC:%.*]], i64 [[TMP8]]
; CHECK-NEXT:    [[TMP11:%.*]] = getelementptr inbounds i32, i32* [[TMP10]], i32 0
; CHECK-NEXT:    [[TMP12:%.*]] = bitcast i32* [[TMP11]] to <vscale x 4 x i32>*
; CHECK-NEXT:    [[WIDE_MASKED_LOAD:%.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>* [[TMP12]], i32 4, <vscale x 4 x i1> [[TMP9]], <vscale x 4 x i32> poison)
; CHECK-NEXT:    call void @llvm.masked.scatter.nxv4i32.nxv4p0i32(<vscale x 4 x i32> [[WIDE_MASKED_LOAD]], <vscale x 4 x i32*> [[BROADCAST_SPLAT4]], i32 4, <vscale x 4 x i1> [[TMP9]])
; CHECK-NEXT:    [[TMP13:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP14:%.*]] = mul i64 [[TMP13]], 4
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[TMP14]]
; CHECK-NEXT:    [[TMP15:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP15]], label %middle.block, label %vector.body
; CHECK:       middle.block:
; CHECK-NEXT:    br i1 true, label %for.end, label %scalar.ph
;

entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %src, i64 %indvars.iv
  %val = load i32, i32* %arrayidx, align 4
  store i32 %val, i32* %dst, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:                                          ; preds = %for.body, %entry
  ret void
}


define void @simple_fdiv(float* noalias %dst, float* noalias %src, i64 %n) #0 {
; CHECK-LABEL: @simple_fdiv(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[UMAX:%.*]] = call i64 @llvm.umax.i64(i64 [[N:%.*]], i64 1)
; CHECK-NEXT:    br i1 false, label %scalar.ph, label %vector.ph
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP0:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP1:%.*]] = mul i64 [[TMP0]], 4
; CHECK-NEXT:    [[TMP2:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP3:%.*]] = mul i64 [[TMP2]], 4
; CHECK-NEXT:    [[TMP4:%.*]] = sub i64 [[TMP3]], 1
; CHECK-NEXT:    [[N_RND_UP:%.*]] = add i64 [[UMAX]], [[TMP4]]
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N_RND_UP]], [[TMP1]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[N_RND_UP]], [[N_MOD_VF]]
; CHECK-NEXT:    [[TRIP_COUNT_MINUS_1:%.*]] = sub i64 [[UMAX]], 1
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i64> poison, i64 [[TRIP_COUNT_MINUS_1]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 4 x i64> [[BROADCAST_SPLATINSERT]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    br label %vector.body
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX1:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT2:%.*]], %vector.body ]
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT3:%.*]] = insertelement <vscale x 4 x i64> poison, i64 [[INDEX1]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT4:%.*]] = shufflevector <vscale x 4 x i64> [[BROADCAST_SPLATINSERT3]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP5:%.*]] = call <vscale x 4 x i64> @llvm.experimental.stepvector.nxv4i64()
; CHECK-NEXT:    [[TMP6:%.*]] = add <vscale x 4 x i64> [[TMP5]], zeroinitializer
; CHECK-NEXT:    [[TMP7:%.*]] = mul <vscale x 4 x i64> [[TMP6]], shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 1, i32 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[INDUCTION:%.*]] = add <vscale x 4 x i64> [[BROADCAST_SPLAT4]], [[TMP7]]
; CHECK-NEXT:    [[TMP8:%.*]] = add i64 [[INDEX1]], 0
; CHECK-NEXT:    [[TMP9:%.*]] = icmp ule <vscale x 4 x i64> [[INDUCTION]], [[BROADCAST_SPLAT]]
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr float, float* [[SRC:%.*]], i64 [[TMP8]]
; CHECK-NEXT:    [[TMP11:%.*]] = getelementptr float, float* [[DST:%.*]], i64 [[TMP8]]
; CHECK-NEXT:    [[TMP12:%.*]] = getelementptr float, float* [[TMP10]], i32 0
; CHECK-NEXT:    [[TMP13:%.*]] = bitcast float* [[TMP12]] to <vscale x 4 x float>*
; CHECK-NEXT:    [[WIDE_MASKED_LOAD:%.*]] = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0nxv4f32(<vscale x 4 x float>* [[TMP13]], i32 4, <vscale x 4 x i1> [[TMP9]], <vscale x 4 x float> poison)
; CHECK-NEXT:    [[TMP14:%.*]] = getelementptr float, float* [[TMP11]], i32 0
; CHECK-NEXT:    [[TMP15:%.*]] = bitcast float* [[TMP14]] to <vscale x 4 x float>*
; CHECK-NEXT:    [[WIDE_MASKED_LOAD5:%.*]] = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0nxv4f32(<vscale x 4 x float>* [[TMP15]], i32 4, <vscale x 4 x i1> [[TMP9]], <vscale x 4 x float> poison)
; CHECK-NEXT:    [[TMP16:%.*]] = fdiv <vscale x 4 x float> [[WIDE_MASKED_LOAD]], [[WIDE_MASKED_LOAD5]]
; CHECK-NEXT:    [[TMP17:%.*]] = bitcast float* [[TMP14]] to <vscale x 4 x float>*
; CHECK-NEXT:    call void @llvm.masked.store.nxv4f32.p0nxv4f32(<vscale x 4 x float> [[TMP16]], <vscale x 4 x float>* [[TMP17]], i32 4, <vscale x 4 x i1> [[TMP9]])
; CHECK-NEXT:    [[TMP18:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP19:%.*]] = mul i64 [[TMP18]], 4
; CHECK-NEXT:    [[INDEX_NEXT2]] = add i64 [[INDEX1]], [[TMP19]]
; CHECK-NEXT:    [[TMP20:%.*]] = icmp eq i64 [[INDEX_NEXT2]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP20]], label %middle.block, label %vector.body, !llvm.loop [[LOOP18:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    br i1 true, label %while.end.loopexit, label %scalar.ph
;
entry:
  br label %while.body

while.body:                                       ; preds = %while.body, %entry
  %index = phi i64 [ %index.next, %while.body ], [ 0, %entry ]
  %gep1 = getelementptr float, float* %src, i64 %index
  %gep2 = getelementptr float, float* %dst, i64 %index
  %val1 = load float, float* %gep1
  %val2 = load float, float* %gep2
  %res = fdiv float %val1, %val2
  store float %res, float* %gep2
  %index.next = add nsw i64 %index, 1
  %cmp10 = icmp ult i64 %index.next, %n
  br i1 %cmp10, label %while.body, label %while.end.loopexit, !llvm.loop !0

while.end.loopexit:                               ; preds = %while.body
  ret void
}


define i32 @add_reduction_i32(i32* %ptr, i64 %n) #0 {
; CHECK-LABEL: @add_reduction_i32(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[UMAX:%.*]] = call i64 @llvm.umax.i64(i64 [[N:%.*]], i64 1)
; CHECK-NEXT:    br i1 false, label %scalar.ph, label %vector.ph
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP0:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP1:%.*]] = mul i64 [[TMP0]], 4
; CHECK-NEXT:    [[TMP2:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP3:%.*]] = mul i64 [[TMP2]], 4
; CHECK-NEXT:    [[TMP4:%.*]] = sub i64 [[TMP3]], 1
; CHECK-NEXT:    [[N_RND_UP:%.*]] = add i64 [[UMAX]], [[TMP4]]
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N_RND_UP]], [[TMP1]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[N_RND_UP]], [[N_MOD_VF]]
; CHECK-NEXT:    [[TRIP_COUNT_MINUS_1:%.*]] = sub i64 [[UMAX]], 1
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i64> poison, i64 [[TRIP_COUNT_MINUS_1]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 4 x i64> [[BROADCAST_SPLATINSERT]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    br label %vector.body
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX1:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT2:%.*]], %vector.body ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <vscale x 4 x i32> [ insertelement (<vscale x 4 x i32> zeroinitializer, i32 0, i32 0), %vector.ph ], [ [[TMP16:%.*]], %vector.body ]
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT3:%.*]] = insertelement <vscale x 4 x i64> poison, i64 [[INDEX1]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT4:%.*]] = shufflevector <vscale x 4 x i64> [[BROADCAST_SPLATINSERT3]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP5:%.*]] = call <vscale x 4 x i64> @llvm.experimental.stepvector.nxv4i64()
; CHECK-NEXT:    [[TMP6:%.*]] = add <vscale x 4 x i64> [[TMP5]], zeroinitializer
; CHECK-NEXT:    [[TMP7:%.*]] = mul <vscale x 4 x i64> [[TMP6]], shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 1, i32 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[INDUCTION:%.*]] = add <vscale x 4 x i64> [[BROADCAST_SPLAT4]], [[TMP7]]
; CHECK-NEXT:    [[TMP8:%.*]] = add i64 [[INDEX1]], 0
; CHECK-NEXT:    [[TMP9:%.*]] = icmp ule <vscale x 4 x i64> [[INDUCTION]], [[BROADCAST_SPLAT]]
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr i32, i32* [[PTR:%.*]], i64 [[TMP8]]
; CHECK-NEXT:    [[TMP11:%.*]] = getelementptr i32, i32* [[TMP10]], i32 0
; CHECK-NEXT:    [[TMP12:%.*]] = bitcast i32* [[TMP11]] to <vscale x 4 x i32>*
; CHECK-NEXT:    [[WIDE_MASKED_LOAD:%.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>* [[TMP12]], i32 4, <vscale x 4 x i1> [[TMP9]], <vscale x 4 x i32> poison)
; CHECK-NEXT:    [[TMP13]] = add <vscale x 4 x i32> [[VEC_PHI]], [[WIDE_MASKED_LOAD]]
; CHECK-NEXT:    [[TMP16:%.*]] = select <vscale x 4 x i1> [[TMP9]], <vscale x 4 x i32> [[TMP13]], <vscale x 4 x i32> [[VEC_PHI]]
; CHECK-NEXT:    [[TMP14:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP15:%.*]] = mul i64 [[TMP14]], 4
; CHECK-NEXT:    [[INDEX_NEXT2]] = add i64 [[INDEX1]], [[TMP15]]
; CHECK-NEXT:    [[TMP17:%.*]] = icmp eq i64 [[INDEX_NEXT2]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP17]], label %middle.block, label %vector.body, !llvm.loop [[LOOP20:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[TMP18:%.*]] = call i32 @llvm.vector.reduce.add.nxv4i32(<vscale x 4 x i32> [[TMP16]])
; CHECK-NEXT:    br i1 true, label %while.end.loopexit, label %scalar.ph
;
entry:
  br label %while.body

while.body:                                       ; preds = %while.body, %entry
  %index = phi i64 [ %index.next, %while.body ], [ 0, %entry ]
  %red = phi i32 [ %red.next, %while.body ], [ 0, %entry ]
  %gep = getelementptr i32, i32* %ptr, i64 %index
  %val = load i32, i32* %gep
  %red.next = add i32 %red, %val
  %index.next = add nsw i64 %index, 1
  %cmp10 = icmp ult i64 %index.next, %n
  br i1 %cmp10, label %while.body, label %while.end.loopexit, !llvm.loop !0

while.end.loopexit:                               ; preds = %while.body
  ret i32 %red.next
}


define float @add_reduction_f32(float* %ptr, i64 %n) #0 {
; CHECK-LABEL: @add_reduction_f32(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[UMAX:%.*]] = call i64 @llvm.umax.i64(i64 [[N:%.*]], i64 1)
; CHECK-NEXT:    br i1 false, label %scalar.ph, label %vector.ph
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP0:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP1:%.*]] = mul i64 [[TMP0]], 4
; CHECK-NEXT:    [[TMP2:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP3:%.*]] = mul i64 [[TMP2]], 4
; CHECK-NEXT:    [[TMP4:%.*]] = sub i64 [[TMP3]], 1
; CHECK-NEXT:    [[N_RND_UP:%.*]] = add i64 [[UMAX]], [[TMP4]]
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N_RND_UP]], [[TMP1]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[N_RND_UP]], [[N_MOD_VF]]
; CHECK-NEXT:    [[TRIP_COUNT_MINUS_1:%.*]] = sub i64 [[UMAX]], 1
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i64> poison, i64 [[TRIP_COUNT_MINUS_1]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 4 x i64> [[BROADCAST_SPLATINSERT]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    br label %vector.body
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX1:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT2:%.*]], %vector.body ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi float [ 0.000000e+00, %vector.ph ], [ [[TMP14:%.*]], %vector.body ]
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT3:%.*]] = insertelement <vscale x 4 x i64> poison, i64 [[INDEX1]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT4:%.*]] = shufflevector <vscale x 4 x i64> [[BROADCAST_SPLATINSERT3]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP5:%.*]] = call <vscale x 4 x i64> @llvm.experimental.stepvector.nxv4i64()
; CHECK-NEXT:    [[TMP6:%.*]] = add <vscale x 4 x i64> [[TMP5]], zeroinitializer
; CHECK-NEXT:    [[TMP7:%.*]] = mul <vscale x 4 x i64> [[TMP6]], shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 1, i32 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[INDUCTION:%.*]] = add <vscale x 4 x i64> [[BROADCAST_SPLAT4]], [[TMP7]]
; CHECK-NEXT:    [[TMP8:%.*]] = add i64 [[INDEX1]], 0
; CHECK-NEXT:    [[TMP9:%.*]] = icmp ule <vscale x 4 x i64> [[INDUCTION]], [[BROADCAST_SPLAT]]
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr float, float* [[PTR:%.*]], i64 [[TMP8]]
; CHECK-NEXT:    [[TMP11:%.*]] = getelementptr float, float* [[TMP10]], i32 0
; CHECK-NEXT:    [[TMP12:%.*]] = bitcast float* [[TMP11]] to <vscale x 4 x float>*
; CHECK-NEXT:    [[WIDE_MASKED_LOAD:%.*]] = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0nxv4f32(<vscale x 4 x float>* [[TMP12]], i32 4, <vscale x 4 x i1> [[TMP9]], <vscale x 4 x float> poison)
; CHECK-NEXT:    [[TMP13:%.*]] = select <vscale x 4 x i1> [[TMP9]], <vscale x 4 x float> [[WIDE_MASKED_LOAD]], <vscale x 4 x float> shufflevector (<vscale x 4 x float> insertelement (<vscale x 4 x float> poison, float -0.000000e+00, i32 0), <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[TMP14]] = call float @llvm.vector.reduce.fadd.nxv4f32(float [[VEC_PHI]], <vscale x 4 x float> [[TMP13]])
; CHECK-NEXT:    [[TMP15:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP16:%.*]] = mul i64 [[TMP15]], 4
; CHECK-NEXT:    [[INDEX_NEXT2]] = add i64 [[INDEX1]], [[TMP16]]
; CHECK-NEXT:    [[TMP17:%.*]] = icmp eq i64 [[INDEX_NEXT2]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP17]], label %middle.block, label %vector.body, !llvm.loop [[LOOP22:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    br i1 true, label %while.end.loopexit, label %scalar.ph
;
entry:
  br label %while.body

while.body:                                       ; preds = %while.body, %entry
  %index = phi i64 [ %index.next, %while.body ], [ 0, %entry ]
  %red = phi float [ %red.next, %while.body ], [ 0.000000, %entry ]
  %gep = getelementptr float, float* %ptr, i64 %index
  %val = load float, float* %gep
  %red.next = fadd float %red, %val
  %index.next = add nsw i64 %index, 1
  %cmp10 = icmp ult i64 %index.next, %n
  br i1 %cmp10, label %while.body, label %while.end.loopexit, !llvm.loop !0

while.end.loopexit:                               ; preds = %while.body
  ret float %red.next
}


; Negative tests where we don't expect tail-folding

; Integer divides can throw exceptions and since we can't scalarize conditional
; divides for scalable vectors we just don't bother vectorizing.
define void @simple_idiv(i32* noalias %dst, i32* noalias %src, i64 %n) #0 {
; CHECK-LABEL: @simple_idiv(
; CHECK-NOT:   vector.body
;
entry:
  br label %while.body

while.body:                                       ; preds = %while.body, %entry
  %index = phi i64 [ %index.next, %while.body ], [ 0, %entry ]
  %gep1 = getelementptr i32, i32* %src, i64 %index
  %gep2 = getelementptr i32, i32* %dst, i64 %index
  %val1 = load i32, i32* %gep1
  %val2 = load i32, i32* %gep2
  %res = udiv i32 %val1, %val2
  store i32 %res, i32* %gep2
  %index.next = add nsw i64 %index, 1
  %cmp10 = icmp ult i64 %index.next, %n
  br i1 %cmp10, label %while.body, label %while.end.loopexit, !llvm.loop !0

while.end.loopexit:                               ; preds = %while.body
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"llvm.loop.vectorize.width", i32 4}
!2 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}

attributes #0 = { "target-features"="+sve" }
