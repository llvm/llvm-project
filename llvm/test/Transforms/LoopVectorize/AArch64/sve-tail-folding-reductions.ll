; RUN: opt -S -hints-allow-reordering=false -loop-vectorize -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue \
; RUN:   < %s | FileCheck %s --check-prefix=CHECK
; RUN: opt -S -hints-allow-reordering=false -loop-vectorize -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue \
; RUN:   -prefer-inloop-reductions < %s | FileCheck %s --check-prefix=CHECK-IN-LOOP

target triple = "aarch64-unknown-linux-gnu"

define i32 @add_reduction_i32(i32* %ptr, i64 %n) #0 {
; CHECK-LABEL: @add_reduction_i32(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[UMAX:%.*]] = call i64 @llvm.umax.i64(i64 [[N:%.*]], i64 1)
; CHECK-NEXT:    [[TMP0:%.*]] = sub i64 -1, [[UMAX]]
; CHECK-NEXT:    [[TMP1:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP2:%.*]] = mul i64 [[TMP1]], 4
; CHECK-NEXT:    [[TMP3:%.*]] = icmp ult i64 [[TMP0]], [[TMP2]]
; CHECK-NEXT:    br i1 [[TMP3]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP4:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP5:%.*]] = mul i64 [[TMP4]], 4
; CHECK-NEXT:    [[TMP6:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP7:%.*]] = mul i64 [[TMP6]], 4
; CHECK-NEXT:    [[TMP8:%.*]] = sub i64 [[TMP7]], 1
; CHECK-NEXT:    [[N_RND_UP:%.*]] = add i64 [[UMAX]], [[TMP8]]
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N_RND_UP]], [[TMP5]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[N_RND_UP]], [[N_MOD_VF]]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK_ENTRY:%.*]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 0, i64 [[UMAX]])
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX1:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT2:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK:%.*]] = phi <vscale x 4 x i1> [ [[ACTIVE_LANE_MASK_ENTRY]], [[VECTOR_PH]] ], [ [[ACTIVE_LANE_MASK_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <vscale x 4 x i32> [ zeroinitializer, [[VECTOR_PH]] ], [ [[TMP14:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP9:%.*]] = add i64 [[INDEX1]], 0
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr i32, i32* [[PTR:%.*]], i64 [[TMP9]]
; CHECK-NEXT:    [[TMP11:%.*]] = getelementptr i32, i32* [[TMP10]], i32 0
; CHECK-NEXT:    [[TMP12:%.*]] = bitcast i32* [[TMP11]] to <vscale x 4 x i32>*
; CHECK-NEXT:    [[WIDE_MASKED_LOAD:%.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>* [[TMP12]], i32 4, <vscale x 4 x i1> [[ACTIVE_LANE_MASK]], <vscale x 4 x i32> poison)
; CHECK-NEXT:    [[TMP13:%.*]] = add <vscale x 4 x i32> [[VEC_PHI]], [[WIDE_MASKED_LOAD]]
; CHECK-NEXT:    [[TMP14]] = select <vscale x 4 x i1> [[ACTIVE_LANE_MASK]], <vscale x 4 x i32> [[TMP13]], <vscale x 4 x i32> [[VEC_PHI]]
; CHECK-NEXT:    [[TMP15:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP16:%.*]] = mul i64 [[TMP15]], 4
; CHECK-NEXT:    [[INDEX_NEXT2]] = add i64 [[INDEX1]], [[TMP16]]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK_NEXT]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[INDEX_NEXT2]], i64 [[UMAX]])
; CHECK-NEXT:    [[TMP17:%.*]] = xor <vscale x 4 x i1> [[ACTIVE_LANE_MASK_NEXT]], shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i32 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[TMP18:%.*]] = extractelement <vscale x 4 x i1> [[TMP17]], i32 0
; CHECK-NEXT:    br i1 [[TMP18]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[TMP19:%.*]] = call i32 @llvm.vector.reduce.add.nxv4i32(<vscale x 4 x i32> [[TMP14]])
; CHECK-NEXT:    br i1 true, label [[WHILE_END_LOOPEXIT:%.*]], label [[SCALAR_PH]]
;
; CHECK-IN-LOOP-LABEL: @add_reduction_i32(
; CHECK-IN-LOOP-NEXT:  entry:
; CHECK-IN-LOOP-NEXT:    [[UMAX:%.*]] = call i64 @llvm.umax.i64(i64 [[N:%.*]], i64 1)
; CHECK-IN-LOOP-NEXT:    [[TMP0:%.*]] = sub i64 -1, [[UMAX]]
; CHECK-IN-LOOP-NEXT:    [[TMP1:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-IN-LOOP-NEXT:    [[TMP2:%.*]] = mul i64 [[TMP1]], 4
; CHECK-IN-LOOP-NEXT:    [[TMP3:%.*]] = icmp ult i64 [[TMP0]], [[TMP2]]
; CHECK-IN-LOOP-NEXT:    br i1 [[TMP3]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK-IN-LOOP:       vector.ph:
; CHECK-IN-LOOP-NEXT:    [[TMP4:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-IN-LOOP-NEXT:    [[TMP5:%.*]] = mul i64 [[TMP4]], 4
; CHECK-IN-LOOP-NEXT:    [[TMP6:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-IN-LOOP-NEXT:    [[TMP7:%.*]] = mul i64 [[TMP6]], 4
; CHECK-IN-LOOP-NEXT:    [[TMP8:%.*]] = sub i64 [[TMP7]], 1
; CHECK-IN-LOOP-NEXT:    [[N_RND_UP:%.*]] = add i64 [[UMAX]], [[TMP8]]
; CHECK-IN-LOOP-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N_RND_UP]], [[TMP5]]
; CHECK-IN-LOOP-NEXT:    [[N_VEC:%.*]] = sub i64 [[N_RND_UP]], [[N_MOD_VF]]
; CHECK-IN-LOOP-NEXT:    [[ACTIVE_LANE_MASK_ENTRY:%.*]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 0, i64 [[UMAX]])
; CHECK-IN-LOOP-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK-IN-LOOP:       vector.body:
; CHECK-IN-LOOP-NEXT:    [[INDEX1:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT2:%.*]], [[VECTOR_BODY]] ]
; CHECK-IN-LOOP-NEXT:    [[ACTIVE_LANE_MASK:%.*]] = phi <vscale x 4 x i1> [ [[ACTIVE_LANE_MASK_ENTRY]], [[VECTOR_PH]] ], [ [[ACTIVE_LANE_MASK_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-IN-LOOP-NEXT:    [[VEC_PHI:%.*]] = phi i32 [ 0, [[VECTOR_PH]] ], [ [[TMP15:%.*]], [[VECTOR_BODY]] ]
; CHECK-IN-LOOP-NEXT:    [[TMP9:%.*]] = add i64 [[INDEX1]], 0
; CHECK-IN-LOOP-NEXT:    [[TMP10:%.*]] = getelementptr i32, i32* [[PTR:%.*]], i64 [[TMP9]]
; CHECK-IN-LOOP-NEXT:    [[TMP11:%.*]] = getelementptr i32, i32* [[TMP10]], i32 0
; CHECK-IN-LOOP-NEXT:    [[TMP12:%.*]] = bitcast i32* [[TMP11]] to <vscale x 4 x i32>*
; CHECK-IN-LOOP-NEXT:    [[WIDE_MASKED_LOAD:%.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>* [[TMP12]], i32 4, <vscale x 4 x i1> [[ACTIVE_LANE_MASK]], <vscale x 4 x i32> poison)
; CHECK-IN-LOOP-NEXT:    [[TMP13:%.*]] = select <vscale x 4 x i1> [[ACTIVE_LANE_MASK]], <vscale x 4 x i32> [[WIDE_MASKED_LOAD]], <vscale x 4 x i32> zeroinitializer
; CHECK-IN-LOOP-NEXT:    [[TMP14:%.*]] = call i32 @llvm.vector.reduce.add.nxv4i32(<vscale x 4 x i32> [[TMP13]])
; CHECK-IN-LOOP-NEXT:    [[TMP15]] = add i32 [[TMP14]], [[VEC_PHI]]
; CHECK-IN-LOOP-NEXT:    [[TMP16:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-IN-LOOP-NEXT:    [[TMP17:%.*]] = mul i64 [[TMP16]], 4
; CHECK-IN-LOOP-NEXT:    [[INDEX_NEXT2]] = add i64 [[INDEX1]], [[TMP17]]
; CHECK-IN-LOOP-NEXT:    [[ACTIVE_LANE_MASK_NEXT]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[INDEX_NEXT2]], i64 [[UMAX]])
; CHECK-IN-LOOP-NEXT:    [[TMP18:%.*]] = xor <vscale x 4 x i1> [[ACTIVE_LANE_MASK_NEXT]], shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i32 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-IN-LOOP-NEXT:    [[TMP19:%.*]] = extractelement <vscale x 4 x i1> [[TMP18]], i32 0
; CHECK-IN-LOOP-NEXT:    br i1 [[TMP19]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
; CHECK-IN-LOOP:       middle.block:
; CHECK-IN-LOOP-NEXT:    br i1 true, label [[WHILE_END_LOOPEXIT:%.*]], label [[SCALAR_PH]]
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
; CHECK-NEXT:    [[TMP0:%.*]] = sub i64 -1, [[UMAX]]
; CHECK-NEXT:    [[TMP1:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP2:%.*]] = mul i64 [[TMP1]], 4
; CHECK-NEXT:    [[TMP3:%.*]] = icmp ult i64 [[TMP0]], [[TMP2]]
; CHECK-NEXT:    br i1 [[TMP3]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP4:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP5:%.*]] = mul i64 [[TMP4]], 4
; CHECK-NEXT:    [[TMP6:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP7:%.*]] = mul i64 [[TMP6]], 4
; CHECK-NEXT:    [[TMP8:%.*]] = sub i64 [[TMP7]], 1
; CHECK-NEXT:    [[N_RND_UP:%.*]] = add i64 [[UMAX]], [[TMP8]]
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N_RND_UP]], [[TMP5]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[N_RND_UP]], [[N_MOD_VF]]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK_ENTRY:%.*]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 0, i64 [[UMAX]])
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX1:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT2:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK:%.*]] = phi <vscale x 4 x i1> [ [[ACTIVE_LANE_MASK_ENTRY]], [[VECTOR_PH]] ], [ [[ACTIVE_LANE_MASK_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi float [ 0.000000e+00, [[VECTOR_PH]] ], [ [[TMP14:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP9:%.*]] = add i64 [[INDEX1]], 0
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr float, float* [[PTR:%.*]], i64 [[TMP9]]
; CHECK-NEXT:    [[TMP11:%.*]] = getelementptr float, float* [[TMP10]], i32 0
; CHECK-NEXT:    [[TMP12:%.*]] = bitcast float* [[TMP11]] to <vscale x 4 x float>*
; CHECK-NEXT:    [[WIDE_MASKED_LOAD:%.*]] = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0nxv4f32(<vscale x 4 x float>* [[TMP12]], i32 4, <vscale x 4 x i1> [[ACTIVE_LANE_MASK]], <vscale x 4 x float> poison)
; CHECK-NEXT:    [[TMP13:%.*]] = select <vscale x 4 x i1> [[ACTIVE_LANE_MASK]], <vscale x 4 x float> [[WIDE_MASKED_LOAD]], <vscale x 4 x float> shufflevector (<vscale x 4 x float> insertelement (<vscale x 4 x float> poison, float -0.000000e+00, i32 0), <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[TMP14]] = call float @llvm.vector.reduce.fadd.nxv4f32(float [[VEC_PHI]], <vscale x 4 x float> [[TMP13]])
; CHECK-NEXT:    [[TMP15:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP16:%.*]] = mul i64 [[TMP15]], 4
; CHECK-NEXT:    [[INDEX_NEXT2]] = add i64 [[INDEX1]], [[TMP16]]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK_NEXT]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[INDEX_NEXT2]], i64 [[UMAX]])
; CHECK-NEXT:    [[TMP17:%.*]] = xor <vscale x 4 x i1> [[ACTIVE_LANE_MASK_NEXT]], shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i32 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[TMP18:%.*]] = extractelement <vscale x 4 x i1> [[TMP17]], i32 0
; CHECK-NEXT:    br i1 [[TMP18]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP4:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    br i1 true, label [[WHILE_END_LOOPEXIT:%.*]], label [[SCALAR_PH]]
;
; CHECK-IN-LOOP-LABEL: @add_reduction_f32(
; CHECK-IN-LOOP-NEXT:  entry:
; CHECK-IN-LOOP-NEXT:    [[UMAX:%.*]] = call i64 @llvm.umax.i64(i64 [[N:%.*]], i64 1)
; CHECK-IN-LOOP-NEXT:    [[TMP0:%.*]] = sub i64 -1, [[UMAX]]
; CHECK-IN-LOOP-NEXT:    [[TMP1:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-IN-LOOP-NEXT:    [[TMP2:%.*]] = mul i64 [[TMP1]], 4
; CHECK-IN-LOOP-NEXT:    [[TMP3:%.*]] = icmp ult i64 [[TMP0]], [[TMP2]]
; CHECK-IN-LOOP-NEXT:    br i1 [[TMP3]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK-IN-LOOP:       vector.ph:
; CHECK-IN-LOOP-NEXT:    [[TMP4:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-IN-LOOP-NEXT:    [[TMP5:%.*]] = mul i64 [[TMP4]], 4
; CHECK-IN-LOOP-NEXT:    [[TMP6:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-IN-LOOP-NEXT:    [[TMP7:%.*]] = mul i64 [[TMP6]], 4
; CHECK-IN-LOOP-NEXT:    [[TMP8:%.*]] = sub i64 [[TMP7]], 1
; CHECK-IN-LOOP-NEXT:    [[N_RND_UP:%.*]] = add i64 [[UMAX]], [[TMP8]]
; CHECK-IN-LOOP-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N_RND_UP]], [[TMP5]]
; CHECK-IN-LOOP-NEXT:    [[N_VEC:%.*]] = sub i64 [[N_RND_UP]], [[N_MOD_VF]]
; CHECK-IN-LOOP-NEXT:    [[ACTIVE_LANE_MASK_ENTRY:%.*]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 0, i64 [[UMAX]])
; CHECK-IN-LOOP-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK-IN-LOOP:       vector.body:
; CHECK-IN-LOOP-NEXT:    [[INDEX1:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT2:%.*]], [[VECTOR_BODY]] ]
; CHECK-IN-LOOP-NEXT:    [[ACTIVE_LANE_MASK:%.*]] = phi <vscale x 4 x i1> [ [[ACTIVE_LANE_MASK_ENTRY]], [[VECTOR_PH]] ], [ [[ACTIVE_LANE_MASK_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-IN-LOOP-NEXT:    [[VEC_PHI:%.*]] = phi float [ 0.000000e+00, [[VECTOR_PH]] ], [ [[TMP14:%.*]], [[VECTOR_BODY]] ]
; CHECK-IN-LOOP-NEXT:    [[TMP9:%.*]] = add i64 [[INDEX1]], 0
; CHECK-IN-LOOP-NEXT:    [[TMP10:%.*]] = getelementptr float, float* [[PTR:%.*]], i64 [[TMP9]]
; CHECK-IN-LOOP-NEXT:    [[TMP11:%.*]] = getelementptr float, float* [[TMP10]], i32 0
; CHECK-IN-LOOP-NEXT:    [[TMP12:%.*]] = bitcast float* [[TMP11]] to <vscale x 4 x float>*
; CHECK-IN-LOOP-NEXT:    [[WIDE_MASKED_LOAD:%.*]] = call <vscale x 4 x float> @llvm.masked.load.nxv4f32.p0nxv4f32(<vscale x 4 x float>* [[TMP12]], i32 4, <vscale x 4 x i1> [[ACTIVE_LANE_MASK]], <vscale x 4 x float> poison)
; CHECK-IN-LOOP-NEXT:    [[TMP13:%.*]] = select <vscale x 4 x i1> [[ACTIVE_LANE_MASK]], <vscale x 4 x float> [[WIDE_MASKED_LOAD]], <vscale x 4 x float> shufflevector (<vscale x 4 x float> insertelement (<vscale x 4 x float> poison, float -0.000000e+00, i32 0), <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-IN-LOOP-NEXT:    [[TMP14]] = call float @llvm.vector.reduce.fadd.nxv4f32(float [[VEC_PHI]], <vscale x 4 x float> [[TMP13]])
; CHECK-IN-LOOP-NEXT:    [[TMP15:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-IN-LOOP-NEXT:    [[TMP16:%.*]] = mul i64 [[TMP15]], 4
; CHECK-IN-LOOP-NEXT:    [[INDEX_NEXT2]] = add i64 [[INDEX1]], [[TMP16]]
; CHECK-IN-LOOP-NEXT:    [[ACTIVE_LANE_MASK_NEXT]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[INDEX_NEXT2]], i64 [[UMAX]])
; CHECK-IN-LOOP-NEXT:    [[TMP17:%.*]] = xor <vscale x 4 x i1> [[ACTIVE_LANE_MASK_NEXT]], shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i32 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-IN-LOOP-NEXT:    [[TMP18:%.*]] = extractelement <vscale x 4 x i1> [[TMP17]], i32 0
; CHECK-IN-LOOP-NEXT:    br i1 [[TMP18]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP4:![0-9]+]]
; CHECK-IN-LOOP:       middle.block:
; CHECK-IN-LOOP-NEXT:    br i1 true, label [[WHILE_END_LOOPEXIT:%.*]], label [[SCALAR_PH]]
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

define i32 @cond_xor_reduction(i32* noalias %a, i32* noalias %cond, i64 %N) #0 {
; CHECK-LABEL: @cond_xor_reduction(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = sub i64 -1, [[N:%.*]]
; CHECK-NEXT:    [[TMP1:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP2:%.*]] = mul i64 [[TMP1]], 4
; CHECK-NEXT:    [[TMP3:%.*]] = icmp ult i64 [[TMP0]], [[TMP2]]
; CHECK-NEXT:    br i1 [[TMP3]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP4:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP5:%.*]] = mul i64 [[TMP4]], 4
; CHECK-NEXT:    [[TMP6:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP7:%.*]] = mul i64 [[TMP6]], 4
; CHECK-NEXT:    [[TMP8:%.*]] = sub i64 [[TMP7]], 1
; CHECK-NEXT:    [[N_RND_UP:%.*]] = add i64 [[N]], [[TMP8]]
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N_RND_UP]], [[TMP5]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[N_RND_UP]], [[N_MOD_VF]]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK_ENTRY:%.*]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 0, i64 [[N]])
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK:%.*]] = phi <vscale x 4 x i1> [ [[ACTIVE_LANE_MASK_ENTRY]], [[VECTOR_PH]] ], [ [[ACTIVE_LANE_MASK_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <vscale x 4 x i32> [ insertelement (<vscale x 4 x i32> zeroinitializer, i32 7, i32 0), [[VECTOR_PH]] ], [ [[TMP21:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP9:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr inbounds i32, i32* [[COND:%.*]], i64 [[TMP9]]
; CHECK-NEXT:    [[TMP11:%.*]] = getelementptr inbounds i32, i32* [[TMP10]], i32 0
; CHECK-NEXT:    [[TMP12:%.*]] = bitcast i32* [[TMP11]] to <vscale x 4 x i32>*
; CHECK-NEXT:    [[WIDE_MASKED_LOAD:%.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>* [[TMP12]], i32 4, <vscale x 4 x i1> [[ACTIVE_LANE_MASK]], <vscale x 4 x i32> poison)
; CHECK-NEXT:    [[TMP13:%.*]] = icmp eq <vscale x 4 x i32> [[WIDE_MASKED_LOAD]], shufflevector (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> poison, i32 5, i32 0), <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[TMP14:%.*]] = getelementptr i32, i32* [[A:%.*]], i64 [[TMP9]]
; CHECK-NEXT:    [[TMP15:%.*]] = select <vscale x 4 x i1> [[ACTIVE_LANE_MASK]], <vscale x 4 x i1> [[TMP13]], <vscale x 4 x i1> zeroinitializer
; CHECK-NEXT:    [[TMP16:%.*]] = getelementptr i32, i32* [[TMP14]], i32 0
; CHECK-NEXT:    [[TMP17:%.*]] = bitcast i32* [[TMP16]] to <vscale x 4 x i32>*
; CHECK-NEXT:    [[WIDE_MASKED_LOAD1:%.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>* [[TMP17]], i32 4, <vscale x 4 x i1> [[TMP15]], <vscale x 4 x i32> poison)
; CHECK-NEXT:    [[TMP18:%.*]] = xor <vscale x 4 x i32> [[VEC_PHI]], [[WIDE_MASKED_LOAD1]]
; CHECK-NEXT:    [[TMP19:%.*]] = xor <vscale x 4 x i1> [[TMP13]], shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i32 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[TMP20:%.*]] = select <vscale x 4 x i1> [[ACTIVE_LANE_MASK]], <vscale x 4 x i1> [[TMP19]], <vscale x 4 x i1> zeroinitializer
; CHECK-NEXT:    [[PREDPHI:%.*]] = select <vscale x 4 x i1> [[TMP15]], <vscale x 4 x i32> [[TMP18]], <vscale x 4 x i32> [[VEC_PHI]]
; CHECK-NEXT:    [[TMP21]] = select <vscale x 4 x i1> [[ACTIVE_LANE_MASK]], <vscale x 4 x i32> [[PREDPHI]], <vscale x 4 x i32> [[VEC_PHI]]
; CHECK-NEXT:    [[TMP22:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP23:%.*]] = mul i64 [[TMP22]], 4
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[TMP23]]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK_NEXT]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[INDEX_NEXT]], i64 [[N]])
; CHECK-NEXT:    [[TMP24:%.*]] = xor <vscale x 4 x i1> [[ACTIVE_LANE_MASK_NEXT]], shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i32 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[TMP25:%.*]] = extractelement <vscale x 4 x i1> [[TMP24]], i32 0
; CHECK-NEXT:    br i1 [[TMP25]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP6:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[TMP26:%.*]] = call i32 @llvm.vector.reduce.xor.nxv4i32(<vscale x 4 x i32> [[TMP21]])
; CHECK-NEXT:    br i1 true, label [[FOR_END:%.*]], label [[SCALAR_PH]]
;
; CHECK-IN-LOOP-LABEL: @cond_xor_reduction(
; CHECK-IN-LOOP-NEXT:  entry:
; CHECK-IN-LOOP-NEXT:    [[TMP0:%.*]] = sub i64 -1, [[N:%.*]]
; CHECK-IN-LOOP-NEXT:    [[TMP1:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-IN-LOOP-NEXT:    [[TMP2:%.*]] = mul i64 [[TMP1]], 4
; CHECK-IN-LOOP-NEXT:    [[TMP3:%.*]] = icmp ult i64 [[TMP0]], [[TMP2]]
; CHECK-IN-LOOP-NEXT:    br i1 [[TMP3]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK-IN-LOOP:       vector.ph:
; CHECK-IN-LOOP-NEXT:    [[TMP4:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-IN-LOOP-NEXT:    [[TMP5:%.*]] = mul i64 [[TMP4]], 4
; CHECK-IN-LOOP-NEXT:    [[TMP6:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-IN-LOOP-NEXT:    [[TMP7:%.*]] = mul i64 [[TMP6]], 4
; CHECK-IN-LOOP-NEXT:    [[TMP8:%.*]] = sub i64 [[TMP7]], 1
; CHECK-IN-LOOP-NEXT:    [[N_RND_UP:%.*]] = add i64 [[N]], [[TMP8]]
; CHECK-IN-LOOP-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N_RND_UP]], [[TMP5]]
; CHECK-IN-LOOP-NEXT:    [[N_VEC:%.*]] = sub i64 [[N_RND_UP]], [[N_MOD_VF]]
; CHECK-IN-LOOP-NEXT:    [[ACTIVE_LANE_MASK_ENTRY:%.*]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 0, i64 [[N]])
; CHECK-IN-LOOP-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK-IN-LOOP:       vector.body:
; CHECK-IN-LOOP-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-IN-LOOP-NEXT:    [[ACTIVE_LANE_MASK:%.*]] = phi <vscale x 4 x i1> [ [[ACTIVE_LANE_MASK_ENTRY]], [[VECTOR_PH]] ], [ [[ACTIVE_LANE_MASK_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-IN-LOOP-NEXT:    [[VEC_PHI:%.*]] = phi i32 [ 7, [[VECTOR_PH]] ], [ [[TMP20:%.*]], [[VECTOR_BODY]] ]
; CHECK-IN-LOOP-NEXT:    [[TMP9:%.*]] = add i64 [[INDEX]], 0
; CHECK-IN-LOOP-NEXT:    [[TMP10:%.*]] = getelementptr inbounds i32, i32* [[COND:%.*]], i64 [[TMP9]]
; CHECK-IN-LOOP-NEXT:    [[TMP11:%.*]] = getelementptr inbounds i32, i32* [[TMP10]], i32 0
; CHECK-IN-LOOP-NEXT:    [[TMP12:%.*]] = bitcast i32* [[TMP11]] to <vscale x 4 x i32>*
; CHECK-IN-LOOP-NEXT:    [[WIDE_MASKED_LOAD:%.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>* [[TMP12]], i32 4, <vscale x 4 x i1> [[ACTIVE_LANE_MASK]], <vscale x 4 x i32> poison)
; CHECK-IN-LOOP-NEXT:    [[TMP13:%.*]] = icmp eq <vscale x 4 x i32> [[WIDE_MASKED_LOAD]], shufflevector (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> poison, i32 5, i32 0), <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-IN-LOOP-NEXT:    [[TMP14:%.*]] = getelementptr i32, i32* [[A:%.*]], i64 [[TMP9]]
; CHECK-IN-LOOP-NEXT:    [[TMP15:%.*]] = select <vscale x 4 x i1> [[ACTIVE_LANE_MASK]], <vscale x 4 x i1> [[TMP13]], <vscale x 4 x i1> zeroinitializer
; CHECK-IN-LOOP-NEXT:    [[TMP16:%.*]] = getelementptr i32, i32* [[TMP14]], i32 0
; CHECK-IN-LOOP-NEXT:    [[TMP17:%.*]] = bitcast i32* [[TMP16]] to <vscale x 4 x i32>*
; CHECK-IN-LOOP-NEXT:    [[WIDE_MASKED_LOAD1:%.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>* [[TMP17]], i32 4, <vscale x 4 x i1> [[TMP15]], <vscale x 4 x i32> poison)
; CHECK-IN-LOOP-NEXT:    [[TMP18:%.*]] = select <vscale x 4 x i1> [[TMP15]], <vscale x 4 x i32> [[WIDE_MASKED_LOAD1]], <vscale x 4 x i32> zeroinitializer
; CHECK-IN-LOOP-NEXT:    [[TMP19:%.*]] = call i32 @llvm.vector.reduce.xor.nxv4i32(<vscale x 4 x i32> [[TMP18]])
; CHECK-IN-LOOP-NEXT:    [[TMP20]] = xor i32 [[TMP19]], [[VEC_PHI]]
; CHECK-IN-LOOP-NEXT:    [[TMP21:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-IN-LOOP-NEXT:    [[TMP22:%.*]] = mul i64 [[TMP21]], 4
; CHECK-IN-LOOP-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[TMP22]]
; CHECK-IN-LOOP-NEXT:    [[ACTIVE_LANE_MASK_NEXT]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[INDEX_NEXT]], i64 [[N]])
; CHECK-IN-LOOP-NEXT:    [[TMP23:%.*]] = xor <vscale x 4 x i1> [[ACTIVE_LANE_MASK_NEXT]], shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i32 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-IN-LOOP-NEXT:    [[TMP24:%.*]] = extractelement <vscale x 4 x i1> [[TMP23]], i32 0
; CHECK-IN-LOOP-NEXT:    br i1 [[TMP24]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP6:![0-9]+]]
; CHECK-IN-LOOP:       middle.block:
; CHECK-IN-LOOP-NEXT:    br i1 true, label [[FOR_END:%.*]], label [[SCALAR_PH]]
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %rdx = phi i32 [ 7, %entry ], [ %res, %for.inc ]
  %arrayidx = getelementptr inbounds i32, i32* %cond, i64 %iv
  %0 = load i32, i32* %arrayidx
  %tobool = icmp eq i32 %0, 5
  br i1 %tobool, label %if.then, label %for.inc

if.then:
  %arrayidx2 = getelementptr inbounds i32, i32* %a, i64 %iv
  %1 = load i32, i32* %arrayidx2
  %xor = xor i32 %rdx, %1
  br label %for.inc

for.inc:
  %res = phi i32 [ %rdx, %for.body ], [ %xor, %if.then ]
  %iv.next = add i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret i32 %res
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"llvm.loop.vectorize.width", i32 4}
!2 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
!3 = distinct !{!3, !4}
!4 = !{!"llvm.loop.vectorize.width", i32 4}

attributes #0 = { "target-features"="+sve" }
