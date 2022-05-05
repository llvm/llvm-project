; RUN: opt -S -loop-vectorize -prefer-predicate-over-epilogue=predicate-dont-vectorize -force-vector-interleave=4 -force-vector-width=4 < %s | FileCheck %s
; RUN: opt -S -loop-vectorize -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue -force-vector-interleave=4 -force-vector-width=4 < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"


define void @simple_memset(i32 %val, i32* %ptr, i64 %n) #0 {
; CHECK-LABEL: @simple_memset(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[UMAX:%.*]] = call i64 @llvm.umax.i64(i64 [[N:%.*]], i64 1)
; CHECK-NEXT:    br i1 false, label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP0:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP1:%.*]] = mul i64 [[TMP0]], 16
; CHECK-NEXT:    [[TMP2:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP3:%.*]] = mul i64 [[TMP2]], 16
; CHECK-NEXT:    [[TMP4:%.*]] = sub i64 [[TMP3]], 1
; CHECK-NEXT:    [[N_RND_UP:%.*]] = add i64 [[UMAX]], [[TMP4]]
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N_RND_UP]], [[TMP1]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[N_RND_UP]], [[N_MOD_VF]]
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[VAL:%.*]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[BROADCAST_SPLATINSERT]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT5:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[VAL]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT6:%.*]] = shufflevector <vscale x 4 x i32> [[BROADCAST_SPLATINSERT5]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT7:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[VAL]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT8:%.*]] = shufflevector <vscale x 4 x i32> [[BROADCAST_SPLATINSERT7]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT9:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[VAL]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT10:%.*]] = shufflevector <vscale x 4 x i32> [[BROADCAST_SPLATINSERT9]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX1:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT11:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP5:%.*]] = add i64 [[INDEX1]], 0
; CHECK-NEXT:    [[TMP6:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP7:%.*]] = mul i64 [[TMP6]], 4
; CHECK-NEXT:    [[TMP8:%.*]] = add i64 [[TMP7]], 0
; CHECK-NEXT:    [[TMP9:%.*]] = mul i64 [[TMP8]], 1
; CHECK-NEXT:    [[TMP10:%.*]] = add i64 [[INDEX1]], [[TMP9]]
; CHECK-NEXT:    [[TMP11:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP12:%.*]] = mul i64 [[TMP11]], 8
; CHECK-NEXT:    [[TMP13:%.*]] = add i64 [[TMP12]], 0
; CHECK-NEXT:    [[TMP14:%.*]] = mul i64 [[TMP13]], 1
; CHECK-NEXT:    [[TMP15:%.*]] = add i64 [[INDEX1]], [[TMP14]]
; CHECK-NEXT:    [[TMP16:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP17:%.*]] = mul i64 [[TMP16]], 12
; CHECK-NEXT:    [[TMP18:%.*]] = add i64 [[TMP17]], 0
; CHECK-NEXT:    [[TMP19:%.*]] = mul i64 [[TMP18]], 1
; CHECK-NEXT:    [[TMP20:%.*]] = add i64 [[INDEX1]], [[TMP19]]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK:%.*]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[TMP5]], i64 [[UMAX]])
; CHECK-NEXT:    [[ACTIVE_LANE_MASK2:%.*]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[TMP10]], i64 [[UMAX]])
; CHECK-NEXT:    [[ACTIVE_LANE_MASK3:%.*]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[TMP15]], i64 [[UMAX]])
; CHECK-NEXT:    [[ACTIVE_LANE_MASK4:%.*]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[TMP20]], i64 [[UMAX]])
; CHECK-NEXT:    [[TMP21:%.*]] = getelementptr i32, i32* [[PTR:%.*]], i64 [[TMP5]]
; CHECK-NEXT:    [[TMP22:%.*]] = getelementptr i32, i32* [[PTR]], i64 [[TMP10]]
; CHECK-NEXT:    [[TMP23:%.*]] = getelementptr i32, i32* [[PTR]], i64 [[TMP15]]
; CHECK-NEXT:    [[TMP24:%.*]] = getelementptr i32, i32* [[PTR]], i64 [[TMP20]]
; CHECK-NEXT:    [[TMP25:%.*]] = getelementptr i32, i32* [[TMP21]], i32 0
; CHECK-NEXT:    [[TMP26:%.*]] = bitcast i32* [[TMP25]] to <vscale x 4 x i32>*
; CHECK-NEXT:    call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> [[BROADCAST_SPLAT]], <vscale x 4 x i32>* [[TMP26]], i32 4, <vscale x 4 x i1> [[ACTIVE_LANE_MASK]])
; CHECK-NEXT:    [[TMP27:%.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT:    [[TMP28:%.*]] = mul i32 [[TMP27]], 4
; CHECK-NEXT:    [[TMP29:%.*]] = getelementptr i32, i32* [[TMP21]], i32 [[TMP28]]
; CHECK-NEXT:    [[TMP30:%.*]] = bitcast i32* [[TMP29]] to <vscale x 4 x i32>*
; CHECK-NEXT:    call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> [[BROADCAST_SPLAT6]], <vscale x 4 x i32>* [[TMP30]], i32 4, <vscale x 4 x i1> [[ACTIVE_LANE_MASK2]])
; CHECK-NEXT:    [[TMP31:%.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT:    [[TMP32:%.*]] = mul i32 [[TMP31]], 8
; CHECK-NEXT:    [[TMP33:%.*]] = getelementptr i32, i32* [[TMP21]], i32 [[TMP32]]
; CHECK-NEXT:    [[TMP34:%.*]] = bitcast i32* [[TMP33]] to <vscale x 4 x i32>*
; CHECK-NEXT:    call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> [[BROADCAST_SPLAT8]], <vscale x 4 x i32>* [[TMP34]], i32 4, <vscale x 4 x i1> [[ACTIVE_LANE_MASK3]])
; CHECK-NEXT:    [[TMP35:%.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT:    [[TMP36:%.*]] = mul i32 [[TMP35]], 12
; CHECK-NEXT:    [[TMP37:%.*]] = getelementptr i32, i32* [[TMP21]], i32 [[TMP36]]
; CHECK-NEXT:    [[TMP38:%.*]] = bitcast i32* [[TMP37]] to <vscale x 4 x i32>*
; CHECK-NEXT:    call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> [[BROADCAST_SPLAT10]], <vscale x 4 x i32>* [[TMP38]], i32 4, <vscale x 4 x i1> [[ACTIVE_LANE_MASK4]])
; CHECK-NEXT:    [[TMP39:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP40:%.*]] = mul i64 [[TMP39]], 16
; CHECK-NEXT:    [[INDEX_NEXT11]] = add i64 [[INDEX1]], [[TMP40]]
; CHECK-NEXT:    [[TMP41:%.*]] = icmp eq i64 [[INDEX_NEXT11]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP41]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    br i1 true, label [[WHILE_END_LOOPEXIT:%.*]], label [[SCALAR_PH]]
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

define void @cond_memset(i32 %val, i32* noalias readonly %cond_ptr, i32* noalias %ptr, i64 %n) #0 {
; CHECK-LABEL: @cond_memset(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[UMAX:%.*]] = call i64 @llvm.umax.i64(i64 [[N:%.*]], i64 1)
; CHECK-NEXT:    br i1 false, label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP0:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP1:%.*]] = mul i64 [[TMP0]], 16
; CHECK-NEXT:    [[TMP2:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP3:%.*]] = mul i64 [[TMP2]], 16
; CHECK-NEXT:    [[TMP4:%.*]] = sub i64 [[TMP3]], 1
; CHECK-NEXT:    [[N_RND_UP:%.*]] = add i64 [[UMAX]], [[TMP4]]
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N_RND_UP]], [[TMP1]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[N_RND_UP]], [[N_MOD_VF]]
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[VAL:%.*]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[BROADCAST_SPLATINSERT]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT8:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[VAL]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT9:%.*]] = shufflevector <vscale x 4 x i32> [[BROADCAST_SPLATINSERT8]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT10:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[VAL]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT11:%.*]] = shufflevector <vscale x 4 x i32> [[BROADCAST_SPLATINSERT10]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT12:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[VAL]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT13:%.*]] = shufflevector <vscale x 4 x i32> [[BROADCAST_SPLATINSERT12]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX1:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT14:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP5:%.*]] = add i64 [[INDEX1]], 0
; CHECK-NEXT:    [[TMP6:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP7:%.*]] = mul i64 [[TMP6]], 4
; CHECK-NEXT:    [[TMP8:%.*]] = add i64 [[TMP7]], 0
; CHECK-NEXT:    [[TMP9:%.*]] = mul i64 [[TMP8]], 1
; CHECK-NEXT:    [[TMP10:%.*]] = add i64 [[INDEX1]], [[TMP9]]
; CHECK-NEXT:    [[TMP11:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP12:%.*]] = mul i64 [[TMP11]], 8
; CHECK-NEXT:    [[TMP13:%.*]] = add i64 [[TMP12]], 0
; CHECK-NEXT:    [[TMP14:%.*]] = mul i64 [[TMP13]], 1
; CHECK-NEXT:    [[TMP15:%.*]] = add i64 [[INDEX1]], [[TMP14]]
; CHECK-NEXT:    [[TMP16:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP17:%.*]] = mul i64 [[TMP16]], 12
; CHECK-NEXT:    [[TMP18:%.*]] = add i64 [[TMP17]], 0
; CHECK-NEXT:    [[TMP19:%.*]] = mul i64 [[TMP18]], 1
; CHECK-NEXT:    [[TMP20:%.*]] = add i64 [[INDEX1]], [[TMP19]]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK:%.*]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[TMP5]], i64 [[UMAX]])
; CHECK-NEXT:    [[ACTIVE_LANE_MASK2:%.*]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[TMP10]], i64 [[UMAX]])
; CHECK-NEXT:    [[ACTIVE_LANE_MASK3:%.*]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[TMP15]], i64 [[UMAX]])
; CHECK-NEXT:    [[ACTIVE_LANE_MASK4:%.*]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[TMP20]], i64 [[UMAX]])
; CHECK-NEXT:    [[TMP21:%.*]] = getelementptr i32, i32* [[COND_PTR:%.*]], i64 [[TMP5]]
; CHECK-NEXT:    [[TMP22:%.*]] = getelementptr i32, i32* [[COND_PTR]], i64 [[TMP10]]
; CHECK-NEXT:    [[TMP23:%.*]] = getelementptr i32, i32* [[COND_PTR]], i64 [[TMP15]]
; CHECK-NEXT:    [[TMP24:%.*]] = getelementptr i32, i32* [[COND_PTR]], i64 [[TMP20]]
; CHECK-NEXT:    [[TMP25:%.*]] = getelementptr i32, i32* [[TMP21]], i32 0
; CHECK-NEXT:    [[TMP26:%.*]] = bitcast i32* [[TMP25]] to <vscale x 4 x i32>*
; CHECK-NEXT:    [[WIDE_MASKED_LOAD:%.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>* [[TMP26]], i32 4, <vscale x 4 x i1> [[ACTIVE_LANE_MASK]], <vscale x 4 x i32> poison)
; CHECK-NEXT:    [[TMP27:%.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT:    [[TMP28:%.*]] = mul i32 [[TMP27]], 4
; CHECK-NEXT:    [[TMP29:%.*]] = getelementptr i32, i32* [[TMP21]], i32 [[TMP28]]
; CHECK-NEXT:    [[TMP30:%.*]] = bitcast i32* [[TMP29]] to <vscale x 4 x i32>*
; CHECK-NEXT:    [[WIDE_MASKED_LOAD5:%.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>* [[TMP30]], i32 4, <vscale x 4 x i1> [[ACTIVE_LANE_MASK2]], <vscale x 4 x i32> poison)
; CHECK-NEXT:    [[TMP31:%.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT:    [[TMP32:%.*]] = mul i32 [[TMP31]], 8
; CHECK-NEXT:    [[TMP33:%.*]] = getelementptr i32, i32* [[TMP21]], i32 [[TMP32]]
; CHECK-NEXT:    [[TMP34:%.*]] = bitcast i32* [[TMP33]] to <vscale x 4 x i32>*
; CHECK-NEXT:    [[WIDE_MASKED_LOAD6:%.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>* [[TMP34]], i32 4, <vscale x 4 x i1> [[ACTIVE_LANE_MASK3]], <vscale x 4 x i32> poison)
; CHECK-NEXT:    [[TMP35:%.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT:    [[TMP36:%.*]] = mul i32 [[TMP35]], 12
; CHECK-NEXT:    [[TMP37:%.*]] = getelementptr i32, i32* [[TMP21]], i32 [[TMP36]]
; CHECK-NEXT:    [[TMP38:%.*]] = bitcast i32* [[TMP37]] to <vscale x 4 x i32>*
; CHECK-NEXT:    [[WIDE_MASKED_LOAD7:%.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>* [[TMP38]], i32 4, <vscale x 4 x i1> [[ACTIVE_LANE_MASK4]], <vscale x 4 x i32> poison)
; CHECK-NEXT:    [[TMP39:%.*]] = icmp ne <vscale x 4 x i32> [[WIDE_MASKED_LOAD]], zeroinitializer
; CHECK-NEXT:    [[TMP40:%.*]] = icmp ne <vscale x 4 x i32> [[WIDE_MASKED_LOAD5]], zeroinitializer
; CHECK-NEXT:    [[TMP41:%.*]] = icmp ne <vscale x 4 x i32> [[WIDE_MASKED_LOAD6]], zeroinitializer
; CHECK-NEXT:    [[TMP42:%.*]] = icmp ne <vscale x 4 x i32> [[WIDE_MASKED_LOAD7]], zeroinitializer
; CHECK-NEXT:    [[TMP43:%.*]] = getelementptr i32, i32* [[PTR:%.*]], i64 [[TMP5]]
; CHECK-NEXT:    [[TMP44:%.*]] = getelementptr i32, i32* [[PTR]], i64 [[TMP10]]
; CHECK-NEXT:    [[TMP45:%.*]] = getelementptr i32, i32* [[PTR]], i64 [[TMP15]]
; CHECK-NEXT:    [[TMP46:%.*]] = getelementptr i32, i32* [[PTR]], i64 [[TMP20]]
; CHECK-NEXT:    [[TMP47:%.*]] = select <vscale x 4 x i1> [[ACTIVE_LANE_MASK]], <vscale x 4 x i1> [[TMP39]], <vscale x 4 x i1> zeroinitializer
; CHECK-NEXT:    [[TMP48:%.*]] = select <vscale x 4 x i1> [[ACTIVE_LANE_MASK2]], <vscale x 4 x i1> [[TMP40]], <vscale x 4 x i1> zeroinitializer
; CHECK-NEXT:    [[TMP49:%.*]] = select <vscale x 4 x i1> [[ACTIVE_LANE_MASK3]], <vscale x 4 x i1> [[TMP41]], <vscale x 4 x i1> zeroinitializer
; CHECK-NEXT:    [[TMP50:%.*]] = select <vscale x 4 x i1> [[ACTIVE_LANE_MASK4]], <vscale x 4 x i1> [[TMP42]], <vscale x 4 x i1> zeroinitializer
; CHECK-NEXT:    [[TMP51:%.*]] = getelementptr i32, i32* [[TMP43]], i32 0
; CHECK-NEXT:    [[TMP52:%.*]] = bitcast i32* [[TMP51]] to <vscale x 4 x i32>*
; CHECK-NEXT:    call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> [[BROADCAST_SPLAT]], <vscale x 4 x i32>* [[TMP52]], i32 4, <vscale x 4 x i1> [[TMP47]])
; CHECK-NEXT:    [[TMP53:%.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT:    [[TMP54:%.*]] = mul i32 [[TMP53]], 4
; CHECK-NEXT:    [[TMP55:%.*]] = getelementptr i32, i32* [[TMP43]], i32 [[TMP54]]
; CHECK-NEXT:    [[TMP56:%.*]] = bitcast i32* [[TMP55]] to <vscale x 4 x i32>*
; CHECK-NEXT:    call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> [[BROADCAST_SPLAT9]], <vscale x 4 x i32>* [[TMP56]], i32 4, <vscale x 4 x i1> [[TMP48]])
; CHECK-NEXT:    [[TMP57:%.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT:    [[TMP58:%.*]] = mul i32 [[TMP57]], 8
; CHECK-NEXT:    [[TMP59:%.*]] = getelementptr i32, i32* [[TMP43]], i32 [[TMP58]]
; CHECK-NEXT:    [[TMP60:%.*]] = bitcast i32* [[TMP59]] to <vscale x 4 x i32>*
; CHECK-NEXT:    call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> [[BROADCAST_SPLAT11]], <vscale x 4 x i32>* [[TMP60]], i32 4, <vscale x 4 x i1> [[TMP49]])
; CHECK-NEXT:    [[TMP61:%.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT:    [[TMP62:%.*]] = mul i32 [[TMP61]], 12
; CHECK-NEXT:    [[TMP63:%.*]] = getelementptr i32, i32* [[TMP43]], i32 [[TMP62]]
; CHECK-NEXT:    [[TMP64:%.*]] = bitcast i32* [[TMP63]] to <vscale x 4 x i32>*
; CHECK-NEXT:    call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> [[BROADCAST_SPLAT13]], <vscale x 4 x i32>* [[TMP64]], i32 4, <vscale x 4 x i1> [[TMP50]])
; CHECK-NEXT:    [[TMP65:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP66:%.*]] = mul i64 [[TMP65]], 16
; CHECK-NEXT:    [[INDEX_NEXT14]] = add i64 [[INDEX1]], [[TMP66]]
; CHECK-NEXT:    [[TMP67:%.*]] = icmp eq i64 [[INDEX_NEXT14]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP67]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP4:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    br i1 true, label [[WHILE_END_LOOPEXIT:%.*]], label [[SCALAR_PH]]
;
entry:
  br label %while.body

while.body:                                       ; preds = %while.body, %entry
  %index = phi i64 [ %index.next, %while.end ], [ 0, %entry ]
  %cond_gep = getelementptr i32, i32* %cond_ptr, i64 %index
  %cond_i32 = load i32, i32* %cond_gep
  %cond_i1 = icmp ne i32 %cond_i32, 0
  br i1 %cond_i1, label %do.store, label %while.end

do.store:
  %gep = getelementptr i32, i32* %ptr, i64 %index
  store i32 %val, i32* %gep
  br label %while.end

while.end:
  %index.next = add nsw i64 %index, 1
  %cmp10 = icmp ult i64 %index.next, %n
  br i1 %cmp10, label %while.body, label %while.end.loopexit, !llvm.loop !0

while.end.loopexit:                               ; preds = %while.body
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}

attributes #0 = { "target-features"="+sve" }
