; RUN: opt -opaque-pointers=0 -S -passes=loop-vectorize -prefer-predicate-over-epilogue=predicate-else-scalar-epilogue -force-vector-interleave=4 -force-vector-width=4 < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"


define void @simple_memset(i32 %val, i32* %ptr, i64 %n) #0 {
; CHECK-LABEL: @simple_memset(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[UMAX:%.*]] = call i64 @llvm.umax.i64(i64 [[N:%.*]], i64 1)
; CHECK-NEXT:    [[TMP2:%.*]] = sub i64 -1, [[UMAX]]
; CHECK-NEXT:    [[TMP0:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP1:%.*]] = mul i64 [[TMP0]], 16
; CHECK-NEXT:    [[TMP3:%.*]] = icmp ult i64 [[TMP2]], [[TMP1]]
; CHECK-NEXT:    br i1 [[TMP3]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP4:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP5:%.*]] = mul i64 [[TMP4]], 16
; CHECK-NEXT:    [[TMP6:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP7:%.*]] = mul i64 [[TMP6]], 16
; CHECK-NEXT:    [[TMP8:%.*]] = sub i64 [[TMP7]], 1
; CHECK-NEXT:    [[N_RND_UP:%.*]] = add i64 [[UMAX]], [[TMP8]]
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N_RND_UP]], [[TMP5]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[N_RND_UP]], [[N_MOD_VF]]
; CHECK-NEXT:    [[TMP9:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP10:%.*]] = mul i64 [[TMP9]], 4
; CHECK-NEXT:    [[INDEX_PART_NEXT:%.*]] = add i64 0, [[TMP10]]
; CHECK-NEXT:    [[TMP11:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP12:%.*]] = mul i64 [[TMP11]], 8
; CHECK-NEXT:    [[INDEX_PART_NEXT1:%.*]] = add i64 0, [[TMP12]]
; CHECK-NEXT:    [[TMP13:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP14:%.*]] = mul i64 [[TMP13]], 12
; CHECK-NEXT:    [[INDEX_PART_NEXT2:%.*]] = add i64 0, [[TMP14]]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK:%.*]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 0, i64 [[UMAX]])
; CHECK-NEXT:    [[ACTIVE_LANE_MASK3:%.*]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[INDEX_PART_NEXT]], i64 [[UMAX]])
; CHECK-NEXT:    [[ACTIVE_LANE_MASK4:%.*]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[INDEX_PART_NEXT1]], i64 [[UMAX]])
; CHECK-NEXT:    [[ACTIVE_LANE_MASK5:%.*]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[INDEX_PART_NEXT2]], i64 [[UMAX]])
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[VAL:%.*]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[BROADCAST_SPLATINSERT]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT11:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[VAL]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT12:%.*]] = shufflevector <vscale x 4 x i32> [[BROADCAST_SPLATINSERT11]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT13:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[VAL]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT14:%.*]] = shufflevector <vscale x 4 x i32> [[BROADCAST_SPLATINSERT13]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT15:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[VAL]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT16:%.*]] = shufflevector <vscale x 4 x i32> [[BROADCAST_SPLATINSERT15]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX6:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT17:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK7:%.*]] = phi <vscale x 4 x i1> [ [[ACTIVE_LANE_MASK]], [[VECTOR_PH]] ], [ [[ACTIVE_LANE_MASK22:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK8:%.*]] = phi <vscale x 4 x i1> [ [[ACTIVE_LANE_MASK3]], [[VECTOR_PH]] ], [ [[ACTIVE_LANE_MASK23:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK9:%.*]] = phi <vscale x 4 x i1> [ [[ACTIVE_LANE_MASK4]], [[VECTOR_PH]] ], [ [[ACTIVE_LANE_MASK24:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK10:%.*]] = phi <vscale x 4 x i1> [ [[ACTIVE_LANE_MASK5]], [[VECTOR_PH]] ], [ [[ACTIVE_LANE_MASK25:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP15:%.*]] = add i64 [[INDEX6]], 0
; CHECK-NEXT:    [[TMP16:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP17:%.*]] = mul i64 [[TMP16]], 4
; CHECK-NEXT:    [[TMP18:%.*]] = add i64 [[TMP17]], 0
; CHECK-NEXT:    [[TMP19:%.*]] = mul i64 [[TMP18]], 1
; CHECK-NEXT:    [[TMP20:%.*]] = add i64 [[INDEX6]], [[TMP19]]
; CHECK-NEXT:    [[TMP21:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP22:%.*]] = mul i64 [[TMP21]], 8
; CHECK-NEXT:    [[TMP23:%.*]] = add i64 [[TMP22]], 0
; CHECK-NEXT:    [[TMP24:%.*]] = mul i64 [[TMP23]], 1
; CHECK-NEXT:    [[TMP25:%.*]] = add i64 [[INDEX6]], [[TMP24]]
; CHECK-NEXT:    [[TMP26:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP27:%.*]] = mul i64 [[TMP26]], 12
; CHECK-NEXT:    [[TMP28:%.*]] = add i64 [[TMP27]], 0
; CHECK-NEXT:    [[TMP29:%.*]] = mul i64 [[TMP28]], 1
; CHECK-NEXT:    [[TMP30:%.*]] = add i64 [[INDEX6]], [[TMP29]]
; CHECK-NEXT:    [[TMP31:%.*]] = getelementptr i32, i32* [[PTR:%.*]], i64 [[TMP15]]
; CHECK-NEXT:    [[TMP32:%.*]] = getelementptr i32, i32* [[PTR]], i64 [[TMP20]]
; CHECK-NEXT:    [[TMP33:%.*]] = getelementptr i32, i32* [[PTR]], i64 [[TMP25]]
; CHECK-NEXT:    [[TMP34:%.*]] = getelementptr i32, i32* [[PTR]], i64 [[TMP30]]
; CHECK-NEXT:    [[TMP35:%.*]] = getelementptr i32, i32* [[TMP31]], i32 0
; CHECK-NEXT:    [[TMP36:%.*]] = bitcast i32* [[TMP35]] to <vscale x 4 x i32>*
; CHECK-NEXT:    call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> [[BROADCAST_SPLAT]], <vscale x 4 x i32>* [[TMP36]], i32 4, <vscale x 4 x i1> [[ACTIVE_LANE_MASK7]])
; CHECK-NEXT:    [[TMP37:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP38:%.*]] = mul i64 [[TMP37]], 4
; CHECK-NEXT:    [[TMP39:%.*]] = getelementptr i32, i32* [[TMP31]], i64 [[TMP38]]
; CHECK-NEXT:    [[TMP40:%.*]] = bitcast i32* [[TMP39]] to <vscale x 4 x i32>*
; CHECK-NEXT:    call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> [[BROADCAST_SPLAT12]], <vscale x 4 x i32>* [[TMP40]], i32 4, <vscale x 4 x i1> [[ACTIVE_LANE_MASK8]])
; CHECK-NEXT:    [[TMP41:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP42:%.*]] = mul i64 [[TMP41]], 8
; CHECK-NEXT:    [[TMP43:%.*]] = getelementptr i32, i32* [[TMP31]], i64 [[TMP42]]
; CHECK-NEXT:    [[TMP44:%.*]] = bitcast i32* [[TMP43]] to <vscale x 4 x i32>*
; CHECK-NEXT:    call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> [[BROADCAST_SPLAT14]], <vscale x 4 x i32>* [[TMP44]], i32 4, <vscale x 4 x i1> [[ACTIVE_LANE_MASK9]])
; CHECK-NEXT:    [[TMP45:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP46:%.*]] = mul i64 [[TMP45]], 12
; CHECK-NEXT:    [[TMP47:%.*]] = getelementptr i32, i32* [[TMP31]], i64 [[TMP46]]
; CHECK-NEXT:    [[TMP48:%.*]] = bitcast i32* [[TMP47]] to <vscale x 4 x i32>*
; CHECK-NEXT:    call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> [[BROADCAST_SPLAT16]], <vscale x 4 x i32>* [[TMP48]], i32 4, <vscale x 4 x i1> [[ACTIVE_LANE_MASK10]])
; CHECK-NEXT:    [[TMP49:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP50:%.*]] = mul i64 [[TMP49]], 16
; CHECK-NEXT:    [[INDEX_NEXT17]] = add i64 [[INDEX6]], [[TMP50]]
; CHECK-NEXT:    [[TMP51:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP52:%.*]] = mul i64 [[TMP51]], 4
; CHECK-NEXT:    [[INDEX_PART_NEXT19:%.*]] = add i64 [[INDEX_NEXT17]], [[TMP52]]
; CHECK-NEXT:    [[TMP53:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP54:%.*]] = mul i64 [[TMP53]], 8
; CHECK-NEXT:    [[INDEX_PART_NEXT20:%.*]] = add i64 [[INDEX_NEXT17]], [[TMP54]]
; CHECK-NEXT:    [[TMP55:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP56:%.*]] = mul i64 [[TMP55]], 12
; CHECK-NEXT:    [[INDEX_PART_NEXT21:%.*]] = add i64 [[INDEX_NEXT17]], [[TMP56]]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK22]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[INDEX_NEXT17]], i64 [[UMAX]])
; CHECK-NEXT:    [[ACTIVE_LANE_MASK23]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[INDEX_PART_NEXT19]], i64 [[UMAX]])
; CHECK-NEXT:    [[ACTIVE_LANE_MASK24]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[INDEX_PART_NEXT20]], i64 [[UMAX]])
; CHECK-NEXT:    [[ACTIVE_LANE_MASK25]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[INDEX_PART_NEXT21]], i64 [[UMAX]])
; CHECK-NEXT:    [[TMP57:%.*]] = xor <vscale x 4 x i1> [[ACTIVE_LANE_MASK22]], shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[TMP58:%.*]] = xor <vscale x 4 x i1> [[ACTIVE_LANE_MASK23]], shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[TMP59:%.*]] = xor <vscale x 4 x i1> [[ACTIVE_LANE_MASK24]], shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[TMP60:%.*]] = xor <vscale x 4 x i1> [[ACTIVE_LANE_MASK25]], shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[TMP61:%.*]] = extractelement <vscale x 4 x i1> [[TMP57]], i32 0
; CHECK-NEXT:    br i1 [[TMP61]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
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
; CHECK-NEXT:    [[TMP2:%.*]] = sub i64 -1, [[UMAX]]
; CHECK-NEXT:    [[TMP0:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP1:%.*]] = mul i64 [[TMP0]], 16
; CHECK-NEXT:    [[TMP3:%.*]] = icmp ult i64 [[TMP2]], [[TMP1]]
; CHECK-NEXT:    br i1 [[TMP3]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP4:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP5:%.*]] = mul i64 [[TMP4]], 16
; CHECK-NEXT:    [[TMP6:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP7:%.*]] = mul i64 [[TMP6]], 16
; CHECK-NEXT:    [[TMP8:%.*]] = sub i64 [[TMP7]], 1
; CHECK-NEXT:    [[N_RND_UP:%.*]] = add i64 [[UMAX]], [[TMP8]]
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N_RND_UP]], [[TMP5]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[N_RND_UP]], [[N_MOD_VF]]
; CHECK-NEXT:    [[TMP9:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP10:%.*]] = mul i64 [[TMP9]], 4
; CHECK-NEXT:    [[INDEX_PART_NEXT:%.*]] = add i64 0, [[TMP10]]
; CHECK-NEXT:    [[TMP11:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP12:%.*]] = mul i64 [[TMP11]], 8
; CHECK-NEXT:    [[INDEX_PART_NEXT1:%.*]] = add i64 0, [[TMP12]]
; CHECK-NEXT:    [[TMP13:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP14:%.*]] = mul i64 [[TMP13]], 12
; CHECK-NEXT:    [[INDEX_PART_NEXT2:%.*]] = add i64 0, [[TMP14]]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK:%.*]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 0, i64 [[UMAX]])
; CHECK-NEXT:    [[ACTIVE_LANE_MASK3:%.*]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[INDEX_PART_NEXT]], i64 [[UMAX]])
; CHECK-NEXT:    [[ACTIVE_LANE_MASK4:%.*]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[INDEX_PART_NEXT1]], i64 [[UMAX]])
; CHECK-NEXT:    [[ACTIVE_LANE_MASK5:%.*]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[INDEX_PART_NEXT2]], i64 [[UMAX]])
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[VAL:%.*]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[BROADCAST_SPLATINSERT]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT14:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[VAL]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT15:%.*]] = shufflevector <vscale x 4 x i32> [[BROADCAST_SPLATINSERT14]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT16:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[VAL]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT17:%.*]] = shufflevector <vscale x 4 x i32> [[BROADCAST_SPLATINSERT16]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT18:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[VAL]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT19:%.*]] = shufflevector <vscale x 4 x i32> [[BROADCAST_SPLATINSERT18]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX6:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT20:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK7:%.*]] = phi <vscale x 4 x i1> [ [[ACTIVE_LANE_MASK]], [[VECTOR_PH]] ], [ [[ACTIVE_LANE_MASK25:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK8:%.*]] = phi <vscale x 4 x i1> [ [[ACTIVE_LANE_MASK3]], [[VECTOR_PH]] ], [ [[ACTIVE_LANE_MASK26:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK9:%.*]] = phi <vscale x 4 x i1> [ [[ACTIVE_LANE_MASK4]], [[VECTOR_PH]] ], [ [[ACTIVE_LANE_MASK27:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK10:%.*]] = phi <vscale x 4 x i1> [ [[ACTIVE_LANE_MASK5]], [[VECTOR_PH]] ], [ [[ACTIVE_LANE_MASK28:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP15:%.*]] = add i64 [[INDEX6]], 0
; CHECK-NEXT:    [[TMP16:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP17:%.*]] = mul i64 [[TMP16]], 4
; CHECK-NEXT:    [[TMP18:%.*]] = add i64 [[TMP17]], 0
; CHECK-NEXT:    [[TMP19:%.*]] = mul i64 [[TMP18]], 1
; CHECK-NEXT:    [[TMP20:%.*]] = add i64 [[INDEX6]], [[TMP19]]
; CHECK-NEXT:    [[TMP21:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP22:%.*]] = mul i64 [[TMP21]], 8
; CHECK-NEXT:    [[TMP23:%.*]] = add i64 [[TMP22]], 0
; CHECK-NEXT:    [[TMP24:%.*]] = mul i64 [[TMP23]], 1
; CHECK-NEXT:    [[TMP25:%.*]] = add i64 [[INDEX6]], [[TMP24]]
; CHECK-NEXT:    [[TMP26:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP27:%.*]] = mul i64 [[TMP26]], 12
; CHECK-NEXT:    [[TMP28:%.*]] = add i64 [[TMP27]], 0
; CHECK-NEXT:    [[TMP29:%.*]] = mul i64 [[TMP28]], 1
; CHECK-NEXT:    [[TMP30:%.*]] = add i64 [[INDEX6]], [[TMP29]]
; CHECK-NEXT:    [[TMP31:%.*]] = getelementptr i32, i32* [[COND_PTR:%.*]], i64 [[TMP15]]
; CHECK-NEXT:    [[TMP32:%.*]] = getelementptr i32, i32* [[COND_PTR]], i64 [[TMP20]]
; CHECK-NEXT:    [[TMP33:%.*]] = getelementptr i32, i32* [[COND_PTR]], i64 [[TMP25]]
; CHECK-NEXT:    [[TMP34:%.*]] = getelementptr i32, i32* [[COND_PTR]], i64 [[TMP30]]
; CHECK-NEXT:    [[TMP35:%.*]] = getelementptr i32, i32* [[TMP31]], i32 0
; CHECK-NEXT:    [[TMP36:%.*]] = bitcast i32* [[TMP35]] to <vscale x 4 x i32>*
; CHECK-NEXT:    [[WIDE_MASKED_LOAD:%.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>* [[TMP36]], i32 4, <vscale x 4 x i1> [[ACTIVE_LANE_MASK7]], <vscale x 4 x i32> poison)
; CHECK-NEXT:    [[TMP37:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP38:%.*]] = mul i64 [[TMP37]], 4
; CHECK-NEXT:    [[TMP39:%.*]] = getelementptr i32, i32* [[TMP31]], i64 [[TMP38]]
; CHECK-NEXT:    [[TMP40:%.*]] = bitcast i32* [[TMP39]] to <vscale x 4 x i32>*
; CHECK-NEXT:    [[WIDE_MASKED_LOAD11:%.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>* [[TMP40]], i32 4, <vscale x 4 x i1> [[ACTIVE_LANE_MASK8]], <vscale x 4 x i32> poison)
; CHECK-NEXT:    [[TMP41:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP42:%.*]] = mul i64 [[TMP41]], 8
; CHECK-NEXT:    [[TMP43:%.*]] = getelementptr i32, i32* [[TMP31]], i64 [[TMP42]]
; CHECK-NEXT:    [[TMP44:%.*]] = bitcast i32* [[TMP43]] to <vscale x 4 x i32>*
; CHECK-NEXT:    [[WIDE_MASKED_LOAD12:%.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>* [[TMP44]], i32 4, <vscale x 4 x i1> [[ACTIVE_LANE_MASK9]], <vscale x 4 x i32> poison)
; CHECK-NEXT:    [[TMP45:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP46:%.*]] = mul i64 [[TMP45]], 12
; CHECK-NEXT:    [[TMP47:%.*]] = getelementptr i32, i32* [[TMP31]], i64 [[TMP46]]
; CHECK-NEXT:    [[TMP48:%.*]] = bitcast i32* [[TMP47]] to <vscale x 4 x i32>*
; CHECK-NEXT:    [[WIDE_MASKED_LOAD13:%.*]] = call <vscale x 4 x i32> @llvm.masked.load.nxv4i32.p0nxv4i32(<vscale x 4 x i32>* [[TMP48]], i32 4, <vscale x 4 x i1> [[ACTIVE_LANE_MASK10]], <vscale x 4 x i32> poison)
; CHECK-NEXT:    [[TMP49:%.*]] = icmp ne <vscale x 4 x i32> [[WIDE_MASKED_LOAD]], zeroinitializer
; CHECK-NEXT:    [[TMP50:%.*]] = icmp ne <vscale x 4 x i32> [[WIDE_MASKED_LOAD11]], zeroinitializer
; CHECK-NEXT:    [[TMP51:%.*]] = icmp ne <vscale x 4 x i32> [[WIDE_MASKED_LOAD12]], zeroinitializer
; CHECK-NEXT:    [[TMP52:%.*]] = icmp ne <vscale x 4 x i32> [[WIDE_MASKED_LOAD13]], zeroinitializer
; CHECK-NEXT:    [[TMP53:%.*]] = getelementptr i32, i32* [[PTR:%.*]], i64 [[TMP15]]
; CHECK-NEXT:    [[TMP54:%.*]] = getelementptr i32, i32* [[PTR]], i64 [[TMP20]]
; CHECK-NEXT:    [[TMP55:%.*]] = getelementptr i32, i32* [[PTR]], i64 [[TMP25]]
; CHECK-NEXT:    [[TMP56:%.*]] = getelementptr i32, i32* [[PTR]], i64 [[TMP30]]
; CHECK-NEXT:    [[TMP57:%.*]] = select <vscale x 4 x i1> [[ACTIVE_LANE_MASK7]], <vscale x 4 x i1> [[TMP49]], <vscale x 4 x i1> zeroinitializer
; CHECK-NEXT:    [[TMP58:%.*]] = select <vscale x 4 x i1> [[ACTIVE_LANE_MASK8]], <vscale x 4 x i1> [[TMP50]], <vscale x 4 x i1> zeroinitializer
; CHECK-NEXT:    [[TMP59:%.*]] = select <vscale x 4 x i1> [[ACTIVE_LANE_MASK9]], <vscale x 4 x i1> [[TMP51]], <vscale x 4 x i1> zeroinitializer
; CHECK-NEXT:    [[TMP60:%.*]] = select <vscale x 4 x i1> [[ACTIVE_LANE_MASK10]], <vscale x 4 x i1> [[TMP52]], <vscale x 4 x i1> zeroinitializer
; CHECK-NEXT:    [[TMP61:%.*]] = getelementptr i32, i32* [[TMP53]], i32 0
; CHECK-NEXT:    [[TMP62:%.*]] = bitcast i32* [[TMP61]] to <vscale x 4 x i32>*
; CHECK-NEXT:    call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> [[BROADCAST_SPLAT]], <vscale x 4 x i32>* [[TMP62]], i32 4, <vscale x 4 x i1> [[TMP57]])
; CHECK-NEXT:    [[TMP63:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP64:%.*]] = mul i64 [[TMP63]], 4
; CHECK-NEXT:    [[TMP65:%.*]] = getelementptr i32, i32* [[TMP53]], i64 [[TMP64]]
; CHECK-NEXT:    [[TMP66:%.*]] = bitcast i32* [[TMP65]] to <vscale x 4 x i32>*
; CHECK-NEXT:    call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> [[BROADCAST_SPLAT15]], <vscale x 4 x i32>* [[TMP66]], i32 4, <vscale x 4 x i1> [[TMP58]])
; CHECK-NEXT:    [[TMP67:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP68:%.*]] = mul i64 [[TMP67]], 8
; CHECK-NEXT:    [[TMP69:%.*]] = getelementptr i32, i32* [[TMP53]], i64 [[TMP68]]
; CHECK-NEXT:    [[TMP70:%.*]] = bitcast i32* [[TMP69]] to <vscale x 4 x i32>*
; CHECK-NEXT:    call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> [[BROADCAST_SPLAT17]], <vscale x 4 x i32>* [[TMP70]], i32 4, <vscale x 4 x i1> [[TMP59]])
; CHECK-NEXT:    [[TMP71:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP72:%.*]] = mul i64 [[TMP71]], 12
; CHECK-NEXT:    [[TMP73:%.*]] = getelementptr i32, i32* [[TMP53]], i64 [[TMP72]]
; CHECK-NEXT:    [[TMP74:%.*]] = bitcast i32* [[TMP73]] to <vscale x 4 x i32>*
; CHECK-NEXT:    call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> [[BROADCAST_SPLAT19]], <vscale x 4 x i32>* [[TMP74]], i32 4, <vscale x 4 x i1> [[TMP60]])
; CHECK-NEXT:    [[TMP75:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP76:%.*]] = mul i64 [[TMP75]], 16
; CHECK-NEXT:    [[INDEX_NEXT20]] = add i64 [[INDEX6]], [[TMP76]]
; CHECK-NEXT:    [[TMP77:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP78:%.*]] = mul i64 [[TMP77]], 4
; CHECK-NEXT:    [[INDEX_PART_NEXT22:%.*]] = add i64 [[INDEX_NEXT20]], [[TMP78]]
; CHECK-NEXT:    [[TMP79:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP80:%.*]] = mul i64 [[TMP79]], 8
; CHECK-NEXT:    [[INDEX_PART_NEXT23:%.*]] = add i64 [[INDEX_NEXT20]], [[TMP80]]
; CHECK-NEXT:    [[TMP81:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP82:%.*]] = mul i64 [[TMP81]], 12
; CHECK-NEXT:    [[INDEX_PART_NEXT24:%.*]] = add i64 [[INDEX_NEXT20]], [[TMP82]]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK25]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[INDEX_NEXT20]], i64 [[UMAX]])
; CHECK-NEXT:    [[ACTIVE_LANE_MASK26]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[INDEX_PART_NEXT22]], i64 [[UMAX]])
; CHECK-NEXT:    [[ACTIVE_LANE_MASK27]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[INDEX_PART_NEXT23]], i64 [[UMAX]])
; CHECK-NEXT:    [[ACTIVE_LANE_MASK28]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[INDEX_PART_NEXT24]], i64 [[UMAX]])
; CHECK-NEXT:    [[TMP83:%.*]] = xor <vscale x 4 x i1> [[ACTIVE_LANE_MASK25]], shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[TMP84:%.*]] = xor <vscale x 4 x i1> [[ACTIVE_LANE_MASK26]], shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[TMP85:%.*]] = xor <vscale x 4 x i1> [[ACTIVE_LANE_MASK27]], shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[TMP86:%.*]] = xor <vscale x 4 x i1> [[ACTIVE_LANE_MASK28]], shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, i64 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer)
; CHECK-NEXT:    [[TMP87:%.*]] = extractelement <vscale x 4 x i1> [[TMP83]], i32 0
; CHECK-NEXT:    br i1 [[TMP87]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP4:![0-9]+]]
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
