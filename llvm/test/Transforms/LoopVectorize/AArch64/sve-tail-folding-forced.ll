; RUN: opt -S -loop-vectorize < %s | FileCheck %s

; These tests ensure that tail-folding is enabled when the predicate.enable
; loop attribute is set to true.

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


attributes #0 = { "target-features"="+sve" }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.predicate.enable", i1 true}
