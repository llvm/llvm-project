; RUN: opt -S -loop-vectorize < %s | FileCheck %s

; These tests ensure that tail-folding is enabled when the predicate.enable
; loop attribute is set to true.

target triple = "aarch64-unknown-linux-gnu"


define void @simple_memset(i32 %val, i32* %ptr, i64 %n) #0 {
; CHECK-LABEL: @simple_memset(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[UMAX:%.*]] = call i64 @llvm.umax.i64(i64 [[N:%.*]], i64 1)
; CHECK-NEXT:    [[TMP0:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP1:%.*]] = mul i64 [[TMP0]], 4
; CHECK-NEXT:    [[TMP2:%.*]] = sub i64 -1, [[UMAX]]
; CHECK-NEXT:    [[TMP3:%.*]] = icmp ult i64 [[TMP2]], [[TMP1]]
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
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[VAL:%.*]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[BROADCAST_SPLATINSERT]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX1:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT2:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP9:%.*]] = add i64 [[INDEX1]], 0
; CHECK-NEXT:    [[ACTIVE_LANE_MASK:%.*]] = call <vscale x 4 x i1> @llvm.get.active.lane.mask.nxv4i1.i64(i64 [[TMP9]], i64 [[UMAX]])
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr i32, i32* [[PTR:%.*]], i64 [[TMP9]]
; CHECK-NEXT:    [[TMP11:%.*]] = getelementptr i32, i32* [[TMP10]], i32 0
; CHECK-NEXT:    [[TMP12:%.*]] = bitcast i32* [[TMP11]] to <vscale x 4 x i32>*
; CHECK-NEXT:    call void @llvm.masked.store.nxv4i32.p0nxv4i32(<vscale x 4 x i32> [[BROADCAST_SPLAT]], <vscale x 4 x i32>* [[TMP12]], i32 4, <vscale x 4 x i1> [[ACTIVE_LANE_MASK]])
; CHECK-NEXT:    [[TMP13:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP14:%.*]] = mul i64 [[TMP13]], 4
; CHECK-NEXT:    [[INDEX_NEXT2]] = add i64 [[INDEX1]], [[TMP14]]
; CHECK-NEXT:    [[TMP15:%.*]] = icmp eq i64 [[INDEX_NEXT2]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP15]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
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


attributes #0 = { "target-features"="+sve" }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.predicate.enable", i1 true}
