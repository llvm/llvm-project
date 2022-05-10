; RUN: opt -loop-vectorize -S < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

define void @trip1024_i64(i64* noalias nocapture noundef %dst, i64* noalias nocapture noundef readonly %src) #0 {
; CHECK-LABEL: @trip1024_i64(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP1:%.*]] = mul i64 [[TMP0]], 2
; CHECK-NEXT:    [[TMP2:%.*]] = icmp ult i64 -1025, [[TMP1]]
; CHECK-NEXT:    br i1 [[TMP2]], label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP3:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP4:%.*]] = mul i64 [[TMP3]], 2
; CHECK-NEXT:    [[TMP5:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP6:%.*]] = mul i64 [[TMP5]], 2
; CHECK-NEXT:    [[TMP7:%.*]] = sub i64 [[TMP6]], 1
; CHECK-NEXT:    [[N_RND_UP:%.*]] = add i64 1024, [[TMP7]]
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = urem i64 [[N_RND_UP]], [[TMP4]]
; CHECK-NEXT:    [[N_VEC:%.*]] = sub i64 [[N_RND_UP]], [[N_MOD_VF]]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK:%.*]] = call <vscale x 2 x i1> @llvm.get.active.lane.mask.nxv2i1.i64(i64 0, i64 1024)
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK1:%.*]] = phi <vscale x 2 x i1> [ [[ACTIVE_LANE_MASK]], [[VECTOR_PH]] ], [ [[ACTIVE_LANE_MASK3:%.*]], [[VECTOR_BODY]] ]
; CHECK-NEXT:    [[TMP8:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP9:%.*]] = getelementptr inbounds i64, i64* [[SRC:%.*]], i64 [[TMP8]]
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr inbounds i64, i64* [[TMP9]], i32 0
; CHECK-NEXT:    [[TMP11:%.*]] = bitcast i64* [[TMP10]] to <vscale x 2 x i64>*
; CHECK-NEXT:    [[WIDE_MASKED_LOAD:%.*]] = call <vscale x 2 x i64> @llvm.masked.load.nxv2i64.p0nxv2i64(<vscale x 2 x i64>* [[TMP11]], i32 8, <vscale x 2 x i1> [[ACTIVE_LANE_MASK1]], <vscale x 2 x i64> poison)
; CHECK-NEXT:    [[TMP12:%.*]] = shl nsw <vscale x 2 x i64> [[WIDE_MASKED_LOAD]], shufflevector (<vscale x 2 x i64> insertelement (<vscale x 2 x i64> poison, i64 1, i32 0), <vscale x 2 x i64> poison, <vscale x 2 x i32> zeroinitializer)
; CHECK-NEXT:    [[TMP13:%.*]] = getelementptr inbounds i64, i64* [[DST:%.*]], i64 [[TMP8]]
; CHECK-NEXT:    [[TMP14:%.*]] = getelementptr inbounds i64, i64* [[TMP13]], i32 0
; CHECK-NEXT:    [[TMP15:%.*]] = bitcast i64* [[TMP14]] to <vscale x 2 x i64>*
; CHECK-NEXT:    [[WIDE_MASKED_LOAD2:%.*]] = call <vscale x 2 x i64> @llvm.masked.load.nxv2i64.p0nxv2i64(<vscale x 2 x i64>* [[TMP15]], i32 8, <vscale x 2 x i1> [[ACTIVE_LANE_MASK1]], <vscale x 2 x i64> poison)
; CHECK-NEXT:    [[TMP16:%.*]] = add nsw <vscale x 2 x i64> [[WIDE_MASKED_LOAD2]], [[TMP12]]
; CHECK-NEXT:    [[TMP17:%.*]] = bitcast i64* [[TMP14]] to <vscale x 2 x i64>*
; CHECK-NEXT:    call void @llvm.masked.store.nxv2i64.p0nxv2i64(<vscale x 2 x i64> [[TMP16]], <vscale x 2 x i64>* [[TMP17]], i32 8, <vscale x 2 x i1> [[ACTIVE_LANE_MASK1]])
; CHECK-NEXT:    [[TMP18:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP19:%.*]] = mul i64 [[TMP18]], 2
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[TMP19]]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK3]] = call <vscale x 2 x i1> @llvm.get.active.lane.mask.nxv2i1.i64(i64 [[INDEX_NEXT]], i64 1024)
; CHECK-NEXT:    [[TMP20:%.*]] = xor <vscale x 2 x i1> [[ACTIVE_LANE_MASK3]], shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer)
; CHECK-NEXT:    [[TMP21:%.*]] = extractelement <vscale x 2 x i1> [[TMP20]], i32 0
; CHECK-NEXT:    br i1 [[TMP21]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !llvm.loop [[LOOP0:![0-9]+]]
; CHECK:       middle.block:
; CHECK-NEXT:    br i1 true, label [[FOR_END:%.*]], label [[SCALAR_PH]]
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.06 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i64, i64* %src, i64 %i.06
  %0 = load i64, i64* %arrayidx, align 8
  %mul = shl nsw i64 %0, 1
  %arrayidx1 = getelementptr inbounds i64, i64* %dst, i64 %i.06
  %1 = load i64, i64* %arrayidx1, align 8
  %add = add nsw i64 %1, %mul
  store i64 %add, i64* %arrayidx1, align 8
  %inc = add nuw nsw i64 %i.06, 1
  %exitcond.not = icmp eq i64 %inc, 1024
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

attributes #0 = { vscale_range(1,16) "target-features"="+sve" optsize }
