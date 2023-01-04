; RUN: opt -passes=loop-vectorize -S < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

define void @trip7_i64(ptr noalias nocapture noundef %dst, ptr noalias nocapture noundef readonly %src) #0 {
; CHECK-LABEL: @trip7_i64(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK:         [[ACTIVE_LANE_MASK:%.*]] = phi <vscale x 2 x i1> [ {{%.*}}, %vector.ph ], [ [[ACTIVE_LANE_MASK_NEXT:%.*]], %vector.body ]
; CHECK:         {{%.*}} = call <vscale x 2 x i64> @llvm.masked.load.nxv2i64.p0(ptr {{%.*}}, i32 8, <vscale x 2 x i1> [[ACTIVE_LANE_MASK]], <vscale x 2 x i64> poison)
; CHECK:         {{%.*}} = call <vscale x 2 x i64> @llvm.masked.load.nxv2i64.p0(ptr {{%.*}}, i32 8, <vscale x 2 x i1> [[ACTIVE_LANE_MASK]], <vscale x 2 x i64> poison)
; CHECK:         call void @llvm.masked.store.nxv2i64.p0(<vscale x 2 x i64> {{%.*}}, ptr {{%.*}}, i32 8, <vscale x 2 x i1> [[ACTIVE_LANE_MASK]])
; CHECK:         [[VSCALE:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[VF:%.*]] = mul i64 [[VSCALE]], 2
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[VF]]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK_NEXT]] = call <vscale x 2 x i1> @llvm.get.active.lane.mask.nxv2i1.i64(i64 [[INDEX_NEXT]], i64 7)
; CHECK-NEXT:    [[ACTIVE_LANE_MASK_NOT:%.*]] = xor <vscale x 2 x i1> [[ACTIVE_LANE_MASK_NEXT]], shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i64 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer)
; CHECK-NEXT:    [[COND:%.*]] = extractelement <vscale x 2 x i1> [[ACTIVE_LANE_MASK_NOT]], i32 0
; CHECK-NEXT:    br i1 [[COND]], label %middle.block, label %vector.body
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.06 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i64, ptr %src, i64 %i.06
  %0 = load i64, ptr %arrayidx, align 8
  %mul = shl nsw i64 %0, 1
  %arrayidx1 = getelementptr inbounds i64, ptr %dst, i64 %i.06
  %1 = load i64, ptr %arrayidx1, align 8
  %add = add nsw i64 %1, %mul
  store i64 %add, ptr %arrayidx1, align 8
  %inc = add nuw nsw i64 %i.06, 1
  %exitcond.not = icmp eq i64 %inc, 7
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

define void @trip5_i8(ptr noalias nocapture noundef %dst, ptr noalias nocapture noundef readonly %src) #0 {
; CHECK-LABEL: @trip5_i8(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[FOR_BODY:%.*]]
; CHECK:       for.body:
; CHECK-NEXT:    [[I_08:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[INC:%.*]], [[FOR_BODY]] ]
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i8, ptr [[SRC:%.*]], i64 [[I_08]]
; CHECK-NEXT:    [[TMP0:%.*]] = load i8, ptr [[ARRAYIDX]], align 1
; CHECK-NEXT:    [[MUL:%.*]] = shl i8 [[TMP0]], 1
; CHECK-NEXT:    [[ARRAYIDX1:%.*]] = getelementptr inbounds i8, ptr [[DST:%.*]], i64 [[I_08]]
; CHECK-NEXT:    [[TMP1:%.*]] = load i8, ptr [[ARRAYIDX1]], align 1
; CHECK-NEXT:    [[ADD:%.*]] = add i8 [[MUL]], [[TMP1]]
; CHECK-NEXT:    store i8 [[ADD]], ptr [[ARRAYIDX1]], align 1
; CHECK-NEXT:    [[INC]] = add nuw nsw i64 [[I_08]], 1
; CHECK-NEXT:    [[EXITCOND_NOT:%.*]] = icmp eq i64 [[INC]], 5
; CHECK-NEXT:    br i1 [[EXITCOND_NOT]], label [[FOR_END:%.*]], label [[FOR_BODY]]
; CHECK:       for.end:
; CHECK-NEXT:    ret void
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.08 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %src, i64 %i.08
  %0 = load i8, ptr %arrayidx, align 1
  %mul = shl i8 %0, 1
  %arrayidx1 = getelementptr inbounds i8, ptr %dst, i64 %i.08
  %1 = load i8, ptr %arrayidx1, align 1
  %add = add i8 %mul, %1
  store i8 %add, ptr %arrayidx1, align 1
  %inc = add nuw nsw i64 %i.08, 1
  %exitcond.not = icmp eq i64 %inc, 5
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

attributes #0 = { vscale_range(1,16) "target-features"="+sve" }
