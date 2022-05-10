; RUN: opt -loop-vectorize -S < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

define void @trip7_i64(i64* noalias nocapture noundef %dst, i64* noalias nocapture noundef readonly %src) #0 {
; CHECK-LABEL: @trip7_i64(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK:         [[ACTIVE_LANE_MASK:%.*]] = phi <vscale x 2 x i1> [ {{%.*}}, %vector.ph ], [ [[ACTIVE_LANE_MASK_NEXT:%.*]], %vector.body ]
; CHECK:         {{%.*}} = call <vscale x 2 x i64> @llvm.masked.load.nxv2i64.p0nxv2i64(<vscale x 2 x i64>* {{%.*}}, i32 8, <vscale x 2 x i1> [[ACTIVE_LANE_MASK]], <vscale x 2 x i64> poison)
; CHECK:         {{%.*}} = call <vscale x 2 x i64> @llvm.masked.load.nxv2i64.p0nxv2i64(<vscale x 2 x i64>* {{%.*}}, i32 8, <vscale x 2 x i1> [[ACTIVE_LANE_MASK]], <vscale x 2 x i64> poison)
; CHECK:         call void @llvm.masked.store.nxv2i64.p0nxv2i64(<vscale x 2 x i64> {{%.*}}, <vscale x 2 x i64>* {{%.*}}, i32 8, <vscale x 2 x i1> [[ACTIVE_LANE_MASK]])
; CHECK:         [[VSCALE:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[VF:%.*]] = mul i64 [[VSCALE]], 2
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[VF]]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK_NEXT]] = call <vscale x 2 x i1> @llvm.get.active.lane.mask.nxv2i1.i64(i64 [[INDEX_NEXT]], i64 7)
; CHECK-NEXT:    [[ACTIVE_LANE_MASK_NOT:%.*]] = xor <vscale x 2 x i1> [[ACTIVE_LANE_MASK_NEXT]], shufflevector (<vscale x 2 x i1> insertelement (<vscale x 2 x i1> poison, i1 true, i32 0), <vscale x 2 x i1> poison, <vscale x 2 x i32> zeroinitializer)
; CHECK-NEXT:    [[COND:%.*]] = extractelement <vscale x 2 x i1> [[ACTIVE_LANE_MASK_NOT]], i32 0
; CHECK-NEXT:    br i1 [[COND]], label %middle.block, label %vector.body
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
  %exitcond.not = icmp eq i64 %inc, 7
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

define void @trip5_i8(i8* noalias nocapture noundef %dst, i8* noalias nocapture noundef readonly %src) #0 {
; CHECK-LABEL: @trip5_i8(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK:         [[ACTIVE_LANE_MASK:%.*]] = phi <vscale x 16 x i1> [ {{%.*}}, %vector.ph ], [ [[ACTIVE_LANE_MASK_NEXT:%.*]], %vector.body ]
; CHECK:         {{%.*}} = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0nxv16i8(<vscale x 16 x i8>* {{%.*}}, i32 1, <vscale x 16 x i1> [[ACTIVE_LANE_MASK]], <vscale x 16 x i8> poison)
; CHECK:         {{%.*}} = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0nxv16i8(<vscale x 16 x i8>* {{%.*}}, i32 1, <vscale x 16 x i1> [[ACTIVE_LANE_MASK]], <vscale x 16 x i8> poison)
; CHECK:         call void @llvm.masked.store.nxv16i8.p0nxv16i8(<vscale x 16 x i8> {{%.*}}, <vscale x 16 x i8>* {{%.*}}, i32 1, <vscale x 16 x i1> [[ACTIVE_LANE_MASK]])
; CHECK:         [[VSCALE:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[VF:%.*]] = mul i64 [[VSCALE]], 16
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], [[VF]]
; CHECK-NEXT:    [[ACTIVE_LANE_MASK_NEXT]] = call <vscale x 16 x i1> @llvm.get.active.lane.mask.nxv16i1.i64(i64 [[INDEX_NEXT]], i64 5)
; CHECK-NEXT:    [[ACTIVE_LANE_MASK_NOT:%.*]] = xor <vscale x 16 x i1> [[ACTIVE_LANE_MASK_NEXT]], shufflevector (<vscale x 16 x i1> insertelement (<vscale x 16 x i1> poison, i1 true, i32 0), <vscale x 16 x i1> poison, <vscale x 16 x i32> zeroinitializer)
; CHECK-NEXT:    br i1 true, label %middle.block, label %vector.body
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.08 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i8, i8* %src, i64 %i.08
  %0 = load i8, i8* %arrayidx, align 1
  %mul = shl i8 %0, 1
  %arrayidx1 = getelementptr inbounds i8, i8* %dst, i64 %i.08
  %1 = load i8, i8* %arrayidx1, align 1
  %add = add i8 %mul, %1
  store i8 %add, i8* %arrayidx1, align 1
  %inc = add nuw nsw i64 %i.08, 1
  %exitcond.not = icmp eq i64 %inc, 5
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

attributes #0 = { vscale_range(1,16) "target-features"="+sve" }
