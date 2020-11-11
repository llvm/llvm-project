; RUN: opt -mtriple=thumbv8.1m.main -mve-tail-predication -tail-predication=enabled -mattr=+mve,+lob %s -S -o - | FileCheck %s

; CHECK-LABEL: mul_v16i8
; CHECK-NOT: %num.elements = add i32 %trip.count.minus.1, 1
; CHECK: vector.body:
; CHECK: %index = phi i32
; CHECK: [[ELEMS:%[^ ]+]] = phi i32 [ %N, %vector.ph ], [ [[REMAINING:%[^ ]+]], %vector.body ]
; CHECK: [[VCTP:%[^ ]+]] = call <16 x i1> @llvm.arm.mve.vctp8(i32 [[ELEMS]])
; CHECK: [[REMAINING]] = sub i32 [[ELEMS]], 16
; CHECK: [[LD0:%[^ ]+]] = tail call <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* {{.*}}, i32 4, <16 x i1> [[VCTP]], <16 x i8> undef)
; CHECK: [[LD1:%[^ ]+]] = tail call <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* {{.*}}, i32 4, <16 x i1> [[VCTP]], <16 x i8> undef)
; CHECK: tail call void @llvm.masked.store.v16i8.p0v16i8(<16 x i8> {{.*}}, <16 x i8>* {{.*}}, i32 4, <16 x i1> [[VCTP]])
define dso_local arm_aapcs_vfpcc void @mul_v16i8(i8* noalias nocapture readonly %a, i8* noalias nocapture readonly %b, i8* noalias nocapture %c, i32 %N) {
entry:
  %cmp8 = icmp eq i32 %N, 0
  %tmp8 = add i32 %N, 15
  %tmp9 = lshr i32 %tmp8, 4
  %tmp10 = shl nuw i32 %tmp9, 4
  %tmp11 = add i32 %tmp10, -16
  %tmp12 = lshr i32 %tmp11, 4
  %tmp13 = add nuw nsw i32 %tmp12, 1
  br i1 %cmp8, label %for.cond.cleanup, label %vector.ph

vector.ph:                                        ; preds = %entry
  %trip.count.minus.1 = add i32 %N, -1
  %broadcast.splatinsert10 = insertelement <16 x i32> undef, i32 %trip.count.minus.1, i32 0
  %broadcast.splat11 = shufflevector <16 x i32> %broadcast.splatinsert10, <16 x i32> undef, <16 x i32> zeroinitializer
  %start = call i32 @llvm.start.loop.iterations.i32(i32 %tmp13)
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %tmp14 = phi i32 [ %start, %vector.ph ], [ %tmp15, %vector.body ]
  %broadcast.splatinsert = insertelement <16 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <16 x i32> %broadcast.splatinsert, <16 x i32> undef, <16 x i32> zeroinitializer
  %induction = or <16 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %tmp = getelementptr inbounds i8, i8* %a, i32 %index

;  %tmp1 = icmp ule <16 x i32> %induction, %broadcast.splat11
  %active.lane.mask = call <16 x i1> @llvm.get.active.lane.mask.v16i1.i32(i32 %index, i32 %N)

  %tmp2 = bitcast i8* %tmp to <16 x i8>*
  %wide.masked.load = tail call <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* %tmp2, i32 4, <16 x i1> %active.lane.mask, <16 x i8> undef)
  %tmp3 = getelementptr inbounds i8, i8* %b, i32 %index
  %tmp4 = bitcast i8* %tmp3 to <16 x i8>*
  %wide.masked.load2 = tail call <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* %tmp4, i32 4, <16 x i1> %active.lane.mask, <16 x i8> undef)
  %mul = mul nsw <16 x i8> %wide.masked.load2, %wide.masked.load
  %tmp6 = getelementptr inbounds i8, i8* %c, i32 %index
  %tmp7 = bitcast i8* %tmp6 to <16 x i8>*
  tail call void @llvm.masked.store.v16i8.p0v16i8(<16 x i8> %mul, <16 x i8>* %tmp7, i32 4, <16 x i1> %active.lane.mask)
  %index.next = add i32 %index, 16
  %tmp15 = call i32 @llvm.loop.decrement.reg.i32(i32 %tmp14, i32 1)
  %tmp16 = icmp ne i32 %tmp15, 0
  br i1 %tmp16, label %vector.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %vector.body, %entry
  ret void
}

; CHECK-LABEL: mul_v8i16
; CHECK-NOT: %num.elements = add i32 %trip.count.minus.1, 1
; CHECK: vector.body:
; CHECK: %index = phi i32
; CHECK: [[ELEMS:%[^ ]+]] = phi i32 [ %N, %vector.ph ], [ [[REMAINING:%[^ ]+]], %vector.body ]
; CHECK: [[VCTP:%[^ ]+]] = call <8 x i1> @llvm.arm.mve.vctp16(i32 [[ELEMS]])
; CHECK: [[REMAINING]] = sub i32 [[ELEMS]], 8
; CHECK: [[LD0:%[^ ]+]] = tail call <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* {{.*}}, i32 4, <8 x i1> [[VCTP]], <8 x i16> undef)
; CHECK: [[LD1:%[^ ]+]] = tail call <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* {{.*}}, i32 4, <8 x i1> [[VCTP]], <8 x i16> undef)
; CHECK: tail call void @llvm.masked.store.v8i16.p0v8i16(<8 x i16> {{.*}}, <8 x i16>* {{.*}}, i32 4, <8 x i1> [[VCTP]])
define dso_local arm_aapcs_vfpcc void @mul_v8i16(i16* noalias nocapture readonly %a, i16* noalias nocapture readonly %b, i16* noalias nocapture %c, i32 %N) {
entry:
  %cmp8 = icmp eq i32 %N, 0
  %tmp8 = add i32 %N, 7
  %tmp9 = lshr i32 %tmp8, 3
  %tmp10 = shl nuw i32 %tmp9, 3
  %tmp11 = add i32 %tmp10, -8
  %tmp12 = lshr i32 %tmp11, 3
  %tmp13 = add nuw nsw i32 %tmp12, 1
  br i1 %cmp8, label %for.cond.cleanup, label %vector.ph

vector.ph:                                        ; preds = %entry
  %trip.count.minus.1 = add i32 %N, -1
  %broadcast.splatinsert10 = insertelement <8 x i32> undef, i32 %trip.count.minus.1, i32 0
  %broadcast.splat11 = shufflevector <8 x i32> %broadcast.splatinsert10, <8 x i32> undef, <8 x i32> zeroinitializer
  %start = call i32 @llvm.start.loop.iterations.i32(i32 %tmp13)
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %tmp14 = phi i32 [ %start, %vector.ph ], [ %tmp15, %vector.body ]
  %broadcast.splatinsert = insertelement <8 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <8 x i32> %broadcast.splatinsert, <8 x i32> undef, <8 x i32> zeroinitializer
  %induction = add <8 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %tmp = getelementptr inbounds i16, i16* %a, i32 %index

;  %tmp1 = icmp ule <8 x i32> %induction, %broadcast.splat11
  %active.lane.mask = call <8 x i1> @llvm.get.active.lane.mask.v8i1.i32(i32 %index, i32 %N)

  %tmp2 = bitcast i16* %tmp to <8 x i16>*
  %wide.masked.load = tail call <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* %tmp2, i32 4, <8 x i1> %active.lane.mask, <8 x i16> undef)
  %tmp3 = getelementptr inbounds i16, i16* %b, i32 %index
  %tmp4 = bitcast i16* %tmp3 to <8 x i16>*
  %wide.masked.load2 = tail call <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* %tmp4, i32 4, <8 x i1> %active.lane.mask, <8 x i16> undef)
  %mul = mul nsw <8 x i16> %wide.masked.load2, %wide.masked.load
  %tmp6 = getelementptr inbounds i16, i16* %c, i32 %index
  %tmp7 = bitcast i16* %tmp6 to <8 x i16>*
  tail call void @llvm.masked.store.v8i16.p0v8i16(<8 x i16> %mul, <8 x i16>* %tmp7, i32 4, <8 x i1> %active.lane.mask)
  %index.next = add i32 %index, 8
  %tmp15 = call i32 @llvm.loop.decrement.reg.i32(i32 %tmp14, i32 1)
  %tmp16 = icmp ne i32 %tmp15, 0
  br i1 %tmp16, label %vector.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %vector.body, %entry
  ret void
}

; CHECK-LABEL: mul_v4i32
; CHECK-NOT: %num.elements = add i32 %trip.count.minus.1, 1
; CHECK: vector.body:
; CHECK: [[ELEMS:%[^ ]+]] = phi i32 [ %N, %vector.ph ], [ [[REMAINING:%[^ ]+]], %vector.body ]
; CHECK: [[VCTP:%[^ ]+]] = call <4 x i1> @llvm.arm.mve.vctp32(i32 [[ELEMS]])
; CHECK: [[REMAINING]] = sub i32 [[ELEMS]], 4
; CHECK: [[LD0:%[^ ]+]] = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* {{.*}}, i32 4, <4 x i1> [[VCTP]], <4 x i32> undef)
; CHECK: [[LD1:%[^ ]+]] = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* {{.*}}, i32 4, <4 x i1> [[VCTP]], <4 x i32> undef)
; CHECK: tail call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> {{.*}}, <4 x i32>* {{.*}}, i32 4, <4 x i1> [[VCTP]])
define dso_local arm_aapcs_vfpcc void @mul_v4i32(i32* noalias nocapture readonly %a, i32* noalias nocapture readonly %b, i32* noalias nocapture %c, i32 %N) {
entry:
  %cmp8 = icmp eq i32 %N, 0
  %tmp8 = add i32 %N, 3
  %tmp9 = lshr i32 %tmp8, 2
  %tmp10 = shl nuw i32 %tmp9, 2
  %tmp11 = add i32 %tmp10, -4
  %tmp12 = lshr i32 %tmp11, 2
  %tmp13 = add nuw nsw i32 %tmp12, 1
  br i1 %cmp8, label %for.cond.cleanup, label %vector.ph

vector.ph:                                        ; preds = %entry
  %trip.count.minus.1 = add i32 %N, -1
  %broadcast.splatinsert10 = insertelement <4 x i32> undef, i32 %trip.count.minus.1, i32 0
  %broadcast.splat11 = shufflevector <4 x i32> %broadcast.splatinsert10, <4 x i32> undef, <4 x i32> zeroinitializer
  %start = call i32 @llvm.start.loop.iterations.i32(i32 %tmp13)
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %tmp14 = phi i32 [ %start, %vector.ph ], [ %tmp15, %vector.body ]
  %broadcast.splatinsert = insertelement <4 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <4 x i32> %broadcast.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  %induction = or <4 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3>
  %tmp = getelementptr inbounds i32, i32* %a, i32 %index
 ; %tmp1 = icmp ule <4 x i32> %induction, %broadcast.splat11
  %tmp2 = bitcast i32* %tmp to <4 x i32>*
  %active.lane.mask = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %index, i32 %N)
  %wide.masked.load = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp2, i32 4, <4 x i1> %active.lane.mask, <4 x i32> undef)
  %tmp3 = getelementptr inbounds i32, i32* %b, i32 %index
  %tmp4 = bitcast i32* %tmp3 to <4 x i32>*
  %wide.masked.load2 = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp4, i32 4, <4 x i1> %active.lane.mask, <4 x i32> undef)
  %mul = mul nsw <4 x i32> %wide.masked.load2, %wide.masked.load
  %tmp6 = getelementptr inbounds i32, i32* %c, i32 %index
  %tmp7 = bitcast i32* %tmp6 to <4 x i32>*
  tail call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %mul, <4 x i32>* %tmp7, i32 4, <4 x i1> %active.lane.mask)
  %index.next = add i32 %index, 4
  %tmp15 = call i32 @llvm.loop.decrement.reg.i32(i32 %tmp14, i32 1)
  %tmp16 = icmp ne i32 %tmp15, 0
  br i1 %tmp16, label %vector.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %vector.body, %entry
  ret void
}

; CHECK-LABEL: split_vector
; CHECK-NOT: %num.elements = add i32 %trip.count.minus.1, 1
; CHECK: vector.body:
; CHECK: %index = phi i32
; CHECK: [[ELEMS:%[^ ]+]] = phi i32 [ %N, %vector.ph ], [ [[REMAINING:%[^ ]+]], %vector.body ]
; CHECK: [[VCTP:%[^ ]+]] = call <4 x i1> @llvm.arm.mve.vctp32(i32 [[ELEMS]])
; CHECK: [[REMAINING]] = sub i32 [[ELEMS]], 4
; CHECK: [[LD0:%[^ ]+]] = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* {{.*}}, i32 4, <4 x i1> [[VCTP]], <4 x i32> undef)
; CHECK: [[LD1:%[^ ]+]] = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* {{.*}}, i32 4, <4 x i1> [[VCTP]], <4 x i32> undef)
; CHECK: tail call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> {{.*}}, <4 x i32>* {{.*}}, i32 4, <4 x i1> [[VCTP]])
define dso_local arm_aapcs_vfpcc void @split_vector(i32* noalias nocapture readonly %a, i32* noalias nocapture readonly %b, i32* noalias nocapture %c, i32 %N) {
entry:
  %cmp8 = icmp eq i32 %N, 0
  %tmp8 = add i32 %N, 3
  %tmp9 = lshr i32 %tmp8, 2
  %tmp10 = shl nuw i32 %tmp9, 2
  %tmp11 = add i32 %tmp10, -4
  %tmp12 = lshr i32 %tmp11, 2
  %tmp13 = add nuw nsw i32 %tmp12, 1
  br i1 %cmp8, label %for.cond.cleanup, label %vector.ph

vector.ph:                                        ; preds = %entry
  %trip.count.minus.1 = add i32 %N, -1
  %broadcast.splatinsert10 = insertelement <4 x i32> undef, i32 %trip.count.minus.1, i32 0
  %broadcast.splat11 = shufflevector <4 x i32> %broadcast.splatinsert10, <4 x i32> undef, <4 x i32> zeroinitializer
  %start = call i32 @llvm.start.loop.iterations.i32(i32 %tmp13)
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %tmp14 = phi i32 [ %start, %vector.ph ], [ %tmp15, %vector.body ]
  %broadcast.splatinsert = insertelement <4 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <4 x i32> %broadcast.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  %induction = add <4 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3>
  %tmp = getelementptr inbounds i32, i32* %a, i32 %index
;  %tmp1 = icmp ule <4 x i32> %induction, %broadcast.splat11
  %active.lane.mask = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %index, i32 %N)
  %tmp2 = bitcast i32* %tmp to <4 x i32>*
  %wide.masked.load = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp2, i32 4, <4 x i1> %active.lane.mask, <4 x i32> undef)
  %extract.1.low = shufflevector <4 x i32> %wide.masked.load, <4 x i32> undef, < 2 x i32> < i32 0, i32 2>
  %extract.1.high = shufflevector <4 x i32> %wide.masked.load, <4 x i32> undef, < 2 x i32> < i32 1, i32 3>
  %tmp3 = getelementptr inbounds i32, i32* %b, i32 %index
  %tmp4 = bitcast i32* %tmp3 to <4 x i32>*
  %wide.masked.load2 = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp4, i32 4, <4 x i1> %active.lane.mask, <4 x i32> undef)
  %extract.2.low = shufflevector <4 x i32> %wide.masked.load2, <4 x i32> undef, < 2 x i32> < i32 0, i32 2>
  %extract.2.high = shufflevector <4 x i32> %wide.masked.load2, <4 x i32> undef, < 2 x i32> < i32 1, i32 3>
  %mul = mul nsw <2 x i32> %extract.1.low, %extract.2.low
  %sub = sub nsw <2 x i32> %extract.1.high, %extract.2.high
  %combine = shufflevector <2 x i32> %mul, <2 x i32> %sub, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp6 = getelementptr inbounds i32, i32* %c, i32 %index
  %tmp7 = bitcast i32* %tmp6 to <4 x i32>*
  tail call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %combine, <4 x i32>* %tmp7, i32 4, <4 x i1> %active.lane.mask)
  %index.next = add i32 %index, 4
  %tmp15 = call i32 @llvm.loop.decrement.reg.i32(i32 %tmp14, i32 1)
  %tmp16 = icmp ne i32 %tmp15, 0
  br i1 %tmp16, label %vector.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %vector.body, %entry
  ret void
}

; One of the loads now uses ult predicate.
; CHECK-LABEL: mismatch_load_pred
; CHECK: [[ELEMS:%[^ ]+]] = phi i32 [ %N, %vector.ph ], [ [[REMAINING:%[^ ]+]], %vector.body ]
; CHECK: [[VCTP:%[^ ]+]] = call <4 x i1> @llvm.arm.mve.vctp32(i32 [[ELEMS]])
; CHECK: [[REMAINING]] = sub i32 [[ELEMS]], 4
; CHECK: [[LD0:%[^ ]+]] = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* {{.*}}, i32 4, <4 x i1> [[VCTP]], <4 x i32> undef)
; CHECK: [[LD1:%[^ ]+]] = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* {{.*}}, i32 4, <4 x i1> %wrong, <4 x i32> undef)
; CHECK: tail call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> {{.*}}, <4 x i32>* {{.*}}, i32 4, <4 x i1> [[VCTP]])
define dso_local arm_aapcs_vfpcc void @mismatch_load_pred(i32* noalias nocapture readonly %a, i32* noalias nocapture readonly %b, i32* noalias nocapture %c, i32 %N) {
entry:
  %cmp8 = icmp eq i32 %N, 0
  %tmp8 = add i32 %N, 3
  %tmp9 = lshr i32 %tmp8, 2
  %tmp10 = shl nuw i32 %tmp9, 2
  %tmp11 = add i32 %tmp10, -4
  %tmp12 = lshr i32 %tmp11, 2
  %tmp13 = add nuw nsw i32 %tmp12, 1
  br i1 %cmp8, label %for.cond.cleanup, label %vector.ph

vector.ph:                                        ; preds = %entry
  %trip.count.minus.1 = add i32 %N, -1
  %broadcast.splatinsert10 = insertelement <4 x i32> undef, i32 %trip.count.minus.1, i32 0
  %broadcast.splat11 = shufflevector <4 x i32> %broadcast.splatinsert10, <4 x i32> undef, <4 x i32> zeroinitializer
  %start = call i32 @llvm.start.loop.iterations.i32(i32 %tmp13)
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %tmp14 = phi i32 [ %start, %vector.ph ], [ %tmp15, %vector.body ]
  %broadcast.splatinsert = insertelement <4 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <4 x i32> %broadcast.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  %induction = add <4 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3>
  %tmp = getelementptr inbounds i32, i32* %a, i32 %index

;  %tmp1 = icmp ule <4 x i32> %induction, %broadcast.splat11
  %active.lane.mask = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %index, i32 %N)

  %wrong = icmp ult <4 x i32> %induction, %broadcast.splat11
  %tmp2 = bitcast i32* %tmp to <4 x i32>*
  %wide.masked.load = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp2, i32 4, <4 x i1> %active.lane.mask, <4 x i32> undef)
  %tmp3 = getelementptr inbounds i32, i32* %b, i32 %index
  %tmp4 = bitcast i32* %tmp3 to <4 x i32>*
  %wide.masked.load12 = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp4, i32 4, <4 x i1> %wrong, <4 x i32> undef)
  %tmp5 = mul nsw <4 x i32> %wide.masked.load12, %wide.masked.load
  %tmp6 = getelementptr inbounds i32, i32* %c, i32 %index
  %tmp7 = bitcast i32* %tmp6 to <4 x i32>*
  tail call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %tmp5, <4 x i32>* %tmp7, i32 4, <4 x i1> %active.lane.mask)
  %index.next = add i32 %index, 4
  %tmp15 = call i32 @llvm.loop.decrement.reg.i32(i32 %tmp14, i32 1)
  %tmp16 = icmp ne i32 %tmp15, 0
  br i1 %tmp16, label %vector.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %vector.body, %entry
  ret void
}

; The store now uses ult predicate.
; CHECK-LABEL: mismatch_store_pred
; CHECK-NOT: %num.elements = add i32 %trip.count.minus.1, 1
; CHECK: vector.body:
; CHECK: %index = phi i32
; CHECK: [[ELEMS:%[^ ]+]] = phi i32 [ %N, %vector.ph ], [ [[REMAINING:%[^ ]+]], %vector.body ]
; CHECK: [[VCTP:%[^ ]+]] = call <4 x i1> @llvm.arm.mve.vctp32(i32 [[ELEMS]])
; CHECK: [[REMAINING]] = sub i32 [[ELEMS]], 4
; CHECK: [[LD0:%[^ ]+]] = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* {{.*}}, i32 4, <4 x i1> [[VCTP]], <4 x i32> undef)
; CHECK: [[LD1:%[^ ]+]] = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* {{.*}}, i32 4, <4 x i1> [[VCTP]], <4 x i32> undef)
; CHECK: tail call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> {{.*}}, <4 x i32>* {{.*}}, i32 4, <4 x i1> %wrong)
define dso_local arm_aapcs_vfpcc void @mismatch_store_pred(i32* noalias nocapture readonly %a, i32* noalias nocapture readonly %b, i32* noalias nocapture %c, i32 %N) {
entry:
  %cmp8 = icmp eq i32 %N, 0
  %tmp8 = add i32 %N, 3
  %tmp9 = lshr i32 %tmp8, 2
  %tmp10 = shl nuw i32 %tmp9, 2
  %tmp11 = add i32 %tmp10, -4
  %tmp12 = lshr i32 %tmp11, 2
  %tmp13 = add nuw nsw i32 %tmp12, 1
  br i1 %cmp8, label %for.cond.cleanup, label %vector.ph

vector.ph:                                        ; preds = %entry
  %trip.count.minus.1 = add i32 %N, -1
  %broadcast.splatinsert10 = insertelement <4 x i32> undef, i32 %trip.count.minus.1, i32 0
  %broadcast.splat11 = shufflevector <4 x i32> %broadcast.splatinsert10, <4 x i32> undef, <4 x i32> zeroinitializer
  %start = call i32 @llvm.start.loop.iterations.i32(i32 %tmp13)
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %tmp14 = phi i32 [ %start, %vector.ph ], [ %tmp15, %vector.body ]
  %broadcast.splatinsert = insertelement <4 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <4 x i32> %broadcast.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  %induction = add <4 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3>
  %tmp = getelementptr inbounds i32, i32* %a, i32 %index

;  %tmp1 = icmp ule <4 x i32> %induction, %broadcast.splat11
  %active.lane.mask = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %index, i32 %N)

  %wrong = icmp ult <4 x i32> %induction, %broadcast.splat11
  %tmp2 = bitcast i32* %tmp to <4 x i32>*
  %wide.masked.load = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp2, i32 4, <4 x i1> %active.lane.mask, <4 x i32> undef)
  %tmp3 = getelementptr inbounds i32, i32* %b, i32 %index
  %tmp4 = bitcast i32* %tmp3 to <4 x i32>*
  %wide.masked.load12 = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp4, i32 4, <4 x i1> %active.lane.mask, <4 x i32> undef)
  %tmp5 = mul nsw <4 x i32> %wide.masked.load12, %wide.masked.load
  %tmp6 = getelementptr inbounds i32, i32* %c, i32 %index
  %tmp7 = bitcast i32* %tmp6 to <4 x i32>*
  tail call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %tmp5, <4 x i32>* %tmp7, i32 4, <4 x i1> %wrong)
  %index.next = add i32 %index, 4
  %tmp15 = call i32 @llvm.loop.decrement.reg.i32(i32 %tmp14, i32 1)
  %tmp16 = icmp ne i32 %tmp15, 0
  br i1 %tmp16, label %vector.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %vector.body, %entry
  ret void
}

; TODO: Multiple intrinsics not yet supported.
; This is currently rejected, because if the vector body is unrolled, the step
; is not what we expect:
;
;   Step value 16 doesn't match vector width 4
;
; CHECK-LABEL: interleave4
; CHECK: vector.body:
; CHECK:  %active.lane.mask = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %index, i32 %N)
; CHECK:  %active.lane.mask{{.*}} = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %v7, i32 %N)
; CHECK:  %active.lane.mask{{.*}} = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %v8, i32 %N)
; CHECK:  %active.lane.mask{{.*}} = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %v9, i32 %N)
;
define dso_local void @interleave4(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C, i32 %N) local_unnamed_addr #0 {
entry:
  %cmp8 = icmp sgt i32 %N, 0
  %v0 = add i32 %N, 15
  %v1 = lshr i32 %v0, 4
  %v2 = shl nuw i32 %v1, 4
  %v3 = add i32 %v2, -16
  %v4 = lshr i32 %v3, 4
  %v5 = add nuw nsw i32 %v4, 1
  br i1 %cmp8, label %vector.ph, label %for.cond.cleanup


vector.ph:
  %trip.count.minus.1 = add i32 %N, -1
  %scevgep = getelementptr i32, i32* %A, i32 8
  %scevgep30 = getelementptr i32, i32* %C, i32 8
  %scevgep37 = getelementptr i32, i32* %B, i32 8
  %start = call i32 @llvm.start.loop.iterations.i32(i32 %v5)
  br label %vector.body

vector.body:
  %lsr.iv38 = phi i32* [ %scevgep39, %vector.body ], [ %scevgep37, %vector.ph ]
  %lsr.iv31 = phi i32* [ %scevgep32, %vector.body ], [ %scevgep30, %vector.ph ]
  %lsr.iv = phi i32* [ %scevgep25, %vector.body ], [ %scevgep, %vector.ph ]
  %index = phi i32 [ 0, %vector.ph ], [ %v14, %vector.body ]
  %v6 = phi i32 [ %start, %vector.ph ], [ %v15, %vector.body ]
  %lsr.iv3840 = bitcast i32* %lsr.iv38 to <4 x i32>*
  %lsr.iv3133 = bitcast i32* %lsr.iv31 to <4 x i32>*
  %lsr.iv26 = bitcast i32* %lsr.iv to <4 x i32>*
  %active.lane.mask = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %index, i32 %N)
  %v7 = add i32 %index, 4
  %active.lane.mask15 = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %v7, i32 %N)
  %v8 = add i32 %v7, 4
  %active.lane.mask16 = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %v8, i32 %N)
  %v9 = add i32 %v8, 4
  %active.lane.mask17 = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %v9, i32 %N)
  %scevgep42 = getelementptr <4 x i32>, <4 x i32>* %lsr.iv3840, i32 -2
  %wide.masked.load = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %scevgep42, i32 4, <4 x i1> %active.lane.mask, <4 x i32> undef)
  %scevgep43 = getelementptr <4 x i32>, <4 x i32>* %lsr.iv3840, i32 -1
  %wide.masked.load18 = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* nonnull %scevgep43, i32 4, <4 x i1> %active.lane.mask15, <4 x i32> undef)
  %wide.masked.load19 = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* nonnull %lsr.iv3840, i32 4, <4 x i1> %active.lane.mask16, <4 x i32> undef)
  %scevgep41 = getelementptr <4 x i32>, <4 x i32>* %lsr.iv3840, i32 1
  %wide.masked.load20 = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* nonnull %scevgep41, i32 4, <4 x i1> %active.lane.mask17, <4 x i32> undef)
  %scevgep34 = getelementptr <4 x i32>, <4 x i32>* %lsr.iv3133, i32 -2
  %wide.masked.load21 = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %scevgep34, i32 4, <4 x i1> %active.lane.mask, <4 x i32> undef)
  %scevgep35 = getelementptr <4 x i32>, <4 x i32>* %lsr.iv3133, i32 -1
  %wide.masked.load22 = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* nonnull %scevgep35, i32 4, <4 x i1> %active.lane.mask15, <4 x i32> undef)
  %wide.masked.load23 = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* nonnull %lsr.iv3133, i32 4, <4 x i1> %active.lane.mask16, <4 x i32> undef)
  %scevgep36 = getelementptr <4 x i32>, <4 x i32>* %lsr.iv3133, i32 1
  %wide.masked.load24 = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* nonnull %scevgep36, i32 4, <4 x i1> %active.lane.mask17, <4 x i32> undef)
  %v10 = add nsw <4 x i32> %wide.masked.load21, %wide.masked.load
  %v11 = add nsw <4 x i32> %wide.masked.load22, %wide.masked.load18
  %v12 = add nsw <4 x i32> %wide.masked.load23, %wide.masked.load19
  %v13 = add nsw <4 x i32> %wide.masked.load24, %wide.masked.load20
  %scevgep27 = getelementptr <4 x i32>, <4 x i32>* %lsr.iv26, i32 -2
  call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %v10, <4 x i32>* %scevgep27, i32 4, <4 x i1> %active.lane.mask)
  %scevgep28 = getelementptr <4 x i32>, <4 x i32>* %lsr.iv26, i32 -1
  call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %v11, <4 x i32>* %scevgep28, i32 4, <4 x i1> %active.lane.mask15)
  call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %v12, <4 x i32>* %lsr.iv26, i32 4, <4 x i1> %active.lane.mask16)
  %scevgep29 = getelementptr <4 x i32>, <4 x i32>* %lsr.iv26, i32 1
  call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %v13, <4 x i32>* %scevgep29, i32 4, <4 x i1> %active.lane.mask17)
  %scevgep25 = getelementptr i32, i32* %lsr.iv, i32 16
  %scevgep32 = getelementptr i32, i32* %lsr.iv31, i32 16
  %scevgep39 = getelementptr i32, i32* %lsr.iv38, i32 16
  %v14 = add i32 %v9, 4
  %v15 = call i32 @llvm.loop.decrement.reg.i32(i32 %v6, i32 1)
  %v16 = icmp ne i32 %v15, 0
  br i1 %v16, label %vector.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void
}

; CHECK-LABEL: const_expected_in_set_loop
; CHECK:       call <4 x i1> @llvm.get.active.lane.mask
; CHECK-NOT:   vctp
; CHECK:       ret void
;
define dso_local void @const_expected_in_set_loop(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C, i32 %N) local_unnamed_addr #0 {
entry:
  %cmp8 = icmp sgt i32 %N, 0
  %0 = add i32 %N, 3
  %1 = lshr i32 %0, 2
  %2 = shl nuw i32 %1, 2
  %3 = add i32 %2, -4
  %4 = lshr i32 %3, 2
  %5 = add nuw nsw i32 %4, 1
  br i1 %cmp8, label %vector.ph, label %for.cond.cleanup

vector.ph:
  %start = call i32 @llvm.start.loop.iterations.i32(i32 %5)
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %lsr.iv17 = phi i32* [ %scevgep18, %vector.body ], [ %A, %vector.ph ]
  %lsr.iv14 = phi i32* [ %scevgep15, %vector.body ], [ %C, %vector.ph ]
  %lsr.iv = phi i32* [ %scevgep, %vector.body ], [ %B, %vector.ph ]
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %6 = phi i32 [ %start, %vector.ph ], [ %8, %vector.body ]
  %lsr.iv13 = bitcast i32* %lsr.iv to <4 x i32>*
  %lsr.iv1416 = bitcast i32* %lsr.iv14 to <4 x i32>*
  %lsr.iv1719 = bitcast i32* %lsr.iv17 to <4 x i32>*

  %active.lane.mask = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %index, i32 42)

  %wide.masked.load = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %lsr.iv13, i32 4, <4 x i1> %active.lane.mask, <4 x i32> undef)
  %wide.masked.load12 = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %lsr.iv1416, i32 4, <4 x i1> %active.lane.mask, <4 x i32> undef)
  %7 = add nsw <4 x i32> %wide.masked.load12, %wide.masked.load
  call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %7, <4 x i32>* %lsr.iv1719, i32 4, <4 x i1> %active.lane.mask)
  %index.next = add i32 %index, 4
  %scevgep = getelementptr i32, i32* %lsr.iv, i32 4
  %scevgep15 = getelementptr i32, i32* %lsr.iv14, i32 4
  %scevgep18 = getelementptr i32, i32* %lsr.iv17, i32 4
  %8 = call i32 @llvm.loop.decrement.reg.i32(i32 %6, i32 1)
  %9 = icmp ne i32 %8, 0
  br i1 %9, label %vector.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %vector.body, %entry
  ret void
}

; CHECK-LABEL: tripcount_arg_not_invariant
; CHECK:       call <4 x i1> @llvm.get.active.lane.mask
; CHECK-NOT:   vctp
; CHECK:       ret void
;
define dso_local void @tripcount_arg_not_invariant(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C, i32 %N) local_unnamed_addr #0 {
entry:
  %cmp8 = icmp sgt i32 %N, 0
  %0 = add i32 %N, 3
  %1 = lshr i32 %0, 2
  %2 = shl nuw i32 %1, 2
  %3 = add i32 %2, -4
  %4 = lshr i32 %3, 2
  %5 = add nuw nsw i32 %4, 1
  br i1 %cmp8, label %vector.ph, label %for.cond.cleanup

vector.ph:                                        ; preds = %entry
  %trip.count.minus.1 = add i32 %N, -1
  %start = call i32 @llvm.start.loop.iterations.i32(i32 %5)
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %lsr.iv17 = phi i32* [ %scevgep18, %vector.body ], [ %A, %vector.ph ]
  %lsr.iv14 = phi i32* [ %scevgep15, %vector.body ], [ %C, %vector.ph ]
  %lsr.iv = phi i32* [ %scevgep, %vector.body ], [ %B, %vector.ph ]
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %6 = phi i32 [ %start, %vector.ph ], [ %8, %vector.body ]

  %lsr.iv13 = bitcast i32* %lsr.iv to <4 x i32>*
  %lsr.iv1416 = bitcast i32* %lsr.iv14 to <4 x i32>*
  %lsr.iv1719 = bitcast i32* %lsr.iv17 to <4 x i32>*

  %active.lane.mask = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %index, i32 %index)

  %wide.masked.load = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %lsr.iv13, i32 4, <4 x i1> %active.lane.mask, <4 x i32> undef)
  %wide.masked.load12 = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %lsr.iv1416, i32 4, <4 x i1> %active.lane.mask, <4 x i32> undef)
  %7 = add nsw <4 x i32> %wide.masked.load12, %wide.masked.load
  call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %7, <4 x i32>* %lsr.iv1719, i32 4, <4 x i1> %active.lane.mask)
  %index.next = add i32 %index, 4
  %scevgep = getelementptr i32, i32* %lsr.iv, i32 4
  %scevgep15 = getelementptr i32, i32* %lsr.iv14, i32 4
  %scevgep18 = getelementptr i32, i32* %lsr.iv17, i32 4
  %8 = call i32 @llvm.loop.decrement.reg.i32(i32 %6, i32 1)
  %9 = icmp ne i32 %8, 0
  ;br i1 %9, label %vector.body, label %for.cond.cleanup
  br i1 %9, label %vector.body, label %vector.ph

for.cond.cleanup:                                 ; preds = %vector.body, %entry
  ret void
}

; CHECK-LABEL: addrec_base_not_zero
; CHECK:       call <4 x i1> @llvm.get.active.lane.mask
; CHECK-NOT:   vctp
; CHECK:       ret void
;
define dso_local void @addrec_base_not_zero(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C, i32 %N) local_unnamed_addr #0 {
entry:
  %cmp8 = icmp sgt i32 %N, 0
  %0 = add i32 %N, 3
  %1 = lshr i32 %0, 2
  %2 = shl nuw i32 %1, 2
  %3 = add i32 %2, -4
  %4 = lshr i32 %3, 2
  %5 = add nuw nsw i32 %4, 1
  br i1 %cmp8, label %vector.ph, label %for.cond.cleanup

vector.ph:                                        ; preds = %entry
  %trip.count.minus.1 = add i32 %N, -1
  %start = call i32 @llvm.start.loop.iterations.i32(i32 %5)
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %lsr.iv17 = phi i32* [ %scevgep18, %vector.body ], [ %A, %vector.ph ]
  %lsr.iv14 = phi i32* [ %scevgep15, %vector.body ], [ %C, %vector.ph ]
  %lsr.iv = phi i32* [ %scevgep, %vector.body ], [ %B, %vector.ph ]

; AddRec base is not 0:
  %index = phi i32 [ 1, %vector.ph ], [ %index.next, %vector.body ]

  %6 = phi i32 [ %start, %vector.ph ], [ %8, %vector.body ]
  %lsr.iv13 = bitcast i32* %lsr.iv to <4 x i32>*
  %lsr.iv1416 = bitcast i32* %lsr.iv14 to <4 x i32>*
  %lsr.iv1719 = bitcast i32* %lsr.iv17 to <4 x i32>*
  %active.lane.mask = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %index, i32 %N)
  %wide.masked.load = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %lsr.iv13, i32 4, <4 x i1> %active.lane.mask, <4 x i32> undef)
  %wide.masked.load12 = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %lsr.iv1416, i32 4, <4 x i1> %active.lane.mask, <4 x i32> undef)
  %7 = add nsw <4 x i32> %wide.masked.load12, %wide.masked.load
  call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %7, <4 x i32>* %lsr.iv1719, i32 4, <4 x i1> %active.lane.mask)
  %index.next = add i32 %index, 4
  %scevgep = getelementptr i32, i32* %lsr.iv, i32 4
  %scevgep15 = getelementptr i32, i32* %lsr.iv14, i32 4
  %scevgep18 = getelementptr i32, i32* %lsr.iv17, i32 4
  %8 = call i32 @llvm.loop.decrement.reg.i32(i32 %6, i32 1)
  %9 = icmp ne i32 %8, 0
  ;br i1 %9, label %vector.body, label %for.cond.cleanup
  br i1 %9, label %vector.body, label %vector.ph

for.cond.cleanup:                                 ; preds = %vector.body, %entry
  ret void
}


declare <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>*, i32 immarg, <16 x i1>, <16 x i8>)
declare void @llvm.masked.store.v16i8.p0v16i8(<16 x i8>, <16 x i8>*, i32 immarg, <16 x i1>)
declare <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>*, i32 immarg, <8 x i1>, <8 x i16>)
declare void @llvm.masked.store.v8i16.p0v8i16(<8 x i16>, <8 x i16>*, i32 immarg, <8 x i1>)
declare <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>*, i32 immarg, <4 x i1>, <4 x i32>)
declare void @llvm.masked.store.v2i64.p0v2i64(<2 x i64>, <2 x i64>*, i32 immarg, <2 x i1>)
declare <2 x i64> @llvm.masked.load.v2i64.p0v2i64(<2 x i64>*, i32 immarg, <2 x i1>, <2 x i64>)
declare void @llvm.masked.store.v4i32.p0v4i32(<4 x i32>, <4 x i32>*, i32 immarg, <4 x i1>)
declare i32 @llvm.start.loop.iterations.i32(i32)
declare i32 @llvm.loop.decrement.reg.i32(i32, i32)
declare <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32, i32)
declare <8 x i1> @llvm.get.active.lane.mask.v8i1.i32(i32, i32)
declare <16 x i1> @llvm.get.active.lane.mask.v16i1.i32(i32, i32)
