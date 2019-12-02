; RUN: opt -mtriple=thumbv8.1m.main -mve-tail-predication -disable-mve-tail-predication=false -mattr=+mve,+lob %s -S -o - | FileCheck %s

; CHECK-LABEL: mul_v16i8
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
  call void @llvm.set.loop.iterations.i32(i32 %tmp13)
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %tmp14 = phi i32 [ %tmp13, %vector.ph ], [ %tmp15, %vector.body ]
  %broadcast.splatinsert = insertelement <16 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <16 x i32> %broadcast.splatinsert, <16 x i32> undef, <16 x i32> zeroinitializer
  %induction = add <16 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %tmp = getelementptr inbounds i8, i8* %a, i32 %index
  %tmp1 = icmp ule <16 x i32> %induction, %broadcast.splat11
  %tmp2 = bitcast i8* %tmp to <16 x i8>*
  %wide.masked.load = tail call <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* %tmp2, i32 4, <16 x i1> %tmp1, <16 x i8> undef)
  %tmp3 = getelementptr inbounds i8, i8* %b, i32 %index
  %tmp4 = bitcast i8* %tmp3 to <16 x i8>*
  %wide.masked.load2 = tail call <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* %tmp4, i32 4, <16 x i1> %tmp1, <16 x i8> undef)
  %mul = mul nsw <16 x i8> %wide.masked.load2, %wide.masked.load
  %tmp6 = getelementptr inbounds i8, i8* %c, i32 %index
  %tmp7 = bitcast i8* %tmp6 to <16 x i8>*
  tail call void @llvm.masked.store.v16i8.p0v16i8(<16 x i8> %mul, <16 x i8>* %tmp7, i32 4, <16 x i1> %tmp1)
  %index.next = add i32 %index, 16
  %tmp15 = call i32 @llvm.loop.decrement.reg.i32.i32.i32(i32 %tmp14, i32 1)
  %tmp16 = icmp ne i32 %tmp15, 0
  br i1 %tmp16, label %vector.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %vector.body, %entry
  ret void
}

; CHECK-LABEL: mul_v8i16
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
  call void @llvm.set.loop.iterations.i32(i32 %tmp13)
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %tmp14 = phi i32 [ %tmp13, %vector.ph ], [ %tmp15, %vector.body ]
  %broadcast.splatinsert = insertelement <8 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <8 x i32> %broadcast.splatinsert, <8 x i32> undef, <8 x i32> zeroinitializer
  %induction = add <8 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %tmp = getelementptr inbounds i16, i16* %a, i32 %index
  %tmp1 = icmp ule <8 x i32> %induction, %broadcast.splat11
  %tmp2 = bitcast i16* %tmp to <8 x i16>*
  %wide.masked.load = tail call <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* %tmp2, i32 4, <8 x i1> %tmp1, <8 x i16> undef)
  %tmp3 = getelementptr inbounds i16, i16* %b, i32 %index
  %tmp4 = bitcast i16* %tmp3 to <8 x i16>*
  %wide.masked.load2 = tail call <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* %tmp4, i32 4, <8 x i1> %tmp1, <8 x i16> undef)
  %mul = mul nsw <8 x i16> %wide.masked.load2, %wide.masked.load
  %tmp6 = getelementptr inbounds i16, i16* %c, i32 %index
  %tmp7 = bitcast i16* %tmp6 to <8 x i16>*
  tail call void @llvm.masked.store.v8i16.p0v8i16(<8 x i16> %mul, <8 x i16>* %tmp7, i32 4, <8 x i1> %tmp1)
  %index.next = add i32 %index, 8
  %tmp15 = call i32 @llvm.loop.decrement.reg.i32.i32.i32(i32 %tmp14, i32 1)
  %tmp16 = icmp ne i32 %tmp15, 0
  br i1 %tmp16, label %vector.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %vector.body, %entry
  ret void
}

; CHECK-LABEL: mul_v4i32
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
  call void @llvm.set.loop.iterations.i32(i32 %tmp13)
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %tmp14 = phi i32 [ %tmp13, %vector.ph ], [ %tmp15, %vector.body ]
  %broadcast.splatinsert = insertelement <4 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <4 x i32> %broadcast.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  %induction = add <4 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3>
  %tmp = getelementptr inbounds i32, i32* %a, i32 %index
  %tmp1 = icmp ule <4 x i32> %induction, %broadcast.splat11
  %tmp2 = bitcast i32* %tmp to <4 x i32>*
  %wide.masked.load = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp2, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %tmp3 = getelementptr inbounds i32, i32* %b, i32 %index
  %tmp4 = bitcast i32* %tmp3 to <4 x i32>*
  %wide.masked.load2 = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp4, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %mul = mul nsw <4 x i32> %wide.masked.load2, %wide.masked.load
  %tmp6 = getelementptr inbounds i32, i32* %c, i32 %index
  %tmp7 = bitcast i32* %tmp6 to <4 x i32>*
  tail call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %mul, <4 x i32>* %tmp7, i32 4, <4 x i1> %tmp1)
  %index.next = add i32 %index, 4
  %tmp15 = call i32 @llvm.loop.decrement.reg.i32.i32.i32(i32 %tmp14, i32 1)
  %tmp16 = icmp ne i32 %tmp15, 0
  br i1 %tmp16, label %vector.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %vector.body, %entry
  ret void
}

; CHECK-LABEL: split_vector
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
  call void @llvm.set.loop.iterations.i32(i32 %tmp13)
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %tmp14 = phi i32 [ %tmp13, %vector.ph ], [ %tmp15, %vector.body ]
  %broadcast.splatinsert = insertelement <4 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <4 x i32> %broadcast.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  %induction = add <4 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3>
  %tmp = getelementptr inbounds i32, i32* %a, i32 %index
  %tmp1 = icmp ule <4 x i32> %induction, %broadcast.splat11
  %tmp2 = bitcast i32* %tmp to <4 x i32>*
  %wide.masked.load = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp2, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %extract.1.low = shufflevector <4 x i32> %wide.masked.load, <4 x i32> undef, < 2 x i32> < i32 0, i32 2>
  %extract.1.high = shufflevector <4 x i32> %wide.masked.load, <4 x i32> undef, < 2 x i32> < i32 1, i32 3>
  %tmp3 = getelementptr inbounds i32, i32* %b, i32 %index
  %tmp4 = bitcast i32* %tmp3 to <4 x i32>*
  %wide.masked.load2 = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp4, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %extract.2.low = shufflevector <4 x i32> %wide.masked.load2, <4 x i32> undef, < 2 x i32> < i32 0, i32 2>
  %extract.2.high = shufflevector <4 x i32> %wide.masked.load2, <4 x i32> undef, < 2 x i32> < i32 1, i32 3>
  %mul = mul nsw <2 x i32> %extract.1.low, %extract.2.low
  %sub = sub nsw <2 x i32> %extract.1.high, %extract.2.high
  %combine = shufflevector <2 x i32> %mul, <2 x i32> %sub, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp6 = getelementptr inbounds i32, i32* %c, i32 %index
  %tmp7 = bitcast i32* %tmp6 to <4 x i32>*
  tail call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %combine, <4 x i32>* %tmp7, i32 4, <4 x i1> %tmp1)
  %index.next = add i32 %index, 4
  %tmp15 = call i32 @llvm.loop.decrement.reg.i32.i32.i32(i32 %tmp14, i32 1)
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
  call void @llvm.set.loop.iterations.i32(i32 %tmp13)
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %tmp14 = phi i32 [ %tmp13, %vector.ph ], [ %tmp15, %vector.body ]
  %broadcast.splatinsert = insertelement <4 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <4 x i32> %broadcast.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  %induction = add <4 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3>
  %tmp = getelementptr inbounds i32, i32* %a, i32 %index
  %tmp1 = icmp ule <4 x i32> %induction, %broadcast.splat11
  %wrong = icmp ult <4 x i32> %induction, %broadcast.splat11
  %tmp2 = bitcast i32* %tmp to <4 x i32>*
  %wide.masked.load = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp2, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %tmp3 = getelementptr inbounds i32, i32* %b, i32 %index
  %tmp4 = bitcast i32* %tmp3 to <4 x i32>*
  %wide.masked.load12 = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp4, i32 4, <4 x i1> %wrong, <4 x i32> undef)
  %tmp5 = mul nsw <4 x i32> %wide.masked.load12, %wide.masked.load
  %tmp6 = getelementptr inbounds i32, i32* %c, i32 %index
  %tmp7 = bitcast i32* %tmp6 to <4 x i32>*
  tail call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %tmp5, <4 x i32>* %tmp7, i32 4, <4 x i1> %tmp1)
  %index.next = add i32 %index, 4
  %tmp15 = call i32 @llvm.loop.decrement.reg.i32.i32.i32(i32 %tmp14, i32 1)
  %tmp16 = icmp ne i32 %tmp15, 0
  br i1 %tmp16, label %vector.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %vector.body, %entry
  ret void
}

; The store now uses ult predicate.
; CHECK-LABEL: mismatch_store_pred
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
  call void @llvm.set.loop.iterations.i32(i32 %tmp13)
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %tmp14 = phi i32 [ %tmp13, %vector.ph ], [ %tmp15, %vector.body ]
  %broadcast.splatinsert = insertelement <4 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <4 x i32> %broadcast.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  %induction = add <4 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3>
  %tmp = getelementptr inbounds i32, i32* %a, i32 %index
  %tmp1 = icmp ule <4 x i32> %induction, %broadcast.splat11
  %wrong = icmp ult <4 x i32> %induction, %broadcast.splat11
  %tmp2 = bitcast i32* %tmp to <4 x i32>*
  %wide.masked.load = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp2, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %tmp3 = getelementptr inbounds i32, i32* %b, i32 %index
  %tmp4 = bitcast i32* %tmp3 to <4 x i32>*
  %wide.masked.load12 = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp4, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %tmp5 = mul nsw <4 x i32> %wide.masked.load12, %wide.masked.load
  %tmp6 = getelementptr inbounds i32, i32* %c, i32 %index
  %tmp7 = bitcast i32* %tmp6 to <4 x i32>*
  tail call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %tmp5, <4 x i32>* %tmp7, i32 4, <4 x i1> %wrong)
  %index.next = add i32 %index, 4
  %tmp15 = call i32 @llvm.loop.decrement.reg.i32.i32.i32(i32 %tmp14, i32 1)
  %tmp16 = icmp ne i32 %tmp15, 0
  br i1 %tmp16, label %vector.body, label %for.cond.cleanup

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
declare void @llvm.set.loop.iterations.i32(i32)
declare i32 @llvm.loop.decrement.reg.i32.i32.i32(i32, i32)

