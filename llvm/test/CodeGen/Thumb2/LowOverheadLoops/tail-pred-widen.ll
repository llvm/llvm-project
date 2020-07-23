; RUN: opt -mtriple=thumbv8.1m.main -mve-tail-predication -tail-predication=enabled -mattr=+mve,+lob %s -S -o - | FileCheck %s

; CHECK-LABEL: expand_v8i16_v8i32
; CHECK-NOT: call i32 @llvm.arm.mve.vctp
define void @expand_v8i16_v8i32(i16* noalias nocapture readonly %a, i16* noalias nocapture readonly %b, i32* noalias nocapture %c, i32 %N) {
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

  ; %tmp1 = icmp ule <8 x i32> %induction, %broadcast.splat11
   %tmp1 = call <8 x i1> @llvm.get.active.lane.mask.v8i1.i32(i32 %index, i32 %trip.count.minus.1)

  %tmp2 = bitcast i16* %tmp to <8 x i16>*
  %wide.masked.load = tail call <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* %tmp2, i32 4, <8 x i1> %tmp1, <8 x i16> undef)
  %tmp3 = getelementptr inbounds i16, i16* %b, i32 %index
  %tmp4 = bitcast i16* %tmp3 to <8 x i16>*
  %wide.masked.load2 = tail call <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* %tmp4, i32 4, <8 x i1> %tmp1, <8 x i16> undef)
  %expand.1 = zext <8 x i16> %wide.masked.load to <8 x i32>
  %expand.2 = zext <8 x i16> %wide.masked.load2 to <8 x i32>
  %mul = mul nsw <8 x i32> %expand.2, %expand.1
  %tmp6 = getelementptr inbounds i32, i32* %c, i32 %index
  %tmp7 = bitcast i32* %tmp6 to <8 x i32>*
  tail call void @llvm.masked.store.v8i32.p0v8i32(<8 x i32> %mul, <8 x i32>* %tmp7, i32 4, <8 x i1> %tmp1)
  %index.next = add i32 %index, 8
  %tmp15 = call i32 @llvm.loop.decrement.reg.i32.i32.i32(i32 %tmp14, i32 1)
  %tmp16 = icmp ne i32 %tmp15, 0
  br i1 %tmp16, label %vector.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %vector.body, %entry
  ret void
}

; CHECK-LABEL: expand_v8i16_v4i32
; CHECK: [[ELEMS:%[^ ]+]] = phi i32 [ %N, %vector.ph ], [ [[ELEMS_REM:%[^ ]+]], %vector.body ]
; CHECK: [[VCTP:%[^ ]+]] = call <8 x i1> @llvm.arm.mve.vctp16(i32 [[ELEMS]])
; CHECK: [[ELEMS_REM]] = sub i32 [[ELEMS]], 8
; CHECK: tail call <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* {{.*}}, i32 4, <8 x i1> [[VCTP]], <8 x i16> undef)
; CHECK: %store.pred = icmp ule <4 x i32> %induction.store
; CHECK: tail call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> {{.*}}, <4 x i32>* {{.*}}, i32 4, <4 x i1> %store.pred)
; CHECK: tail call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> {{.*}}, <4 x i32>* {{.*}}, i32 4, <4 x i1> %store.pred)
define void @expand_v8i16_v4i32(i16* readonly %a, i16* readonly %b, i32* %c, i32* %d, i32 %N) {
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
  %broadcast.splatinsert10.store = insertelement <4 x i32> undef, i32 %trip.count.minus.1, i32 0
  %broadcast.splat11.store = shufflevector <4 x i32> %broadcast.splatinsert10.store, <4 x i32> undef, <4 x i32> zeroinitializer
  call void @llvm.set.loop.iterations.i32(i32 %tmp13)
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %store.idx = phi i32 [ 0, %vector.ph ], [ %store.idx.next, %vector.body ]
  %tmp14 = phi i32 [ %tmp13, %vector.ph ], [ %tmp15, %vector.body ]
  %broadcast.splatinsert = insertelement <8 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <8 x i32> %broadcast.splatinsert, <8 x i32> undef, <8 x i32> zeroinitializer
  %induction = add <8 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %tmp = getelementptr inbounds i16, i16* %a, i32 %index

  ; %tmp1 = icmp ule <8 x i32> %induction, %broadcast.splat11
  %tmp1 = call <8 x i1> @llvm.get.active.lane.mask.v8i1.i32(i32 %index, i32 %trip.count.minus.1)

  %tmp2 = bitcast i16* %tmp to <8 x i16>*
  %wide.masked.load = tail call <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* %tmp2, i32 4, <8 x i1> %tmp1, <8 x i16> undef)
  %tmp3 = getelementptr inbounds i16, i16* %b, i32 %index
  %tmp4 = bitcast i16* %tmp3 to <8 x i16>*
  %wide.masked.load2 = tail call <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* %tmp4, i32 4, <8 x i1> %tmp1, <8 x i16> undef)
  %extract.2.low = shufflevector <8 x i16> %wide.masked.load2, <8 x i16> undef, < 4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %extract.2.high = shufflevector <8 x i16> %wide.masked.load2, <8 x i16> undef, < 4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %expand.1 = zext <4 x i16> %extract.2.low to <4 x i32>
  %expand.2 = zext <4 x i16> %extract.2.high to <4 x i32>
  %mul = mul nsw <4 x i32> %expand.2, %expand.1
  %sub = mul nsw <4 x i32> %expand.1, %expand.2
  %broadcast.splatinsert.store = insertelement <4 x i32> undef, i32 %store.idx, i32 0
  %broadcast.splat.store = shufflevector <4 x i32> %broadcast.splatinsert.store, <4 x i32> undef, <4 x i32> zeroinitializer
  %induction.store = add <4 x i32> %broadcast.splat.store, <i32 0, i32 1, i32 2, i32 3>
  %store.pred = icmp ule <4 x i32> %induction.store, %broadcast.splat11.store
  %tmp6 = getelementptr inbounds i32, i32* %c, i32 %store.idx
  %tmp7 = bitcast i32* %tmp6 to <4 x i32>*
  tail call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %mul, <4 x i32>* %tmp7, i32 4, <4 x i1> %store.pred)
  %gep = getelementptr inbounds i32, i32* %d, i32 %store.idx
  %cast.gep = bitcast i32* %gep to <4 x i32>*
  tail call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %sub, <4 x i32>* %cast.gep, i32 4, <4 x i1> %store.pred)
  %store.idx.next = add i32 %store.idx, 4
  %index.next = add i32 %index, 8
  %tmp15 = call i32 @llvm.loop.decrement.reg.i32.i32.i32(i32 %tmp14, i32 1)
  %tmp16 = icmp ne i32 %tmp15, 0
  br i1 %tmp16, label %vector.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %vector.body, %entry
  ret void
}

; CHECK-LABEL: expand_v4i32_v4i64
; CHECK-NOT: call i32 @llvm.arm.mve.vctp
define void @expand_v4i32_v4i64(i32* noalias nocapture readonly %a, i32* noalias nocapture readonly %b, i64* noalias nocapture %c, i32 %N) {
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

  ; %tmp1 = icmp ule <4 x i32> %induction, %broadcast.splat11
  %tmp1 = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %index, i32 %trip.count.minus.1)

  %tmp2 = bitcast i32* %tmp to <4 x i32>*
  %wide.masked.load = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp2, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %tmp3 = getelementptr inbounds i32, i32* %b, i32 %index
  %tmp4 = bitcast i32* %tmp3 to <4 x i32>*
  %wide.masked.load2 = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp4, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %expand.1 = zext <4 x i32> %wide.masked.load to <4 x i64>
  %expand.2 = zext <4 x i32> %wide.masked.load2 to <4 x i64>
  %mul = mul nsw <4 x i64> %expand.2, %expand.1
  %tmp6 = getelementptr inbounds i64, i64* %c, i32 %index
  %tmp7 = bitcast i64* %tmp6 to <4 x i64>*
  tail call void @llvm.masked.store.v4i64.p0v4i64(<4 x i64> %mul, <4 x i64>* %tmp7, i32 4, <4 x i1> %tmp1)
  %index.next = add i32 %index, 4
  %tmp15 = call i32 @llvm.loop.decrement.reg.i32.i32.i32(i32 %tmp14, i32 1)
  %tmp16 = icmp ne i32 %tmp15, 0
  br i1 %tmp16, label %vector.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %vector.body, %entry
  ret void
}

declare <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>*, i32 immarg, <8 x i1>, <8 x i16>)
declare void @llvm.masked.store.v8i32.p0v8i32(<8 x i32>, <8 x i32>*, i32 immarg, <8 x i1>)
declare <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>*, i32 immarg, <4 x i1>, <4 x i32>)
declare void @llvm.masked.store.v4i32.p0v4i32(<4 x i32>, <4 x i32>*, i32 immarg, <4 x i1>)
declare void @llvm.masked.store.v4i64.p0v4i64(<4 x i64>, <4 x i64>*, i32 immarg, <4 x i1>)
declare void @llvm.set.loop.iterations.i32(i32)
declare i32 @llvm.loop.decrement.reg.i32.i32.i32(i32, i32)
declare <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32, i32)
declare <8 x i1> @llvm.get.active.lane.mask.v8i1.i32(i32, i32)
