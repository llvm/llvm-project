; RUN: opt -mtriple=thumbv8.1m.main -mve-tail-predication -disable-mve-tail-predication=false -mattr=+mve,+lob %s -S -o - | FileCheck %s

; The following functions should all fail to become tail-predicated.
; CHECK-NOT: call i32 @llvm.arm.vctp

; trip.count.minus.1 has been inserted into element 1, not 0.
define dso_local arm_aapcs_vfpcc void @wrong_ph_insert_0(i32* noalias nocapture readonly %a, i32* noalias nocapture readonly %b, i32* noalias nocapture %c, i32 %N) {
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
  %broadcast.splatinsert10 = insertelement <4 x i32> undef, i32 %trip.count.minus.1, i32 1
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
  %wide.masked.load12 = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp4, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
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

; The insert isn't using an undef for operand 0.
define dso_local arm_aapcs_vfpcc void @wrong_ph_insert_def(i32* noalias nocapture readonly %a, i32* noalias nocapture readonly %b, i32* noalias nocapture %c, i32 %N) {
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
  %broadcast.splatinsert10 = insertelement <4 x i32> <i32 1, i32 1, i32 1, i32 1>, i32 %trip.count.minus.1, i32 0
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
  %wide.masked.load12 = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp4, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
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

; The shuffle uses a defined value for operand 1.
define dso_local arm_aapcs_vfpcc void @wrong_ph_shuffle_1(i32* noalias nocapture readonly %a, i32* noalias nocapture readonly %b, i32* noalias nocapture %c, i32 %N) {
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
  %broadcast.splat11 = shufflevector <4 x i32> %broadcast.splatinsert10, <4 x i32> <i32 1, i32 1, i32 1, i32 1>, <4 x i32> zeroinitializer
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
  %wide.masked.load12 = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp4, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
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

; The shuffle uses a non zero value for operand 2.
define dso_local arm_aapcs_vfpcc void @wrong_ph_shuffle_2(i32* noalias nocapture readonly %a, i32* noalias nocapture readonly %b, i32* noalias nocapture %c, i32 %N) {
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
  %broadcast.splat11 = shufflevector <4 x i32> %broadcast.splatinsert10, <4 x i32> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
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
  %wide.masked.load12 = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp4, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
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

; %N - 2
define dso_local arm_aapcs_vfpcc void @trip_count_minus_2(i32* noalias nocapture readonly %a, i32* noalias nocapture readonly %b, i32* noalias nocapture %c, i32 %N) {
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
  %trip.count.minus.2 = add i32 %N, -2
  %broadcast.splatinsert10 = insertelement <4 x i32> undef, i32 %trip.count.minus.2, i32 1
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
  %wide.masked.load12 = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp4, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
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

; index has been inserted at element 1, not 0.
define dso_local arm_aapcs_vfpcc void @wrong_loop_insert(i32* noalias nocapture readonly %a, i32* noalias nocapture readonly %b, i32* noalias nocapture %c, i32 %N) {
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
  %broadcast.splatinsert = insertelement <4 x i32> undef, i32 %index, i32 1
  %broadcast.splat = shufflevector <4 x i32> %broadcast.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  %induction = add <4 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3>
  %tmp = getelementptr inbounds i32, i32* %a, i32 %index
  %tmp1 = icmp ule <4 x i32> %induction, %broadcast.splat11
  %tmp2 = bitcast i32* %tmp to <4 x i32>*
  %wide.masked.load = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp2, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %tmp3 = getelementptr inbounds i32, i32* %b, i32 %index
  %tmp4 = bitcast i32* %tmp3 to <4 x i32>*
  %wide.masked.load12 = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp4, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
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

define dso_local arm_aapcs_vfpcc void @wrong_loop_invalid_index_splat(i32* noalias nocapture readonly %a, i32* noalias nocapture readonly %b, i32* noalias nocapture %c, i32 %N) {
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
  %incorrect = add i32 %index, 1
  %broadcast.splatinsert = insertelement <4 x i32> undef, i32 %incorrect, i32 0
  %broadcast.splat = shufflevector <4 x i32> %broadcast.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  %induction = add <4 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3>
  %tmp = getelementptr inbounds i32, i32* %a, i32 %index
  %tmp1 = icmp ule <4 x i32> %induction, %broadcast.splat11
  %tmp2 = bitcast i32* %tmp to <4 x i32>*
  %wide.masked.load = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp2, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %tmp3 = getelementptr inbounds i32, i32* %b, i32 %index
  %tmp4 = bitcast i32* %tmp3 to <4 x i32>*
  %wide.masked.load12 = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp4, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
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

; Now using ult, not ule for the vector icmp
define dso_local arm_aapcs_vfpcc void @wrong_pred_opcode(i32* noalias nocapture readonly %a, i32* noalias nocapture readonly %b, i32* noalias nocapture %c, i32 %N) {
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
  %tmp1 = icmp ult <4 x i32> %induction, %broadcast.splat11
  %tmp2 = bitcast i32* %tmp to <4 x i32>*
  %wide.masked.load = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp2, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %tmp3 = getelementptr inbounds i32, i32* %b, i32 %index
  %tmp4 = bitcast i32* %tmp3 to <4 x i32>*
  %wide.masked.load12 = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp4, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
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

; The add in the body uses 1, 2, 3, 4
define void @wrong_body_broadcast_splat(i32* noalias nocapture readonly %a, i32* noalias nocapture readonly %b, i32* noalias nocapture %c, i32 %N) {
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
  %induction = add <4 x i32> %broadcast.splat, <i32 1, i32 2, i32 3, i32 4>
  %tmp = getelementptr inbounds i32, i32* %a, i32 %index
  %tmp1 = icmp ule <4 x i32> %induction, %broadcast.splat11
  %tmp2 = bitcast i32* %tmp to <4 x i32>*
  %wide.masked.load = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp2, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %tmp3 = getelementptr inbounds i32, i32* %b, i32 %index
  %tmp4 = bitcast i32* %tmp3 to <4 x i32>*
  %wide.masked.load12 = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp4, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
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

; Using a variable for the loop body broadcast.
define void @wrong_body_broadcast_splat_2(i32* noalias nocapture readonly %a, i32* noalias nocapture readonly %b, i32* noalias nocapture %c, i32 %N, <4 x i32> %offsets) {
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
  %induction = add <4 x i32> %broadcast.splat, %offsets
  %tmp = getelementptr inbounds i32, i32* %a, i32 %index
  %tmp1 = icmp ule <4 x i32> %induction, %broadcast.splat11
  %tmp2 = bitcast i32* %tmp to <4 x i32>*
  %wide.masked.load = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp2, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %tmp3 = getelementptr inbounds i32, i32* %b, i32 %index
  %tmp4 = bitcast i32* %tmp3 to <4 x i32>*
  %wide.masked.load12 = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp4, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
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

; adding 5, instead of 4, to index. 
define void @wrong_index_add(i32* noalias nocapture readonly %a, i32* noalias nocapture readonly %b, i32* noalias nocapture %c, i32 %N) {
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
  %wide.masked.load12 = tail call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp4, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %tmp5 = mul nsw <4 x i32> %wide.masked.load12, %wide.masked.load
  %tmp6 = getelementptr inbounds i32, i32* %c, i32 %index
  %tmp7 = bitcast i32* %tmp6 to <4 x i32>*
  tail call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %tmp5, <4 x i32>* %tmp7, i32 4, <4 x i1> %tmp1)
  %index.next = add i32 %index, 5
  %tmp15 = call i32 @llvm.loop.decrement.reg.i32.i32.i32(i32 %tmp14, i32 1)
  %tmp16 = icmp ne i32 %tmp15, 0
  br i1 %tmp16, label %vector.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %vector.body, %entry
  ret void
}

declare <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>*, i32 immarg, <4 x i1>, <4 x i32>) #1
declare void @llvm.masked.store.v4i32.p0v4i32(<4 x i32>, <4 x i32>*, i32 immarg, <4 x i1>) #2
declare void @llvm.set.loop.iterations.i32(i32) #3
declare i32 @llvm.loop.decrement.reg.i32.i32.i32(i32, i32) #3

