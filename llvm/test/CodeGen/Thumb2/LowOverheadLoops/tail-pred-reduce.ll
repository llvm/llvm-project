; RUN: opt -mtriple=thumbv8.1m.main -mve-tail-predication -tail-predication=enabled -mattr=+mve %s -S -o - | FileCheck %s

; CHECK-LABEL: reduction_i32
; CHECK: phi i32 [ 0, %vector.ph ]
; CHECK: phi <8 x i16> [ zeroinitializer, %vector.ph ]
; CHECK: phi i32
; CHECK: [[PHI:%[^ ]+]] = phi i32 [ %N, %vector.ph ], [ [[ELEMS:%[^ ]+]], %vector.body ]
; CHECK: [[VCTP:%[^ ]+]] = call <8 x i1> @llvm.arm.mve.vctp16(i32 [[PHI]])
; CHECK: [[ELEMS]] = sub i32 [[PHI]], 8
; CHECK: call <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* %tmp4, i32 4, <8 x i1> [[VCTP]], <8 x i16> undef)
; CHECK: call <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* %tmp6, i32 4, <8 x i1> [[VCTP]], <8 x i16> undef)
define i16 @reduction_i32(i16* nocapture readonly %A, i16* nocapture readonly %B, i32 %N) {
entry:
  %cmp8 = icmp eq i32 %N, 0
  br i1 %cmp8, label %for.cond.cleanup, label %vector.ph

vector.ph:
  %tmp = add i32 %N, -1
  %n.rnd.up = add i32 %tmp, 8
  %n.vec = and i32 %n.rnd.up, -8
  %broadcast.splatinsert1 = insertelement <8 x i32> undef, i32 %tmp, i32 0
  %broadcast.splat2 = shufflevector <8 x i32> %broadcast.splatinsert1, <8 x i32> undef, <8 x i32> zeroinitializer
  %0 = add i32 %n.vec, -8
  %1 = lshr i32 %0, 3
  %2 = add i32 %1, 1
  call void @llvm.set.loop.iterations.i32(i32 %2)
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph], [ %index.next, %vector.body ]
  %vec.phi = phi <8 x i16> [ zeroinitializer, %vector.ph], [ %tmp8, %vector.body ]
  %3 = phi i32 [ %2, %vector.ph], [ %4, %vector.body ]
  %broadcast.splatinsert = insertelement <8 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <8 x i32> %broadcast.splatinsert, <8 x i32> undef, <8 x i32> zeroinitializer
  %induction = add <8 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %tmp2 = getelementptr inbounds i16, i16* %A, i32 %index

  ; %tmp3 = icmp ule <8 x i32> %induction, %broadcast.splat2
  %tmp3 = call <8 x i1> @llvm.get.active.lane.mask.v8i1.i32(i32 %index, i32 %N)

  %tmp4 = bitcast i16* %tmp2 to <8 x i16>*
  %wide.masked.load = call <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* %tmp4, i32 4, <8 x i1> %tmp3, <8 x i16> undef)
  %tmp5 = getelementptr inbounds i16, i16* %B, i32 %index
  %tmp6 = bitcast i16* %tmp5 to <8 x i16>*
  %wide.masked.load3 = call <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* %tmp6, i32 4, <8 x i1> %tmp3, <8 x i16> undef)
  %tmp7 = add <8 x i16> %wide.masked.load, %vec.phi
  %tmp8 = add <8 x i16> %tmp7, %wide.masked.load3
  %index.next = add i32 %index, 8
  %4 = call i32 @llvm.loop.decrement.reg.i32.i32.i32(i32 %3, i32 1)
  %5 = icmp ne i32 %4, 0
  br i1 %5, label %vector.body, label %middle.block

middle.block:                                     ; preds = %vector.body
  %vec.phi.lcssa = phi <8 x i16> [ %vec.phi, %vector.body ]
  %.lcssa3 = phi <8 x i1> [ %tmp3, %vector.body ]
  %.lcssa = phi <8 x i16> [ %tmp8, %vector.body ]
  %tmp10 = select <8 x i1> %.lcssa3, <8 x i16> %.lcssa, <8 x i16> %vec.phi.lcssa
  %rdx.shuf = shufflevector <8 x i16> %tmp10, <8 x i16> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx = add <8 x i16> %rdx.shuf, %tmp10
  %rdx.shuf4 = shufflevector <8 x i16> %bin.rdx, <8 x i16> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx5 = add <8 x i16> %rdx.shuf4, %bin.rdx
  %rdx.shuf6 = shufflevector <8 x i16> %bin.rdx5, <8 x i16> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx7 = add <8 x i16> %rdx.shuf6, %bin.rdx5
  %tmp11 = extractelement <8 x i16> %bin.rdx7, i32 0
  ret i16 %tmp11

for.cond.cleanup:
  %res.0 = phi i16 [ 0, %entry ]
  ret i16 %res.0
}

; CHECK-LABEL: reduction_i32_with_scalar
; CHECK: vector.body:
; CHECK: %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK: %vec.phi = phi <8 x i16> [ zeroinitializer, %vector.ph ], [ %{{.*}}, %vector.body ]
; CHECK: %{{.*}} = phi i32 [ %{{.*}}, %vector.ph ], [ %{{.*}}, %vector.body ]
; CHECK: [[PHI:%[^ ]+]] = phi i32 [ %N, %vector.ph ], [ [[ELEMS:%[^ ]+]], %vector.body ]
; CHECK: [[VCTP:%[^ ]+]] = call <8 x i1> @llvm.arm.mve.vctp16(i32 [[PHI]])
; CHECK: [[ELEMS]] = sub i32 [[PHI]], 8
; CHECK: call <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* %tmp4, i32 4, <8 x i1> [[VCTP]], <8 x i16> undef)
define i16 @reduction_i32_with_scalar(i16* nocapture readonly %A, i16 %B, i32 %N) local_unnamed_addr {
entry:
  %cmp8 = icmp eq i32 %N, 0
  br i1 %cmp8, label %for.cond.cleanup, label %vector.ph

vector.ph:
  %tmp = add i32 %N, -1
  %n.rnd.up = add nuw nsw i32 %tmp, 8
  %n.vec = and i32 %n.rnd.up, -8
  %broadcast.splatinsert1 = insertelement <8 x i32> undef, i32 %tmp, i32 0
  %broadcast.splat2 = shufflevector <8 x i32> %broadcast.splatinsert1, <8 x i32> undef, <8 x i32> zeroinitializer
  %broadcast.splatinsert3 = insertelement <8 x i16> undef, i16 %B, i32 0
  %broadcast.splat4 = shufflevector <8 x i16> %broadcast.splatinsert3, <8 x i16> undef, <8 x i32> zeroinitializer
  %0 = add i32 %n.vec, -8
  %1 = lshr i32 %0, 3
  %2 = add nuw nsw i32 %1, 1
  call void @llvm.set.loop.iterations.i32(i32 %2)
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph], [ %index.next, %vector.body ]
  %vec.phi = phi <8 x i16> [ zeroinitializer, %vector.ph], [ %tmp6, %vector.body ]
  %3 = phi i32 [ %2, %vector.ph], [ %4, %vector.body ]
  %broadcast.splatinsert = insertelement <8 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <8 x i32> %broadcast.splatinsert, <8 x i32> undef, <8 x i32> zeroinitializer
  %induction = add <8 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %tmp2 = getelementptr inbounds i16, i16* %A, i32 %index

  ; %tmp3 = icmp ule <8 x i32> %induction, %broadcast.splat2
  %tmp3 = call <8 x i1> @llvm.get.active.lane.mask.v8i1.i32(i32 %index, i32 %N)

  %tmp4 = bitcast i16* %tmp2 to <8 x i16>*
  %wide.masked.load = call <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* %tmp4, i32 4, <8 x i1> %tmp3, <8 x i16> undef)
  %tmp5 = add <8 x i16> %vec.phi, %broadcast.splat4
  %tmp6 = add <8 x i16> %tmp5, %wide.masked.load
  %index.next = add nuw nsw i32 %index, 8
  %4 = call i32 @llvm.loop.decrement.reg.i32.i32.i32(i32 %3, i32 1)
  %5 = icmp ne i32 %4, 0
  br i1 %5, label %vector.body, label %middle.block

middle.block:                                     ; preds = %vector.body
  %tmp8 = select <8 x i1> %tmp3, <8 x i16> %tmp6, <8 x i16> %vec.phi
  %rdx.shuf = shufflevector <8 x i16> %tmp8, <8 x i16> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx = add <8 x i16> %rdx.shuf, %tmp8
  %rdx.shuf5 = shufflevector <8 x i16> %bin.rdx, <8 x i16> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx6 = add <8 x i16> %rdx.shuf5, %bin.rdx
  %rdx.shuf7 = shufflevector <8 x i16> %bin.rdx6, <8 x i16> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx8 = add <8 x i16> %rdx.shuf7, %bin.rdx6
  %tmp9 = extractelement <8 x i16> %bin.rdx8, i32 0
  ret i16 %tmp9

for.cond.cleanup:
  %res.0 = phi i16 [ 0, %entry ]
  ret i16 %res.0
}

; The vector loop is not guarded with an entry check (N == 0). Check that
; despite this we can still calculate a precise enough range so that the
; the overflow checks for get.active.active.lane.mask don't reject
; tail-predication.
;
; CHECK-LABEL: @reduction_not_guarded
;
; CHECK:     vector.body:
; CHECK:     @llvm.arm.mve.vctp
; CHECK-NOT: @llvm.get.active.lane.mask.v8i1.i32
; CHECK:     ret
;
define i16 @reduction_not_guarded(i16* nocapture readonly %A, i16 %B, i32 %N) local_unnamed_addr {
entry:
  %tmp = add i32 %N, -1
  %n.rnd.up = add nuw nsw i32 %tmp, 8
  %n.vec = and i32 %n.rnd.up, -8
  %broadcast.splatinsert1 = insertelement <8 x i32> undef, i32 %tmp, i32 0
  %broadcast.splat2 = shufflevector <8 x i32> %broadcast.splatinsert1, <8 x i32> undef, <8 x i32> zeroinitializer
  %broadcast.splatinsert3 = insertelement <8 x i16> undef, i16 %B, i32 0
  %broadcast.splat4 = shufflevector <8 x i16> %broadcast.splatinsert3, <8 x i16> undef, <8 x i32> zeroinitializer
  %0 = add i32 %n.vec, -8
  %1 = lshr i32 %0, 3
  %2 = add nuw nsw i32 %1, 1
  call void @llvm.set.loop.iterations.i32(i32 %2)
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %entry], [ %index.next, %vector.body ]
  %vec.phi = phi <8 x i16> [ zeroinitializer, %entry], [ %tmp6, %vector.body ]
  %3 = phi i32 [ %2, %entry ], [ %4, %vector.body ]
  %broadcast.splatinsert = insertelement <8 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <8 x i32> %broadcast.splatinsert, <8 x i32> undef, <8 x i32> zeroinitializer
  %induction = add <8 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %tmp2 = getelementptr inbounds i16, i16* %A, i32 %index

  ; %tmp3 = icmp ule <8 x i32> %induction, %broadcast.splat2
  %tmp3 = call <8 x i1> @llvm.get.active.lane.mask.v8i1.i32(i32 %index, i32 %N)

  %tmp4 = bitcast i16* %tmp2 to <8 x i16>*
  %wide.masked.load = call <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* %tmp4, i32 4, <8 x i1> %tmp3, <8 x i16> undef)
  %tmp5 = add <8 x i16> %vec.phi, %broadcast.splat4
  %tmp6 = add <8 x i16> %tmp5, %wide.masked.load
  %index.next = add nuw nsw i32 %index, 8
  %4 = call i32 @llvm.loop.decrement.reg.i32.i32.i32(i32 %3, i32 1)
  %5 = icmp ne i32 %4, 0
  br i1 %5, label %vector.body, label %middle.block

middle.block:                                     ; preds = %vector.body
  %tmp8 = select <8 x i1> %tmp3, <8 x i16> %tmp6, <8 x i16> %vec.phi
  %rdx.shuf = shufflevector <8 x i16> %tmp8, <8 x i16> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx = add <8 x i16> %rdx.shuf, %tmp8
  %rdx.shuf5 = shufflevector <8 x i16> %bin.rdx, <8 x i16> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx6 = add <8 x i16> %rdx.shuf5, %bin.rdx
  %rdx.shuf7 = shufflevector <8 x i16> %bin.rdx6, <8 x i16> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx8 = add <8 x i16> %rdx.shuf7, %bin.rdx6
  %tmp9 = extractelement <8 x i16> %bin.rdx8, i32 0
  ret i16 %tmp9
}

; CHECK-LABEL: @Correlation
; CHECK:       vector.body:
; CHECK:       @llvm.arm.mve.vctp
; CHECK-NOT:   %active.lane.mask = call <4 x i1> @llvm.get.active.lane.mask
;
define dso_local void @Correlation(i16* nocapture readonly %Input, i16* nocapture %Output, i16 signext %Size, i16 signext %N, i16 signext %Scale) local_unnamed_addr #0 {
entry:
  %conv = sext i16 %N to i32
  %cmp36 = icmp sgt i16 %N, 0
  br i1 %cmp36, label %for.body.lr.ph, label %for.end17

for.body.lr.ph:
  %conv2 = sext i16 %Size to i32
  %conv1032 = zext i16 %Scale to i32
  %0 = add i32 %conv2, 3
  br label %for.body

for.body:
  %lsr.iv51 = phi i32 [ %lsr.iv.next, %for.end ], [ %0, %for.body.lr.ph ]
  %lsr.iv46 = phi i16* [ %scevgep47, %for.end ], [ %Input, %for.body.lr.ph ]
  %i.037 = phi i32 [ 0, %for.body.lr.ph ], [ %inc16, %for.end ]
  %1 = mul nsw i32 %i.037, -1
  %2 = add i32 %0, %1
  %3 = lshr i32 %2, 2
  %4 = shl nuw i32 %3, 2
  %5 = add i32 %4, -4
  %6 = lshr i32 %5, 2
  %7 = add nuw nsw i32 %6, 1
  %8 = sub i32 %conv2, %i.037
  %cmp433 = icmp slt i32 %i.037, %conv2
  br i1 %cmp433, label %vector.ph, label %for.end

vector.ph:                                        ; preds = %for.body
  %trip.count.minus.1 = add i32 %8, -1
  call void @llvm.set.loop.iterations.i32(i32 %7)
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %lsr.iv48 = phi i16* [ %scevgep49, %vector.body ], [ %lsr.iv46, %vector.ph ]
  %lsr.iv = phi i16* [ %scevgep, %vector.body ], [ %Input, %vector.ph ]
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %vec.phi = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ %16, %vector.body ]
  %9 = phi i32 [ %7, %vector.ph ], [ %17, %vector.body ]
  %lsr.iv4850 = bitcast i16* %lsr.iv48 to <4 x i16>*
  %lsr.iv45 = bitcast i16* %lsr.iv to <4 x i16>*
  %active.lane.mask = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %index, i32 %8)
  %wide.masked.load = call <4 x i16> @llvm.masked.load.v4i16.p0v4i16(<4 x i16>* %lsr.iv45, i32 2, <4 x i1> %active.lane.mask, <4 x i16> undef)
  %10 = sext <4 x i16> %wide.masked.load to <4 x i32>
  %wide.masked.load42 = call <4 x i16> @llvm.masked.load.v4i16.p0v4i16(<4 x i16>* %lsr.iv4850, i32 2, <4 x i1> %active.lane.mask, <4 x i16> undef)
  %11 = sext <4 x i16> %wide.masked.load42 to <4 x i32>
  %12 = mul nsw <4 x i32> %11, %10
  %13 = insertelement <4 x i32> undef, i32 %conv1032, i32 0
  %14 = shufflevector <4 x i32> %13, <4 x i32> undef, <4 x i32> zeroinitializer
  %15 = ashr <4 x i32> %12, %14
  %16 = add <4 x i32> %15, %vec.phi
  %index.next = add i32 %index, 4
  %scevgep = getelementptr i16, i16* %lsr.iv, i32 4
  %scevgep49 = getelementptr i16, i16* %lsr.iv48, i32 4
  %17 = call i32 @llvm.loop.decrement.reg.i32(i32 %9, i32 1)
  %18 = icmp ne i32 %17, 0
  br i1 %18, label %vector.body, label %middle.block

middle.block:                                     ; preds = %vector.body
  %19 = select <4 x i1> %active.lane.mask, <4 x i32> %16, <4 x i32> %vec.phi
  %20 = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %19)
  br label %for.end

for.end:                                          ; preds = %middle.block, %for.body
  %Sum.0.lcssa = phi i32 [ 0, %for.body ], [ %20, %middle.block ]
  %21 = lshr i32 %Sum.0.lcssa, 16
  %conv13 = trunc i32 %21 to i16
  %arrayidx14 = getelementptr inbounds i16, i16* %Output, i32 %i.037
  store i16 %conv13, i16* %arrayidx14, align 2
  %inc16 = add nuw nsw i32 %i.037, 1
  %scevgep47 = getelementptr i16, i16* %lsr.iv46, i32 1
  %lsr.iv.next = add i32 %lsr.iv51, -1
  %exitcond39 = icmp eq i32 %inc16, %conv
  br i1 %exitcond39, label %for.end17, label %for.body

for.end17:                                        ; preds = %for.end, %entry
  ret void
}

declare <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>*, i32 immarg, <8 x i1>, <8 x i16>)
declare void @llvm.set.loop.iterations.i32(i32)
declare i32 @llvm.loop.decrement.reg.i32.i32.i32(i32, i32)
declare <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32, i32)
declare <8 x i1> @llvm.get.active.lane.mask.v8i1.i32(i32, i32)
declare i32 @llvm.vector.reduce.add.v4i32(<4 x i32>)
declare i32 @llvm.loop.decrement.reg.i32(i32, i32)
declare <4 x i16> @llvm.masked.load.v4i16.p0v4i16(<4 x i16>*, i32 immarg, <4 x i1>, <4 x i16>)
