; RUN: opt -mtriple=thumbv8.1m.main -mve-tail-predication -disable-mve-tail-predication=false -mattr=+mve %s -S -o - | FileCheck %s

; CHECK-LABEL: reduction_i32
; CHECK: phi i32 [ 0, %entry ]
; CHECK: phi <8 x i16> [ zeroinitializer, %entry ]
; CHECK: phi i32
; CHECK: [[PHI:%[^ ]+]] = phi i32 [ %N, %entry ], [ [[ELEMS:%[^ ]+]], %vector.body ]
; CHECK: [[VCTP:%[^ ]+]] = call <8 x i1> @llvm.arm.mve.vctp16(i32 [[PHI]])
; CHECK: [[ELEMS]] = sub i32 [[PHI]], 8
; CHECK: call <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* %tmp4, i32 4, <8 x i1> [[VCTP]], <8 x i16> undef)
; CHECK: call <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* %tmp6, i32 4, <8 x i1> [[VCTP]], <8 x i16> undef)
define i16 @reduction_i32(i16* nocapture readonly %A, i16* nocapture readonly %B, i32 %N) {
entry:
  %tmp = add i32 %N, -1
  %n.rnd.up = add nuw nsw i32 %tmp, 8
  %n.vec = and i32 %n.rnd.up, -8
  %broadcast.splatinsert1 = insertelement <8 x i32> undef, i32 %tmp, i32 0
  %broadcast.splat2 = shufflevector <8 x i32> %broadcast.splatinsert1, <8 x i32> undef, <8 x i32> zeroinitializer
  %0 = add i32 %n.vec, -8
  %1 = lshr i32 %0, 3
  %2 = add nuw nsw i32 %1, 1
  call void @llvm.set.loop.iterations.i32(i32 %2)
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i32 [ 0, %entry ], [ %index.next, %vector.body ]
  %vec.phi = phi <8 x i16> [ zeroinitializer, %entry ], [ %tmp8, %vector.body ]
  %3 = phi i32 [ %2, %entry ], [ %4, %vector.body ]
  %broadcast.splatinsert = insertelement <8 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <8 x i32> %broadcast.splatinsert, <8 x i32> undef, <8 x i32> zeroinitializer
  %induction = add <8 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %tmp2 = getelementptr inbounds i16, i16* %A, i32 %index
  %tmp3 = icmp ule <8 x i32> %induction, %broadcast.splat2
  %tmp4 = bitcast i16* %tmp2 to <8 x i16>*
  %wide.masked.load = call <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* %tmp4, i32 4, <8 x i1> %tmp3, <8 x i16> undef)
  %tmp5 = getelementptr inbounds i16, i16* %B, i32 %index
  %tmp6 = bitcast i16* %tmp5 to <8 x i16>*
  %wide.masked.load3 = call <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* %tmp6, i32 4, <8 x i1> %tmp3, <8 x i16> undef)
  %tmp7 = add <8 x i16> %wide.masked.load, %vec.phi
  %tmp8 = add <8 x i16> %tmp7, %wide.masked.load3
  %index.next = add nuw nsw i32 %index, 8
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
}

; CHECK-LABEL: reduction_i32_with_scalar
; CHECK: phi i32 [ 0, %entry ]
; CHECK: phi <8 x i16> [ zeroinitializer, %entry ]
; CHECK: phi i32
; CHECK: [[PHI:%[^ ]+]] = phi i32 [ %N, %entry ], [ [[ELEMS:%[^ ]+]], %vector.body ]
; CHECK: [[VCTP:%[^ ]+]] = call <8 x i1> @llvm.arm.mve.vctp16(i32 [[PHI]])
; CHECK: [[ELEMS]] = sub i32 [[PHI]], 8
; CHECK: call <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* %tmp4, i32 4, <8 x i1> [[VCTP]], <8 x i16> undef)
define i16 @reduction_i32_with_scalar(i16* nocapture readonly %A, i16 %B, i32 %N) local_unnamed_addr {
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

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i32 [ 0, %entry ], [ %index.next, %vector.body ]
  %vec.phi = phi <8 x i16> [ zeroinitializer, %entry ], [ %tmp6, %vector.body ]
  %3 = phi i32 [ %2, %entry ], [ %4, %vector.body ]
  %broadcast.splatinsert = insertelement <8 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <8 x i32> %broadcast.splatinsert, <8 x i32> undef, <8 x i32> zeroinitializer
  %induction = add <8 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %tmp2 = getelementptr inbounds i16, i16* %A, i32 %index
  %tmp3 = icmp ule <8 x i32> %induction, %broadcast.splat2
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

declare <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>*, i32 immarg, <8 x i1>, <8 x i16>)
declare void @llvm.set.loop.iterations.i32(i32)
declare i32 @llvm.loop.decrement.reg.i32.i32.i32(i32, i32)


