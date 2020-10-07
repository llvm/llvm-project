
; RUN: opt -mtriple=thumbv8.1m.main -mve-tail-predication -tail-predication=enabled -mattr=+mve %s -S -o - | FileCheck %s

; CHECK-LABEL: vec_mul_reduce_add

; CHECK: vector.ph:
; CHECK:  call void @llvm.set.loop.iterations.i32
; CHECK:  br label %vector.body

; CHECK: vector.body:
; CHECK: [[ELTS:%[^ ]+]] = phi i32 [ %N, %vector.ph ], [ [[SUB:%[^ ]+]], %vector.body ]
; CHECK: [[VCTP:%[^ ]+]] = call <4 x i1> @llvm.arm.mve.vctp32(i32 [[ELTS]])
; CHECK: [[SUB]] = sub i32 [[ELTS]], 4
; CHECK: call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* {{.*}}, i32 4, <4 x i1> [[VCTP]]
; CHECK: call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* {{.*}}, i32 4, <4 x i1> [[VCTP]],

; CHECK: middle.block:
; CHECK: [[VPSEL:%[^ ]+]] = select <4 x i1> [[VCTP]],
; CHECK: call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> [[VPSEL]])

define i32 @vec_mul_reduce_add(i32* noalias nocapture readonly %a, i32* noalias nocapture readonly %b, i32 %N) {
entry:
  %cmp8 = icmp eq i32 %N, 0
  %0 = add i32 %N, 3
  %1 = lshr i32 %0, 2
  %2 = shl nuw i32 %1, 2
  %3 = add i32 %2, -4
  %4 = lshr i32 %3, 2
  %5 = add nuw nsw i32 %4, 1
  br i1 %cmp8, label %for.cond.cleanup, label %vector.ph

vector.ph:                                        ; preds = %entry
  %trip.count.minus.1 = add i32 %N, -1
  %broadcast.splatinsert11 = insertelement <4 x i32> undef, i32 %trip.count.minus.1, i32 0
  %broadcast.splat12 = shufflevector <4 x i32> %broadcast.splatinsert11, <4 x i32> undef, <4 x i32> zeroinitializer
  call void @llvm.set.loop.iterations.i32(i32 %5)
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %lsr.iv2 = phi i32* [ %scevgep3, %vector.body ], [ %a, %vector.ph ]
  %lsr.iv = phi i32* [ %scevgep, %vector.body ], [ %b, %vector.ph ]
  %vec.phi = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ %9, %vector.body ]
  %6 = phi i32 [ %5, %vector.ph ], [ %10, %vector.body ]
  %lsr.iv24 = bitcast i32* %lsr.iv2 to <4 x i32>*
  %lsr.iv1 = bitcast i32* %lsr.iv to <4 x i32>*
  %broadcast.splatinsert = insertelement <4 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <4 x i32> %broadcast.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  %induction = add <4 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3>

  ; %7 = icmp ule <4 x i32> %induction, %broadcast.splat12
  %7 = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %index, i32 %N)

  %wide.masked.load = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %lsr.iv24, i32 4, <4 x i1> %7, <4 x i32> undef)
  %wide.masked.load13 = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %lsr.iv1, i32 4, <4 x i1> %7, <4 x i32> undef)
  %8 = mul nsw <4 x i32> %wide.masked.load13, %wide.masked.load
  %9 = add nsw <4 x i32> %8, %vec.phi
  %index.next = add i32 %index, 4
  %scevgep = getelementptr i32, i32* %lsr.iv, i32 4
  %scevgep3 = getelementptr i32, i32* %lsr.iv2, i32 4
  %10 = call i32 @llvm.loop.decrement.reg.i32.i32.i32(i32 %6, i32 1)
  %11 = icmp ne i32 %10, 0
  br i1 %11, label %vector.body, label %middle.block

middle.block:                                     ; preds = %vector.body
  %12 = select <4 x i1> %7, <4 x i32> %9, <4 x i32> %vec.phi
  %13 = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %12)
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %middle.block, %entry
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %13, %middle.block ]
  ret i32 %res.0.lcssa
}

declare <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>*, i32 immarg, <4 x i1>, <4 x i32>)
declare i32 @llvm.vector.reduce.add.v4i32(<4 x i32>)
declare void @llvm.set.loop.iterations.i32(i32)
declare i32 @llvm.loop.decrement.reg.i32.i32.i32(i32, i32)
declare <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32, i32)
