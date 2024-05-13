; RUN: opt -mtriple=thumbv8.1m.main -mve-tail-predication -tail-predication=enabled -mattr=+mve,+lob %s -S -o - | FileCheck %s

; TODO: We should be able to generate a vctp for the loads.
; CHECK-LABEL: trunc_v4i32_v4i16
; CHECK-NOT: vcpt
define void @trunc_v4i32_v4i16(ptr readonly %a, ptr readonly %b, ptr %c, i32 %N) {
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
  %tmp = getelementptr inbounds i32, ptr %a, i32 %index
  %tmp1 = icmp ule <4 x i32> %induction, %broadcast.splat11
  %wide.masked.load = tail call <4 x i32> @llvm.masked.load.v4i32.p0(ptr %tmp, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %tmp3 = getelementptr inbounds i32, ptr %b, i32 %index
  %wide.masked.load2 = tail call <4 x i32> @llvm.masked.load.v4i32.p0(ptr %tmp3, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %mul = mul nsw <4 x i32> %wide.masked.load2, %wide.masked.load
  %trunc = trunc <4 x i32> %mul to <4 x i16>
  %tmp6 = getelementptr inbounds i16, ptr %c, i32 %index
  tail call void @llvm.masked.store.v4i16.p0(<4 x i16> %trunc, ptr %tmp6, i32 4, <4 x i1> %tmp1)
  %index.next = add i32 %index, 4
  %tmp15 = call i32 @llvm.loop.decrement.reg.i32.i32.i32(i32 %tmp14, i32 1)
  %tmp16 = icmp ne i32 %tmp15, 0
  br i1 %tmp16, label %vector.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %vector.body, %entry
  ret void
}

declare <4 x i32> @llvm.masked.load.v4i32.p0(ptr, i32 immarg, <4 x i1>, <4 x i32>)
declare void @llvm.masked.store.v4i16.p0(<4 x i16>, ptr, i32 immarg, <4 x i1>)
declare i32 @llvm.start.loop.iterations.i32(i32)
declare i32 @llvm.loop.decrement.reg.i32.i32.i32(i32, i32)
