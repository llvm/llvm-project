; RUN: opt -mtriple=thumbv8.1m.main -mve-tail-predication -tail-predication=enabled -mattr=+mve,+lob %s -S -o - | FileCheck %s

; TODO: The unrolled pattern is preventing the transform
; CHECK-LABEL: mul_v16i8_unroll
; CHECK-NOT: call i32 @llvm.arm.vcpt
define void @mul_v16i8_unroll(i8* noalias nocapture readonly %a, i8* noalias nocapture readonly %b, i8* noalias nocapture %c, i32 %N) {
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
  %xtraiter = and i32 %tmp13, 1
  %0 = icmp ult i32 %tmp12, 1
  br i1 %0, label %for.cond.cleanup.loopexit.unr-lcssa, label %vector.ph.new

vector.ph.new:                                    ; preds = %vector.ph
  call void @llvm.set.loop.iterations.i32(i32 %tmp13)
  %unroll_iter = sub i32 %tmp13, %xtraiter
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph.new
  %index = phi i32 [ 0, %vector.ph.new ], [ %index.next.1, %vector.body ]
  %niter = phi i32 [ %unroll_iter, %vector.ph.new ], [ %niter.nsub.1, %vector.body ]
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
  %index.next = add nuw nsw i32 %index, 16
  %niter.nsub = sub i32 %niter, 1
  %broadcast.splatinsert.1 = insertelement <16 x i32> undef, i32 %index.next, i32 0
  %broadcast.splat.1 = shufflevector <16 x i32> %broadcast.splatinsert.1, <16 x i32> undef, <16 x i32> zeroinitializer
  %induction.1 = add <16 x i32> %broadcast.splat.1, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %tmp.1 = getelementptr inbounds i8, i8* %a, i32 %index.next
  %tmp1.1 = icmp ule <16 x i32> %induction.1, %broadcast.splat11
  %tmp2.1 = bitcast i8* %tmp.1 to <16 x i8>*
  %wide.masked.load.1 = tail call <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* %tmp2.1, i32 4, <16 x i1> %tmp1.1, <16 x i8> undef)
  %tmp3.1 = getelementptr inbounds i8, i8* %b, i32 %index.next
  %tmp4.1 = bitcast i8* %tmp3.1 to <16 x i8>*
  %wide.masked.load2.1 = tail call <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* %tmp4.1, i32 4, <16 x i1> %tmp1.1, <16 x i8> undef)
  %mul.1 = mul nsw <16 x i8> %wide.masked.load2.1, %wide.masked.load.1
  %tmp6.1 = getelementptr inbounds i8, i8* %c, i32 %index.next
  %tmp7.1 = bitcast i8* %tmp6.1 to <16 x i8>*
  tail call void @llvm.masked.store.v16i8.p0v16i8(<16 x i8> %mul.1, <16 x i8>* %tmp7.1, i32 4, <16 x i1> %tmp1.1)
  %index.next.1 = add i32 %index.next, 16
  %niter.nsub.1 = call i32 @llvm.loop.decrement.reg.i32.i32.i32(i32 %niter.nsub, i32 1)
  %niter.ncmp.1 = icmp ne i32 %niter.nsub.1, 0
  br i1 %niter.ncmp.1, label %vector.body, label %for.cond.cleanup.loopexit.unr-lcssa.loopexit

for.cond.cleanup.loopexit.unr-lcssa.loopexit:     ; preds = %vector.body
  %index.unr.ph = phi i32 [ %index.next.1, %vector.body ]
  %tmp14.unr.ph = phi i32 [ -2, %vector.body ]
  br label %for.cond.cleanup.loopexit.unr-lcssa

for.cond.cleanup.loopexit.unr-lcssa:              ; preds = %for.cond.cleanup.loopexit.unr-lcssa.loopexit, %vector.ph
  %index.unr = phi i32 [ 0, %vector.ph ], [ %index.unr.ph, %for.cond.cleanup.loopexit.unr-lcssa.loopexit ]
  %tmp14.unr = phi i32 [ %tmp13, %vector.ph ], [ %tmp14.unr.ph, %for.cond.cleanup.loopexit.unr-lcssa.loopexit ]
  %lcmp.mod = icmp ne i32 %xtraiter, 0
  br i1 %lcmp.mod, label %vector.body.epil.preheader, label %for.cond.cleanup.loopexit

vector.body.epil.preheader:                       ; preds = %for.cond.cleanup.loopexit.unr-lcssa
  br label %vector.body.epil

vector.body.epil:                                 ; preds = %vector.body.epil.preheader
  %index.epil = phi i32 [ %index.unr, %vector.body.epil.preheader ]
  %tmp14.epil = phi i32 [ %tmp14.unr, %vector.body.epil.preheader ]
  %broadcast.splatinsert.epil = insertelement <16 x i32> undef, i32 %index.epil, i32 0
  %broadcast.splat.epil = shufflevector <16 x i32> %broadcast.splatinsert.epil, <16 x i32> undef, <16 x i32> zeroinitializer
  %induction.epil = add <16 x i32> %broadcast.splat.epil, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %tmp.epil = getelementptr inbounds i8, i8* %a, i32 %index.epil
  %tmp1.epil = icmp ule <16 x i32> %induction.epil, %broadcast.splat11
  %tmp2.epil = bitcast i8* %tmp.epil to <16 x i8>*
  %wide.masked.load.epil = tail call <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* %tmp2.epil, i32 4, <16 x i1> %tmp1.epil, <16 x i8> undef)
  %tmp3.epil = getelementptr inbounds i8, i8* %b, i32 %index.epil
  %tmp4.epil = bitcast i8* %tmp3.epil to <16 x i8>*
  %wide.masked.load2.epil = tail call <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* %tmp4.epil, i32 4, <16 x i1> %tmp1.epil, <16 x i8> undef)
  %mul.epil = mul nsw <16 x i8> %wide.masked.load2.epil, %wide.masked.load.epil
  %tmp6.epil = getelementptr inbounds i8, i8* %c, i32 %index.epil
  %tmp7.epil = bitcast i8* %tmp6.epil to <16 x i8>*
  tail call void @llvm.masked.store.v16i8.p0v16i8(<16 x i8> %mul.epil, <16 x i8>* %tmp7.epil, i32 4, <16 x i1> %tmp1.epil)
  %index.next.epil = add i32 %index.epil, 16
  %tmp15.epil = add nuw nsw i32 %tmp14.epil, -1
  %tmp16.epil = icmp ne i32 %tmp15.epil, 0
  br label %for.cond.cleanup.loopexit.epilog-lcssa

for.cond.cleanup.loopexit.epilog-lcssa:           ; preds = %vector.body.epil
  br label %for.cond.cleanup.loopexit

for.cond.cleanup.loopexit:                        ; preds = %for.cond.cleanup.loopexit.unr-lcssa, %for.cond.cleanup.loopexit.epilog-lcssa
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void
}

declare <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>*, i32 immarg, <16 x i1>, <16 x i8>) #1
declare void @llvm.masked.store.v16i8.p0v16i8(<16 x i8>, <16 x i8>*, i32 immarg, <16 x i1>) #2
declare void @llvm.set.loop.iterations.i32(i32) #3
declare i32 @llvm.loop.decrement.reg.i32.i32.i32(i32, i32) #3

