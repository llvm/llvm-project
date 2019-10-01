; RUN: llc -mtriple=armv8.1m.main -mattr=+mve -enable-arm-maskedldst=true -disable-mve-tail-predication=false --verify-machineinstrs %s -o - | FileCheck %s

; CHECK-LABEL: mul_reduce_add
; CHECK:      dls lr,
; CHECK: [[LOOP:.LBB[0-9_]+]]:
; CHECK:      vctp.32	[[ELEMS:r[0-9]+]]
; CHECK:      vpstt	
; CHECK-NEXT: vldrwt.u32
; CHECK-NEXT: vldrwt.u32
; CHECK:      mov [[ELEMS_OUT:r[0-9]+]], [[ELEMS]]
; CHECK:      sub{{.*}} [[ELEMS]], #4
; CHECK:      le	lr, [[LOOP]]
; CHECK:      vctp.32	[[ELEMS_OUT]]
; CHECK:      vpsel
; CHECK:      vaddv.u32	r0
define dso_local i32 @mul_reduce_add(i32* noalias nocapture readonly %a, i32* noalias nocapture readonly %b, i32 %N) {
entry:
  %cmp8 = icmp eq i32 %N, 0
  br i1 %cmp8, label %for.cond.cleanup, label %vector.ph

vector.ph:                                        ; preds = %entry
  %n.rnd.up = add i32 %N, 3
  %n.vec = and i32 %n.rnd.up, -4
  %trip.count.minus.1 = add i32 %N, -1
  %broadcast.splatinsert11 = insertelement <4 x i32> undef, i32 %trip.count.minus.1, i32 0
  %broadcast.splat12 = shufflevector <4 x i32> %broadcast.splatinsert11, <4 x i32> undef, <4 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %vec.phi = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ %6, %vector.body ]
  %broadcast.splatinsert = insertelement <4 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <4 x i32> %broadcast.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  %induction = add <4 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3>
  %0 = getelementptr inbounds i32, i32* %a, i32 %index
  %1 = icmp ule <4 x i32> %induction, %broadcast.splat12
  %2 = bitcast i32* %0 to <4 x i32>*
  %wide.masked.load = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %2, i32 4, <4 x i1> %1, <4 x i32> undef)
  %3 = getelementptr inbounds i32, i32* %b, i32 %index
  %4 = bitcast i32* %3 to <4 x i32>*
  %wide.masked.load13 = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %4, i32 4, <4 x i1> %1, <4 x i32> undef)
  %5 = mul nsw <4 x i32> %wide.masked.load13, %wide.masked.load
  %6 = add nsw <4 x i32> %5, %vec.phi
  %index.next = add i32 %index, 4
  %7 = icmp eq i32 %index.next, %n.vec
  br i1 %7, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
  %8 = select <4 x i1> %1, <4 x i32> %6, <4 x i32> %vec.phi
  %9 = call i32 @llvm.experimental.vector.reduce.add.v4i32(<4 x i32> %8)
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %middle.block, %entry
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %9, %middle.block ]
  ret i32 %res.0.lcssa
}

; CHECK-LABEL: mul_reduce_add_const
; CHECK:    dls lr
; CHECK: [[LOOP:.LBB[0-9_]+]]:
; CHECK:      vctp.32 [[ELEMS:r[0-9]+]]
; CHECK:      vpst	
; CHECK-NEXT: vldrwt.u32 q{{.*}}, [r0]
; CHECK:      mov [[ELEMS_OUT:r[0-9]+]], [[ELEMS]]
; CHECK:      sub{{.*}} [[ELEMS]], #4
; CHECK:      le lr, [[LOOP]]
; CHECK:      vctp.32 [[ELEMS_OUT]]
; CHECK:      vpsel
define dso_local i32 @mul_reduce_add_const(i32* noalias nocapture readonly %a, i32 %b, i32 %N) {
entry:
  %cmp6 = icmp eq i32 %N, 0
  br i1 %cmp6, label %for.cond.cleanup, label %vector.ph

vector.ph:                                        ; preds = %entry
  %n.rnd.up = add i32 %N, 3
  %n.vec = and i32 %n.rnd.up, -4
  %trip.count.minus.1 = add i32 %N, -1
  %broadcast.splatinsert9 = insertelement <4 x i32> undef, i32 %trip.count.minus.1, i32 0
  %broadcast.splat10 = shufflevector <4 x i32> %broadcast.splatinsert9, <4 x i32> undef, <4 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %vec.phi = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ %3, %vector.body ]
  %broadcast.splatinsert = insertelement <4 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <4 x i32> %broadcast.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  %induction = add <4 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3>
  %0 = getelementptr inbounds i32, i32* %a, i32 %index
  %1 = icmp ule <4 x i32> %induction, %broadcast.splat10
  %2 = bitcast i32* %0 to <4 x i32>*
  %wide.masked.load = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %2, i32 4, <4 x i1> %1, <4 x i32> undef)
  %3 = add nsw <4 x i32> %wide.masked.load, %vec.phi
  %index.next = add i32 %index, 4
  %4 = icmp eq i32 %index.next, %n.vec
  br i1 %4, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
  %5 = select <4 x i1> %1, <4 x i32> %3, <4 x i32> %vec.phi
  %6 = call i32 @llvm.experimental.vector.reduce.add.v4i32(<4 x i32> %5)
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %middle.block, %entry
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %6, %middle.block ]
  ret i32 %res.0.lcssa
}

; CHECK-LABEL: add_reduce_add_const
; CHECK:      dls lr, lr
; CHECK: [[LOOP:.LBB[0-9_]+]]:
; CHECK:      vctp.32 [[ELEMS:r[0-9]+]]
; CHECK:      vpst	
; CHECK-NEXT: vldrwt.u32 q{{.*}}, [r0]
; CHECK:      mov [[ELEMS_OUT:r[0-9]+]], [[ELEMS]]
; CHECK:      sub{{.*}} [[ELEMS]], #4
; CHECK:      vadd.i32
; CHECK:      le lr, [[LOOP]]
; CHECK:      vctp.32 [[ELEMS_OUT]]
; CHECK:      vpsel
define dso_local i32 @add_reduce_add_const(i32* noalias nocapture readonly %a, i32 %b, i32 %N) {
entry:
  %cmp6 = icmp eq i32 %N, 0
  br i1 %cmp6, label %for.cond.cleanup, label %vector.ph

vector.ph:                                        ; preds = %entry
  %n.rnd.up = add i32 %N, 3
  %n.vec = and i32 %n.rnd.up, -4
  %trip.count.minus.1 = add i32 %N, -1
  %broadcast.splatinsert9 = insertelement <4 x i32> undef, i32 %trip.count.minus.1, i32 0
  %broadcast.splat10 = shufflevector <4 x i32> %broadcast.splatinsert9, <4 x i32> undef, <4 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %vec.phi = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ %3, %vector.body ]
  %broadcast.splatinsert = insertelement <4 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <4 x i32> %broadcast.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  %induction = add <4 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3>
  %0 = getelementptr inbounds i32, i32* %a, i32 %index
  %1 = icmp ule <4 x i32> %induction, %broadcast.splat10
  %2 = bitcast i32* %0 to <4 x i32>*
  %wide.masked.load = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %2, i32 4, <4 x i1> %1, <4 x i32> undef)
  %3 = add nsw <4 x i32> %wide.masked.load, %vec.phi
  %index.next = add i32 %index, 4
  %4 = icmp eq i32 %index.next, %n.vec
  br i1 %4, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
  %5 = select <4 x i1> %1, <4 x i32> %3, <4 x i32> %vec.phi
  %6 = call i32 @llvm.experimental.vector.reduce.add.v4i32(<4 x i32> %5)
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %middle.block, %entry
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %6, %middle.block ]
  ret i32 %res.0.lcssa
}

; CHECK-LABEL: vector_mul_const
; CHECK:      dls lr, lr
; CHECK: [[LOOP:.LBB[0-9_]+]]:
; CHECK:      vctp.32 [[ELEMS:r[0-9]+]]
; CHECK:      sub{{.*}} [[ELEMS]], #4
; CHECK:      vpst	
; CHECK-NEXT: vldrwt.u32 q{{.*}}, [r1]
; CHECK:      vmul.i32
; CHECK:      vpst	
; CHECK-NEXT: vstrwt.32 q{{.*}}, [r0]
; CHECK:      le lr, [[LOOP]]
define dso_local void @vector_mul_const(i32* noalias nocapture %a, i32* noalias nocapture readonly %b, i32 %c, i32 %N) {
entry:
  %cmp6 = icmp eq i32 %N, 0
  br i1 %cmp6, label %for.cond.cleanup, label %vector.ph

vector.ph:                                        ; preds = %entry
  %n.rnd.up = add i32 %N, 3
  %n.vec = and i32 %n.rnd.up, -4
  %trip.count.minus.1 = add i32 %N, -1
  %broadcast.splatinsert8 = insertelement <4 x i32> undef, i32 %trip.count.minus.1, i32 0
  %broadcast.splat9 = shufflevector <4 x i32> %broadcast.splatinsert8, <4 x i32> undef, <4 x i32> zeroinitializer
  %broadcast.splatinsert10 = insertelement <4 x i32> undef, i32 %c, i32 0
  %broadcast.splat11 = shufflevector <4 x i32> %broadcast.splatinsert10, <4 x i32> undef, <4 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %broadcast.splatinsert = insertelement <4 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <4 x i32> %broadcast.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  %induction = add <4 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3>
  %0 = getelementptr inbounds i32, i32* %b, i32 %index
  %1 = icmp ule <4 x i32> %induction, %broadcast.splat9
  %2 = bitcast i32* %0 to <4 x i32>*
  %wide.masked.load = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %2, i32 4, <4 x i1> %1, <4 x i32> undef)
  %3 = mul nsw <4 x i32> %wide.masked.load, %broadcast.splat11
  %4 = getelementptr inbounds i32, i32* %a, i32 %index
  %5 = bitcast i32* %4 to <4 x i32>*
  call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %3, <4 x i32>* %5, i32 4, <4 x i1> %1)
  %index.next = add i32 %index, 4
  %6 = icmp eq i32 %index.next, %n.vec
  br i1 %6, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %vector.body, %entry
  ret void
}

; CHECK-LABEL: vector_add_const
; CHECK:      dls lr, lr
; CHECK: [[LOOP:.LBB[0-9_]+]]:
; CHECK:      vctp.32 [[ELEMS:r[0-9]+]]
; CHECK:      sub{{.*}} [[ELEMS]], #4
; CHECK:      vpst	
; CHECK-NEXT: vldrwt.u32 q{{.*}}, [r1]
; CHECK:      vadd.i32
; CHECK:      vpst	
; CHECK-NEXT: vstrwt.32 q{{.*}}, [r0]
; CHECK:      le lr, [[LOOP]]
define dso_local void @vector_add_const(i32* noalias nocapture %a, i32* noalias nocapture readonly %b, i32 %c, i32 %N) {
entry:
  %cmp6 = icmp eq i32 %N, 0
  br i1 %cmp6, label %for.cond.cleanup, label %vector.ph

vector.ph:                                        ; preds = %entry
  %n.rnd.up = add i32 %N, 3
  %n.vec = and i32 %n.rnd.up, -4
  %trip.count.minus.1 = add i32 %N, -1
  %broadcast.splatinsert8 = insertelement <4 x i32> undef, i32 %trip.count.minus.1, i32 0
  %broadcast.splat9 = shufflevector <4 x i32> %broadcast.splatinsert8, <4 x i32> undef, <4 x i32> zeroinitializer
  %broadcast.splatinsert10 = insertelement <4 x i32> undef, i32 %c, i32 0
  %broadcast.splat11 = shufflevector <4 x i32> %broadcast.splatinsert10, <4 x i32> undef, <4 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %broadcast.splatinsert = insertelement <4 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <4 x i32> %broadcast.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  %induction = add <4 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3>
  %0 = getelementptr inbounds i32, i32* %b, i32 %index
  %1 = icmp ule <4 x i32> %induction, %broadcast.splat9
  %2 = bitcast i32* %0 to <4 x i32>*
  %wide.masked.load = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %2, i32 4, <4 x i1> %1, <4 x i32> undef)
  %3 = add nsw <4 x i32> %wide.masked.load, %broadcast.splat11
  %4 = getelementptr inbounds i32, i32* %a, i32 %index
  %5 = bitcast i32* %4 to <4 x i32>*
  call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %3, <4 x i32>* %5, i32 4, <4 x i1> %1)
  %index.next = add i32 %index, 4
  %6 = icmp eq i32 %index.next, %n.vec
  br i1 %6, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %vector.body, %entry
  ret void
}

declare <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>*, i32 immarg, <4 x i1>, <4 x i32>)
declare void @llvm.masked.store.v4i32.p0v4i32(<4 x i32>, <4 x i32>*, i32 immarg, <4 x i1>) #4
declare i32 @llvm.experimental.vector.reduce.add.v4i32(<4 x i32>)

