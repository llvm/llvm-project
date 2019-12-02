; RUN: llc -mtriple=thumbv8.1m.main -mattr=+mve -enable-arm-maskedldst=true -disable-mve-tail-predication=false --verify-machineinstrs %s -o - | FileCheck %s

; CHECK-LABEL: vpsel_mul_reduce_add
; CHECK:      dls lr, lr
; CHECK:  [[LOOP:.LBB[0-9_]+]]:
; CHECK:      vctp.32 [[ELEMS:r[0-9]+]]
; CHECK:      mov [[ELEMS_OUT:r[0-9]+]], [[ELEMS]]
; CHECK:      vstr p0, [sp
; CHECK:      vpstt	
; CHECK-NEXT: vldrwt.u32
; CHECK-NEXT: vldrwt.u32
; CHECK:      vcmp.i32
; CHECK:      vpsel
; CHECK:      vldr p0, [sp
; CHECK:      vpst	
; CHECK-NEXT: vldrwt.u32 q{{.*}}, [r0]
; CHECK:      sub{{.*}} [[ELEMS]], [[ELEMS_OUT]], #4
; CHECK:      le lr, [[LOOP]]
; CHECK:      vctp.32	[[ELEMS_OUT]]
; CHECK-NEXT: vpsel
; CHECK-NEXT: vaddv.u32
define dso_local i32 @vpsel_mul_reduce_add(i32* noalias nocapture readonly %a, i32* noalias nocapture readonly %b, i32* noalias nocapture readonly %c, i32 %N) {
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
  %vec.phi = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ %add, %vector.body ]
  %broadcast.splatinsert = insertelement <4 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <4 x i32> %broadcast.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  %induction = add <4 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3>
  %tmp = getelementptr inbounds i32, i32* %a, i32 %index
  %tmp1 = icmp ule <4 x i32> %induction, %broadcast.splat12
  %tmp2 = bitcast i32* %tmp to <4 x i32>*
  %wide.masked.load.a = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp2, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %tmp3 = getelementptr inbounds i32, i32* %b, i32 %index
  %tmp4 = bitcast i32* %tmp3 to <4 x i32>*
  %wide.masked.load.b = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp4, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %tmp5 = getelementptr inbounds i32, i32* %c, i32 %index
  %tmp6 = bitcast i32* %tmp5 to <4 x i32>*
  %wide.masked.load.c = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp6, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %rem = urem i32 %index, 16
  %rem.broadcast.splatinsert = insertelement <4 x i32> undef, i32 %rem, i32 0
  %rem.broadcast.splat = shufflevector <4 x i32> %rem.broadcast.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  %cmp = icmp eq <4 x i32> %rem.broadcast.splat, <i32 0, i32 0, i32 0, i32 0>
  %wide.masked.load = select <4 x i1> %cmp, <4 x i32> %wide.masked.load.b, <4 x i32> %wide.masked.load.c
  %mul = mul nsw <4 x i32> %wide.masked.load, %wide.masked.load.a
  %add = add nsw <4 x i32> %mul, %vec.phi
  %index.next = add i32 %index, 4
  %tmp7 = icmp eq i32 %index.next, %n.vec
  br i1 %tmp7, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
  %tmp8 = select <4 x i1> %tmp1, <4 x i32> %add, <4 x i32> %vec.phi
  %tmp9 = call i32 @llvm.experimental.vector.reduce.add.v4i32(<4 x i32> %tmp8)
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %middle.block, %entry
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %tmp9, %middle.block ]
  ret i32 %res.0.lcssa
}

; CHECK-LABEL: vpsel_mul_reduce_add_2
; CHECK:      dls lr, lr
; CHECK:  [[LOOP:.LBB[0-9_]+]]:
; CHECK:      vctp.32 [[ELEMS:r[0-9]+]]
; CHECK:      mov [[ELEMS_OUT:r[0-9]+]], [[ELEMS]]
; CHECK:      vstr p0, [sp
; CHECK:      vpstt
; CHECK-NEXT: vldrwt.u32
; CHECK-NEXT: vldrwt.u32
; CHECK;      vsub
; CHECK:      vpst
; CHECK-NEXT: vldrwt.u32
; CHECK:      vcmp.i32
; CHECK:      vpsel
; CHECK:      vldr p0, [sp
; CHECK:      vpst	
; CHECK-NEXT: vldrwt.u32 q{{.*}}, [r0]
; CHECK:      sub{{.*}} [[ELEMS]], [[ELEMS_OUT]], #4
; CHECK:      le lr, [[LOOP]]
; CHECK:      vctp.32	[[ELEMS_OUT]]
; CHECK-NEXT: vpsel
; CHECK-NEXT: vaddv.u32
define dso_local i32 @vpsel_mul_reduce_add_2(i32* noalias nocapture readonly %a, i32* noalias nocapture readonly %b,
                                         i32* noalias nocapture readonly %c, i32* noalias nocapture readonly %d, i32 %N) {
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
  %vec.phi = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ %add, %vector.body ]
  %broadcast.splatinsert = insertelement <4 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <4 x i32> %broadcast.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  %induction = add <4 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3>
  %tmp = getelementptr inbounds i32, i32* %a, i32 %index
  %tmp1 = icmp ule <4 x i32> %induction, %broadcast.splat12
  %tmp2 = bitcast i32* %tmp to <4 x i32>*
  %wide.masked.load.a = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp2, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %tmp3 = getelementptr inbounds i32, i32* %b, i32 %index
  %tmp4 = bitcast i32* %tmp3 to <4 x i32>*
  %wide.masked.load.b = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp4, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %tmp5 = getelementptr inbounds i32, i32* %c, i32 %index
  %tmp6 = bitcast i32* %tmp5 to <4 x i32>*
  %wide.masked.load.c = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp6, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %tmp7 = getelementptr inbounds i32, i32* %d, i32 %index
  %tmp8 = bitcast i32* %tmp7 to <4 x i32>*
  %wide.masked.load.d = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp8, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %sub = sub <4 x i32> %wide.masked.load.c, %wide.masked.load.d
  %rem = urem i32 %index, 16
  %rem.broadcast.splatinsert = insertelement <4 x i32> undef, i32 %rem, i32 0
  %rem.broadcast.splat = shufflevector <4 x i32> %rem.broadcast.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  %cmp = icmp eq <4 x i32> %rem.broadcast.splat, <i32 0, i32 0, i32 0, i32 0>
  %sel = select <4 x i1> %cmp, <4 x i32> %sub, <4 x i32> %wide.masked.load.b
  %mul = mul  <4 x i32> %sel, %wide.masked.load.a
  %add = add  <4 x i32> %mul, %vec.phi
  %index.next = add i32 %index, 4
  %cmp.exit = icmp eq i32 %index.next, %n.vec
  br i1 %cmp.exit, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
  %acc = select <4 x i1> %tmp1, <4 x i32> %add, <4 x i32> %vec.phi
  %reduce = call i32 @llvm.experimental.vector.reduce.add.v4i32(<4 x i32> %acc)
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %middle.block, %entry
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %reduce, %middle.block ]
  ret i32 %res.0.lcssa
}

; CHECK-LABEL: and_mul_reduce_add
; CHECK:      dls lr, lr
; CHECK:  [[LOOP:.LBB[0-9_]+]]:
; CHECK:      vctp.32 [[ELEMS:r[0-9]+]]
; CHECK:      vpstt	
; CHECK-NEXT: vldrwt.u32
; CHECK-NEXT: vldrwt.u32
; CHECK:      mov [[ELEMS_OUT:r[0-9]+]], [[ELEMS]]
; CHECK:      sub{{.*}} [[ELEMS]],{{.*}}#4
; CHECK:      vpsttt
; CHECK-NEXT: vcmpt.i32	eq, {{.*}}, zr
; CHECK-NEXT: vldrwt.u32 q{{.*}}, [r3]
; CHECK-NEXT: vldrwt.u32 q{{.*}}, [r2]
; CHECK:      le lr, [[LOOP]]
; CHECK:      vctp.32 [[ELEMS_OUT]]
; CHECK:      vpsel
define dso_local i32 @and_mul_reduce_add(i32* noalias nocapture readonly %a, i32* noalias nocapture readonly %b,
                                         i32* noalias nocapture readonly %c, i32* noalias nocapture readonly %d, i32 %N) {
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
  %vec.phi = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ %add, %vector.body ]
  %broadcast.splatinsert = insertelement <4 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <4 x i32> %broadcast.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  %induction = add <4 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3>
  %tmp = getelementptr inbounds i32, i32* %a, i32 %index
  %tmp1 = icmp ule <4 x i32> %induction, %broadcast.splat12
  %tmp2 = bitcast i32* %tmp to <4 x i32>*
  %wide.masked.load.a = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp2, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %tmp3 = getelementptr inbounds i32, i32* %b, i32 %index
  %tmp4 = bitcast i32* %tmp3 to <4 x i32>*
  %wide.masked.load.b = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp4, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %sub = sub <4 x i32> %wide.masked.load.a, %wide.masked.load.b
  %cmp = icmp eq <4 x i32> %sub, <i32 0, i32 0, i32 0, i32 0>
  %mask = and <4 x i1> %cmp, %tmp1
  %tmp5 = getelementptr inbounds i32, i32* %c, i32 %index
  %tmp6 = bitcast i32* %tmp5 to <4 x i32>*
  %wide.masked.load.c = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp6, i32 4, <4 x i1> %mask, <4 x i32> undef)
  %tmp7 = getelementptr inbounds i32, i32* %d, i32 %index
  %tmp8 = bitcast i32* %tmp7 to <4 x i32>*
  %wide.masked.load.d = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp8, i32 4, <4 x i1> %mask, <4 x i32> undef)
  %mul = mul  <4 x i32> %wide.masked.load.c, %wide.masked.load.d
  %add = add  <4 x i32> %mul, %vec.phi
  %index.next = add i32 %index, 4
  %cmp.exit = icmp eq i32 %index.next, %n.vec
  br i1 %cmp.exit, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
  %acc = select <4 x i1> %tmp1, <4 x i32> %add, <4 x i32> %vec.phi
  %reduce = call i32 @llvm.experimental.vector.reduce.add.v4i32(<4 x i32> %acc)
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %middle.block, %entry
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %reduce, %middle.block ]
  ret i32 %res.0.lcssa
}

; TODO: Why does p0 get reloaded from the stack into p0, just to be vmrs'd?
; CHECK-LABEL: or_mul_reduce_add
; CHECK:      dls lr, lr
; CHECK:  [[LOOP:.LBB[0-9_]+]]:
; CHECK:      vctp.32 [[ELEMS:r[0-9]+]]
; CHECK:      vstr p0, [sp
; CHECK:      mov [[ELEMS_OUT:r[0-9]+]], [[ELEMS]]
; CHECK:      vpstt	
; CHECK-NEXT: vldrwt.u32
; CHECK-NEXT: vldrwt.u32
; CHECK:      vcmp.i32	eq, {{.*}}, zr
; CHECK:      vmrs [[VCMP:r[0-9]+]], p0
; CHECK:      vldr p0, [sp
; CHECK:      vmrs [[VCTP:r[0-9]+]], p0
; CHECK:      orr{{.*}} [[VCMP]], [[VCTP]]
; CHECK:      sub{{.*}} [[ELEMS:r[0-9]+]], [[ELEMS_OUT]], #4
; CHECK-NEXT: vmsr p0
; CHECK-NEXT: vpstt
; CHECK-NEXT: vldrwt.u32 q{{.*}}, [r3]
; CHECK-NEXT: vldrwt.u32 q{{.*}}, [r2]
; CHECK:      le lr, [[LOOP]]
; CHECK:      vctp.32 [[ELEMS_OUT]]
; CHECK:      vpsel
define dso_local i32 @or_mul_reduce_add(i32* noalias nocapture readonly %a, i32* noalias nocapture readonly %b,
                                        i32* noalias nocapture readonly %c, i32* noalias nocapture readonly %d, i32 %N) {
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
  %vec.phi = phi <4 x i32> [ zeroinitializer, %vector.ph ], [ %add, %vector.body ]
  %broadcast.splatinsert = insertelement <4 x i32> undef, i32 %index, i32 0
  %broadcast.splat = shufflevector <4 x i32> %broadcast.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  %induction = add <4 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3>
  %tmp = getelementptr inbounds i32, i32* %a, i32 %index
  %tmp1 = icmp ule <4 x i32> %induction, %broadcast.splat12
  %tmp2 = bitcast i32* %tmp to <4 x i32>*
  %wide.masked.load.a = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp2, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %tmp3 = getelementptr inbounds i32, i32* %b, i32 %index
  %tmp4 = bitcast i32* %tmp3 to <4 x i32>*
  %wide.masked.load.b = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp4, i32 4, <4 x i1> %tmp1, <4 x i32> undef)
  %sub = sub <4 x i32> %wide.masked.load.a, %wide.masked.load.b
  %cmp = icmp eq <4 x i32> %sub, <i32 0, i32 0, i32 0, i32 0>
  %mask = or <4 x i1> %cmp, %tmp1
  %tmp5 = getelementptr inbounds i32, i32* %c, i32 %index
  %tmp6 = bitcast i32* %tmp5 to <4 x i32>*
  %wide.masked.load.c = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp6, i32 4, <4 x i1> %mask, <4 x i32> undef)
  %tmp7 = getelementptr inbounds i32, i32* %d, i32 %index
  %tmp8 = bitcast i32* %tmp7 to <4 x i32>*
  %wide.masked.load.d = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %tmp8, i32 4, <4 x i1> %mask, <4 x i32> undef)
  %mul = mul  <4 x i32> %wide.masked.load.c, %wide.masked.load.d
  %add = add  <4 x i32> %mul, %vec.phi
  %index.next = add i32 %index, 4
  %cmp.exit = icmp eq i32 %index.next, %n.vec
  br i1 %cmp.exit, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
  %acc = select <4 x i1> %tmp1, <4 x i32> %add, <4 x i32> %vec.phi
  %reduce = call i32 @llvm.experimental.vector.reduce.add.v4i32(<4 x i32> %acc)
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %middle.block, %entry
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %reduce, %middle.block ]
  ret i32 %res.0.lcssa
}

; Function Attrs: argmemonly nounwind readonly willreturn
declare <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>*, i32 immarg, <4 x i1>, <4 x i32>)

; Function Attrs: nounwind readnone willreturn
declare i32 @llvm.experimental.vector.reduce.add.v4i32(<4 x i32>)
