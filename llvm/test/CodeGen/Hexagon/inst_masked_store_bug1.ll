;; REQUIRES: asserts
;; RUN: llc --mtriple=hexagon -mattr=+hvxv79,+hvx-length128b %s -o - | FileCheck %s
;; Sanity check for lowering masked scatter without assertion errors.

define void @outer_product(ptr  %aptr, ptr  %bptr, ptr %cptr, i32 %T, i32 %W) {
entry:
  %W.ripple.bcast.splatinsert = insertelement <8 x i32> poison, i32 %W, i64 0
  %W.ripple.bcast.splat = shufflevector <8 x i32> %W.ripple.bcast.splatinsert, <8 x i32> poison, <8 x i32> zeroinitializer
  %div1194 = lshr i32 %T, 3
  %cmp84.not = icmp ult i32 %T, 8
  br i1 %cmp84.not, label %for.end49, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %div10195 = lshr i32 %W, 3
  %cmp1782.not = icmp ult i32 %W, 8
  %arrayidx27.ripple.LS.dim.slope = mul <8 x i32> %W.ripple.bcast.splat, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %arrayidx27.ripple.LS.dim.slope.ripple.bcast = shufflevector <8 x i32> %arrayidx27.ripple.LS.dim.slope, <8 x i32> poison, <64 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  %arrayidx27.ripple.LS.slope = add <64 x i32> %arrayidx27.ripple.LS.dim.slope.ripple.bcast, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %invariant.gep196 = getelementptr i8, ptr %cptr, <64 x i32> %arrayidx27.ripple.LS.slope
  br label %for.body

for.body:                                         ; preds = %for.end, %for.body.preheader
  %ripple.par.iv.085 = phi i32 [ %add48, %for.end ], [ 0, %for.body.preheader ]
  %mul2 = shl i32 %ripple.par.iv.085, 3
  br i1 %cmp1782.not, label %for.end, label %for.body18.lr.ph

for.body18.lr.ph:                                 ; preds = %for.body
  %arrayidx = getelementptr inbounds nuw i8, ptr %aptr, i32 %mul2
  %mul25 = mul i32 %mul2, %W
  %gep197 = getelementptr i8, <64 x ptr> %invariant.gep196, i32 %mul25
  br label %for.body18

for.body18:                                       ; preds = %for.body18, %for.body18.lr.ph
  %ripple.par.iv15.083 = phi i32 [ 0, %for.body18.lr.ph ], [ %add28, %for.body18 ]
  %mul19 = shl i32 %ripple.par.iv15.083, 3
  %.ripple.LS.instance184 = load <8 x i8>, ptr %arrayidx, align 1
  %.ripple.LS.instance184.ripple.bcast = shufflevector <8 x i8> %.ripple.LS.instance184, <8 x i8> poison, <64 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  %arrayidx21 = getelementptr inbounds nuw i8, ptr %bptr, i32 %mul19
  %.ripple.LS.instance = load <8 x i8>, ptr %arrayidx21, align 1
  %.ripple.LS.instance.ripple.bcast = shufflevector <8 x i8> %.ripple.LS.instance, <8 x i8> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %mul23.ripple.LS.instance = mul <64 x i8> %.ripple.LS.instance.ripple.bcast, %.ripple.LS.instance184.ripple.bcast
  %gep = getelementptr i8, <64 x ptr> %gep197, i32 %mul19
  tail call void @llvm.masked.scatter.v64i8.v64p0(<64 x i8> %mul23.ripple.LS.instance, <64 x ptr> %gep, i32 1, <64 x i1> splat (i1 true))
  %add28 = add nuw i32 %ripple.par.iv15.083, 1
  %cmp17 = icmp ult i32 %add28, %div10195
  br i1 %cmp17, label %for.body18, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body18
  %0 = shl i32 %add28, 3
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %for.body
  %ripple.par.iv15.0.lcssa = phi i32 [ 0, %for.body ], [ %0, %for.end.loopexit ]
  %add30.ripple.bcast.splatinsert = insertelement <8 x i32> poison, i32 %ripple.par.iv15.0.lcssa, i64 0
  %add30.ripple.bcast.splat = shufflevector <8 x i32> %add30.ripple.bcast.splatinsert, <8 x i32> poison, <8 x i32> zeroinitializer
  %add30.ripple.LS.instance = or disjoint <8 x i32> %add30.ripple.bcast.splat, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %cmp32.ripple.LS.instance = icmp ne i32 %ripple.par.iv15.0.lcssa, %W
  %cmp32.ripple.LS.instance.ripple.bcast.splatinsert = insertelement <8 x i1> poison, i1 %cmp32.ripple.LS.instance, i64 0
  %cmp32.ripple.LS.instance.ripple.bcast.splat = shufflevector <8 x i1> %cmp32.ripple.LS.instance.ripple.bcast.splatinsert, <8 x i1> poison, <8 x i32> zeroinitializer
  %cmp33.ripple.vectorized = icmp ult <8 x i32> %add30.ripple.LS.instance, %W.ripple.bcast.splat
  %or.cond.ripple.LS.instance = select <8 x i1> %cmp32.ripple.LS.instance.ripple.bcast.splat, <8 x i1> %cmp33.ripple.vectorized, <8 x i1> zeroinitializer
  %or.cond.ripple.LS.instance.ripple.bcast = shufflevector <8 x i1> %or.cond.ripple.LS.instance, <8 x i1> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %or.cond.ripple.LS.instance.ripple.reducelog2.shuffle = shufflevector <8 x i1> %or.cond.ripple.LS.instance, <8 x i1> <i1 poison, i1 poison, i1 poison, i1 poison, i1 poison, i1 poison, i1 poison, i1 false>, <8 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 15>
  %or.cond.ripple.LS.instance.ripple.reducelog2.operator = or <8 x i1> %or.cond.ripple.LS.instance, %or.cond.ripple.LS.instance.ripple.reducelog2.shuffle
  %or.cond.ripple.LS.instance.ripple.reducelog2.shuffle189 = shufflevector <8 x i1> %or.cond.ripple.LS.instance.ripple.reducelog2.operator, <8 x i1> <i1 poison, i1 poison, i1 poison, i1 poison, i1 poison, i1 poison, i1 false, i1 false>, <8 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 14, i32 15>
  %or.cond.ripple.LS.instance.ripple.reducelog2.operator190 = or <8 x i1> %or.cond.ripple.LS.instance.ripple.reducelog2.operator, %or.cond.ripple.LS.instance.ripple.reducelog2.shuffle189
  %or.cond.ripple.LS.instance.ripple.reducelog2.shuffle191 = shufflevector <8 x i1> %or.cond.ripple.LS.instance.ripple.reducelog2.operator190, <8 x i1> poison, <8 x i32> <i32 4, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %or.cond.ripple.LS.instance.ripple.reducelog2.operator192 = or <8 x i1> %or.cond.ripple.LS.instance.ripple.reducelog2.operator190, %or.cond.ripple.LS.instance.ripple.reducelog2.shuffle191
  %ripple.red.extract.ripple.bcast.splat = shufflevector <8 x i1> %or.cond.ripple.LS.instance.ripple.reducelog2.operator192, <8 x i1> poison, <8 x i32> zeroinitializer
  %arrayidx34.ripple.branch.clone = getelementptr inbounds nuw i8, ptr %aptr, i32 %mul2
  %.ripple.LS.instance188.ripple.branch.clone.ripple.masked.load = tail call <8 x i8> @llvm.masked.load.v8i8.p0(ptr %arrayidx34.ripple.branch.clone, i32 1, <8 x i1> %ripple.red.extract.ripple.bcast.splat, <8 x i8> poison)
  %.ripple.LS.instance188.ripple.bcast.ripple.branch.clone = shufflevector <8 x i8> %.ripple.LS.instance188.ripple.branch.clone.ripple.masked.load, <8 x i8> poison, <64 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  %arrayidx36.ripple.branch.clone = getelementptr inbounds nuw i8, ptr %bptr, i32 %ripple.par.iv15.0.lcssa
  %.ripple.LS.instance187.ripple.branch.clone.ripple.masked.load = tail call <8 x i8> @llvm.masked.load.v8i8.p0(ptr %arrayidx36.ripple.branch.clone, i32 1, <8 x i1> %or.cond.ripple.LS.instance, <8 x i8> poison)
  %.ripple.LS.instance187.ripple.bcast.ripple.branch.clone = shufflevector <8 x i8> %.ripple.LS.instance187.ripple.branch.clone.ripple.masked.load, <8 x i8> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %mul38.ripple.LS.instance.ripple.branch.clone = mul <64 x i8> %.ripple.LS.instance187.ripple.bcast.ripple.branch.clone, %.ripple.LS.instance188.ripple.bcast.ripple.branch.clone
  %mul40.ripple.branch.clone = mul i32 %mul2, %W
  %1 = getelementptr i8, ptr %cptr, i32 %mul40.ripple.branch.clone
  %arrayidx42.ripple.branch.clone = getelementptr i8, ptr %1, i32 %ripple.par.iv15.0.lcssa
  %arrayidx42.ripple.LS.instance.ripple.branch.clone = getelementptr i8, ptr %arrayidx42.ripple.branch.clone, <64 x i32> %arrayidx27.ripple.LS.slope
  tail call void @llvm.masked.scatter.v64i8.v64p0(<64 x i8> %mul38.ripple.LS.instance.ripple.branch.clone, <64 x ptr> %arrayidx42.ripple.LS.instance.ripple.branch.clone, i32 1, <64 x i1> %or.cond.ripple.LS.instance.ripple.bcast)
  %add48 = add nuw i32 %ripple.par.iv.085, 1
  %cmp = icmp ult i32 %add48, %div1194
  br i1 %cmp, label %for.body, label %for.end49

for.end49:                                        ; preds = %for.end, %entry
  ret void
}

;; CHECK: outer_product
;; CHECK: {{r[0-9]+}} = lsr({{r[0-9]+}},#3)
;; CHECK: {{q[0-9]+}} = vand({{v[0-9]+}},{{r[0-9]+}})
;; CHECK: {{v[0-9]+}} = vmux(q0,{{v[0-9]+}},{{v[0-9]+}})
;; CHECK: vmem{{.*}} = {{v[0-9]+}}
