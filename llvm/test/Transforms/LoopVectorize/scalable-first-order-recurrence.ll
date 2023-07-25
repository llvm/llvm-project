; RUN: opt -passes=loop-vectorize -scalable-vectorization=on -force-vector-width=4 -force-vector-interleave=1 -force-target-supports-scalable-vectors=true -S < %s | FileCheck %s --check-prefix=CHECK-VF4UF1
; RUN: opt -passes=loop-vectorize -scalable-vectorization=on -force-vector-width=4 -force-vector-interleave=2 -force-target-supports-scalable-vectors=true -S < %s | FileCheck %s --check-prefix=CHECK-VF4UF2

; void recurrence_1(int *a, int *b, int n) {
;   for(int i = 0; i < n; i++)
;     b[i] =  a[i] + a[i - 1]
; }
;
define i32 @recurrence_1(ptr nocapture readonly %a, ptr nocapture %b, i32 %n) {
; CHECK-VF4UF1-LABEL: @recurrence_1
; CHECK-VF4UF1: for.preheader
; CHECK-VF4UF1: %[[SUB_1:.*]] = add i32 %n, -1
; CHECK-VF4UF1: %[[ZEXT:.*]] = zext i32 %[[SUB_1]] to i64
; CHECK-VF4UF1: %[[ADD:.*]] = add nuw nsw i64 %[[ZEXT]], 1
; CHECK-VF4UF1: vector.ph:
; CHECK-VF4UF1: %[[VSCALE1:.*]] = call i32 @llvm.vscale.i32()
; CHECK-VF4UF1: %[[MUL1:.*]] = mul i32 %[[VSCALE1]], 4
; CHECK-VF4UF1: %[[SUB1:.*]] = sub i32 %[[MUL1]], 1
; CHECK-VF4UF1: %[[VEC_RECUR_INIT:.*]] = insertelement <vscale x 4 x i32> poison, i32 %pre_load, i32 %[[SUB1]]
; CHECK-VF4UF1: vector.body:
; CHECK-VF4UF1: %[[INDEX:.*]] = phi i64 [ 0, %vector.ph ], [ %[[NEXT_IDX:.*]], %vector.body ]
; CHECK-VF4UF1: %[[VEC_RECUR:.*]] = phi <vscale x 4 x i32> [ %[[VEC_RECUR_INIT]], %vector.ph ], [ %[[LOAD:.*]], %vector.body ]
; CHECK-VF4UF1: %[[LOAD]] = load <vscale x 4 x i32>, ptr
; CHECK-VF4UF1: %[[SPLICE:.*]] = call <vscale x 4 x i32> @llvm.experimental.vector.splice.nxv4i32(<vscale x 4 x i32> %[[VEC_RECUR]], <vscale x 4 x i32> %[[LOAD]], i32 -1)
; CHECK-VF4UF1: middle.block:
; CHECK-VF4UF1: %[[VSCALE2:.*]] = call i32 @llvm.vscale.i32()
; CHECK-VF4UF1: %[[MUL2:.*]] = mul i32 %[[VSCALE2]], 4
; CHECK-VF4UF1: %[[SUB2:.*]] = sub i32 %[[MUL2]], 1
; CHECK-VF4UF1: %[[VEC_RECUR_EXT:.*]] = extractelement <vscale x 4 x i32> %[[LOAD]], i32 %[[SUB2]]
; CHECK-VF4UF1: %[[SUB3:.*]] = sub i32 %[[MUL2]], 2
; CHECK-VF4UF1: %[[VEC_RECUR_FOR_PHI:.*]] =  extractelement <vscale x 4 x i32> %[[LOAD]], i32 %[[SUB3]]
entry:
  br label %for.preheader

for.preheader:
  %pre_load = load i32, ptr %a
  br label %scalar.body

scalar.body:
  %0 = phi i32 [ %pre_load, %for.preheader ], [ %1, %scalar.body ]
  %indvars.iv = phi i64 [ 0, %for.preheader ], [ %indvars.iv.next, %scalar.body ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx32 = getelementptr inbounds i32, ptr %a, i64 %indvars.iv.next
  %1 = load i32, ptr %arrayidx32
  %arrayidx34 = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %add35 = add i32 %1, %0
  store i32 %add35, ptr %arrayidx34
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.exit, label %scalar.body, !llvm.loop !0

for.exit:
  ret i32 %0
}

; int recurrence_2(int *a, int n) {
;   int minmax;
;   for (int i = 0; i < n; ++i)
;     minmax = min(minmax, max(a[i] - a[i-1], 0));
;   return minmax;
; }
;
define i32 @recurrence_2(ptr nocapture readonly %a, i32 %n) {
; CHECK-VF4UF1-LABEL: @recurrence_2
; CHECK-VF4UF1: vector.ph:
; CHECK-VF4UF1: %[[VSCALE1:.*]] = call i32 @llvm.vscale.i32()
; CHECK-VF4UF1: %[[MUL1:.*]] = mul i32 %[[VSCALE1]], 4
; CHECK-VF4UF1: %[[SUB1:.*]] = sub i32 %[[MUL1]], 1
; CHECK-VF4UF1: %[[VEC_RECUR_INIT:.*]] = insertelement <vscale x 4 x i32> poison, i32 %.pre, i32 %[[SUB1]]
; CHECK-VF4UF1: vector.body:
; CHECK-VF4UF1: %[[VEC_RECUR:.*]] = phi <vscale x 4 x i32> [ %[[VEC_RECUR_INIT]], %vector.ph ], [ %[[LOAD:.*]], %vector.body ]
; CHECK-VF4UF1: %[[LOAD]] = load <vscale x 4 x i32>, ptr
; CHECK-VF4UF1: %[[REVERSE:.*]] = call <vscale x 4 x i32> @llvm.experimental.vector.splice.nxv4i32(<vscale x 4 x i32> %[[VEC_RECUR]], <vscale x 4 x i32> %[[LOAD]], i32 -1)
; CHECK-VF4UF1: middle.block:
; CHECK-VF4UF1: %[[VSCALE2:.*]] = call i32 @llvm.vscale.i32()
; CHECK-VF4UF1: %[[MUL2:.*]] = mul i32 %[[VSCALE2]], 4
; CHECK-VF4UF1: %[[SUB2:.*]] = sub i32 %[[MUL2]], 1
; CHECK-VF4UF1: %[[VEC_RECUR_EXT:.*]] = extractelement <vscale x 4 x i32> %[[LOAD]], i32 %[[SUB2]]
entry:
  %cmp27 = icmp sgt i32 %n, 0
  br i1 %cmp27, label %for.preheader, label %for.cond.cleanup

for.preheader:
  %arrayidx2.phi.trans.insert = getelementptr inbounds i32, ptr %a, i64 -1
  %.pre = load i32, ptr %arrayidx2.phi.trans.insert, align 4
  br label %scalar.body

for.cond.cleanup.loopexit:
  %minmax.0.cond.lcssa = phi i32 [ %minmax.0.cond, %scalar.body ]
  br label %for.cond.cleanup

for.cond.cleanup:
  %minmax.0.lcssa = phi i32 [ undef, %entry ], [ %minmax.0.cond.lcssa, %for.cond.cleanup.loopexit ]
  ret i32 %minmax.0.lcssa

scalar.body:
  %0 = phi i32 [ %.pre, %for.preheader ], [ %1, %scalar.body ]
  %indvars.iv = phi i64 [ 0, %for.preheader ], [ %indvars.iv.next, %scalar.body ]
  %minmax.028 = phi i32 [ undef, %for.preheader ], [ %minmax.0.cond, %scalar.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx, align 4
  %sub3 = sub nsw i32 %1, %0
  %cmp4 = icmp sgt i32 %sub3, 0
  %cond = select i1 %cmp4, i32 %sub3, i32 0
  %cmp5 = icmp slt i32 %minmax.028, %cond
  %minmax.0.cond = select i1 %cmp5, i32 %minmax.028, i32 %cond
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %scalar.body, !llvm.loop !0
}

define void @recurrence_3(ptr nocapture readonly %a, ptr nocapture %b, i32 %n, float %f, i16 %p) {
; CHECK-VF4UF1: vector.ph:
; CHECK-VF4UF1: %[[VSCALE1:.*]] = call i32 @llvm.vscale.i32()
; CHECK-VF4UF1: %[[MUL1:.*]] = mul i32 %[[VSCALE1]], 4
; CHECK-VF4UF1: %[[SUB1:.*]] = sub i32 %[[MUL1]], 1
; CHECK-VF4UF1: %vector.recur.init = insertelement <vscale x 4 x i16> poison, i16 %0, i32 %[[SUB1]]
; CHECK-VF4UF1: vector.body:
; CHECK-VF4UF1: %vector.recur = phi <vscale x 4 x i16> [ %vector.recur.init, %vector.ph ], [ %[[L1:.*]], %vector.body ]
; CHECK-VF4UF1: %[[L1]] = load <vscale x 4 x i16>, ptr
; CHECK-VF4UF1: %[[SPLICE:.*]] = call <vscale x 4 x i16> @llvm.experimental.vector.splice.nxv4i16(<vscale x 4 x i16> %vector.recur, <vscale x 4 x i16> %[[L1]], i32 -1)
; Check also that the casts were not moved needlessly.
; CHECK-VF4UF1: sitofp <vscale x 4 x i16> %[[L1]] to <vscale x 4 x double>
; CHECK-VF4UF1: sitofp <vscale x 4 x i16> %[[SPLICE]] to <vscale x 4 x double>
; CHECK-VF4UF1: middle.block:
; CHECK-VF4UF1: %[[VSCALE2:.*]] = call i32 @llvm.vscale.i32()
; CHECK-VF4UF1: %[[MUL2:.*]] = mul i32 %[[VSCALE2]], 4
; CHECK-VF4UF1: %[[SUB2:.*]] = sub i32 %[[MUL2]], 1
; CHECK-VF4UF1: %vector.recur.extract = extractelement <vscale x 4 x i16> %[[L1]], i32 %[[SUB2]]
entry:
  %0 = load i16, ptr %a, align 2
  %conv = sitofp i16 %0 to double
  %conv1 = fpext float %f to double
  %conv2 = sitofp i16 %p to double
  %mul = fmul fast double %conv2, %conv1
  %sub = fsub fast double %conv, %mul
  store double %sub, ptr %b, align 8
  %cmp25 = icmp sgt i32 %n, 1
  br i1 %cmp25, label %for.preheader, label %for.end

for.preheader:
  br label %scalar.body

scalar.body:
  %1 = phi i16 [ %0, %for.preheader ], [ %2, %scalar.body ]
  %iv = phi i64 [ %iv.next, %scalar.body ], [ 1, %for.preheader ]
  %arrayidx5 = getelementptr inbounds i16, ptr %a, i64 %iv
  %2 = load i16, ptr %arrayidx5, align 2
  %conv6 = sitofp i16 %2 to double
  %conv11 = sitofp i16 %1 to double
  %mul12 = fmul fast double %conv11, %conv1
  %sub13 = fsub fast double %conv6, %mul12
  %arrayidx15 = getelementptr inbounds double, ptr %b, i64 %iv
  store double %sub13, ptr %arrayidx15, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %lftr.wideiv = trunc i64 %iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end.loopexit, label %scalar.body, !llvm.loop !0

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

define i64 @constant_folded_previous_value() {
; CHECK-VF4UF2-LABEL: @constant_folded_previous_value
; CHECK-VF4UF2: vector.body
; CHECK-VF4UF2: %[[VECTOR_RECUR:.*]] = phi <vscale x 4 x i64> [ %vector.recur.init, %vector.ph ], [ shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 1, i64 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer), %vector.body ]
; CHECK-VF4UF2: %[[SPLICE1:.*]] = call <vscale x 4 x i64> @llvm.experimental.vector.splice.nxv4i64(<vscale x 4 x i64> %vector.recur, <vscale x 4 x i64> shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 1, i64 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer), i32 -1)
; CHECK-VF4UF2: %[[SPLICE2:.*]] = call <vscale x 4 x i64> @llvm.experimental.vector.splice.nxv4i64(<vscale x 4 x i64> shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 1, i64 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer), <vscale x 4 x i64> shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 1, i64 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer), i32 -1)
; CHECK-VF4UF2: br i1 {{.*}}, label %middle.block, label %vector.body
entry:
  br label %scalar.body

scalar.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %scalar.body ]
  %tmp2 = phi i64 [ 0, %entry ], [ %tmp3, %scalar.body ]
  %tmp3 = add i64 0, 1
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp eq i64 %i.next, undef
  br i1 %cond, label %for.end, label %scalar.body, !llvm.loop !0

for.end:
  ret i64 %tmp2
}

; We vectorize this first order recurrence, by generating two
; extracts for the phi `val.phi` - one at the last index and
; another at the second last index. We need these 2 extracts because
; the first order recurrence phi is used outside the loop, so we require the phi
; itself and not its update (addx).
define i32 @extract_second_last_iteration(ptr %cval, i32 %x)  {
; CHECK-VF4UF2-LABEL: @extract_second_last_iteration
; CHECK-VF4UF2: vector.ph
; CHECK-VF4UF2: call i32 @llvm.vscale.i32()
; CHECK-VF4UF2: call i32 @llvm.vscale.i32()
; CHECK-VF4UF2: %[[VSCALE1:.*]] = call i32 @llvm.vscale.i32()
; CHECK-VF4UF2: %[[MUL1:.*]] = mul i32 %[[VSCALE1]], 4
; CHECK-VF4UF2: %[[SUB1:.*]] = sub i32 %[[MUL1]], 1
; CHECK-VF4UF2: %[[VEC_RECUR_INIT:.*]] = insertelement <vscale x 4 x i32> poison, i32 0, i32 %[[SUB1]]
; CHECK-VF4UF2: %[[SPLAT_INS1:.*]] = insertelement <vscale x 4 x i32> poison, i32 %x, i64 0
; CHECK-VF4UF2: %[[SPLAT1:.*]] = shufflevector <vscale x 4 x i32> %[[SPLAT_INS1]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-VF4UF2: %[[SPLAT_INS2:.*]] = insertelement <vscale x 4 x i32> poison, i32 %x, i64 0
; CHECK-VF4UF2: %[[SPLAT2:.*]] = shufflevector <vscale x 4 x i32> %[[SPLAT_INS2]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; ; CHECK-VF4UF2: vector.body
; CHECK-VF4UF2: %[[VEC_RECUR:.*]] = phi <vscale x 4 x i32> [ %[[VEC_RECUR_INIT]], %vector.ph ], [ %[[ADD2:.*]], %vector.body ]
; CHECK-VF4UF2: %[[ADD1:.*]] = add <vscale x 4 x i32> %{{.*}}, %[[SPLAT1]]
; CHECK-VF4UF2: %[[ADD2]] = add <vscale x 4 x i32> %{{.*}}, %[[SPLAT2]]
; CHECK-VF4UF2: middle.block
; CHECK-VF4UF2: %[[VSCALE2:.*]] = call i32 @llvm.vscale.i32()
; CHECK-VF4UF2: %[[MUL2:.*]] = mul i32 %[[VSCALE2]], 4
; CHECK-VF4UF2: %[[SUB2:.*]] = sub i32 %[[MUL2]], 1
; CHECK-VF4UF2: %vector.recur.extract = extractelement <vscale x 4 x i32> %[[ADD2]], i32 %[[SUB2]]
; CHECK-VF4UF2: %[[SUB3:.*]] = sub i32 %[[MUL2]], 2
; CHECK-VF4UF2: %vector.recur.extract.for.phi = extractelement <vscale x 4 x i32> %[[ADD2]], i32 %[[SUB3]]
entry:
  br label %for.body

for.body:
  %inc.phi = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %val.phi = phi i32 [ 0, %entry ], [ %addx, %for.body ]
  %inc = add i32 %inc.phi, 1
  %bc = zext i32 %inc.phi to i64
  %addx = add i32 %inc.phi, %x
  %cmp = icmp eq i32 %inc.phi, 95
  br i1 %cmp, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret i32 %val.phi
}

; void sink_after(short *a, int n, int *b) {
;   for(int i = 0; i < n; i++)
;     b[i] = (aptr a[i + 1]);
; }

; Check that the sext sank after the load in the vector loop.
define void @sink_after(ptr %a, ptr %b, i64 %n) {
; CHECK-VF4UF1-LABEL: @sink_after
; CHECK-VF4UF1: vector.body
; CHECK-VF4UF1: %[[VEC_RECUR:.*]] = phi <vscale x 4 x i16> [ %vector.recur.init, %vector.ph ], [ %[[LOAD:.*]], %vector.body ]
; CHECK-VF4UF1: %[[LOAD]] = load <vscale x 4 x i16>, ptr
; CHECK-VF4UF1-NEXT: %[[SPLICE:.*]] = call <vscale x 4 x i16> @llvm.experimental.vector.splice.nxv4i16(<vscale x 4 x i16> %[[VEC_RECUR]], <vscale x 4 x i16> %[[LOAD]], i32 -1)
; CHECK-VF4UF1-NEXT: sext <vscale x 4 x i16> %[[SPLICE]] to <vscale x 4 x i32>
; CHECK-VF4UF1-NEXT: sext <vscale x 4 x i16> %[[LOAD]] to <vscale x 4 x i32>
entry:
  %.pre = load i16, ptr %a
  br label %for.body

for.body:
  %0 = phi i16 [ %.pre, %entry ], [ %1, %for.body ]
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %conv = sext i16 %0 to i32
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx2 = getelementptr inbounds i16, ptr %a, i64 %indvars.iv.next
  %1 = load i16, ptr %arrayidx2
  %conv3 = sext i16 %1 to i32
  %mul = mul nsw i32 %conv3, %conv
  %arrayidx5 = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  store i32 %mul, ptr %arrayidx5
  %exitcond = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
