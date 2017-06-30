; RUN: opt < %s -loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -dce -instcombine -S | FileCheck %s
; RUN: opt < %s -loop-vectorize -force-vector-width=4 -force-vector-interleave=2 -dce -instcombine -S | FileCheck %s --check-prefix=UNROLL
; RUN: opt < %s -loop-vectorize -force-vector-width=4 -force-vector-interleave=2 -S | FileCheck %s --check-prefix=UNROLL-NO-IC
; RUN: opt < %s -loop-vectorize -force-vector-width=1 -force-vector-interleave=2 -S | FileCheck %s --check-prefix=UNROLL-NO-VF
; RUN: opt < %s -loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -S | FileCheck %s --check-prefix=SINK-AFTER
; RUN: opt < %s -loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -S | FileCheck %s --check-prefix=NO-SINK-AFTER

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

; void recurrence_1(int *a, int *b, int n) {
;   for(int i = 0; i < n; i++)
;     b[i] =  a[i] + a[i - 1]
; }
;
; CHECK-LABEL: @recurrence_1(
; CHECK:       vector.ph:
; CHECK:         %vector.recur.init = insertelement <4 x i32> undef, i32 %pre_load, i32 3
; CHECK:       vector.body:
; CHECK:         %vector.recur = phi <4 x i32> [ %vector.recur.init, %vector.ph ], [ [[L1:%[a-zA-Z0-9.]+]], %vector.body ]
; CHECK:         [[L1]] = load <4 x i32>
; CHECK:         {{.*}} = shufflevector <4 x i32> %vector.recur, <4 x i32> [[L1]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK:       middle.block:
; CHECK:         %vector.recur.extract = extractelement <4 x i32> [[L1]], i32 3
; CHECK:       scalar.ph:
; CHECK:         %scalar.recur.init = phi i32 [ %vector.recur.extract, %middle.block ], [ %pre_load, %vector.memcheck ], [ %pre_load, %min.iters.checked ], [ %pre_load, %for.preheader ]
; CHECK:       scalar.body:
; CHECK:         %scalar.recur = phi i32 [ %scalar.recur.init, %scalar.ph ], [ {{.*}}, %scalar.body ]
;
; UNROLL-LABEL: @recurrence_1(
; UNROLL:       vector.body:
; UNROLL:         %vector.recur = phi <4 x i32> [ %vector.recur.init, %vector.ph ], [ [[L2:%[a-zA-Z0-9.]+]], %vector.body ]
; UNROLL:         [[L1:%[a-zA-Z0-9.]+]] = load <4 x i32>
; UNROLL:         [[L2]] = load <4 x i32>
; UNROLL:         {{.*}} = shufflevector <4 x i32> %vector.recur, <4 x i32> [[L1]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; UNROLL:         {{.*}} = shufflevector <4 x i32> [[L1]], <4 x i32> [[L2]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; UNROLL:       middle.block:
; UNROLL:         %vector.recur.extract = extractelement <4 x i32> [[L2]], i32 3
;
define void @recurrence_1(i32* nocapture readonly %a, i32* nocapture %b, i32 %n) {
entry:
  br label %for.preheader

for.preheader:
  %arrayidx.phi.trans.insert = getelementptr inbounds i32, i32* %a, i64 0
  %pre_load = load i32, i32* %arrayidx.phi.trans.insert
  br label %scalar.body

scalar.body:
  %0 = phi i32 [ %pre_load, %for.preheader ], [ %1, %scalar.body ]
  %indvars.iv = phi i64 [ 0, %for.preheader ], [ %indvars.iv.next, %scalar.body ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx32 = getelementptr inbounds i32, i32* %a, i64 %indvars.iv.next
  %1 = load i32, i32* %arrayidx32
  %arrayidx34 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %add35 = add i32 %1, %0
  store i32 %add35, i32* %arrayidx34
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.exit, label %scalar.body

for.exit:
  ret void
}

; int recurrence_2(int *a, int n) {
;   int minmax;
;   for (int i = 0; i < n; ++i)
;     minmax = min(minmax, max(a[i] - a[i-1], 0));
;   return minmax;
; }
;
; CHECK-LABEL: @recurrence_2(
; CHECK:       vector.ph:
; CHECK:         %vector.recur.init = insertelement <4 x i32> undef, i32 %.pre, i32 3
; CHECK:       vector.body:
; CHECK:         %vector.recur = phi <4 x i32> [ %vector.recur.init, %vector.ph ], [ [[L1:%[a-zA-Z0-9.]+]], %vector.body ]
; CHECK:         [[L1]] = load <4 x i32>
; CHECK:         {{.*}} = shufflevector <4 x i32> %vector.recur, <4 x i32> [[L1]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK:       middle.block:
; CHECK:         %vector.recur.extract = extractelement <4 x i32> [[L1]], i32 3
; CHECK:       scalar.ph:
; CHECK:         %scalar.recur.init = phi i32 [ %vector.recur.extract, %middle.block ], [ %.pre, %min.iters.checked ], [ %.pre, %for.preheader ]
; CHECK:       scalar.body:
; CHECK:         %scalar.recur = phi i32 [ %scalar.recur.init, %scalar.ph ], [ {{.*}}, %scalar.body ]
;
; UNROLL-LABEL: @recurrence_2(
; UNROLL:       vector.body:
; UNROLL:         %vector.recur = phi <4 x i32> [ %vector.recur.init, %vector.ph ], [ [[L2:%[a-zA-Z0-9.]+]], %vector.body ]
; UNROLL:         [[L1:%[a-zA-Z0-9.]+]] = load <4 x i32>
; UNROLL:         [[L2]] = load <4 x i32>
; UNROLL:         {{.*}} = shufflevector <4 x i32> %vector.recur, <4 x i32> [[L1]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; UNROLL:         {{.*}} = shufflevector <4 x i32> [[L1]], <4 x i32> [[L2]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; UNROLL:       middle.block:
; UNROLL:         %vector.recur.extract = extractelement <4 x i32> [[L2]], i32 3
;
define i32 @recurrence_2(i32* nocapture readonly %a, i32 %n) {
entry:
  %cmp27 = icmp sgt i32 %n, 0
  br i1 %cmp27, label %for.preheader, label %for.cond.cleanup

for.preheader:
  %arrayidx2.phi.trans.insert = getelementptr inbounds i32, i32* %a, i64 -1
  %.pre = load i32, i32* %arrayidx2.phi.trans.insert, align 4
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
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %1 = load i32, i32* %arrayidx, align 4
  %sub3 = sub nsw i32 %1, %0
  %cmp4 = icmp sgt i32 %sub3, 0
  %cond = select i1 %cmp4, i32 %sub3, i32 0
  %cmp5 = icmp slt i32 %minmax.028, %cond
  %minmax.0.cond = select i1 %cmp5, i32 %minmax.028, i32 %cond
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %scalar.body
}

; void recurrence_3(short *a, double *b, int n, float f, short p) {
;   b[0] = (double)a[0] - f * (double)p;
;   for (int i = 1; i < n; i++)
;     b[i] = (double)a[i] - f * (double)a[i - 1];
; }
;
; CHECK-LABEL: @recurrence_3(
; CHECK:       vector.ph:
; CHECK:         %vector.recur.init = insertelement <4 x i16> undef, i16 %0, i32 3
; CHECK:       vector.body:
; CHECK:         %vector.recur = phi <4 x i16> [ %vector.recur.init, %vector.ph ], [ [[L1:%[a-zA-Z0-9.]+]], %vector.body ]
; CHECK:         [[L1]] = load <4 x i16>
; CHECK:         {{.*}} = shufflevector <4 x i16> %vector.recur, <4 x i16> [[L1]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK:       middle.block:
; CHECK:         %vector.recur.extract = extractelement <4 x i16> [[L1]], i32 3
; CHECK:       scalar.ph:
; CHECK:         %scalar.recur.init = phi i16 [ %vector.recur.extract, %middle.block ], [ %0, %vector.memcheck ], [ %0, %min.iters.checked ], [ %0, %for.preheader ]
; CHECK:       scalar.body:
; CHECK:         %scalar.recur = phi i16 [ %scalar.recur.init, %scalar.ph ], [ {{.*}}, %scalar.body ]
;
; UNROLL-LABEL: @recurrence_3(
; UNROLL:       vector.body:
; UNROLL:         %vector.recur = phi <4 x i16> [ %vector.recur.init, %vector.ph ], [ [[L2:%[a-zA-Z0-9.]+]], %vector.body ]
; UNROLL:         [[L1:%[a-zA-Z0-9.]+]] = load <4 x i16>
; UNROLL:         [[L2]] = load <4 x i16>
; UNROLL:         {{.*}} = shufflevector <4 x i16> %vector.recur, <4 x i16> [[L1]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; UNROLL:         {{.*}} = shufflevector <4 x i16> [[L1]], <4 x i16> [[L2]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; UNROLL:       middle.block:
; UNROLL:         %vector.recur.extract = extractelement <4 x i16> [[L2]], i32 3
;
define void @recurrence_3(i16* nocapture readonly %a, double* nocapture %b, i32 %n, float %f, i16 %p) {
entry:
  %0 = load i16, i16* %a, align 2
  %conv = sitofp i16 %0 to double
  %conv1 = fpext float %f to double
  %conv2 = sitofp i16 %p to double
  %mul = fmul fast double %conv2, %conv1
  %sub = fsub fast double %conv, %mul
  store double %sub, double* %b, align 8
  %cmp25 = icmp sgt i32 %n, 1
  br i1 %cmp25, label %for.preheader, label %for.end

for.preheader:
  br label %scalar.body

scalar.body:
  %1 = phi i16 [ %0, %for.preheader ], [ %2, %scalar.body ]
  %advars.iv = phi i64 [ %advars.iv.next, %scalar.body ], [ 1, %for.preheader ]
  %arrayidx5 = getelementptr inbounds i16, i16* %a, i64 %advars.iv
  %2 = load i16, i16* %arrayidx5, align 2
  %conv6 = sitofp i16 %2 to double
  %conv11 = sitofp i16 %1 to double
  %mul12 = fmul fast double %conv11, %conv1
  %sub13 = fsub fast double %conv6, %mul12
  %arrayidx15 = getelementptr inbounds double, double* %b, i64 %advars.iv
  store double %sub13, double* %arrayidx15, align 8
  %advars.iv.next = add nuw nsw i64 %advars.iv, 1
  %lftr.wideiv = trunc i64 %advars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end.loopexit, label %scalar.body

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

; void PR26734(short *a, int *b, int *c, int d, short *e) {
;   for (; d != 21; d++) {
;     *b &= *c;
;     *e = *a - 6;
;     *c = *e;
;   }
; }
;
; CHECK-LABEL: @PR26734(
; CHECK-NOT:   vector.ph:
; CHECK:       }
;
define void @PR26734(i16* %a, i32* %b, i32* %c, i32 %d, i16* %e) {
entry:
  %cmp4 = icmp eq i32 %d, 21
  br i1 %cmp4, label %entry.for.end_crit_edge, label %for.body.lr.ph

entry.for.end_crit_edge:
  %.pre = load i32, i32* %b, align 4
  br label %for.end

for.body.lr.ph:
  %0 = load i16, i16* %a, align 2
  %sub = add i16 %0, -6
  %conv2 = sext i16 %sub to i32
  %c.promoted = load i32, i32* %c, align 4
  %b.promoted = load i32, i32* %b, align 4
  br label %for.body

for.body:
  %inc7 = phi i32 [ %d, %for.body.lr.ph ], [ %inc, %for.body ]
  %and6 = phi i32 [ %b.promoted, %for.body.lr.ph ], [ %and, %for.body ]
  %conv25 = phi i32 [ %c.promoted, %for.body.lr.ph ], [ %conv2, %for.body ]
  %and = and i32 %and6, %conv25
  %inc = add nsw i32 %inc7, 1
  %cmp = icmp eq i32 %inc, 21
  br i1 %cmp, label %for.cond.for.end_crit_edge, label %for.body

for.cond.for.end_crit_edge:
  %and.lcssa = phi i32 [ %and, %for.body ]
  store i32 %conv2, i32* %c, align 4
  store i32 %and.lcssa, i32* %b, align 4
  store i16 %sub, i16* %e, align 2
  br label %for.end

for.end:
  ret void
}

; int PR27246() {
;   unsigned int e, n;
;   for (int i = 1; i < 49; ++i) {
;     for (int k = i; k > 1; --k)
;       e = k;
;     n = e;
;   }
;   return n;
; }
;
; CHECK-LABEL: @PR27246(
; CHECK-NOT:   vector.ph:
; CHECK:       }
;
define i32 @PR27246() {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:
  %i.016 = phi i32 [ 1, %entry ], [ %inc, %for.cond.cleanup3 ]
  %e.015 = phi i32 [ undef, %entry ], [ %e.1.lcssa, %for.cond.cleanup3 ]
  br label %for.cond1

for.cond.cleanup:
  %e.1.lcssa.lcssa = phi i32 [ %e.1.lcssa, %for.cond.cleanup3 ]
  ret i32 %e.1.lcssa.lcssa

for.cond1:
  %e.1 = phi i32 [ %k.0, %for.cond1 ], [ %e.015, %for.cond1.preheader ]
  %k.0 = phi i32 [ %dec, %for.cond1 ], [ %i.016, %for.cond1.preheader ]
  %cmp2 = icmp sgt i32 %k.0, 1
  %dec = add nsw i32 %k.0, -1
  br i1 %cmp2, label %for.cond1, label %for.cond.cleanup3

for.cond.cleanup3:
  %e.1.lcssa = phi i32 [ %e.1, %for.cond1 ]
  %inc = add nuw nsw i32 %i.016, 1
  %exitcond = icmp eq i32 %inc, 49
  br i1 %exitcond, label %for.cond.cleanup, label %for.cond1.preheader
}

; UNROLL-NO-IC-LABEL: @PR30183(
; UNROLL-NO-IC:       vector.ph:
; UNROLL-NO-IC-NEXT:    [[VECTOR_RECUR_INIT:%.*]] = insertelement <4 x i32> undef, i32 [[PRE_LOAD:%.*]], i32 3
; UNROLL-NO-IC-NEXT:    br label %vector.body
; UNROLL-NO-IC:       vector.body:
; UNROLL-NO-IC-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; UNROLL-NO-IC-NEXT:    [[VECTOR_RECUR:%.*]] = phi <4 x i32> [ [[VECTOR_RECUR_INIT]], %vector.ph ], [ [[TMP42:%.*]], %vector.body ]
; UNROLL-NO-IC:         [[TMP27:%.*]] = load i32, i32* {{.*}}
; UNROLL-NO-IC-NEXT:    [[TMP28:%.*]] = load i32, i32* {{.*}}
; UNROLL-NO-IC-NEXT:    [[TMP29:%.*]] = load i32, i32* {{.*}}
; UNROLL-NO-IC-NEXT:    [[TMP30:%.*]] = load i32, i32* {{.*}}
; UNROLL-NO-IC-NEXT:    [[TMP35:%.*]] = insertelement <4 x i32> undef, i32 [[TMP27]], i32 0
; UNROLL-NO-IC-NEXT:    [[TMP36:%.*]] = insertelement <4 x i32> [[TMP35]], i32 [[TMP28]], i32 1
; UNROLL-NO-IC-NEXT:    [[TMP37:%.*]] = insertelement <4 x i32> [[TMP36]], i32 [[TMP29]], i32 2
; UNROLL-NO-IC-NEXT:    [[TMP38:%.*]] = insertelement <4 x i32> [[TMP37]], i32 [[TMP30]], i32 3
; UNROLL-NO-IC-NEXT:    [[TMP31:%.*]] = load i32, i32* {{.*}}
; UNROLL-NO-IC-NEXT:    [[TMP32:%.*]] = load i32, i32* {{.*}}
; UNROLL-NO-IC-NEXT:    [[TMP33:%.*]] = load i32, i32* {{.*}}
; UNROLL-NO-IC-NEXT:    [[TMP34:%.*]] = load i32, i32* {{.*}}
; UNROLL-NO-IC-NEXT:    [[TMP39:%.*]] = insertelement <4 x i32> undef, i32 [[TMP31]], i32 0
; UNROLL-NO-IC-NEXT:    [[TMP40:%.*]] = insertelement <4 x i32> [[TMP39]], i32 [[TMP32]], i32 1
; UNROLL-NO-IC-NEXT:    [[TMP41:%.*]] = insertelement <4 x i32> [[TMP40]], i32 [[TMP33]], i32 2
; UNROLL-NO-IC-NEXT:    [[TMP42]] = insertelement <4 x i32> [[TMP41]], i32 [[TMP34]], i32 3
; UNROLL-NO-IC-NEXT:    [[TMP43:%.*]] = shufflevector <4 x i32> [[VECTOR_RECUR]], <4 x i32> [[TMP38]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; UNROLL-NO-IC-NEXT:    [[TMP44:%.*]] = shufflevector <4 x i32> [[TMP38]], <4 x i32> [[TMP42]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; UNROLL-NO-IC-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], 8
; UNROLL-NO-IC:         br i1 {{.*}}, label %middle.block, label %vector.body
;
define void @PR30183(i32 %pre_load, i32* %a, i32* %b, i64 %n) {
entry:
  br label %scalar.body

scalar.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %scalar.body ]
  %tmp0 = phi i32 [ %pre_load, %entry ], [ %tmp2, %scalar.body ]
  %i.next = add nuw nsw i64 %i, 2
  %tmp1 = getelementptr inbounds i32, i32* %a, i64 %i.next
  %tmp2 = load i32, i32* %tmp1
  %cond = icmp eq i64 %i.next,%n
  br i1 %cond, label %for.end, label %scalar.body

for.end:
  ret void
}

; UNROLL-NO-IC-LABEL: @constant_folded_previous_value(
; UNROLL-NO-IC:       vector.body:
; UNROLL-NO-IC:         [[VECTOR_RECUR:%.*]] = phi <4 x i64> [ <i64 undef, i64 undef, i64 undef, i64 0>, %vector.ph ], [ <i64 1, i64 1, i64 1, i64 1>, %vector.body ]
; UNROLL-NO-IC-NEXT:    [[TMP0:%.*]] = shufflevector <4 x i64> [[VECTOR_RECUR]], <4 x i64> <i64 1, i64 1, i64 1, i64 1>, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; UNROLL-NO-IC:         br i1 {{.*}}, label %middle.block, label %vector.body
;
define void @constant_folded_previous_value() {
entry:
  br label %scalar.body

scalar.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %scalar.body ]
  %tmp2 = phi i64 [ 0, %entry ], [ %tmp3, %scalar.body ]
  %tmp3 = add i64 0, 1
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp eq i64 %i.next, undef
  br i1 %cond, label %for.end, label %scalar.body

for.end:
  ret void
}

; We vectorize this first order recurrence, by generating two
; extracts for the phi `val.phi` - one at the last index and 
; another at the second last index. We need these 2 extracts because 
; the first order recurrence phi is used outside the loop, so we require the phi
; itself and not its update (addx).
; UNROLL-NO-IC-LABEL: extract_second_last_iteration
; UNROLL-NO-IC: vector.body
; UNROLL-NO-IC:   %step.add = add <4 x i32> %vec.ind, <i32 4, i32 4, i32 4, i32 4>
; UNROLL-NO-IC:   %[[L1:.+]] = add <4 x i32> %vec.ind, %broadcast.splat
; UNROLL-NO-IC:   %[[L2:.+]] = add <4 x i32> %step.add, %broadcast.splat
; UNROLL-NO-IC:   %index.next = add i32 %index, 8
; UNROLL-NO-IC:   icmp eq i32 %index.next, 96
; UNROLL-NO-IC: middle.block
; UNROLL-NO-IC:   icmp eq i32 96, 96
; UNROLL-NO-IC:   %vector.recur.extract = extractelement <4 x i32> %[[L2]], i32 3
; UNROLL-NO-IC:   %vector.recur.extract.for.phi = extractelement <4 x i32> %[[L2]], i32 2
; UNROLL-NO-IC: for.end
; UNROLL-NO-IC:   %val.phi.lcssa = phi i32 [ %scalar.recur, %for.body ], [ %vector.recur.extract.for.phi, %middle.block ]
; Check the case when unrolled but not vectorized.
; UNROLL-NO-VF-LABEL: extract_second_last_iteration
; UNROLL-NO-VF: vector.body:
; UNROLL-NO-VF:   %induction = add i32 %index, 0
; UNROLL-NO-VF:   %induction1 = add i32 %index, 1
; UNROLL-NO-VF:   %[[L1:.+]] = add i32 %induction, %x
; UNROLL-NO-VF:   %[[L2:.+]] = add i32 %induction1, %x
; UNROLL-NO-VF:   %index.next = add i32 %index, 2
; UNROLL-NO-VF:   icmp eq i32 %index.next, 96
; UNROLL-NO-VF: for.end:
; UNROLL-NO-VF:   %val.phi.lcssa = phi i32 [ %scalar.recur, %for.body ], [ %[[L1]], %middle.block ]
define i32 @extract_second_last_iteration(i32* %cval, i32 %x)  {
entry:
  br label %for.body

for.body:
  %inc.phi = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %val.phi = phi i32 [ 0, %entry ], [ %addx, %for.body ]
  %inc = add i32 %inc.phi, 1
  %bc = zext i32 %inc.phi to i64
  %addx = add i32 %inc.phi, %x
  %cmp = icmp eq i32 %inc.phi, 95
  br i1 %cmp, label %for.end, label %for.body

for.end:
  ret i32 %val.phi
}

; We vectorize this first order recurrence, with a set of insertelements for
; each unrolled part. Make sure these insertelements are generated in-order,
; because the shuffle of the first order recurrence will be added after the
; insertelement of the last part UF - 1, assuming the latter appears after the
; insertelements of all other parts.
;
; int PR33613(double *b, double j, int d) {
;   int a = 0;
;   for(int i = 0; i < 10240; i++, b+=25) {
;     double f = b[d]; // Scalarize to form insertelements
;     if (j * f)
;       a++;
;     j = f;
;   }
;   return a;
; }
;
; UNROLL-NO-IC-LABEL: @PR33613(
; UNROLL-NO-IC:     vector.body:
; UNROLL-NO-IC:       [[VECTOR_RECUR:%.*]] = phi <4 x double>
; UNROLL-NO-IC:       shufflevector <4 x double> [[VECTOR_RECUR]], <4 x double> {{.*}}, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; UNROLL-NO-IC-NEXT:  shufflevector <4 x double> {{.*}}, <4 x double> {{.*}}, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; UNROLL-NO-IC-NOT:   insertelement <4 x double>
; UNROLL-NO-IC:     middle.block:
;
define i32 @PR33613(double* %b, double %j, i32 %d) {
entry:
  %idxprom = sext i32 %d to i64
  br label %for.body

for.cond.cleanup:
  %a.1.lcssa = phi i32 [ %a.1, %for.body ]
  ret i32 %a.1.lcssa

for.body:
  %b.addr.012 = phi double* [ %b, %entry ], [ %add.ptr, %for.body ]
  %i.011 = phi i32 [ 0, %entry ], [ %inc1, %for.body ]
  %a.010 = phi i32 [ 0, %entry ], [ %a.1, %for.body ]
  %j.addr.09 = phi double [ %j, %entry ], [ %0, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %b.addr.012, i64 %idxprom
  %0 = load double, double* %arrayidx, align 8
  %mul = fmul double %j.addr.09, %0
  %tobool = fcmp une double %mul, 0.000000e+00
  %inc = zext i1 %tobool to i32
  %a.1 = add nsw i32 %a.010, %inc
  %inc1 = add nuw nsw i32 %i.011, 1
  %add.ptr = getelementptr inbounds double, double* %b.addr.012, i64 25
  %exitcond = icmp eq i32 %inc1, 10240
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; void sink_after(short *a, int n, int *b) {
;   for(int i = 0; i < n; i++)
;     b[i] = (a[i] * a[i + 1]);
; }
;
; SINK-AFTER-LABEL: sink_after
; Check that the sext sank after the load in the vector loop.
; SINK-AFTER: vector.body
; SINK-AFTER:   %vector.recur = phi <4 x i16> [ %vector.recur.init, %vector.ph ], [ %wide.load, %vector.body ]
; SINK-AFTER:   %wide.load = load <4 x i16>
; SINK-AFTER:   %[[VSHUF:.+]] = shufflevector <4 x i16> %vector.recur, <4 x i16> %wide.load, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; SINK-AFTER:   %[[VCONV:.+]] = sext <4 x i16> %[[VSHUF]] to <4 x i32>
; SINK-AFTER:   %[[VCONV3:.+]] = sext <4 x i16> %wide.load to <4 x i32>
; SINK-AFTER:   mul nsw <4 x i32> %[[VCONV3]], %[[VCONV]]
; Check also that the sext sank after the load in the scalar loop.
; SINK-AFTER: for.body
; SINK-AFTER:   %scalar.recur = phi i16 [ %scalar.recur.init, %scalar.ph ], [ %[[LOAD:.+]], %for.body ]
; SINK-AFTER:   %[[LOAD]] = load i16, i16* %arrayidx2
; SINK-AFTER:   %[[CONV:.+]] = sext i16 %scalar.recur to i32
; SINK-AFTER:   %[[CONV3:.+]] = sext i16 %[[LOAD]] to i32
; SINK-AFTER:   %mul = mul nsw i32 %[[CONV3]], %[[CONV]]
;
define void @sink_after(i16* %a, i32* %b, i64 %n) {
entry:
  %.pre = load i16, i16* %a
  br label %for.body

for.body:
  %0 = phi i16 [ %.pre, %entry ], [ %1, %for.body ]
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %conv = sext i16 %0 to i32
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx2 = getelementptr inbounds i16, i16* %a, i64 %indvars.iv.next
  %1 = load i16, i16* %arrayidx2
  %conv3 = sext i16 %1 to i32
  %mul = mul nsw i32 %conv3, %conv
  %arrayidx5 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  store i32 %mul, i32* %arrayidx5
  %exitcond = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

; void no_sink_after(short *a, int n, int *b) {
;   for(int i = 0; i < n; i++)
;     b[i] = ((a[i] + 2) * a[i + 1]);
; }
;
; NO-SINK-AFTER-LABEL: no_sink_after
; NO-SINK-AFTER-NOT:   vector.ph:
; NO-SINK-AFTER:       }
;
define void @no_sink_after(i16* %a, i32* %b, i64 %n) {
entry:
  %.pre = load i16, i16* %a
  br label %for.body

for.body:
  %0 = phi i16 [ %.pre, %entry ], [ %1, %for.body ]
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %conv = sext i16 %0 to i32
  %add = add nsw i32 %conv, 2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx2 = getelementptr inbounds i16, i16* %a, i64 %indvars.iv.next
  %1 = load i16, i16* %arrayidx2
  %conv3 = sext i16 %1 to i32
  %mul = mul nsw i32 %add, %conv3
  %arrayidx5 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  store i32 %mul, i32* %arrayidx5
  %exitcond = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
