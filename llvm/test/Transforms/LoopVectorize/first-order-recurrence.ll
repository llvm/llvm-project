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
; CHECK:         %vector.recur.init = insertelement <4 x i32> poison, i32 %pre_load, i32 3
; CHECK:       vector.body:
; CHECK:         %vector.recur = phi <4 x i32> [ %vector.recur.init, %vector.ph ], [ [[L1:%[a-zA-Z0-9.]+]], %vector.body ]
; CHECK:         [[L1]] = load <4 x i32>
; CHECK:         {{.*}} = shufflevector <4 x i32> %vector.recur, <4 x i32> [[L1]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK:       middle.block:
; CHECK:         %vector.recur.extract = extractelement <4 x i32> [[L1]], i32 3
; CHECK:       scalar.ph:
; CHECK:         %scalar.recur.init = phi i32 [ %pre_load, %vector.memcheck ], [ %pre_load, %for.preheader ], [ %vector.recur.extract, %middle.block ]
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
; CHECK:         %vector.recur.init = insertelement <4 x i32> poison, i32 %.pre, i32 3
; CHECK:       vector.body:
; CHECK:         %vector.recur = phi <4 x i32> [ %vector.recur.init, %vector.ph ], [ [[L1:%[a-zA-Z0-9.]+]], %vector.body ]
; CHECK:         [[L1]] = load <4 x i32>
; CHECK:         {{.*}} = shufflevector <4 x i32> %vector.recur, <4 x i32> [[L1]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK:       middle.block:
; CHECK:         %vector.recur.extract = extractelement <4 x i32> [[L1]], i32 3
; CHECK:       scalar.ph:
; CHECK:         %scalar.recur.init = phi i32 [ %.pre, %for.preheader ], [ %vector.recur.extract, %middle.block ]
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
  %minmax.0.lcssa = phi i32 [ poison, %entry ], [ %minmax.0.cond.lcssa, %for.cond.cleanup.loopexit ]
  ret i32 %minmax.0.lcssa

scalar.body:
  %0 = phi i32 [ %.pre, %for.preheader ], [ %1, %scalar.body ]
  %indvars.iv = phi i64 [ 0, %for.preheader ], [ %indvars.iv.next, %scalar.body ]
  %minmax.028 = phi i32 [ poison, %for.preheader ], [ %minmax.0.cond, %scalar.body ]
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
; CHECK:         %vector.recur.init = insertelement <4 x i16> poison, i16 %0, i32 3
; CHECK:       vector.body:
; CHECK:         %vector.recur = phi <4 x i16> [ %vector.recur.init, %vector.ph ], [ [[L1:%[a-zA-Z0-9.]+]], %vector.body ]
; CHECK:         [[L1]] = load <4 x i16>
; CHECK:         [[SHUF:%[a-zA-Z0-9.]+]] = shufflevector <4 x i16> %vector.recur, <4 x i16> [[L1]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; Check also that the casts were not moved needlessly.
; CHECK:         sitofp <4 x i16> [[L1]] to <4 x double>
; CHECK:         sitofp <4 x i16> [[SHUF]] to <4 x double> 
; CHECK:       middle.block:
; CHECK:         %vector.recur.extract = extractelement <4 x i16> [[L1]], i32 3
; CHECK:       scalar.ph:
; CHECK:         %scalar.recur.init = phi i16 [ %0, %vector.memcheck ], [ %0, %for.preheader ], [ %vector.recur.extract, %middle.block ]
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
  %e.015 = phi i32 [ poison, %entry ], [ %e.1.lcssa, %for.cond.cleanup3 ]
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
; UNROLL-NO-IC:         [[VECTOR_RECUR_INIT:%.*]] = insertelement <4 x i32> poison, i32 [[PRE_LOAD:%.*]], i32 3
; UNROLL-NO-IC-NEXT:    br label %vector.body
; UNROLL-NO-IC:       vector.body:
; UNROLL-NO-IC-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; UNROLL-NO-IC-NEXT:    [[VECTOR_RECUR:%.*]] = phi <4 x i32> [ [[VECTOR_RECUR_INIT]], %vector.ph ], [ [[TMP42:%.*]], %vector.body ]
; UNROLL-NO-IC:         [[TMP27:%.*]] = load i32, i32* {{.*}}
; UNROLL-NO-IC-NEXT:    [[TMP28:%.*]] = load i32, i32* {{.*}}
; UNROLL-NO-IC-NEXT:    [[TMP29:%.*]] = load i32, i32* {{.*}}
; UNROLL-NO-IC-NEXT:    [[TMP30:%.*]] = load i32, i32* {{.*}}
; UNROLL-NO-IC-NEXT:    [[TMP35:%.*]] = insertelement <4 x i32> poison, i32 [[TMP27]], i32 0
; UNROLL-NO-IC-NEXT:    [[TMP36:%.*]] = insertelement <4 x i32> [[TMP35]], i32 [[TMP28]], i32 1
; UNROLL-NO-IC-NEXT:    [[TMP37:%.*]] = insertelement <4 x i32> [[TMP36]], i32 [[TMP29]], i32 2
; UNROLL-NO-IC-NEXT:    [[TMP38:%.*]] = insertelement <4 x i32> [[TMP37]], i32 [[TMP30]], i32 3
; UNROLL-NO-IC-NEXT:    [[TMP31:%.*]] = load i32, i32* {{.*}}
; UNROLL-NO-IC-NEXT:    [[TMP32:%.*]] = load i32, i32* {{.*}}
; UNROLL-NO-IC-NEXT:    [[TMP33:%.*]] = load i32, i32* {{.*}}
; UNROLL-NO-IC-NEXT:    [[TMP34:%.*]] = load i32, i32* {{.*}}
; UNROLL-NO-IC-NEXT:    [[TMP39:%.*]] = insertelement <4 x i32> poison, i32 [[TMP31]], i32 0
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
; UNROLL-NO-IC:         [[VECTOR_RECUR:%.*]] = phi <4 x i64> [ <i64 poison, i64 poison, i64 poison, i64 0>, %vector.ph ], [ <i64 1, i64 1, i64 1, i64 1>, %vector.body ]
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

; PR34711: given three consecutive instructions such that the first will be
; widened, the second is a cast that will be widened and needs to sink after the
; third, and the third is a first-order-recurring load that will be replicated
; instead of widened. Although the cast and the first instruction will both be
; widened, and are originally adjacent to each other, make sure the replicated
; load ends up appearing between them.
;
; void PR34711(short[2] *a, int *b, int *c, int n) {
;   for(int i = 0; i < n; i++) {
;     c[i] = 7;
;     b[i] = (a[i][0] * a[i][1]);
;   }
; }
;
; SINK-AFTER-LABEL: @PR34711
; Check that the sext sank after the load in the vector loop.
; SINK-AFTER: vector.body
; SINK-AFTER:   %vector.recur = phi <4 x i16> [ %vector.recur.init, %vector.ph ], [ {{.*}}, %vector.body ]
; SINK-AFTER:   %[[VSHUF:.+]] = shufflevector <4 x i16> %vector.recur, <4 x i16> %{{.*}}, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; SINK-AFTER:   %[[VCONV:.+]] = sext <4 x i16> %[[VSHUF]] to <4 x i32>
; SINK-AFTER:   %[[VCONV3:.+]] = sext <4 x i16> {{.*}} to <4 x i32>
; SINK-AFTER:   mul nsw <4 x i32> %[[VCONV3]], %[[VCONV]]
;
define void @PR34711([2 x i16]* %a, i32* %b, i32* %c, i64 %n) {
entry:
  %pre.index = getelementptr inbounds [2 x i16], [2 x i16]* %a, i64 0, i64 0
  %.pre = load i16, i16* %pre.index
  br label %for.body

for.body:
  %0 = phi i16 [ %.pre, %entry ], [ %1, %for.body ]
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arraycidx = getelementptr inbounds i32, i32* %c, i64 %indvars.iv
  %cur.index = getelementptr inbounds [2 x i16], [2 x i16]* %a, i64 %indvars.iv, i64 1
  store i32 7, i32* %arraycidx   ; 1st instruction, to be widened.
  %conv = sext i16 %0 to i32     ; 2nd, cast to sink after third.
  %1 = load i16, i16* %cur.index ; 3rd, first-order-recurring load not widened.
  %conv3 = sext i16 %1 to i32
  %mul = mul nsw i32 %conv3, %conv
  %arrayidx5 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  store i32 %mul, i32* %arrayidx5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
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

; Do not sink branches: While branches are if-converted and do not require
; sinking, instructions with side effects (e.g. loads) conditioned by those
; branches will become users of the condition bit after vectorization and would
; need to be sunk if the loop is vectorized.
define void @do_not_sink_branch(i32 %x, i32* %in, i32* %out, i32 %tc) local_unnamed_addr #0 {
; NO-SINK-AFTER-LABEL: do_not_sink_branch
; NO-SINK-AFTER-NOT:   vector.ph:
; NO-SINK-AFTER:       }
entry:
  %cmp530 = icmp slt i32 0, %tc
  br label %for.body4

for.body4:                                        ; preds = %cond.end, %entry
  %indvars.iv = phi i32 [ 0, %entry ], [ %indvars.iv.next, %cond.end ]
  %cmp534 = phi i1 [ %cmp530, %entry ], [ %cmp5, %cond.end ]
  br i1 %cmp534, label %cond.true, label %cond.end

cond.true:                                        ; preds = %for.body4
  %arrayidx7 = getelementptr inbounds i32, i32* %in, i32 %indvars.iv
  %in.val = load i32, i32* %arrayidx7, align 4
  br label %cond.end

cond.end:                                         ; preds = %for.body4, %cond.true
  %cond = phi i32 [ %in.val, %cond.true ], [ 0, %for.body4 ]
  %arrayidx8 = getelementptr inbounds i32, i32* %out, i32 %indvars.iv
  store i32 %cond, i32* %arrayidx8, align 4
  %indvars.iv.next = add nuw nsw i32 %indvars.iv, 1
  %cmp5 = icmp slt i32 %indvars.iv.next, %tc
  %exitcond = icmp eq i32 %indvars.iv.next, %x
  br i1 %exitcond, label %for.end12.loopexit, label %for.body4

for.end12.loopexit:                               ; preds = %cond.end
  ret void
}

; Dead instructions, like the exit condition are not part of the actual VPlan
; and do not need to be sunk. PR44634.
define void @sink_dead_inst() {
; SINK-AFTER-LABEL: define void @sink_dead_inst(
; SINK-AFTER-LABEL: vector.body:                                      ; preds = %vector.body, %vector.ph
; SINK-AFTER-NEXT:    %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; SINK-AFTER-NEXT:    %vec.ind = phi <4 x i16> [ <i16 -27, i16 -26, i16 -25, i16 -24>, %vector.ph ], [ %vec.ind.next, %vector.body ]
; SINK-AFTER-NEXT:    %vector.recur = phi <4 x i16> [ <i16 poison, i16 poison, i16 poison, i16 0>, %vector.ph ], [ %3, %vector.body ]
; SINK-AFTER-NEXT:    %vector.recur2 = phi <4 x i32> [ <i32 poison, i32 poison, i32 poison, i32 -27>, %vector.ph ], [ %1, %vector.body ]
; SINK-AFTER-NEXT:    %0 = add <4 x i16> %vec.ind, <i16 1, i16 1, i16 1, i16 1>
; SINK-AFTER-NEXT:    %1 = zext <4 x i16> %0 to <4 x i32>
; SINK-AFTER-NEXT:    %2 = shufflevector <4 x i32> %vector.recur2, <4 x i32> %1, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; SINK-AFTER-NEXT:    %3 = add <4 x i16> %0, <i16 5, i16 5, i16 5, i16 5>
; SINK-AFTER-NEXT:    %4 = shufflevector <4 x i16> %vector.recur, <4 x i16> %3, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; SINK-AFTER-NEXT:    %5 = sub <4 x i16> %4, <i16 10, i16 10, i16 10, i16 10>
; SINK-AFTER-NEXT:    %index.next = add i32 %index, 4
; SINK-AFTER-NEXT:    %vec.ind.next = add <4 x i16> %vec.ind, <i16 4, i16 4, i16 4, i16 4>
; SINK-AFTER-NEXT:    %6 = icmp eq i32 %index.next, 40
; SINK-AFTER-NEXT:    br i1 %6, label %middle.block, label %vector.body, !llvm.loop !43
;
entry:
  br label %for.cond

for.cond:
  %iv = phi i16 [ -27, %entry ], [ %iv.next, %for.cond ]
  %rec.1 = phi i16 [ 0, %entry ], [ %rec.1.prev, %for.cond ]
  %rec.2 = phi i32 [ -27, %entry ], [ %rec.2.prev, %for.cond ]
  %use.rec.1 = sub i16 %rec.1, 10
  %cmp = icmp eq i32 %rec.2, 15
  %iv.next = add i16 %iv, 1
  %rec.2.prev = zext i16 %iv.next to i32
  %rec.1.prev = add i16 %iv.next, 5
  br i1 %cmp, label %for.end, label %for.cond

for.end:
  ret void
}

define i32 @sink_into_replication_region(i32 %y) {
; CHECK-LABEL: @sink_into_replication_region(
; CHECK-NEXT:  bb:
; CHECK-NEXT:    [[TMP0:%.*]] = icmp sgt i32 [[Y:%.*]], 1
; CHECK-NEXT:    [[TMP1:%.*]] = select i1 [[TMP0]], i32 [[Y]], i32 1
; CHECK-NEXT:    br i1 false, label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[N_RND_UP:%.*]] = add nuw i32 [[TMP1]], 3
; CHECK-NEXT:    [[N_VEC:%.*]] = and i32 [[N_RND_UP]], -4
; CHECK-NEXT:    [[TRIP_COUNT_MINUS_1:%.*]] = add nsw i32 [[TMP1]], -1
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <4 x i32> poison, i32 [[TRIP_COUNT_MINUS_1]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <4 x i32> [[BROADCAST_SPLATINSERT]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i32 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[PRED_UDIV_CONTINUE9:%.*]] ]
; CHECK-NEXT:    [[VECTOR_RECUR:%.*]] = phi <4 x i32> [ <i32 poison, i32 poison, i32 poison, i32 0>, [[VECTOR_PH]] ], [ [[TMP21:%.*]], [[PRED_UDIV_CONTINUE9]] ]
; CHECK-NEXT:    [[VEC_PHI1:%.*]] = phi <4 x i32> [ zeroinitializer, [[VECTOR_PH]] ], [ [[TMP23:%.*]], [[PRED_UDIV_CONTINUE9]] ]
; CHECK-NEXT:    [[OFFSET_IDX:%.*]] = sub i32 [[Y]], [[INDEX]]
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT2:%.*]] = insertelement <4 x i32> poison, i32 [[INDEX]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT3:%.*]] = shufflevector <4 x i32> [[BROADCAST_SPLATINSERT2]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[VEC_IV:%.*]] = or <4 x i32> [[BROADCAST_SPLAT3]], <i32 0, i32 1, i32 2, i32 3>
; CHECK-NEXT:    [[TMP2:%.*]] = icmp ule <4 x i32> [[VEC_IV]], [[BROADCAST_SPLAT]]
; CHECK-NEXT:    [[TMP3:%.*]] = extractelement <4 x i1> [[TMP2]], i32 0
; CHECK-NEXT:    br i1 [[TMP3]], label [[PRED_UDIV_IF:%.*]], label [[PRED_UDIV_CONTINUE:%.*]]
; CHECK:       pred.udiv.if:
; CHECK-NEXT:    [[TMP4:%.*]] = udiv i32 219220132, [[OFFSET_IDX]]
; CHECK-NEXT:    [[TMP5:%.*]] = insertelement <4 x i32> poison, i32 [[TMP4]], i32 0
; CHECK-NEXT:    br label [[PRED_UDIV_CONTINUE]]
; CHECK:       pred.udiv.continue:
; CHECK-NEXT:    [[TMP6:%.*]] = phi <4 x i32> [ poison, [[VECTOR_BODY]] ], [ [[TMP5]], [[PRED_UDIV_IF]] ]
; CHECK-NEXT:    [[TMP7:%.*]] = extractelement <4 x i1> [[TMP2]], i32 1
; CHECK-NEXT:    br i1 [[TMP7]], label [[PRED_UDIV_IF4:%.*]], label [[PRED_UDIV_CONTINUE5:%.*]]
; CHECK:       pred.udiv.if4:
; CHECK-NEXT:    [[TMP8:%.*]] = add i32 [[OFFSET_IDX]], -1
; CHECK-NEXT:    [[TMP9:%.*]] = udiv i32 219220132, [[TMP8]]
; CHECK-NEXT:    [[TMP10:%.*]] = insertelement <4 x i32> [[TMP6]], i32 [[TMP9]], i32 1
; CHECK-NEXT:    br label [[PRED_UDIV_CONTINUE5]]
; CHECK:       pred.udiv.continue5:
; CHECK-NEXT:    [[TMP11:%.*]] = phi <4 x i32> [ [[TMP6]], [[PRED_UDIV_CONTINUE]] ], [ [[TMP10]], [[PRED_UDIV_IF4]] ]
; CHECK-NEXT:    [[TMP12:%.*]] = extractelement <4 x i1> [[TMP2]], i32 2
; CHECK-NEXT:    br i1 [[TMP12]], label [[PRED_UDIV_IF6:%.*]], label [[PRED_UDIV_CONTINUE7:%.*]]
; CHECK:       pred.udiv.if6:
; CHECK-NEXT:    [[TMP13:%.*]] = add i32 [[OFFSET_IDX]], -2
; CHECK-NEXT:    [[TMP14:%.*]] = udiv i32 219220132, [[TMP13]]
; CHECK-NEXT:    [[TMP15:%.*]] = insertelement <4 x i32> [[TMP11]], i32 [[TMP14]], i32 2
; CHECK-NEXT:    br label [[PRED_UDIV_CONTINUE7]]
; CHECK:       pred.udiv.continue7:
; CHECK-NEXT:    [[TMP16:%.*]] = phi <4 x i32> [ [[TMP11]], [[PRED_UDIV_CONTINUE5]] ], [ [[TMP15]], [[PRED_UDIV_IF6]] ]
; CHECK-NEXT:    [[TMP17:%.*]] = extractelement <4 x i1> [[TMP2]], i32 3
; CHECK-NEXT:    br i1 [[TMP17]], label [[PRED_UDIV_IF8:%.*]], label [[PRED_UDIV_CONTINUE9]]
; CHECK:       pred.udiv.if8:
; CHECK-NEXT:    [[TMP18:%.*]] = add i32 [[OFFSET_IDX]], -3
; CHECK-NEXT:    [[TMP19:%.*]] = udiv i32 219220132, [[TMP18]]
; CHECK-NEXT:    [[TMP20:%.*]] = insertelement <4 x i32> [[TMP16]], i32 [[TMP19]], i32 3
; CHECK-NEXT:    br label [[PRED_UDIV_CONTINUE9]]
; CHECK:       pred.udiv.continue9:
; CHECK-NEXT:    [[TMP21]] = phi <4 x i32> [ [[TMP16]], [[PRED_UDIV_CONTINUE7]] ], [ [[TMP20]], [[PRED_UDIV_IF8]] ]
; CHECK-NEXT:    [[TMP22:%.*]] = shufflevector <4 x i32> [[VECTOR_RECUR]], <4 x i32> [[TMP21]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP23]] = add <4 x i32> [[VEC_PHI1]], [[TMP22]]
; CHECK-NEXT:    [[TMP24:%.*]] = select <4 x i1> [[TMP2]], <4 x i32> [[TMP23]], <4 x i32> [[VEC_PHI1]]
; CHECK-NEXT:    [[INDEX_NEXT]] = add i32 [[INDEX]], 4
; CHECK-NEXT:    [[TMP25:%.*]] = icmp eq i32 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP25]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !prof !45, [[LOOP46:!llvm.loop !.*]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[RDX_SHUF:%.*]] = shufflevector <4 x i32> [[TMP24]], <4 x i32> poison, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
; CHECK-NEXT:    [[BIN_RDX:%.*]] = add <4 x i32> [[TMP24]], [[RDX_SHUF]]
; CHECK-NEXT:    [[RDX_SHUF10:%.*]] = shufflevector <4 x i32> [[BIN_RDX]], <4 x i32> poison, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
; CHECK-NEXT:    [[BIN_RDX11:%.*]] = add <4 x i32> [[BIN_RDX]], [[RDX_SHUF10]]
; CHECK-NEXT:    [[TMP26:%.*]] = extractelement <4 x i32> [[BIN_RDX11]], i32 0
; CHECK-NEXT:    br i1 true, label [[BB1:%.*]], label [[SCALAR_PH]]
; CHECK:       scalar.ph:
; CHECK-NEXT:    br label [[BB2:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    [[TMP:%.*]] = phi i32 [ undef, [[BB2]] ], [ [[TMP26]], [[MIDDLE_BLOCK]] ]
; CHECK-NEXT:    ret i32 [[TMP]]
; CHECK:       bb2:
; CHECK-NEXT:    br i1 undef, label [[BB1]], label [[BB2]], !prof !47, [[LOOP48:!llvm.loop !.*]]
;
bb:
  br label %bb2

  bb1:                                              ; preds = %bb2
  %tmp = phi i32 [ %tmp6, %bb2 ]
  ret i32 %tmp

  bb2:                                              ; preds = %bb2, %bb
  %tmp3 = phi i32 [ %tmp8, %bb2 ], [ %y, %bb ]
  %tmp4 = phi i32 [ %tmp7, %bb2 ], [ 0, %bb ]
  %tmp5 = phi i32 [ %tmp6, %bb2 ], [ 0, %bb ]
  %tmp6 = add i32 %tmp5, %tmp4
  %tmp7 = udiv i32 219220132, %tmp3
  %tmp8 = add nsw i32 %tmp3, -1
  %tmp9 = icmp slt i32 %tmp3, 2
  br i1 %tmp9, label %bb1, label %bb2, !prof !2
}

define i32 @sink_into_replication_region_multiple(i32 *%x, i32 %y) {
; CHECK-LABEL: @sink_into_replication_region_multiple(
; CHECK-NEXT:  bb:
; CHECK-NEXT:    [[TMP0:%.*]] = icmp sgt i32 [[Y:%.*]], 1
; CHECK-NEXT:    [[TMP1:%.*]] = select i1 [[TMP0]], i32 [[Y]], i32 1
; CHECK-NEXT:    br i1 false, label [[SCALAR_PH:%.*]], label [[VECTOR_PH:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[N_RND_UP:%.*]] = add nuw i32 [[TMP1]], 3
; CHECK-NEXT:    [[N_VEC:%.*]] = and i32 [[N_RND_UP]], -4
; CHECK-NEXT:    [[TRIP_COUNT_MINUS_1:%.*]] = add nsw i32 [[TMP1]], -1
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <4 x i32> poison, i32 [[TRIP_COUNT_MINUS_1]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <4 x i32> [[BROADCAST_SPLATINSERT]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i32 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[PRED_STORE_CONTINUE16:%.*]] ]
; CHECK-NEXT:    [[VEC_IND2:%.*]] = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, [[VECTOR_PH]] ], [ [[VEC_IND_NEXT3:%.*]], [[PRED_STORE_CONTINUE16]] ]
; CHECK-NEXT:    [[VECTOR_RECUR:%.*]] = phi <4 x i32> [ <i32 poison, i32 poison, i32 poison, i32 0>, [[VECTOR_PH]] ], [ [[TMP21:%.*]], [[PRED_STORE_CONTINUE16]] ]
; CHECK-NEXT:    [[VEC_PHI4:%.*]] = phi <4 x i32> [ zeroinitializer, [[VECTOR_PH]] ], [ [[TMP23:%.*]], [[PRED_STORE_CONTINUE16]] ]
; CHECK-NEXT:    [[OFFSET_IDX:%.*]] = sub i32 [[Y]], [[INDEX]]
; CHECK-NEXT:    [[TMP2:%.*]] = add i32 [[OFFSET_IDX]], -1
; CHECK-NEXT:    [[TMP3:%.*]] = add i32 [[OFFSET_IDX]], -2
; CHECK-NEXT:    [[TMP4:%.*]] = add i32 [[OFFSET_IDX]], -3
; CHECK-NEXT:    [[TMP5:%.*]] = icmp ule <4 x i32> [[VEC_IND2]], [[BROADCAST_SPLAT]]
; CHECK-NEXT:    [[TMP6:%.*]] = extractelement <4 x i1> [[TMP5]], i32 0
; CHECK-NEXT:    br i1 [[TMP6]], label [[PRED_UDIV_IF:%.*]], label [[PRED_UDIV_CONTINUE:%.*]]
; CHECK:       pred.udiv.if:
; CHECK-NEXT:    [[TMP7:%.*]] = udiv i32 219220132, [[OFFSET_IDX]]
; CHECK-NEXT:    [[TMP8:%.*]] = insertelement <4 x i32> poison, i32 [[TMP7]], i32 0
; CHECK-NEXT:    br label [[PRED_UDIV_CONTINUE]]
; CHECK:       pred.udiv.continue:
; CHECK-NEXT:    [[TMP9:%.*]] = phi <4 x i32> [ poison, [[VECTOR_BODY]] ], [ [[TMP8]], [[PRED_UDIV_IF]] ]
; CHECK-NEXT:    [[TMP10:%.*]] = extractelement <4 x i1> [[TMP5]], i32 1
; CHECK-NEXT:    br i1 [[TMP10]], label [[PRED_UDIV_IF5:%.*]], label [[PRED_UDIV_CONTINUE6:%.*]]
; CHECK:       pred.udiv.if5:
; CHECK-NEXT:    [[TMP11:%.*]] = udiv i32 219220132, [[TMP2]]
; CHECK-NEXT:    [[TMP12:%.*]] = insertelement <4 x i32> [[TMP9]], i32 [[TMP11]], i32 1
; CHECK-NEXT:    br label [[PRED_UDIV_CONTINUE6]]
; CHECK:       pred.udiv.continue6:
; CHECK-NEXT:    [[TMP13:%.*]] = phi <4 x i32> [ [[TMP9]], [[PRED_UDIV_CONTINUE]] ], [ [[TMP12]], [[PRED_UDIV_IF5]] ]
; CHECK-NEXT:    [[TMP14:%.*]] = extractelement <4 x i1> [[TMP5]], i32 2
; CHECK-NEXT:    br i1 [[TMP14]], label [[PRED_UDIV_IF7:%.*]], label [[PRED_UDIV_CONTINUE8:%.*]]
; CHECK:       pred.udiv.if7:
; CHECK-NEXT:    [[TMP15:%.*]] = udiv i32 219220132, [[TMP3]]
; CHECK-NEXT:    [[TMP16:%.*]] = insertelement <4 x i32> [[TMP13]], i32 [[TMP15]], i32 2
; CHECK-NEXT:    br label [[PRED_UDIV_CONTINUE8]]
; CHECK:       pred.udiv.continue8:
; CHECK-NEXT:    [[TMP17:%.*]] = phi <4 x i32> [ [[TMP13]], [[PRED_UDIV_CONTINUE6]] ], [ [[TMP16]], [[PRED_UDIV_IF7]] ]
; CHECK-NEXT:    [[TMP18:%.*]] = extractelement <4 x i1> [[TMP5]], i32 3
; CHECK-NEXT:    br i1 [[TMP18]], label [[PRED_UDIV_IF9:%.*]], label [[PRED_UDIV_CONTINUE10:%.*]]
; CHECK:       pred.udiv.if9:
; CHECK-NEXT:    [[TMP19:%.*]] = udiv i32 219220132, [[TMP4]]
; CHECK-NEXT:    [[TMP20:%.*]] = insertelement <4 x i32> [[TMP17]], i32 [[TMP19]], i32 3
; CHECK-NEXT:    br label [[PRED_UDIV_CONTINUE10]]
; CHECK:       pred.udiv.continue10:
; CHECK-NEXT:    [[TMP21]] = phi <4 x i32> [ [[TMP17]], [[PRED_UDIV_CONTINUE8]] ], [ [[TMP20]], [[PRED_UDIV_IF9]] ]
; CHECK-NEXT:    [[TMP22:%.*]] = shufflevector <4 x i32> [[VECTOR_RECUR]], <4 x i32> [[TMP21]], <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:    [[TMP23]] = add <4 x i32> [[VEC_PHI4]], [[TMP22]]
; CHECK-NEXT:    [[TMP24:%.*]] = extractelement <4 x i1> [[TMP5]], i32 0
; CHECK-NEXT:    br i1 [[TMP24]], label [[PRED_STORE_IF:%.*]], label [[PRED_STORE_CONTINUE:%.*]]
; CHECK:       pred.store.if:
; CHECK-NEXT:    [[TMP25:%.*]] = sext i32 [[INDEX]] to i64
; CHECK-NEXT:    [[TMP26:%.*]] = getelementptr inbounds i32, i32* [[X:%.*]], i64 [[TMP25]]
; CHECK-NEXT:    store i32 [[OFFSET_IDX]], i32* [[TMP26]], align 4
; CHECK-NEXT:    br label [[PRED_STORE_CONTINUE]]
; CHECK:       pred.store.continue:
; CHECK-NEXT:    [[TMP27:%.*]] = extractelement <4 x i1> [[TMP5]], i32 1
; CHECK-NEXT:    br i1 [[TMP27]], label [[PRED_STORE_IF11:%.*]], label [[PRED_STORE_CONTINUE12:%.*]]
; CHECK:       pred.store.if11:
; CHECK-NEXT:    [[TMP28:%.*]] = or i32 [[INDEX]], 1
; CHECK-NEXT:    [[TMP29:%.*]] = sext i32 [[TMP28]] to i64
; CHECK-NEXT:    [[TMP30:%.*]] = getelementptr inbounds i32, i32* [[X]], i64 [[TMP29]]
; CHECK-NEXT:    store i32 [[TMP2]], i32* [[TMP30]], align 4
; CHECK-NEXT:    br label [[PRED_STORE_CONTINUE12]]
; CHECK:       pred.store.continue12:
; CHECK-NEXT:    [[TMP31:%.*]] = extractelement <4 x i1> [[TMP5]], i32 2
; CHECK-NEXT:    br i1 [[TMP31]], label [[PRED_STORE_IF13:%.*]], label [[PRED_STORE_CONTINUE14:%.*]]
; CHECK:       pred.store.if13:
; CHECK-NEXT:    [[TMP32:%.*]] = or i32 [[INDEX]], 2
; CHECK-NEXT:    [[TMP33:%.*]] = sext i32 [[TMP32]] to i64
; CHECK-NEXT:    [[TMP34:%.*]] = getelementptr inbounds i32, i32* [[X]], i64 [[TMP33]]
; CHECK-NEXT:    store i32 [[TMP3]], i32* [[TMP34]], align 4
; CHECK-NEXT:    br label [[PRED_STORE_CONTINUE14]]
; CHECK:       pred.store.continue14:
; CHECK-NEXT:    [[TMP35:%.*]] = extractelement <4 x i1> [[TMP5]], i32 3
; CHECK-NEXT:    br i1 [[TMP35]], label [[PRED_STORE_IF15:%.*]], label [[PRED_STORE_CONTINUE16]]
; CHECK:       pred.store.if15:
; CHECK-NEXT:    [[TMP36:%.*]] = or i32 [[INDEX]], 3
; CHECK-NEXT:    [[TMP37:%.*]] = sext i32 [[TMP36]] to i64
; CHECK-NEXT:    [[TMP38:%.*]] = getelementptr inbounds i32, i32* [[X]], i64 [[TMP37]]
; CHECK-NEXT:    store i32 [[TMP4]], i32* [[TMP38]], align 4
; CHECK-NEXT:    br label [[PRED_STORE_CONTINUE16]]
; CHECK:       pred.store.continue16:
; CHECK-NEXT:    [[TMP39:%.*]] = select <4 x i1> [[TMP5]], <4 x i32> [[TMP23]], <4 x i32> [[VEC_PHI4]]
; CHECK-NEXT:    [[INDEX_NEXT]] = add i32 [[INDEX]], 4
; CHECK-NEXT:    [[VEC_IND_NEXT3]] = add <4 x i32> [[VEC_IND2]], <i32 4, i32 4, i32 4, i32 4>
; CHECK-NEXT:    [[TMP40:%.*]] = icmp eq i32 [[INDEX_NEXT]], [[N_VEC]]
; CHECK-NEXT:    br i1 [[TMP40]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !prof !45, [[LOOP49:!llvm.loop !.*]]
; CHECK:       middle.block:
; CHECK-NEXT:    [[RDX_SHUF:%.*]] = shufflevector <4 x i32> [[TMP39]], <4 x i32> poison, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
; CHECK-NEXT:    [[BIN_RDX:%.*]] = add <4 x i32> [[TMP39]], [[RDX_SHUF]]
; CHECK-NEXT:    [[RDX_SHUF17:%.*]] = shufflevector <4 x i32> [[BIN_RDX]], <4 x i32> poison, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
; CHECK-NEXT:    [[BIN_RDX18:%.*]] = add <4 x i32> [[BIN_RDX]], [[RDX_SHUF17]]
; CHECK-NEXT:    [[TMP41:%.*]] = extractelement <4 x i32> [[BIN_RDX18]], i32 0
; CHECK-NEXT:    br i1 true, label [[BB1:%.*]], label [[SCALAR_PH]]
; CHECK:       scalar.ph:
; CHECK-NEXT:    br label [[BB2:%.*]]
; CHECK:       bb1:
; CHECK-NEXT:    [[TMP:%.*]] = phi i32 [ undef, [[BB2]] ], [ [[TMP41]], [[MIDDLE_BLOCK]] ]
; CHECK-NEXT:    ret i32 [[TMP]]
; CHECK:       bb2:
; CHECK-NEXT:    br i1 undef, label [[BB1]], label [[BB2]], !prof !47, [[LOOP50:!llvm.loop !.*]]
;
bb:
  br label %bb2

  bb1:                                              ; preds = %bb2
  %tmp = phi i32 [ %tmp6, %bb2 ]
  ret i32 %tmp

  bb2:                                              ; preds = %bb2, %bb
  %tmp3 = phi i32 [ %tmp8, %bb2 ], [ %y, %bb ]
  %iv = phi i32 [ %iv.next, %bb2 ], [ 0, %bb ]
  %tmp4 = phi i32 [ %tmp7, %bb2 ], [ 0, %bb ]
  %tmp5 = phi i32 [ %tmp6, %bb2 ], [ 0, %bb ]
  %g = getelementptr inbounds i32, i32* %x, i32 %iv
  %tmp6 = add i32 %tmp5, %tmp4
  %tmp7 = udiv i32 219220132, %tmp3
  store i32 %tmp3, i32* %g, align 4
  %tmp8 = add nsw i32 %tmp3, -1
  %iv.next = add nsw i32 %iv, 1
  %tmp9 = icmp slt i32 %tmp3, 2
  br i1 %tmp9, label %bb1, label %bb2, !prof !2
}

!2 = !{!"branch_weights", i32 1, i32 1}
