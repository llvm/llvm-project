; RUN: opt < %s -loop-vectorize -force-vector-interleave=1 -force-vector-width=2 -S | FileCheck %s
; RUN: opt < %s -loop-vectorize -force-vector-interleave=1 -force-vector-width=2 -instcombine -S | FileCheck %s --check-prefix=IND
; RUN: opt < %s -loop-vectorize -force-vector-interleave=2 -force-vector-width=2 -instcombine -S | FileCheck %s --check-prefix=UNROLL
; RUN: opt < %s -loop-vectorize -force-vector-interleave=2 -force-vector-width=2 -S | FileCheck %s --check-prefix=UNROLL-NO-IC
; RUN: opt < %s -loop-vectorize -force-vector-interleave=2 -force-vector-width=4 -enable-interleaved-mem-accesses -instcombine -S | FileCheck %s --check-prefix=INTERLEAVE

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Make sure that we can handle multiple integer induction variables.
;
; CHECK-LABEL: @multi_int_induction(
; CHECK:       vector.body:
; CHECK-NEXT:    %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK-NEXT:    %vec.ind = phi <2 x i32> [ <i32 190, i32 191>, %vector.ph ], [ %vec.ind.next, %vector.body ]
; CHECK:         [[TMP3:%.*]] = add i64 %index, 0
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr inbounds i32, i32* %A, i64 [[TMP3]]
; CHECK-NEXT:    [[TMP5:%.*]] = getelementptr inbounds i32, i32* [[TMP4]], i32 0
; CHECK-NEXT:    [[TMP6:%.*]] = bitcast i32* [[TMP5]] to <2 x i32>*
; CHECK-NEXT:    store <2 x i32> %vec.ind, <2 x i32>* [[TMP6]], align 4
; CHECK:         %index.next = add i64 %index, 2
; CHECK-NEXT:    %vec.ind.next = add <2 x i32> %vec.ind, <i32 2, i32 2>
; CHECK:         br i1 {{.*}}, label %middle.block, label %vector.body
define void @multi_int_induction(i32* %A, i32 %N) {
for.body.lr.ph:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %count.09 = phi i32 [ 190, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %count.09, i32* %arrayidx2, align 4
  %inc = add nsw i32 %count.09, 1
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, %N
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret void
}

; Make sure we remove unneeded vectorization of induction variables.
; In order for instcombine to cleanup the vectorized induction variables that we
; create in the loop vectorizer we need to perform some form of redundancy
; elimination to get rid of multiple uses.

; IND-LABEL: scalar_use

; IND:     br label %vector.body
; IND:     vector.body:
;   Vectorized induction variable.
; IND-NOT:  insertelement <2 x i64>
; IND-NOT:  shufflevector <2 x i64>
; IND:     br {{.*}}, label %vector.body

define void @scalar_use(float* %a, float %b, i64 %offset, i64 %offset2, i64 %n) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %ind.sum = add i64 %iv, %offset
  %arr.idx = getelementptr inbounds float, float* %a, i64 %ind.sum
  %l1 = load float, float* %arr.idx, align 4
  %ind.sum2 = add i64 %iv, %offset2
  %arr.idx2 = getelementptr inbounds float, float* %a, i64 %ind.sum2
  %l2 = load float, float* %arr.idx2, align 4
  %m = fmul fast float %b, %l2
  %ad = fadd fast float %l1, %m
  store float %ad, float* %arr.idx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %loopexit, label %for.body

loopexit:
  ret void
}

; Make sure we don't create a vector induction phi node that is unused.
; Scalarize the step vectors instead.
;
; for (int i = 0; i < n; ++i)
;   sum += a[i];
;
; CHECK-LABEL: @scalarize_induction_variable_01(
; CHECK: vector.body:
; CHECK:   %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK:   %[[i0:.+]] = add i64 %index, 0
; CHECK:   getelementptr inbounds i64, i64* %a, i64 %[[i0]]
;
; UNROLL-NO-IC-LABEL: @scalarize_induction_variable_01(
; UNROLL-NO-IC: vector.body:
; UNROLL-NO-IC:   %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; UNROLL-NO-IC:   %[[i0:.+]] = add i64 %index, 0
; UNROLL-NO-IC:   %[[i2:.+]] = add i64 %index, 2
; UNROLL-NO-IC:   getelementptr inbounds i64, i64* %a, i64 %[[i0]]
; UNROLL-NO-IC:   getelementptr inbounds i64, i64* %a, i64 %[[i2]]
;
; IND-LABEL: @scalarize_induction_variable_01(
; IND:     vector.body:
; IND:       %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; IND-NOT:   add i64 {{.*}}, 2
; IND:       getelementptr inbounds i64, i64* %a, i64 %index
;
; UNROLL-LABEL: @scalarize_induction_variable_01(
; UNROLL:     vector.body:
; UNROLL:       %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; UNROLL-NOT:   add i64 {{.*}}, 4
; UNROLL:       %[[g1:.+]] = getelementptr inbounds i64, i64* %a, i64 %index
; UNROLL:       getelementptr inbounds i64, i64* %[[g1]], i64 2

define i64 @scalarize_induction_variable_01(i64 *%a, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %sum = phi i64 [ %2, %for.body ], [ 0, %entry ]
  %0 = getelementptr inbounds i64, i64* %a, i64 %i
  %1 = load i64, i64* %0, align 8
  %2 = add i64 %1, %sum
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  %3  = phi i64 [ %2, %for.body ]
  ret i64 %3
}

; Make sure we scalarize the step vectors used for the pointer arithmetic. We
; can't easily simplify vectorized step vectors.
;
; float s = 0;
; for (int i ; 0; i < n; i += 8)
;   s += (a[i] + b[i] + 1.0f);
;
; CHECK-LABEL: @scalarize_induction_variable_02(
; CHECK: vector.body:
; CHECK:   %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK:   %offset.idx = mul i64 %index, 8
; CHECK:   %[[i0:.+]] = add i64 %offset.idx, 0
; CHECK:   %[[i1:.+]] = add i64 %offset.idx, 8
; CHECK:   getelementptr inbounds float, float* %a, i64 %[[i0]]
; CHECK:   getelementptr inbounds float, float* %a, i64 %[[i1]]
; CHECK:   getelementptr inbounds float, float* %b, i64 %[[i0]]
; CHECK:   getelementptr inbounds float, float* %b, i64 %[[i1]]
;
; UNROLL-NO-IC-LABEL: @scalarize_induction_variable_02(
; UNROLL-NO-IC: vector.body:
; UNROLL-NO-IC:   %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; UNROLL-NO-IC:   %offset.idx = mul i64 %index, 8
; UNROLL-NO-IC:   %[[i0:.+]] = add i64 %offset.idx, 0
; UNROLL-NO-IC:   %[[i1:.+]] = add i64 %offset.idx, 8
; UNROLL-NO-IC:   %[[i2:.+]] = add i64 %offset.idx, 16
; UNROLL-NO-IC:   %[[i3:.+]] = add i64 %offset.idx, 24
; UNROLL-NO-IC:   getelementptr inbounds float, float* %a, i64 %[[i0]]
; UNROLL-NO-IC:   getelementptr inbounds float, float* %a, i64 %[[i1]]
; UNROLL-NO-IC:   getelementptr inbounds float, float* %a, i64 %[[i2]]
; UNROLL-NO-IC:   getelementptr inbounds float, float* %a, i64 %[[i3]]
; UNROLL-NO-IC:   getelementptr inbounds float, float* %b, i64 %[[i0]]
; UNROLL-NO-IC:   getelementptr inbounds float, float* %b, i64 %[[i1]]
; UNROLL-NO-IC:   getelementptr inbounds float, float* %b, i64 %[[i2]]
; UNROLL-NO-IC:   getelementptr inbounds float, float* %b, i64 %[[i3]]
;
; IND-LABEL: @scalarize_induction_variable_02(
; IND: vector.body:
; IND:   %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; IND:   %[[i0:.+]] = shl i64 %index, 3
; IND:   %[[i1:.+]] = or i64 %[[i0]], 8
; IND:   getelementptr inbounds float, float* %a, i64 %[[i0]]
; IND:   getelementptr inbounds float, float* %a, i64 %[[i1]]
;
; UNROLL-LABEL: @scalarize_induction_variable_02(
; UNROLL: vector.body:
; UNROLL:   %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; UNROLL:   %[[i0:.+]] = shl i64 %index, 3
; UNROLL:   %[[i1:.+]] = or i64 %[[i0]], 8
; UNROLL:   %[[i2:.+]] = or i64 %[[i0]], 16
; UNROLL:   %[[i3:.+]] = or i64 %[[i0]], 24
; UNROLL:   getelementptr inbounds float, float* %a, i64 %[[i0]]
; UNROLL:   getelementptr inbounds float, float* %a, i64 %[[i1]]
; UNROLL:   getelementptr inbounds float, float* %a, i64 %[[i2]]
; UNROLL:   getelementptr inbounds float, float* %a, i64 %[[i3]]

define float @scalarize_induction_variable_02(float* %a, float* %b, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %s = phi float [ 0.0, %entry ], [ %6, %for.body ]
  %0 = getelementptr inbounds float, float* %a, i64 %i
  %1 = load float, float* %0, align 4
  %2 = getelementptr inbounds float, float* %b, i64 %i
  %3 = load float, float* %2, align 4
  %4 = fadd fast float %s, 1.0
  %5 = fadd fast float %4, %1
  %6 = fadd fast float %5, %3
  %i.next = add nuw nsw i64 %i, 8
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  %s.lcssa = phi float [ %6, %for.body ]
  ret float %s.lcssa
}

; Make sure we scalarize the step vectors used for the pointer arithmetic. We
; can't easily simplify vectorized step vectors. (Interleaved accesses.)
;
; for (int i = 0; i < n; ++i)
;   a[i].f ^= y;
;
; INTERLEAVE-LABEL: @scalarize_induction_variable_03(
; INTERLEAVE: vector.body:
; INTERLEAVE:   %[[i0:.+]] = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; INTERLEAVE:   %[[i1:.+]] = or i64 %[[i0]], 1
; INTERLEAVE:   %[[i2:.+]] = or i64 %[[i0]], 2
; INTERLEAVE:   %[[i3:.+]] = or i64 %[[i0]], 3
; INTERLEAVE:   %[[i4:.+]] = or i64 %[[i0]], 4
; INTERLEAVE:   %[[i5:.+]] = or i64 %[[i0]], 5
; INTERLEAVE:   %[[i6:.+]] = or i64 %[[i0]], 6
; INTERLEAVE:   %[[i7:.+]] = or i64 %[[i0]], 7
; INTERLEAVE:   getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %[[i0]], i32 1
; INTERLEAVE:   getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %[[i1]], i32 1
; INTERLEAVE:   getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %[[i2]], i32 1
; INTERLEAVE:   getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %[[i3]], i32 1
; INTERLEAVE:   getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %[[i4]], i32 1
; INTERLEAVE:   getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %[[i5]], i32 1
; INTERLEAVE:   getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %[[i6]], i32 1
; INTERLEAVE:   getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %[[i7]], i32 1

%pair.i32 = type { i32, i32 }
define void @scalarize_induction_variable_03(%pair.i32 *%p, i32 %y, i64 %n) {
entry:
  br label %for.body

for.body:
  %i  = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %f = getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %i, i32 1
  %0 = load i32, i32* %f, align 8
  %1 = xor i32 %0, %y
  store i32 %1, i32* %f, align 8
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

; Make sure we scalarize the step vectors used for the pointer arithmetic. We
; can't easily simplify vectorized step vectors. (Interleaved accesses.)
;
; for (int i = 0; i < n; ++i)
;   p[i].f = a[i * 4]
;
; INTERLEAVE-LABEL: @scalarize_induction_variable_04(
; INTERLEAVE: vector.body:
; INTERLEAVE:   %[[i0:.+]] = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; INTERLEAVE:   %[[i1:.+]] = or i64 %[[i0]], 1
; INTERLEAVE:   %[[i2:.+]] = or i64 %[[i0]], 2
; INTERLEAVE:   %[[i3:.+]] = or i64 %[[i0]], 3
; INTERLEAVE:   %[[i4:.+]] = or i64 %[[i0]], 4
; INTERLEAVE:   %[[i5:.+]] = or i64 %[[i0]], 5
; INTERLEAVE:   %[[i6:.+]] = or i64 %[[i0]], 6
; INTERLEAVE:   %[[i7:.+]] = or i64 %[[i0]], 7
; INTERLEAVE:   getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %[[i0]], i32 1
; INTERLEAVE:   getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %[[i1]], i32 1
; INTERLEAVE:   getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %[[i2]], i32 1
; INTERLEAVE:   getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %[[i3]], i32 1
; INTERLEAVE:   getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %[[i4]], i32 1
; INTERLEAVE:   getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %[[i5]], i32 1
; INTERLEAVE:   getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %[[i6]], i32 1
; INTERLEAVE:   getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %[[i7]], i32 1

define void @scalarize_induction_variable_04(i32* %a, %pair.i32* %p, i32 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry]
  %0 = shl nsw i64 %i, 2
  %1 = getelementptr inbounds i32, i32* %a, i64 %0
  %2 = load i32, i32* %1, align 1
  %3 = getelementptr inbounds %pair.i32, %pair.i32* %p, i64 %i, i32 1
  store i32 %2, i32* %3, align 1
  %i.next = add nuw nsw i64 %i, 1
  %4 = trunc i64 %i.next to i32
  %cond = icmp eq i32 %4, %n
  br i1 %cond, label %for.end, label %for.body

for.end:
  ret void
}

; PR30542. Ensure we generate all the scalar steps for the induction variable.
; The scalar induction variable is used by a getelementptr instruction
; (uniform), and a udiv (non-uniform).
;
; int sum = 0;
; for (int i = 0; i < n; ++i) {
;   int x = a[i];
;   if (c)
;     x /= i;
;   sum += x;
; }
;
; CHECK-LABEL: @scalarize_induction_variable_05(
; CHECK: vector.body:
; CHECK:   %index = phi i32 [ 0, %vector.ph ], [ %index.next, %pred.udiv.continue{{[0-9]+}} ]
; CHECK:   %[[I0:.+]] = add i32 %index, 0
; CHECK:   getelementptr inbounds i32, i32* %a, i32 %[[I0]]
; CHECK: pred.udiv.if:
; CHECK:   udiv i32 {{.*}}, %[[I0]]
; CHECK: pred.udiv.if{{[0-9]+}}:
; CHECK:   %[[I1:.+]] = add i32 %index, 1
; CHECK:   udiv i32 {{.*}}, %[[I1]]
;
; UNROLL-NO_IC-LABEL: @scalarize_induction_variable_05(
; UNROLL-NO-IC: vector.body:
; UNROLL-NO-IC:   %index = phi i32 [ 0, %vector.ph ], [ %index.next, %pred.udiv.continue{{[0-9]+}} ]
; UNROLL-NO-IC:   %[[I0:.+]] = add i32 %index, 0
; UNROLL-NO-IC:   %[[I2:.+]] = add i32 %index, 2
; UNROLL-NO-IC:   getelementptr inbounds i32, i32* %a, i32 %[[I0]]
; UNROLL-NO-IC:   getelementptr inbounds i32, i32* %a, i32 %[[I2]]
; UNROLL-NO-IC: pred.udiv.if:
; UNROLL-NO-IC:   udiv i32 {{.*}}, %[[I0]]
; UNROLL-NO-IC: pred.udiv.if{{[0-9]+}}:
; UNROLL-NO-IC:   %[[I1:.+]] = add i32 %index, 1
; UNROLL-NO-IC:   udiv i32 {{.*}}, %[[I1]]
; UNROLL-NO-IC: pred.udiv.if{{[0-9]+}}:
; UNROLL-NO-IC:   udiv i32 {{.*}}, %[[I2]]
; UNROLL-NO-IC: pred.udiv.if{{[0-9]+}}:
; UNROLL-NO-IC:   %[[I3:.+]] = add i32 %index, 3
; UNROLL-NO-IC:   udiv i32 {{.*}}, %[[I3]]
;
; IND-LABEL: @scalarize_induction_variable_05(
; IND: vector.body:
; IND:   %index = phi i32 [ 0, %vector.ph ], [ %index.next, %pred.udiv.continue{{[0-9]+}} ]
; IND:   %[[E0:.+]] = sext i32 %index to i64
; IND:   getelementptr inbounds i32, i32* %a, i64 %[[E0]]
; IND: pred.udiv.if:
; IND:   udiv i32 {{.*}}, %index
; IND: pred.udiv.if{{[0-9]+}}:
; IND:   %[[I1:.+]] = or i32 %index, 1
; IND:   udiv i32 {{.*}}, %[[I1]]
;
; UNROLL-LABEL: @scalarize_induction_variable_05(
; UNROLL: vector.body:
; UNROLL:   %index = phi i32 [ 0, %vector.ph ], [ %index.next, %pred.udiv.continue{{[0-9]+}} ]
; UNROLL:   %[[I2:.+]] = or i32 %index, 2
; UNROLL:   %[[E0:.+]] = sext i32 %index to i64
; UNROLL:   %[[G0:.+]] = getelementptr inbounds i32, i32* %a, i64 %[[E0]]
; UNROLL:   getelementptr inbounds i32, i32* %[[G0]], i64 2
; UNROLL: pred.udiv.if:
; UNROLL:   udiv i32 {{.*}}, %index
; UNROLL: pred.udiv.if{{[0-9]+}}:
; UNROLL:   %[[I1:.+]] = or i32 %index, 1
; UNROLL:   udiv i32 {{.*}}, %[[I1]]
; UNROLL: pred.udiv.if{{[0-9]+}}:
; UNROLL:   udiv i32 {{.*}}, %[[I2]]
; UNROLL: pred.udiv.if{{[0-9]+}}:
; UNROLL:   %[[I3:.+]] = or i32 %index, 3
; UNROLL:   udiv i32 {{.*}}, %[[I3]]

define i32 @scalarize_induction_variable_05(i32* %a, i32 %x, i1 %c, i32 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %i.next, %if.end ]
  %sum = phi i32 [ 0, %entry ], [ %tmp4, %if.end ]
  %tmp0 = getelementptr inbounds i32, i32* %a, i32 %i
  %tmp1 = load i32, i32* %tmp0, align 4
  br i1 %c, label %if.then, label %if.end

if.then:
  %tmp2 = udiv i32 %tmp1, %i
  br label %if.end

if.end:
  %tmp3 = phi i32 [ %tmp2, %if.then ], [ %tmp1, %for.body ]
  %tmp4 = add i32 %tmp3, %sum
  %i.next = add nuw nsw i32 %i, 1
  %cond = icmp slt i32 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  %tmp5  = phi i32 [ %tmp4, %if.end ]
  ret i32 %tmp5
}

; Ensure we generate both a vector and a scalar induction variable. In this
; test, the induction variable is used by an instruction that will be
; vectorized (trunc) as well as an instruction that will remain in scalar form
; (gepelementptr).
;
; CHECK-LABEL: @iv_vector_and_scalar_users(
; CHECK: vector.body:
; CHECK:   %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK:   %vec.ind = phi <2 x i64> [ <i64 0, i64 1>, %vector.ph ], [ %vec.ind.next, %vector.body ]
; CHECK:   %vec.ind1 = phi <2 x i32> [ <i32 0, i32 1>, %vector.ph ], [ %vec.ind.next2, %vector.body ]
; CHECK:   %[[i0:.+]] = add i64 %index, 0
; CHECK:   %[[i1:.+]] = add i64 %index, 1
; CHECK:   getelementptr inbounds %pair.i16, %pair.i16* %p, i64 %[[i0]], i32 1
; CHECK:   getelementptr inbounds %pair.i16, %pair.i16* %p, i64 %[[i1]], i32 1
; CHECK:   %index.next = add i64 %index, 2
; CHECK:   %vec.ind.next = add <2 x i64> %vec.ind, <i64 2, i64 2>
; CHECK:   %vec.ind.next2 = add <2 x i32> %vec.ind1, <i32 2, i32 2>
;
; IND-LABEL: @iv_vector_and_scalar_users(
; IND: vector.body:
; IND:   %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; IND:   %vec.ind1 = phi <2 x i32> [ <i32 0, i32 1>, %vector.ph ], [ %vec.ind.next2, %vector.body ]
; IND:   %[[i1:.+]] = or i64 %index, 1
; IND:   getelementptr inbounds %pair.i16, %pair.i16* %p, i64 %index, i32 1
; IND:   getelementptr inbounds %pair.i16, %pair.i16* %p, i64 %[[i1]], i32 1
; IND:   %index.next = add i64 %index, 2
; IND:   %vec.ind.next2 = add <2 x i32> %vec.ind1, <i32 2, i32 2>
;
; UNROLL-LABEL: @iv_vector_and_scalar_users(
; UNROLL: vector.body:
; UNROLL:   %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; UNROLL:   %vec.ind2 = phi <2 x i32> [ <i32 0, i32 1>, %vector.ph ], [ %vec.ind.next5, %vector.body ]
; UNROLL:   %[[i1:.+]] = or i64 %index, 1
; UNROLL:   %[[i2:.+]] = or i64 %index, 2
; UNROLL:   %[[i3:.+]] = or i64 %index, 3
; UNROLL:   %[[add:.+]]= add <2 x i32> %[[splat:.+]], <i32 2, i32 poison>
; UNROLL:   getelementptr inbounds %pair.i16, %pair.i16* %p, i64 %index, i32 1
; UNROLL:   getelementptr inbounds %pair.i16, %pair.i16* %p, i64 %[[i1]], i32 1
; UNROLL:   getelementptr inbounds %pair.i16, %pair.i16* %p, i64 %[[i2]], i32 1
; UNROLL:   getelementptr inbounds %pair.i16, %pair.i16* %p, i64 %[[i3]], i32 1
; UNROLL:   %index.next = add i64 %index, 4
; UNROLL:   %vec.ind.next5 = add <2 x i32> %vec.ind2, <i32 4, i32 4>

%pair.i16 = type { i16, i16 }
define void @iv_vector_and_scalar_users(%pair.i16* %p, i32 %a, i32 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %0 = trunc i64 %i to i32
  %1 = add i32 %a, %0
  %2 = trunc i32 %1 to i16
  %3 = getelementptr inbounds %pair.i16, %pair.i16* %p, i64 %i, i32 1
  store i16 %2, i16* %3, align 2
  %i.next = add nuw nsw i64 %i, 1
  %4 = trunc i64 %i.next to i32
  %cond = icmp eq i32 %4, %n
  br i1 %cond, label %for.end, label %for.body

for.end:
  ret void
}

; Make sure that the loop exit count computation does not overflow for i8 and
; i16. The exit count of these loops is i8/i16 max + 1. If we don't cast the
; induction variable to a bigger type the exit count computation will overflow
; to 0.
; PR17532

; CHECK-LABEL: i8_loop
; CHECK: icmp eq i32 {{.*}}, 256
define i32 @i8_loop() nounwind readnone ssp uwtable {
  br label %1

; <label>:1                                       ; preds = %1, %0
  %a.0 = phi i32 [ 1, %0 ], [ %2, %1 ]
  %b.0 = phi i8 [ 0, %0 ], [ %3, %1 ]
  %2 = and i32 %a.0, 4
  %3 = add i8 %b.0, -1
  %4 = icmp eq i8 %3, 0
  br i1 %4, label %5, label %1

; <label>:5                                       ; preds = %1
  ret i32 %2
}

; CHECK-LABEL: i16_loop
; CHECK: icmp eq i32 {{.*}}, 65536

define i32 @i16_loop() nounwind readnone ssp uwtable {
  br label %1

; <label>:1                                       ; preds = %1, %0
  %a.0 = phi i32 [ 1, %0 ], [ %2, %1 ]
  %b.0 = phi i16 [ 0, %0 ], [ %3, %1 ]
  %2 = and i32 %a.0, 4
  %3 = add i16 %b.0, -1
  %4 = icmp eq i16 %3, 0
  br i1 %4, label %5, label %1

; <label>:5                                       ; preds = %1
  ret i32 %2
}

; This loop has a backedge taken count of i32_max. We need to check for this
; condition and branch directly to the scalar loop.

; CHECK-LABEL: max_i32_backedgetaken
; CHECK:  br i1 true, label %scalar.ph, label %vector.ph

; CHECK: middle.block:
; CHECK:  %[[v9:.+]] = call i32 @llvm.vector.reduce.and.v2i32(<2 x i32>
; CHECK: scalar.ph:
; CHECK:  %bc.resume.val = phi i32 [ 0, %middle.block ], [ 0, %[[v0:.+]] ]
; CHECK:  %bc.merge.rdx = phi i32 [ 1, %[[v0:.+]] ], [ %[[v9]], %middle.block ]

define i32 @max_i32_backedgetaken() nounwind readnone ssp uwtable {

  br label %1

; <label>:1                                       ; preds = %1, %0
  %a.0 = phi i32 [ 1, %0 ], [ %2, %1 ]
  %b.0 = phi i32 [ 0, %0 ], [ %3, %1 ]
  %2 = and i32 %a.0, 4
  %3 = add i32 %b.0, -1
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %5, label %1

; <label>:5                                       ; preds = %1
  ret i32 %2
}

; When generating the overflow check we must sure that the induction start value
; is defined before the branch to the scalar preheader.

; CHECK-LABEL: testoverflowcheck
; CHECK: entry
; CHECK: %[[LOAD:.*]] = load i8
; CHECK: br

; CHECK: scalar.ph
; CHECK: phi i8 [ %{{.*}}, %middle.block ], [ %[[LOAD]], %entry ]

@e = global i8 1, align 1
@d = common global i32 0, align 4
@c = common global i32 0, align 4
define i32 @testoverflowcheck() {
entry:
  %.pr.i = load i8, i8* @e, align 1
  %0 = load i32, i32* @d, align 4
  %c.promoted.i = load i32, i32* @c, align 4
  br label %cond.end.i

cond.end.i:
  %inc4.i = phi i8 [ %.pr.i, %entry ], [ %inc.i, %cond.end.i ]
  %and3.i = phi i32 [ %c.promoted.i, %entry ], [ %and.i, %cond.end.i ]
  %and.i = and i32 %0, %and3.i
  %inc.i = add i8 %inc4.i, 1
  %tobool.i = icmp eq i8 %inc.i, 0
  br i1 %tobool.i, label %loopexit, label %cond.end.i

loopexit:
  ret i32 %and.i
}

; The SCEV expression of %sphi is (zext i8 {%t,+,1}<%loop> to i32)
; In order to recognize %sphi as an induction PHI and vectorize this loop,
; we need to convert the SCEV expression into an AddRecExpr.
; The expression gets converted to {zext i8 %t to i32,+,1}.

; CHECK-LABEL: wrappingindvars1
; CHECK-LABEL: vector.scevcheck
; CHECK-LABEL: vector.ph
; CHECK: %[[START:.*]] = add <2 x i32> %{{.*}}, <i32 0, i32 1>
; CHECK-LABEL: vector.body
; CHECK: %[[PHI:.*]] = phi <2 x i32> [ %[[START]], %vector.ph ], [ %[[STEP:.*]], %vector.body ]
; CHECK: %[[STEP]] = add <2 x i32> %[[PHI]], <i32 2, i32 2>
define void @wrappingindvars1(i8 %t, i32 %len, i32 *%A) {
 entry:
  %st = zext i8 %t to i16
  %ext = zext i8 %t to i32
  %ecmp = icmp ult i16 %st, 42
  br i1 %ecmp, label %loop, label %exit

 loop:

  %idx = phi i8 [ %t, %entry ], [ %idx.inc, %loop ]
  %idx.b = phi i32 [ 0, %entry ], [ %idx.b.inc, %loop ]
  %sphi = phi i32 [ %ext, %entry ], [%idx.inc.ext, %loop]

  %ptr = getelementptr inbounds i32, i32* %A, i8 %idx
  store i32 %sphi, i32* %ptr

  %idx.inc = add i8 %idx, 1
  %idx.inc.ext = zext i8 %idx.inc to i32
  %idx.b.inc = add nuw nsw i32 %idx.b, 1

  %c = icmp ult i32 %idx.b, %len
  br i1 %c, label %loop, label %exit

 exit:
  ret void
}

; The SCEV expression of %sphi is (4 * (zext i8 {%t,+,1}<%loop> to i32))
; In order to recognize %sphi as an induction PHI and vectorize this loop,
; we need to convert the SCEV expression into an AddRecExpr.
; The expression gets converted to ({4 * (zext %t to i32),+,4}).
; CHECK-LABEL: wrappingindvars2
; CHECK-LABEL: vector.scevcheck
; CHECK-LABEL: vector.ph
; CHECK: %[[START:.*]] = add <2 x i32> %{{.*}}, <i32 0, i32 4>
; CHECK-LABEL: vector.body
; CHECK: %[[PHI:.*]] = phi <2 x i32> [ %[[START]], %vector.ph ], [ %[[STEP:.*]], %vector.body ]
; CHECK: %[[STEP]] = add <2 x i32> %[[PHI]], <i32 8, i32 8>
define void @wrappingindvars2(i8 %t, i32 %len, i32 *%A) {

entry:
  %st = zext i8 %t to i16
  %ext = zext i8 %t to i32
  %ext.mul = mul i32 %ext, 4

  %ecmp = icmp ult i16 %st, 42
  br i1 %ecmp, label %loop, label %exit

 loop:

  %idx = phi i8 [ %t, %entry ], [ %idx.inc, %loop ]
  %sphi = phi i32 [ %ext.mul, %entry ], [%mul, %loop]
  %idx.b = phi i32 [ 0, %entry ], [ %idx.b.inc, %loop ]

  %ptr = getelementptr inbounds i32, i32* %A, i8 %idx
  store i32 %sphi, i32* %ptr

  %idx.inc = add i8 %idx, 1
  %idx.inc.ext = zext i8 %idx.inc to i32
  %mul = mul i32 %idx.inc.ext, 4
  %idx.b.inc = add nuw nsw i32 %idx.b, 1

  %c = icmp ult i32 %idx.b, %len
  br i1 %c, label %loop, label %exit

 exit:
  ret void
}

; Check that we generate vectorized IVs in the pre-header
; instead of widening the scalar IV inside the loop, when
; we know how to do that.
; IND-LABEL: veciv
; IND: vector.body:
; IND: %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; IND: %vec.ind = phi <2 x i32> [ <i32 0, i32 1>, %vector.ph ], [ %vec.ind.next, %vector.body ]
; IND: %index.next = add i32 %index, 2
; IND: %vec.ind.next = add <2 x i32> %vec.ind, <i32 2, i32 2>
; IND: %[[CMP:.*]] = icmp eq i32 %index.next
; IND: br i1 %[[CMP]]
; UNROLL-LABEL: veciv
; UNROLL: vector.body:
; UNROLL: %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; UNROLL: %vec.ind = phi <2 x i32> [ <i32 0, i32 1>, %vector.ph ], [ %vec.ind.next, %vector.body ]
; UNROLL: %step.add = add <2 x i32> %vec.ind, <i32 2, i32 2>
; UNROLL: %index.next = add i32 %index, 4
; UNROLL: %vec.ind.next = add <2 x i32> %vec.ind, <i32 4, i32 4>
; UNROLL: %[[CMP:.*]] = icmp eq i32 %index.next
; UNROLL: br i1 %[[CMP]]
define void @veciv(i32* nocapture %a, i32 %start, i32 %k) {
for.body.preheader:
  br label %for.body

for.body:
  %indvars.iv = phi i32 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i32 %indvars.iv
  store i32 %indvars.iv, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i32 %indvars.iv, 1
  %exitcond = icmp eq i32 %indvars.iv.next, %k
  br i1 %exitcond, label %exit, label %for.body

exit:
  ret void
}

; IND-LABEL: trunciv
; IND: vector.body:
; IND: %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; IND: %[[VECIND:.*]] = phi <2 x i32> [ <i32 0, i32 1>, %vector.ph ], [ %[[STEPADD:.*]], %vector.body ]
; IND: %index.next = add i64 %index, 2
; IND: %[[STEPADD]] = add <2 x i32> %[[VECIND]], <i32 2, i32 2>
; IND: %[[CMP:.*]] = icmp eq i64 %index.next
; IND: br i1 %[[CMP]]
define void @trunciv(i32* nocapture %a, i32 %start, i64 %k) {
for.body.preheader:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %trunc.iv = trunc i64 %indvars.iv to i32
  %arrayidx = getelementptr inbounds i32, i32* %a, i32 %trunc.iv
  store i32 %trunc.iv, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %k
  br i1 %exitcond, label %exit, label %for.body

exit:
  ret void
}

; CHECK-LABEL: @nonprimary(
; CHECK: vector.ph:
; CHECK:   %[[INSERT:.*]] = insertelement <2 x i32> poison, i32 %i, i32 0
; CHECK:   %[[SPLAT:.*]] = shufflevector <2 x i32> %[[INSERT]], <2 x i32> poison, <2 x i32> zeroinitializer
; CHECK:   %[[START:.*]] = add <2 x i32> %[[SPLAT]], <i32 0, i32 1>
; CHECK: vector.body:
; CHECK:   %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK:   %vec.ind = phi <2 x i32> [ %[[START]], %vector.ph ], [ %vec.ind.next, %vector.body ]
; CHECK:   %offset.idx = add i32 %i, %index
; CHECK:   %[[A1:.*]] = add i32 %offset.idx, 0
; CHECK:   %[[G1:.*]] = getelementptr inbounds i32, i32* %a, i32 %[[A1]]
; CHECK:   %[[G3:.*]] = getelementptr inbounds i32, i32* %[[G1]], i32 0
; CHECK:   %[[B1:.*]] = bitcast i32* %[[G3]] to <2 x i32>*
; CHECK:   store <2 x i32> %vec.ind, <2 x i32>* %[[B1]]
; CHECK:   %index.next = add i32 %index, 2
; CHECK:   %vec.ind.next = add <2 x i32> %vec.ind, <i32 2, i32 2>
; CHECK:   %[[CMP:.*]] = icmp eq i32 %index.next, %n.vec
; CHECK:   br i1 %[[CMP]]
;
; IND-LABEL: @nonprimary(
; IND: vector.ph:
; IND:   %[[INSERT:.*]] = insertelement <2 x i32> poison, i32 %i, i32 0
; IND:   %[[SPLAT:.*]] = shufflevector <2 x i32> %[[INSERT]], <2 x i32> poison, <2 x i32> zeroinitializer
; IND:   %[[START:.*]] = add <2 x i32> %[[SPLAT]], <i32 0, i32 1>
; IND: vector.body:
; IND:   %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; IND:   %vec.ind = phi <2 x i32> [ %[[START]], %vector.ph ], [ %vec.ind.next, %vector.body ]
; IND:   %[[A1:.*]] = add i32 %index, %i
; IND:   %[[S1:.*]] = sext i32 %[[A1]] to i64
; IND:   %[[G1:.*]] = getelementptr inbounds i32, i32* %a, i64 %[[S1]]
; IND:   %[[B1:.*]] = bitcast i32* %[[G1]] to <2 x i32>*
; IND:   store <2 x i32> %vec.ind, <2 x i32>* %[[B1]]
; IND:   %index.next = add i32 %index, 2
; IND:   %vec.ind.next = add <2 x i32> %vec.ind, <i32 2, i32 2>
; IND:   %[[CMP:.*]] = icmp eq i32 %index.next, %n.vec
; IND:   br i1 %[[CMP]]
;
; UNROLL-LABEL: @nonprimary(
; UNROLL: vector.ph:
; UNROLL:   %[[INSERT:.*]] = insertelement <2 x i32> poison, i32 %i, i32 0
; UNROLL:   %[[SPLAT:.*]] = shufflevector <2 x i32> %[[INSERT]], <2 x i32> poison, <2 x i32> zeroinitializer
; UNROLL:   %[[START:.*]] = add <2 x i32> %[[SPLAT]], <i32 0, i32 1>
; UNROLL: vector.body:
; UNROLL:   %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; UNROLL:   %vec.ind = phi <2 x i32> [ %[[START]], %vector.ph ], [ %vec.ind.next, %vector.body ]
; UNROLL:   %step.add = add <2 x i32> %vec.ind, <i32 2, i32 2>
; UNROLL:   %[[A1:.*]] = add i32 %index, %i
; UNROLL:   %[[S1:.*]] = sext i32 %[[A1]] to i64
; UNROLL:   %[[G1:.*]] = getelementptr inbounds i32, i32* %a, i64 %[[S1]]
; UNROLL:   %[[B1:.*]] = bitcast i32* %[[G1]] to <2 x i32>*
; UNROLL:   store <2 x i32> %vec.ind, <2 x i32>* %[[B1]]
; UNROLL:   %[[G2:.*]] = getelementptr inbounds i32, i32* %[[G1]], i64 2
; UNROLL:   %[[B2:.*]] = bitcast i32* %[[G2]] to <2 x i32>*
; UNROLL:   store <2 x i32> %step.add, <2 x i32>* %[[B2]]
; UNROLL:   %index.next = add i32 %index, 4
; UNROLL:   %vec.ind.next = add <2 x i32> %vec.ind, <i32 4, i32 4>
; UNROLL:   %[[CMP:.*]] = icmp eq i32 %index.next, %n.vec
; UNROLL:   br i1 %[[CMP]]
define void @nonprimary(i32* nocapture %a, i32 %start, i32 %i, i32 %k) {
for.body.preheader:
  br label %for.body

for.body:
  %indvars.iv = phi i32 [ %indvars.iv.next, %for.body ], [ %i, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i32 %indvars.iv
  store i32 %indvars.iv, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i32 %indvars.iv, 1
  %exitcond = icmp eq i32 %indvars.iv.next, %k
  br i1 %exitcond, label %exit, label %for.body

exit:
  ret void
}

; CHECK-LABEL: @non_primary_iv_trunc(
; CHECK:       vector.body:
; CHECK-NEXT:    %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK:         [[VEC_IND:%.*]] = phi <2 x i32> [ <i32 0, i32 2>, %vector.ph ], [ [[VEC_IND_NEXT:%.*]], %vector.body ]
; CHECK:         [[TMP3:%.*]] = add i64 %index, 0
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr inbounds i32, i32* %a, i64 [[TMP3]]
; CHECK-NEXT:    [[TMP5:%.*]] = getelementptr inbounds i32, i32* [[TMP4]], i32 0
; CHECK-NEXT:    [[TMP6:%.*]] = bitcast i32* [[TMP5]] to <2 x i32>*
; CHECK-NEXT:    store <2 x i32> [[VEC_IND]], <2 x i32>* [[TMP6]], align 4
; CHECK-NEXT:    %index.next = add i64 %index, 2
; CHECK:         [[VEC_IND_NEXT]] = add <2 x i32> [[VEC_IND]], <i32 4, i32 4>
; CHECK:         br i1 {{.*}}, label %middle.block, label %vector.body
define void @non_primary_iv_trunc(i32* %a, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %j = phi i64 [ %j.next, %for.body ], [ 0, %entry ]
  %tmp0 = getelementptr inbounds i32, i32* %a, i64 %i
  %tmp1 = trunc i64 %j to i32
  store i32 %tmp1, i32* %tmp0, align 4
  %i.next = add nuw nsw i64 %i, 1
  %j.next = add nuw nsw i64 %j, 2
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}

; PR32419. Ensure we transform truncated non-primary induction variables. In
; the test case below we replace %tmp1 with a new induction variable. Because
; the truncated value is non-primary, we must compute an offset from the
; primary induction variable.
;
; CHECK-LABEL: @PR32419(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i32 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %[[PRED_UREM_CONTINUE4:.*]] ]
; CHECK:         [[OFFSET_IDX:%.*]] = add i32 -20, [[INDEX]]
; CHECK-NEXT:    [[TMP1:%.*]] = trunc i32 [[OFFSET_IDX]] to i16
; CHECK:         [[TMP8:%.*]] = add i16 [[TMP1]], 0
; CHECK-NEXT:    [[TMP9:%.*]] = urem i16 %b, [[TMP8]]
; CHECK:         [[TMP15:%.*]] = add i16 [[TMP1]], 1
; CHECK-NEXT:    [[TMP16:%.*]] = urem i16 %b, [[TMP15]]
; CHECK:       [[PRED_UREM_CONTINUE4]]:
; CHECK:         br i1 {{.*}}, label %middle.block, label %vector.body
;
define i32 @PR32419(i32 %a, i16 %b) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [ -20, %entry ], [ %i.next, %for.inc ]
  %tmp0 = phi i32 [ %a, %entry ], [ %tmp6, %for.inc ]
  %tmp1 = trunc i32 %i to i16
  %tmp2 = icmp eq i16 %tmp1, 0
  br i1 %tmp2, label %for.inc, label %for.cond

for.cond:
  %tmp3 = urem i16 %b, %tmp1
  br label %for.inc

for.inc:
  %tmp4 = phi i16 [ %tmp3, %for.cond ], [ 0, %for.body ]
  %tmp5 = sext i16 %tmp4 to i32
  %tmp6 = or i32 %tmp0, %tmp5
  %i.next = add nsw i32 %i, 1
  %cond = icmp eq i32 %i.next, 0
  br i1 %cond, label %for.end, label %for.body

for.end:
  %tmp7 = phi i32 [ %tmp6, %for.inc ]
  ret i32 %tmp7
}

; Ensure that the shuffle vector for first order recurrence is inserted
; correctly after all the phis. These new phis correspond to new IVs 
; that are generated by optimizing non-free truncs of IVs to IVs themselves 
define i64 @trunc_with_first_order_recurrence() {
; CHECK-LABEL: trunc_with_first_order_recurrence
; CHECK-LABEL: vector.body:
; CHECK-NEXT:    %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK-NEXT:    %vec.phi = phi <2 x i64>
; CHECK-NEXT:    %vec.ind = phi <2 x i64> [ <i64 1, i64 2>, %vector.ph ], [ %vec.ind.next, %vector.body ]
; CHECK-NEXT:    %vec.ind2 = phi <2 x i32> [ <i32 1, i32 2>, %vector.ph ], [ %vec.ind.next3, %vector.body ]
; CHECK-NEXT:    %vector.recur = phi <2 x i32> [ <i32 poison, i32 42>, %vector.ph ], [ %vec.ind5, %vector.body ]
; CHECK-NEXT:    %vec.ind5 = phi <2 x i32> [ <i32 1, i32 2>, %vector.ph ], [ %vec.ind.next6, %vector.body ]
; CHECK-NEXT:    %vec.ind7 = phi <2 x i32> [ <i32 1, i32 2>, %vector.ph ], [ %vec.ind.next8, %vector.body ]
; CHECK-NEXT:    shufflevector <2 x i32> %vector.recur, <2 x i32> %vec.ind5, <2 x i32> <i32 1, i32 2>
entry:
  br label %loop

exit:                                        ; preds = %loop
  %.lcssa = phi i64 [ %c23, %loop ]
  ret i64 %.lcssa

loop:                                         ; preds = %loop, %entry
  %c5 = phi i64 [ %c23, %loop ], [ 0, %entry ]
  %indvars.iv = phi i64 [ %indvars.iv.next, %loop ], [ 1, %entry ]
  %x = phi i32 [ %c24, %loop ], [ 1, %entry ]
  %y = phi i32 [ %c6, %loop ], [ 42, %entry ]
  %c6 = trunc i64 %indvars.iv to i32
  %c8 = mul i32 %x, %c6
  %c9 = add i32 %c8, 42
  %c10 = add i32 %y, %c6
  %c11 = add i32 %c10, %c9
  %c12 = sext i32 %c11 to i64
  %c13 = add i64 %c5, %c12
  %indvars.iv.tr = trunc i64 %indvars.iv to i32
  %c14 = shl i32 %indvars.iv.tr, 1
  %c15 = add i32 %c9, %c14
  %c16 = sext i32 %c15 to i64
  %c23 = add i64 %c13, %c16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %c24 = add nuw nsw i32 %x, 1
  %exitcond.i = icmp eq i64 %indvars.iv.next, 114
  br i1 %exitcond.i, label %exit, label %loop

}
