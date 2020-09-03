; RUN: opt -S -loop-vectorize -force-vector-width=4 -force-vector-interleave=1 < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

; Float pattern:
;   Check vectorization of reduction code which has an fadd instruction after
;   an fcmp instruction which compares an array element and 0.
;
; float fcmp_0_fadd_select1(float * restrict x, const int N) {
;   float sum = 0.
;   for (int i = 0; i < N; ++i)
;     if (x[i] > (float)0.)
;       sum += x[i];
;   return sum;
; }

; CHECK-LABEL: @fcmp_0_fadd_select1(
; CHECK: %[[V1:.*]] = fcmp fast ogt <4 x float> %[[V0:.*]], zeroinitializer
; CHECK: %[[V3:.*]] = fadd fast <4 x float> %[[V0]], %[[V2:.*]]
; CHECK: select <4 x i1> %[[V1]], <4 x float> %[[V3]], <4 x float> %[[V2]]
define float @fcmp_0_fadd_select1(float* noalias %x, i32 %N) nounwind readonly {
entry:
  %cmp.1 = icmp sgt i32 %N, 0
  br i1 %cmp.1, label %for.header, label %for.end

for.header:                                       ; preds = %entry
  %zext = zext i32 %N to i64
  br label %for.body

for.body:                                         ; preds = %header, %for.body
  %indvars.iv = phi i64 [ 0, %for.header ], [ %indvars.iv.next, %for.body ]
  %sum.1 = phi float [ 0.000000e+00, %for.header ], [ %sum.2, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %x, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %cmp.2 = fcmp fast ogt float %0, 0.000000e+00
  %add = fadd fast float %0, %sum.1
  %sum.2 = select i1 %cmp.2, float %add, float %sum.1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %zext
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %1 = phi float [ 0.000000e+00, %entry ], [ %sum.2, %for.body ]
  ret float %1
}

; Double pattern:
;   Check vectorization of reduction code which has an fadd instruction after
;   an fcmp instruction which compares an array element and 0.
;
; double fcmp_0_fadd_select2(double * restrict x, const int N) {
;   double sum = 0.
;   for (int i = 0; i < N; ++i)
;     if (x[i] > 0.)
;       sum += x[i];
;   return sum;
; }

; CHECK-LABEL: @fcmp_0_fadd_select2(
; CHECK: %[[V1:.*]] = fcmp fast ogt <4 x double> %[[V0:.*]], zeroinitializer
; CHECK: %[[V3:.*]] = fadd fast <4 x double> %[[V0]], %[[V2:.*]]
; CHECK: select <4 x i1> %[[V1]], <4 x double> %[[V3]], <4 x double> %[[V2]]
define double @fcmp_0_fadd_select2(double* noalias %x, i32 %N) nounwind readonly {
entry:
  %cmp.1 = icmp sgt i32 %N, 0
  br i1 %cmp.1, label %for.header, label %for.end

for.header:                                       ; preds = %entry
  %zext = zext i32 %N to i64
  br label %for.body

for.body:                                         ; preds = %header, %for.body
  %indvars.iv = phi i64 [ 0, %for.header ], [ %indvars.iv.next, %for.body ]
  %sum.1 = phi double [ 0.000000e+00, %for.header ], [ %sum.2, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %x, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 4
  %cmp.2 = fcmp fast ogt double %0, 0.000000e+00
  %add = fadd fast double %0, %sum.1
  %sum.2 = select i1 %cmp.2, double %add, double %sum.1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %zext
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %1 = phi double [ 0.000000e+00, %entry ], [ %sum.2, %for.body ]
  ret double %1
}

; Float pattern:
;   Check vectorization of reduction code which has an fadd instruction after
;   an fcmp instruction which compares an array element and a floating-point
;   value.
;
; float fcmp_val_fadd_select1(float * restrict x, float y, const int N) {
;   float sum = 0.
;   for (int i = 0; i < N; ++i)
;     if (x[i] > y)
;       sum += x[i];
;   return sum;
; }

; CHECK-LABEL: @fcmp_val_fadd_select1(
; CHECK: %[[V1:.*]] = fcmp fast ogt <4 x float> %[[V0:.*]], %broadcast.splat
; CHECK: %[[V3:.*]] = fadd fast <4 x float> %[[V0]], %[[V2:.*]]
; CHECK: select <4 x i1> %[[V1]], <4 x float> %[[V3]], <4 x float> %[[V2]]
define float @fcmp_val_fadd_select1(float* noalias %x, float %y, i32 %N) nounwind readonly {
entry:
  %cmp.1 = icmp sgt i32 %N, 0
  br i1 %cmp.1, label %for.header, label %for.end

for.header:                                       ; preds = %entry
  %zext = zext i32 %N to i64
  br label %for.body

for.body:                                         ; preds = %header, %for.body
  %indvars.iv = phi i64 [ 0, %for.header ], [ %indvars.iv.next, %for.body ]
  %sum.1 = phi float [ 0.000000e+00, %for.header ], [ %sum.2, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %x, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %cmp.2 = fcmp fast ogt float %0, %y
  %add = fadd fast float %0, %sum.1
  %sum.2 = select i1 %cmp.2, float %add, float %sum.1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %zext
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %1 = phi float [ 0.000000e+00, %entry ], [ %sum.2, %for.body ]
  ret float %1
}

; Double pattern:
;   Check vectorization of reduction code which has an fadd instruction after
;   an fcmp instruction which compares an array element and a floating-point
;   value.
;
; double fcmp_val_fadd_select2(double * restrict x, double y, const int N) {
;   double sum = 0.
;   for (int i = 0; i < N; ++i)
;     if (x[i] > y)
;       sum += x[i];
;   return sum;
; }

; CHECK-LABEL: @fcmp_val_fadd_select2(
; CHECK: %[[V1:.*]] = fcmp fast ogt <4 x double> %[[V0:.*]], %broadcast.splat
; CHECK: %[[V3:.*]] = fadd fast <4 x double> %[[V0]], %[[V2:.*]]
; CHECK: select <4 x i1> %[[V1]], <4 x double> %[[V3]], <4 x double> %[[V2]]
define double @fcmp_val_fadd_select2(double* noalias %x, double %y, i32 %N) nounwind readonly {
entry:
  %cmp.1 = icmp sgt i32 %N, 0
  br i1 %cmp.1, label %for.header, label %for.end

for.header:                                       ; preds = %entry
  %zext = zext i32 %N to i64
  br label %for.body

for.body:                                         ; preds = %header, %for.body
  %indvars.iv = phi i64 [ 0, %for.header ], [ %indvars.iv.next, %for.body ]
  %sum.1 = phi double [ 0.000000e+00, %for.header ], [ %sum.2, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %x, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 4
  %cmp.2 = fcmp fast ogt double %0, %y
  %add = fadd fast double %0, %sum.1
  %sum.2 = select i1 %cmp.2, double %add, double %sum.1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %zext
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %1 = phi double [ 0.000000e+00, %entry ], [ %sum.2, %for.body ]
  ret double %1
}

; Float pattern:
;   Check vectorization of reduction code which has an fadd instruction after
;   an fcmp instruction which compares an array element and another array
;   element.
;
; float fcmp_array_elm_fadd_select1(float * restrict x, float * restrict y,
;                                   const int N) {
;   float sum = 0.
;   for (int i = 0; i < N; ++i)
;     if (x[i] > y[i])
;       sum += x[i];
;   return sum;
; }

; CHECK-LABEL: @fcmp_array_elm_fadd_select1(
; CHECK: %[[V2:.*]] = fcmp fast ogt <4 x float> %[[V0:.*]], %[[V1:.*]]
; CHECK: %[[V4:.*]] = fadd fast <4 x float> %[[V0]], %[[V3:.*]]
; CHECK: select <4 x i1> %[[V2]], <4 x float> %[[V4]], <4 x float> %[[V3]]
define float @fcmp_array_elm_fadd_select1(float* noalias %x, float* noalias %y, i32 %N) nounwind readonly {
entry:
  %cmp.1 = icmp sgt i32 %N, 0
  br i1 %cmp.1, label %for.header, label %for.end

for.header:                                       ; preds = %entry
  %zext = zext i32 %N to i64
  br label %for.body

for.body:                                         ; preds = %for.body, %for.header
  %indvars.iv = phi i64 [ 0, %for.header ], [ %indvars.iv.next, %for.body ]
  %sum.1 = phi float [ 0.000000e+00, %for.header ], [ %sum.2, %for.body ]
  %arrayidx.1 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  %0 = load float, float* %arrayidx.1, align 4
  %arrayidx.2 = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %1 = load float, float* %arrayidx.2, align 4
  %cmp.2 = fcmp fast ogt float %0, %1
  %add = fadd fast float %0, %sum.1
  %sum.2 = select i1 %cmp.2, float %add, float %sum.1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %zext
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %2 = phi float [ 0.000000e+00, %entry ], [ %sum.2, %for.body ]
  ret float %2
}

; Double pattern:
;   Check vectorization of reduction code which has an fadd instruction after
;   an fcmp instruction which compares an array element and another array
;   element.
;
; double fcmp_array_elm_fadd_select2(double * restrict x, double * restrict y,
;                                    const int N) {
;   double sum = 0.
;   for (int i = 0; i < N; ++i)
;     if (x[i] > y[i])
;       sum += x[i];
;   return sum;
; }

; CHECK-LABEL: @fcmp_array_elm_fadd_select2(
; CHECK: %[[V2:.*]] = fcmp fast ogt <4 x double> %[[V0:.*]], %[[V1:.*]]
; CHECK: %[[V4:.*]] = fadd fast <4 x double> %[[V0]], %[[V3:.*]]
; CHECK: select <4 x i1> %[[V2]], <4 x double> %[[V4]], <4 x double> %[[V3]]
define double @fcmp_array_elm_fadd_select2(double* noalias %x, double* noalias %y, i32 %N) nounwind readonly {
entry:
  %cmp.1 = icmp sgt i32 %N, 0
  br i1 %cmp.1, label %for.header, label %for.end

for.header:                                       ; preds = %entry
  %zext = zext i32 %N to i64
  br label %for.body

for.body:                                         ; preds = %for.body, %for.header
  %indvars.iv = phi i64 [ 0, %for.header ], [ %indvars.iv.next, %for.body ]
  %sum.1 = phi double [ 0.000000e+00, %for.header ], [ %sum.2, %for.body ]
  %arrayidx.1 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  %0 = load double, double* %arrayidx.1, align 4
  %arrayidx.2 = getelementptr inbounds double, double* %y, i64 %indvars.iv
  %1 = load double, double* %arrayidx.2, align 4
  %cmp.2 = fcmp fast ogt double %0, %1
  %add = fadd fast double %0, %sum.1
  %sum.2 = select i1 %cmp.2, double %add, double %sum.1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %zext
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %2 = phi double [ 0.000000e+00, %entry ], [ %sum.2, %for.body ]
  ret double %2
}

; Float pattern:
;   Check vectorization of reduction code which has an fsub instruction after
;   an fcmp instruction which compares an array element and 0.
;
; float fcmp_0_fsub_select1(float * restrict x, const int N) {
;   float sum = 0.
;   for (int i = 0; i < N; ++i)
;     if (x[i] > (float)0.)
;       sum -= x[i];
;   return sum;
; }

; CHECK-LABEL: @fcmp_0_fsub_select1(
; CHECK: %[[V1:.*]] = fcmp fast ogt <4 x float> %[[V0:.*]], zeroinitializer
; CHECK: %[[V3:.*]] = fsub fast <4 x float> %[[V2:.*]], %[[V0]]
; CHECK: select <4 x i1> %[[V1]], <4 x float> %[[V3]], <4 x float> %[[V2]]
define float @fcmp_0_fsub_select1(float* noalias %x, i32 %N) nounwind readonly {
entry:
  %cmp.1 = icmp sgt i32 %N, 0
  br i1 %cmp.1, label %for.header, label %for.end

for.header:                                       ; preds = %entry
  %zext = zext i32 %N to i64
  br label %for.body

for.body:                                         ; preds = %for.body, %for.header
  %indvars.iv = phi i64 [ 0, %for.header ], [ %indvars.iv.next, %for.body ]
  %sum.1 = phi float [ 0.000000e+00, %for.header ], [ %sum.2, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %x, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %cmp.2 = fcmp fast ogt float %0, 0.000000e+00
  %sub = fsub fast float %sum.1, %0
  %sum.2 = select i1 %cmp.2, float %sub, float %sum.1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %zext
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %1 = phi float [ 0.000000e+00, %entry ], [ %sum.2, %for.body ]
  ret float %1
}

; Float pattern:
;   Check that is not vectorized if fp-instruction has no fast-math property.
; float fcmp_0_fsub_select1_novectorize(float * restrict x, const int N) {
;   float sum = 0.
;   for (int i = 0; i < N; ++i)
;     if (x[i] > (float)0.)
;       sum -= x[i];
;   return sum;
; }

; CHECK-LABEL: @fcmp_0_fsub_select1_novectorize(
; CHECK-NOT: <4 x float>
define float @fcmp_0_fsub_select1_novectorize(float* noalias %x, i32 %N) nounwind readonly {
entry:
  %cmp.1 = icmp sgt i32 %N, 0
  br i1 %cmp.1, label %for.header, label %for.end

for.header:                                       ; preds = %entry
  %zext = zext i32 %N to i64
  br label %for.body

for.body:                                         ; preds = %for.body, %for.header
  %indvars.iv = phi i64 [ 0, %for.header ], [ %indvars.iv.next, %for.body ]
  %sum.1 = phi float [ 0.000000e+00, %for.header ], [ %sum.2, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %x, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %cmp.2 = fcmp ogt float %0, 0.000000e+00
  %sub = fsub float %sum.1, %0
  %sum.2 = select i1 %cmp.2, float %sub, float %sum.1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %zext
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %1 = phi float [ 0.000000e+00, %entry ], [ %sum.2, %for.body ]
  ret float %1
}

; Double pattern:
;   Check vectorization of reduction code which has an fsub instruction after
;   an fcmp instruction which compares an array element and 0.
;
; double fcmp_0_fsub_select2(double * restrict x, const int N) {
;   double sum = 0.
;   for (int i = 0; i < N; ++i)
;     if (x[i] > 0.)
;       sum -= x[i];
;   return sum;
; }

; CHECK-LABEL: @fcmp_0_fsub_select2(
; CHECK: %[[V1:.*]] = fcmp fast ogt <4 x double> %[[V0:.*]], zeroinitializer
; CHECK: %[[V3:.*]] = fsub fast <4 x double> %[[V2:.*]], %[[V0]]
; CHECK: select <4 x i1> %[[V1]], <4 x double> %[[V3]], <4 x double> %[[V2]]
define double @fcmp_0_fsub_select2(double* noalias %x, i32 %N) nounwind readonly {
entry:
  %cmp.1 = icmp sgt i32 %N, 0
  br i1 %cmp.1, label %for.header, label %for.end

for.header:                                       ; preds = %entry
  %zext = zext i32 %N to i64
  br label %for.body

for.body:                                         ; preds = %for.body, %for.header
  %indvars.iv = phi i64 [ 0, %for.header ], [ %indvars.iv.next, %for.body ]
  %sum.1 = phi double [ 0.000000e+00, %for.header ], [ %sum.2, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %x, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 4
  %cmp.2 = fcmp fast ogt double %0, 0.000000e+00
  %sub = fsub fast double %sum.1, %0
  %sum.2 = select i1 %cmp.2, double %sub, double %sum.1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %zext
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %1 = phi double [ 0.000000e+00, %entry ], [ %sum.2, %for.body ]
  ret double %1
}

; Double pattern:
; Check that is not vectorized if fp-instruction has no fast-math property. 
;
; double fcmp_0_fsub_select2_notvectorize(double * restrict x, const int N) {
;   double sum = 0.                                              
;   for (int i = 0; i < N; ++i)
;     if (x[i] > 0.)
;       sum -= x[i];
;   return sum;
; }

; CHECK-LABEL: @fcmp_0_fsub_select2_notvectorize(
; CHECK-NOT: <4 x doubole>
define double @fcmp_0_fsub_select2_notvectorize(double* noalias %x, i32 %N) nounwind readonly {
entry:
  %cmp.1 = icmp sgt i32 %N, 0
  br i1 %cmp.1, label %for.header, label %for.end

for.header:                                       ; preds = %entry
  %zext = zext i32 %N to i64
  br label %for.body

for.body:                                         ; preds = %for.body, %for.header
  %indvars.iv = phi i64 [ 0, %for.header ], [ %indvars.iv.next, %for.body ]
  %sum.1 = phi double [ 0.000000e+00, %for.header ], [ %sum.2, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %x, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 4
  %cmp.2 = fcmp ogt double %0, 0.000000e+00
  %sub = fsub double %sum.1, %0
  %sum.2 = select i1 %cmp.2, double %sub, double %sum.1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %zext
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %1 = phi double [ 0.000000e+00, %entry ], [ %sum.2, %for.body ]
  ret double %1
}

; Float pattern:
;   Check vectorization of reduction code which has an fmul instruction after
;   an fcmp instruction which compares an array element and 0.
;
; float fcmp_0_fmult_select1(float * restrict x, const int N) {
;   float sum = 0.
;   for (int i = 0; i < N; ++i)
;     if (x[i] > (float)0.)
;       sum *= x[i];
;   return sum;
; }

; CHECK-LABEL: @fcmp_0_fmult_select1(
; CHECK: %[[V1:.*]] = fcmp fast ogt <4 x float> %[[V0:.*]], zeroinitializer
; CHECK: %[[V3:.*]] = fmul fast <4 x float> %[[V2:.*]], %[[V0]]
; CHECK: select <4 x i1> %[[V1]], <4 x float> %[[V3]], <4 x float> %[[V2]]
define float @fcmp_0_fmult_select1(float* noalias %x, i32 %N) nounwind readonly {
entry:
  %cmp.1 = icmp sgt i32 %N, 0
  br i1 %cmp.1, label %for.header, label %for.end

for.header:                                       ; preds = %entry
  %zext = zext i32 %N to i64
  br label %for.body

for.body:                                         ; preds = %for.body, %for.header
  %indvars.iv = phi i64 [ 0, %for.header ], [ %indvars.iv.next, %for.body ]
  %sum.1 = phi float [ 0.000000e+00, %for.header ], [ %sum.2, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %x, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %cmp.2 = fcmp fast ogt float %0, 0.000000e+00
  %mult = fmul fast float %sum.1, %0
  %sum.2 = select i1 %cmp.2, float %mult, float %sum.1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %zext
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %1 = phi float [ 0.000000e+00, %entry ], [ %sum.2, %for.body ]
  ret float %1
}

; Float pattern:
;   Check that is not vectorized if fp-instruction has no fast-math property. 
;
; float fcmp_0_fmult_select1_notvectorize(float * restrict x, const int N) {
;   float sum = 0.
;   for (int i = 0; i < N; ++i)
;     if (x[i] > (float)0.)
;       sum *= x[i];
;   return sum;
; }

; CHECK-LABEL: @fcmp_0_fmult_select1_notvectorize(
; CHECK-NOT: <4 x float>
define float @fcmp_0_fmult_select1_notvectorize(float* noalias %x, i32 %N) nounwind readonly {
entry:
  %cmp.1 = icmp sgt i32 %N, 0
  br i1 %cmp.1, label %for.header, label %for.end

for.header:                                       ; preds = %entry
  %zext = zext i32 %N to i64
  br label %for.body

for.body:                                         ; preds = %for.body, %for.header
  %indvars.iv = phi i64 [ 0, %for.header ], [ %indvars.iv.next, %for.body ]
  %sum.1 = phi float [ 0.000000e+00, %for.header ], [ %sum.2, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %x, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %cmp.2 = fcmp ogt float %0, 0.000000e+00
  %mult = fmul float %sum.1, %0
  %sum.2 = select i1 %cmp.2, float %mult, float %sum.1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %zext
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %1 = phi float [ 0.000000e+00, %entry ], [ %sum.2, %for.body ]
  ret float %1
}

; Double pattern:
;   Check vectorization of reduction code which has an fmul instruction after
;   an fcmp instruction which compares an array element and 0.
;
; double fcmp_0_fmult_select2(double * restrict x, const int N) {
;   double sum = 0.
;   for (int i = 0; i < N; ++i)
;     if (x[i] > 0.)
;       sum *= x[i];
;   return sum;
; }

; CHECK-LABEL: @fcmp_0_fmult_select2(
; CHECK: %[[V1:.*]] = fcmp fast ogt <4 x double> %[[V0:.*]], zeroinitializer
; CHECK: %[[V3:.*]] = fmul fast <4 x double> %[[V2:.*]], %[[V0]]
; CHECK: select <4 x i1> %[[V1]], <4 x double> %[[V3]], <4 x double> %[[V2]]
define double @fcmp_0_fmult_select2(double* noalias %x, i32 %N) nounwind readonly {
entry:
  %cmp.1 = icmp sgt i32 %N, 0
  br i1 %cmp.1, label %for.header, label %for.end

for.header:                                       ; preds = %entry
  %zext = zext i32 %N to i64
  br label %for.body

for.body:                                         ; preds = %for.body, %for.header
  %indvars.iv = phi i64 [ 0, %for.header ], [ %indvars.iv.next, %for.body ]
  %sum.1 = phi double [ 0.000000e+00, %for.header ], [ %sum.2, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %x, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 4
  %cmp.2 = fcmp fast ogt double %0, 0.000000e+00
  %mult = fmul fast double %sum.1, %0
  %sum.2 = select i1 %cmp.2, double %mult, double %sum.1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %zext
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %1 = phi double [ 0.000000e+00, %entry ], [ %sum.2, %for.body ]
  ret double %1
}

; Double pattern:
;   Check that is not vectorized if fp-instruction has no fast-math property.
;
; double fcmp_0_fmult_select2_notvectorize(double * restrict x, const int N) {
;   double sum = 0.
;   for (int i = 0; i < N; ++i)
;     if (x[i] > 0.)
;       sum *= x[i];
;   return sum;
; }

; CHECK-LABEL: @fcmp_0_fmult_select2_notvectorize(
; CHECK-NOT: <4 x double>
define double @fcmp_0_fmult_select2_notvectorize(double* noalias %x, i32 %N) nounwind readonly {
entry:
  %cmp.1 = icmp sgt i32 %N, 0
  br i1 %cmp.1, label %for.header, label %for.end

for.header:                                       ; preds = %entry
  %zext = zext i32 %N to i64
  br label %for.body

for.body:                                         ; preds = %for.body, %for.header
  %indvars.iv = phi i64 [ 0, %for.header ], [ %indvars.iv.next, %for.body ]
  %sum.1 = phi double [ 0.000000e+00, %for.header ], [ %sum.2, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %x, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 4
  %cmp.2 = fcmp ogt double %0, 0.000000e+00
  %mult = fmul double %sum.1, %0
  %sum.2 = select i1 %cmp.2, double %mult, double %sum.1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %zext
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %1 = phi double [ 0.000000e+00, %entry ], [ %sum.2, %for.body ]
  ret double %1
}

; Float multi pattern
;   Check vectorisation of reduction code with a pair of selects to different
;   fadd patterns.
;
; float fcmp_multi(float *a, int n) {
;   float sum=0.0;
;   for (int i=0;i<n;i++) {
;     if (a[i]>1.0)
;       sum+=a[i];
;     else if (a[i]<3.0)
;       sum+=2*a[i];
;     else
;       sum+=3*a[i];
;   }
;   return sum;
; }

; CHECK-LABEL: @fcmp_multi(
; CHECK: %[[C1:.*]] = fcmp ogt <4 x float> %[[V0:.*]], <float 1.000000e+00,
; CHECK: %[[C2:.*]] = fcmp olt <4 x float> %[[V0]], <float 3.000000e+00,
; CHECK-DAG: %[[M1:.*]] = fmul fast <4 x float> %[[V0]], <float 3.000000e+00,
; CHECK-DAG: %[[M2:.*]] = fmul fast <4 x float> %[[V0]], <float 2.000000e+00,
; CHECK: %[[C11:.*]] = xor <4 x i1> %[[C1]], <i1 true,
; CHECK-DAG: %[[C12:.*]] = and <4 x i1> %[[C2]], %[[C11]]
; CHECK-DAG: %[[C21:.*]] = xor <4 x i1> %[[C2]], <i1 true,
; CHECK: %[[C22:.*]] = and <4 x i1> %[[C21]], %[[C11]]
; CHECK: %[[S1:.*]] = select <4 x i1> %[[C22]], <4 x float> %[[M1]], <4 x float> %[[M2]]
; CHECK: %[[S2:.*]] = select <4 x i1> %[[C1]], <4 x float> %[[V0]], <4 x float> %[[S1]]
; CHECK: fadd fast <4 x float> %[[S2]],
define float @fcmp_multi(float* nocapture readonly %a, i32 %n) nounwind readonly {
entry:
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.body:                                         ; preds = %for.inc, %for.body.preheader
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.inc ]
  %sum.011 = phi float [ 0.000000e+00, %for.body.preheader ], [ %sum.1, %for.inc ]
  %arrayidx = getelementptr inbounds float, float* %a, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %cmp1 = fcmp ogt float %0, 1.000000e+00
  br i1 %cmp1, label %for.inc, label %if.else

if.else:                                          ; preds = %for.body
  %cmp8 = fcmp olt float %0, 3.000000e+00
  br i1 %cmp8, label %if.then10, label %if.else14

if.then10:                                        ; preds = %if.else
  %mul = fmul fast float %0, 2.000000e+00
  br label %for.inc

if.else14:                                        ; preds = %if.else
  %mul17 = fmul fast float %0, 3.000000e+00
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.else14, %if.then10
  %.pn = phi float [ %mul, %if.then10 ], [ %mul17, %if.else14 ], [ %0, %for.body ]
  %sum.1 = fadd fast float %.pn, %sum.011
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  %sum.0.lcssa = phi float [ 0.000000e+00, %entry ], [ %sum.1, %for.inc ]
  ret float %sum.0.lcssa
}

; Float fadd + fsub patterns
;   Check vectorisation of reduction code with a pair of selects to different
;   instructions { fadd, fsub } but equivalent (change in constant).
;
; float fcmp_multi(float *a, int n) {
;   float sum=0.0;
;   for (int i=0;i<n;i++) {
;     if (a[i]>1.0)
;       sum+=a[i];
;     else if (a[i]<3.0)
;       sum-=a[i];
;   }
;   return sum;
; }

; CHECK-LABEL: @fcmp_fadd_fsub(
; CHECK: %[[C1:.*]] = fcmp ogt <4 x float> %[[V0:.*]], <float 1.000000e+00,
; CHECK: %[[C2:.*]] = fcmp olt <4 x float> %[[V0]], <float 3.000000e+00,
; CHECK-DAG: %[[SUB:.*]] = fsub fast <4 x float>
; CHECK-DAG: %[[ADD:.*]] = fadd fast <4 x float>
; CHECK: %[[C11:.*]] = xor <4 x i1> %[[C1]], <i1 true,
; CHECK-DAG: %[[C12:.*]] = and <4 x i1> %[[C2]], %[[C11]]
; CHECK-DAG: %[[C21:.*]] = xor <4 x i1> %[[C2]], <i1 true,
; CHECK: %[[C22:.*]] = and <4 x i1> %[[C21]], %[[C11]]
; CHECK: %[[S1:.*]] = select <4 x i1> %[[C12]], <4 x float> %[[SUB]], <4 x float> %[[ADD]]
; CHECK: %[[S2:.*]] = select <4 x i1> %[[C22]], {{.*}} <4 x float> %[[S1]]
define float @fcmp_fadd_fsub(float* nocapture readonly %a, i32 %n) nounwind readonly {
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.body:                                         ; preds = %for.inc, %for.body.preheader
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.inc ]
  %sum.010 = phi float [ 0.000000e+00, %for.body.preheader ], [ %sum.1, %for.inc ]
  %arrayidx = getelementptr inbounds float, float* %a, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %cmp1 = fcmp ogt float %0, 1.000000e+00
  br i1 %cmp1, label %if.then, label %if.else

if.then:                                          ; preds = %for.body
  %add = fadd fast float %0, %sum.010
  br label %for.inc

if.else:                                          ; preds = %for.body
  %cmp8 = fcmp olt float %0, 3.000000e+00
  br i1 %cmp8, label %if.then10, label %for.inc

if.then10:                                        ; preds = %if.else
  %sub = fsub fast float %sum.010, %0
  br label %for.inc

for.inc:                                          ; preds = %if.then, %if.then10, %if.else
  %sum.1 = phi float [ %add, %if.then ], [ %sub, %if.then10 ], [ %sum.010, %if.else ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  %sum.0.lcssa = phi float [ 0.000000e+00, %entry ], [ %sum.1, %for.inc ]
  ret float %sum.0.lcssa
}

; Float fadd + fmul patterns
;   Check lack of vectorisation of reduction code with a pair of non-compatible
;   instructions { fadd, fmul }.
;
; float fcmp_multi(float *a, int n) {
;   float sum=0.0;
;   for (int i=0;i<n;i++) {
;     if (a[i]>1.0)
;       sum+=a[i];
;     else if (a[i]<3.0)
;       sum*=a[i];
;   }
;   return sum;
; }

; CHECK-LABEL: @fcmp_fadd_fmul(
; CHECK-NOT: <4 x float>
define float @fcmp_fadd_fmul(float* nocapture readonly %a, i32 %n) nounwind readonly {
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.body:                                         ; preds = %for.inc, %for.body.preheader
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.inc ]
  %sum.010 = phi float [ 0.000000e+00, %for.body.preheader ], [ %sum.1, %for.inc ]
  %arrayidx = getelementptr inbounds float, float* %a, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %cmp1 = fcmp ogt float %0, 1.000000e+00
  br i1 %cmp1, label %if.then, label %if.else

if.then:                                          ; preds = %for.body
  %add = fadd fast float %0, %sum.010
  br label %for.inc

if.else:                                          ; preds = %for.body
  %cmp8 = fcmp olt float %0, 3.000000e+00
  br i1 %cmp8, label %if.then10, label %for.inc

if.then10:                                        ; preds = %if.else
  %mul = fmul fast float %0, %sum.010
  br label %for.inc

for.inc:                                          ; preds = %if.then, %if.then10, %if.else
  %sum.1 = phi float [ %add, %if.then ], [ %mul, %if.then10 ], [ %sum.010, %if.else ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc, %entry
  %sum.0.lcssa = phi float [ 0.000000e+00, %entry ], [ %sum.1, %for.inc ]
  ret float %sum.0.lcssa
}

; Float fadd + store patterns
;   Check lack of vectorisation of reduction code with a store back, given it
;   has loop dependency on a[i].
;
; float fcmp_store_back(float a[], int LEN) {
;     float sum = 0.0;
;     for (int i = 0; i < LEN; i++) {
;       sum += a[i];
;       a[i] = sum;
;     }
;     return sum;
; }

; CHECK-LABEL: @fcmp_store_back(
; CHECK-NOT: <4 x float>
define float @fcmp_store_back(float* nocapture %a, i32 %LEN) nounwind readonly {
entry:
  %cmp7 = icmp sgt i32 %LEN, 0
  br i1 %cmp7, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %LEN to i64
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %sum.08 = phi float [ 0.000000e+00, %for.body.preheader ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %a, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %add = fadd fast float %0, %sum.08
  store float %add, float* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %sum.0.lcssa = phi float [ 0.000000e+00, %entry ], [ %add, %for.body ]
  ret float %sum.0.lcssa
}
