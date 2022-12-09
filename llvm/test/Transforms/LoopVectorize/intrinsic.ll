; RUN: opt < %s -passes=loop-vectorize,dce,instcombine -force-vector-interleave=1 -force-vector-width=4 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

;CHECK-LABEL: @sqrt_f32(
;CHECK: llvm.sqrt.v4f32
;CHECK: ret void
define void @sqrt_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %call = tail call float @llvm.sqrt.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.sqrt.f32(float) nounwind readnone

;CHECK-LABEL: @sqrt_f64(
;CHECK: llvm.sqrt.v4f64
;CHECK: ret void
define void @sqrt_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double, double* %y, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %call = tail call double @llvm.sqrt.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.sqrt.f64(double) nounwind readnone

;CHECK-LABEL: @sin_f32(
;CHECK: llvm.sin.v4f32
;CHECK: ret void
define void @sin_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %call = tail call float @llvm.sin.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.sin.f32(float) nounwind readnone

;CHECK-LABEL: @sin_f64(
;CHECK: llvm.sin.v4f64
;CHECK: ret void
define void @sin_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double, double* %y, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %call = tail call double @llvm.sin.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.sin.f64(double) nounwind readnone

;CHECK-LABEL: @cos_f32(
;CHECK: llvm.cos.v4f32
;CHECK: ret void
define void @cos_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %call = tail call float @llvm.cos.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.cos.f32(float) nounwind readnone

;CHECK-LABEL: @cos_f64(
;CHECK: llvm.cos.v4f64
;CHECK: ret void
define void @cos_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double, double* %y, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %call = tail call double @llvm.cos.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.cos.f64(double) nounwind readnone

;CHECK-LABEL: @exp_f32(
;CHECK: llvm.exp.v4f32
;CHECK: ret void
define void @exp_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %call = tail call float @llvm.exp.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.exp.f32(float) nounwind readnone

;CHECK-LABEL: @exp_f64(
;CHECK: llvm.exp.v4f64
;CHECK: ret void
define void @exp_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double, double* %y, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %call = tail call double @llvm.exp.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.exp.f64(double) nounwind readnone

;CHECK-LABEL: @exp2_f32(
;CHECK: llvm.exp2.v4f32
;CHECK: ret void
define void @exp2_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %call = tail call float @llvm.exp2.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.exp2.f32(float) nounwind readnone

;CHECK-LABEL: @exp2_f64(
;CHECK: llvm.exp2.v4f64
;CHECK: ret void
define void @exp2_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double, double* %y, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %call = tail call double @llvm.exp2.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.exp2.f64(double) nounwind readnone

;CHECK-LABEL: @log_f32(
;CHECK: llvm.log.v4f32
;CHECK: ret void
define void @log_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %call = tail call float @llvm.log.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.log.f32(float) nounwind readnone

;CHECK-LABEL: @log_f64(
;CHECK: llvm.log.v4f64
;CHECK: ret void
define void @log_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double, double* %y, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %call = tail call double @llvm.log.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.log.f64(double) nounwind readnone

;CHECK-LABEL: @log10_f32(
;CHECK: llvm.log10.v4f32
;CHECK: ret void
define void @log10_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %call = tail call float @llvm.log10.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.log10.f32(float) nounwind readnone

;CHECK-LABEL: @log10_f64(
;CHECK: llvm.log10.v4f64
;CHECK: ret void
define void @log10_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double, double* %y, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %call = tail call double @llvm.log10.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.log10.f64(double) nounwind readnone

;CHECK-LABEL: @log2_f32(
;CHECK: llvm.log2.v4f32
;CHECK: ret void
define void @log2_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %call = tail call float @llvm.log2.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.log2.f32(float) nounwind readnone

;CHECK-LABEL: @log2_f64(
;CHECK: llvm.log2.v4f64
;CHECK: ret void
define void @log2_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double, double* %y, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %call = tail call double @llvm.log2.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.log2.f64(double) nounwind readnone

;CHECK-LABEL: @fabs_f32(
;CHECK: llvm.fabs.v4f32
;CHECK: ret void
define void @fabs_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %call = tail call float @llvm.fabs.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.fabs.f32(float) nounwind readnone

define void @fabs_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double, double* %y, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %call = tail call double @llvm.fabs(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.fabs(double) nounwind readnone

;CHECK-LABEL: @copysign_f32(
;CHECK: llvm.copysign.v4f32
;CHECK: ret void
define void @copysign_f32(i32 %n, float* noalias %y, float* noalias %x, float* noalias %z) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, float* %z, i64 %indvars.iv
  %1 = load float, float* %arrayidx1, align 4
  %call = tail call float @llvm.copysign.f32(float %0, float %1) nounwind readnone
  %arrayidx2 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.copysign.f32(float, float) nounwind readnone

define void @copysign_f64(i32 %n, double* noalias %y, double* noalias %x, double* noalias %z) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double, double* %y, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds double, double* %z, i64 %indvars.iv
  %1 = load double, double* %arrayidx, align 8
  %call = tail call double @llvm.copysign(double %0, double %1) nounwind readnone
  %arrayidx2 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.copysign(double, double) nounwind readnone

;CHECK-LABEL: @floor_f32(
;CHECK: llvm.floor.v4f32
;CHECK: ret void
define void @floor_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %call = tail call float @llvm.floor.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.floor.f32(float) nounwind readnone

;CHECK-LABEL: @floor_f64(
;CHECK: llvm.floor.v4f64
;CHECK: ret void
define void @floor_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double, double* %y, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %call = tail call double @llvm.floor.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.floor.f64(double) nounwind readnone

;CHECK-LABEL: @ceil_f32(
;CHECK: llvm.ceil.v4f32
;CHECK: ret void
define void @ceil_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %call = tail call float @llvm.ceil.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.ceil.f32(float) nounwind readnone

;CHECK-LABEL: @ceil_f64(
;CHECK: llvm.ceil.v4f64
;CHECK: ret void
define void @ceil_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double, double* %y, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %call = tail call double @llvm.ceil.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.ceil.f64(double) nounwind readnone

;CHECK-LABEL: @trunc_f32(
;CHECK: llvm.trunc.v4f32
;CHECK: ret void
define void @trunc_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %call = tail call float @llvm.trunc.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.trunc.f32(float) nounwind readnone

;CHECK-LABEL: @trunc_f64(
;CHECK: llvm.trunc.v4f64
;CHECK: ret void
define void @trunc_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double, double* %y, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %call = tail call double @llvm.trunc.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.trunc.f64(double) nounwind readnone

;CHECK-LABEL: @rint_f32(
;CHECK: llvm.rint.v4f32
;CHECK: ret void
define void @rint_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %call = tail call float @llvm.rint.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.rint.f32(float) nounwind readnone

;CHECK-LABEL: @rint_f64(
;CHECK: llvm.rint.v4f64
;CHECK: ret void
define void @rint_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double, double* %y, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %call = tail call double @llvm.rint.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.rint.f64(double) nounwind readnone

;CHECK-LABEL: @nearbyint_f32(
;CHECK: llvm.nearbyint.v4f32
;CHECK: ret void
define void @nearbyint_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %call = tail call float @llvm.nearbyint.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.nearbyint.f32(float) nounwind readnone

;CHECK-LABEL: @nearbyint_f64(
;CHECK: llvm.nearbyint.v4f64
;CHECK: ret void
define void @nearbyint_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double, double* %y, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %call = tail call double @llvm.nearbyint.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.nearbyint.f64(double) nounwind readnone

;CHECK-LABEL: @round_f32(
;CHECK: llvm.round.v4f32
;CHECK: ret void
define void @round_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %call = tail call float @llvm.round.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.round.f32(float) nounwind readnone

;CHECK-LABEL: @round_f64(
;CHECK: llvm.round.v4f64
;CHECK: ret void
define void @round_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double, double* %y, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %call = tail call double @llvm.round.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.round.f64(double) nounwind readnone

;CHECK-LABEL: @roundeven_f32(
;CHECK: llvm.roundeven.v4f32
;CHECK: ret void
define void @roundeven_f32(i32 %n, float* noalias %y, float* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %call = tail call float @llvm.roundeven.f32(float %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.roundeven.f32(float) nounwind readnone

;CHECK-LABEL: @roundeven_f64(
;CHECK: llvm.roundeven.v4f64
;CHECK: ret void
define void @roundeven_f64(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double, double* %y, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %call = tail call double @llvm.roundeven.f64(double %0) nounwind readnone
  %arrayidx2 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx2, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.roundeven.f64(double) nounwind readnone

;CHECK-LABEL: @fma_f32(
;CHECK: llvm.fma.v4f32
;CHECK: ret void
define void @fma_f32(i32 %n, float* noalias %y, float* noalias %x, float* noalias %z, float* noalias %w) nounwind uwtable {
entry:
  %cmp12 = icmp sgt i32 %n, 0
  br i1 %cmp12, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, float* %w, i64 %indvars.iv
  %1 = load float, float* %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds float, float* %z, i64 %indvars.iv
  %2 = load float, float* %arrayidx4, align 4
  %3 = tail call float @llvm.fma.f32(float %0, float %2, float %1)
  %arrayidx6 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %3, float* %arrayidx6, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.fma.f32(float, float, float) nounwind readnone

;CHECK-LABEL: @fma_f64(
;CHECK: llvm.fma.v4f64
;CHECK: ret void
define void @fma_f64(i32 %n, double* noalias %y, double* noalias %x, double* noalias %z, double* noalias %w) nounwind uwtable {
entry:
  %cmp12 = icmp sgt i32 %n, 0
  br i1 %cmp12, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double, double* %y, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %arrayidx2 = getelementptr inbounds double, double* %w, i64 %indvars.iv
  %1 = load double, double* %arrayidx2, align 8
  %arrayidx4 = getelementptr inbounds double, double* %z, i64 %indvars.iv
  %2 = load double, double* %arrayidx4, align 8
  %3 = tail call double @llvm.fma.f64(double %0, double %2, double %1)
  %arrayidx6 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  store double %3, double* %arrayidx6, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.fma.f64(double, double, double) nounwind readnone

;CHECK-LABEL: @fmuladd_f32(
;CHECK: llvm.fmuladd.v4f32
;CHECK: ret void
define void @fmuladd_f32(i32 %n, float* noalias %y, float* noalias %x, float* noalias %z, float* noalias %w) nounwind uwtable {
entry:
  %cmp12 = icmp sgt i32 %n, 0
  br i1 %cmp12, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, float* %w, i64 %indvars.iv
  %1 = load float, float* %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds float, float* %z, i64 %indvars.iv
  %2 = load float, float* %arrayidx4, align 4
  %3 = tail call float @llvm.fmuladd.f32(float %0, float %2, float %1)
  %arrayidx6 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %3, float* %arrayidx6, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.fmuladd.f32(float, float, float) nounwind readnone

;CHECK-LABEL: @fmuladd_f64(
;CHECK: llvm.fmuladd.v4f64
;CHECK: ret void
define void @fmuladd_f64(i32 %n, double* noalias %y, double* noalias %x, double* noalias %z, double* noalias %w) nounwind uwtable {
entry:
  %cmp12 = icmp sgt i32 %n, 0
  br i1 %cmp12, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double, double* %y, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %arrayidx2 = getelementptr inbounds double, double* %w, i64 %indvars.iv
  %1 = load double, double* %arrayidx2, align 8
  %arrayidx4 = getelementptr inbounds double, double* %z, i64 %indvars.iv
  %2 = load double, double* %arrayidx4, align 8
  %3 = tail call double @llvm.fmuladd.f64(double %0, double %2, double %1)
  %arrayidx6 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  store double %3, double* %arrayidx6, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare double @llvm.fmuladd.f64(double, double, double) nounwind readnone

;CHECK-LABEL: @pow_f32(
;CHECK: llvm.pow.v4f32
;CHECK: ret void
define void @pow_f32(i32 %n, float* noalias %y, float* noalias %x, float* noalias %z) nounwind uwtable {
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, float* %z, i64 %indvars.iv
  %1 = load float, float* %arrayidx2, align 4
  %call = tail call float @llvm.pow.f32(float %0, float %1) nounwind readnone
  %arrayidx4 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx4, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.pow.f32(float, float) nounwind readnone

;CHECK-LABEL: @pow_f64(
;CHECK: llvm.pow.v4f64
;CHECK: ret void
define void @pow_f64(i32 %n, double* noalias %y, double* noalias %x, double* noalias %z) nounwind uwtable {
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double, double* %y, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %arrayidx2 = getelementptr inbounds double, double* %z, i64 %indvars.iv
  %1 = load double, double* %arrayidx2, align 8
  %call = tail call double @llvm.pow.f64(double %0, double %1) nounwind readnone
  %arrayidx4 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx4, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; CHECK: fabs_libm
; CHECK:  call <4 x float> @llvm.fabs.v4f32
; CHECK: ret void
define void @fabs_libm(float* nocapture %x) nounwind {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %x, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %call = tail call float @fabsf(float %0) nounwind readnone
  store float %call, float* %arrayidx, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

declare float @fabsf(float) nounwind readnone

declare double @llvm.pow.f64(double, double) nounwind readnone



; Make sure we don't replace calls to functions with standard library function
; signatures but defined with internal linkage.

define internal float @roundf(float %x) nounwind readnone {
  ret float 0.00000000
}
; CHECK-LABEL: internal_round
; CHECK-NOT:  load <4 x float>

define void @internal_round(float* nocapture %x) nounwind {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %x, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %call = tail call float @roundf(float %0) nounwind readnone
  store float %call, float* %arrayidx, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; Make sure we don't replace calls to functions with standard library names but
; different signatures.

declare void @round(double %f)

; CHECK-LABEL: wrong_signature
; CHECK-NOT:  load <4 x double>

define void @wrong_signature(double* nocapture %x) nounwind {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %x, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 4
  store double %0, double* %arrayidx, align 4
  tail call void @round(double %0) nounwind readnone
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

declare double @llvm.powi.f64.i32(double %Val, i32 %power) nounwind readnone

;CHECK-LABEL: @powi_f64(
;CHECK: llvm.powi.v4f64
;CHECK: ret void
define void @powi_f64(i32 %n, double* noalias %y, double* noalias %x, i32 %P) nounwind uwtable {
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double, double* %y, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %call = tail call double @llvm.powi.f64.i32(double %0, i32  %P) nounwind readnone
  %arrayidx4 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx4, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

;CHECK-LABEL: @powi_f64_neg(
;CHECK-NOT: llvm.powi.v4f64
;CHECK: ret void
define void @powi_f64_neg(i32 %n, double* noalias %y, double* noalias %x) nounwind uwtable {
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double, double* %y, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %1 = trunc i64 %indvars.iv to i32
  %call = tail call double @llvm.powi.f64.i32(double %0, i32  %1) nounwind readnone
  %arrayidx4 = getelementptr inbounds double, double* %x, i64 %indvars.iv
  store double %call, double* %arrayidx4, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare i64  @llvm.cttz.i64 (i64, i1) nounwind readnone

;CHECK-LABEL: @cttz_f64(
;CHECK: llvm.cttz.v4i64
;CHECK: ret void
define void @cttz_f64(i32 %n, i64* noalias %y, i64* noalias %x) nounwind uwtable {
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i64, i64* %y, i64 %indvars.iv
  %0 = load i64, i64* %arrayidx, align 8
  %call = tail call i64 @llvm.cttz.i64(i64 %0, i1 true) nounwind readnone
  %arrayidx4 = getelementptr inbounds i64, i64* %x, i64 %indvars.iv
  store i64 %call, i64* %arrayidx4, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare i64  @llvm.ctlz.i64 (i64, i1) nounwind readnone

;CHECK-LABEL: @ctlz_f64(
;CHECK: llvm.ctlz.v4i64
;CHECK: ret void
define void @ctlz_f64(i32 %n, i64* noalias %y, i64* noalias %x) nounwind uwtable {
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i64, i64* %y, i64 %indvars.iv
  %0 = load i64, i64* %arrayidx, align 8
  %call = tail call i64 @llvm.ctlz.i64(i64 %0, i1 true) nounwind readnone
  %arrayidx4 = getelementptr inbounds i64, i64* %x, i64 %indvars.iv
  store i64 %call, i64* %arrayidx4, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare i64 @llvm.abs.i64 (i64, i1) nounwind readnone

define void @abs_i64(i32 %n, i64* noalias %y, i64* noalias %x) nounwind uwtable {
;CHECK-LABEL: @abs_i64(
;CHECK: llvm.abs.v4i64(<4 x i64> [[WIDE_LOADX:%.*]], i1 true)
;CHECK: ret void
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i64, i64* %y, i64 %indvars.iv
  %0 = load i64, i64* %arrayidx, align 8
  %call = tail call i64 @llvm.abs.i64(i64 %0, i1 true) nounwind readnone
  %arrayidx4 = getelementptr inbounds i64, i64* %x, i64 %indvars.iv
  store i64 %call, i64* %arrayidx4, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare i32 @llvm.smin.i32 (i32, i32)

define void @smin_i32(i32 %n, i32* noalias %x, i32* noalias %y) {
; CHECK-LABEL: @smin_i32(
; CHECK:         call <4 x i32> @llvm.smin.v4i32(<4 x i32> [[WIDE_LOADX:%.*]], <4 x i32> [[WIDE_LOADY:%.*]])
; CHECK:         ret void
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop, label %end

loop:
  %iv = phi i32 [ %iv.next, %loop ], [ 0, %entry ]
  %xi = getelementptr inbounds i32, i32* %x, i32 %iv
  %yi = getelementptr inbounds i32, i32* %y, i32 %iv
  %xld = load i32, i32* %xi, align 4
  %yld = load i32, i32* %yi, align 4
  %call = tail call i32 @llvm.smin.i32(i32 %xld, i32 %yld)
  store i32 %call, i32* %xi, align 4
  %iv.next = add i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %end, label %loop

end:
  ret void
}

declare i32 @llvm.smax.i32 (i32, i32)

define void @smax_i32(i32 %n, i32* noalias %x, i32* noalias %y) {
; CHECK-LABEL: @smax_i32(
; CHECK:         call <4 x i32> @llvm.smax.v4i32(<4 x i32> [[WIDE_LOADX:%.*]], <4 x i32> [[WIDE_LOADY:%.*]])
; CHECK:         ret void
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop, label %end

loop:
  %iv = phi i32 [ %iv.next, %loop ], [ 0, %entry ]
  %xi = getelementptr inbounds i32, i32* %x, i32 %iv
  %yi = getelementptr inbounds i32, i32* %y, i32 %iv
  %xld = load i32, i32* %xi, align 4
  %yld = load i32, i32* %yi, align 4
  %call = tail call i32 @llvm.smax.i32(i32 %xld, i32 %yld)
  store i32 %call, i32* %xi, align 4
  %iv.next = add i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %end, label %loop

end:
  ret void
}

declare i32 @llvm.umin.i32 (i32, i32)

define void @umin_i32(i32 %n, i32* noalias %x, i32* noalias %y) {
; CHECK-LABEL: @umin_i32(
; CHECK:         call <4 x i32> @llvm.umin.v4i32(<4 x i32> [[WIDE_LOADX:%.*]], <4 x i32> [[WIDE_LOADY:%.*]])
; CHECK:         ret void
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop, label %end

loop:
  %iv = phi i32 [ %iv.next, %loop ], [ 0, %entry ]
  %xi = getelementptr inbounds i32, i32* %x, i32 %iv
  %yi = getelementptr inbounds i32, i32* %y, i32 %iv
  %xld = load i32, i32* %xi, align 4
  %yld = load i32, i32* %yi, align 4
  %call = tail call i32 @llvm.umin.i32(i32 %xld, i32 %yld)
  store i32 %call, i32* %xi, align 4
  %iv.next = add i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %end, label %loop

end:
  ret void
}

declare i32 @llvm.umax.i32 (i32, i32)

define void @umax_i32(i32 %n, i32* noalias %x, i32* noalias %y) {
; CHECK-LABEL: @umax_i32(
; CHECK:         call <4 x i32> @llvm.umax.v4i32(<4 x i32> [[WIDE_LOADX:%.*]], <4 x i32> [[WIDE_LOADY:%.*]])
; CHECK:         ret void
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop, label %end

loop:
  %iv = phi i32 [ %iv.next, %loop ], [ 0, %entry ]
  %xi = getelementptr inbounds i32, i32* %x, i32 %iv
  %yi = getelementptr inbounds i32, i32* %y, i32 %iv
  %xld = load i32, i32* %xi, align 4
  %yld = load i32, i32* %yi, align 4
  %call = tail call i32 @llvm.umax.i32(i32 %xld, i32 %yld)
  store i32 %call, i32* %xi, align 4
  %iv.next = add i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %end, label %loop

end:
  ret void
}

declare i32 @llvm.fshl.i32 (i32, i32, i32)

define void @fshl_i32(i32 %n, i32* noalias %x, i32* noalias %y, i32 %shAmt) {
; CHECK-LABEL: @fshl_i32(
; CHECK:         call <4 x i32> @llvm.fshl.v4i32(<4 x i32> [[WIDE_LOADX:%.*]], <4 x i32> [[WIDE_LOADY:%.*]], <4 x i32> [[SPLAT:%.*]])
; CHECK:         ret void
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop, label %end

loop:
  %iv = phi i32 [ %iv.next, %loop ], [ 0, %entry ]
  %xi = getelementptr inbounds i32, i32* %x, i32 %iv
  %yi = getelementptr inbounds i32, i32* %y, i32 %iv
  %xld = load i32, i32* %xi, align 4
  %yld = load i32, i32* %yi, align 4
  %call = tail call i32 @llvm.fshl.i32(i32 %xld, i32 %yld, i32 %shAmt)
  store i32 %call, i32* %xi, align 4
  %iv.next = add i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %end, label %loop

end:
  ret void
}

declare i32 @llvm.fshr.i32 (i32, i32, i32)

define void @fshr_i32(i32 %n, i32* noalias %x, i32* noalias %y, i32 %shAmt) {
; CHECK-LABEL: @fshr_i32(
; CHECK:         call <4 x i32> @llvm.fshr.v4i32(<4 x i32> [[WIDE_LOADX:%.*]], <4 x i32> [[WIDE_LOADY:%.*]], <4 x i32> [[SPLAT:%.*]])
; CHECK:         ret void
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop, label %end

loop:
  %iv = phi i32 [ %iv.next, %loop ], [ 0, %entry ]
  %xi = getelementptr inbounds i32, i32* %x, i32 %iv
  %yi = getelementptr inbounds i32, i32* %y, i32 %iv
  %xld = load i32, i32* %xi, align 4
  %yld = load i32, i32* %yi, align 4
  %call = tail call i32 @llvm.fshr.i32(i32 %xld, i32 %yld, i32 %shAmt)
  store i32 %call, i32* %xi, align 4
  %iv.next = add i32 %iv, 1
  %exitcond = icmp eq i32 %iv.next, %n
  br i1 %exitcond, label %end, label %loop

end:
  ret void
}

declare float @llvm.minnum.f32(float, float) nounwind readnone

;CHECK-LABEL: @minnum_f32(
;CHECK: llvm.minnum.v4f32
;CHECK: ret void
define void @minnum_f32(i32 %n, float* noalias %y, float* noalias %x, float* noalias %z) nounwind uwtable {
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, float* %z, i64 %indvars.iv
  %1 = load float, float* %arrayidx2, align 4
  %call = tail call float @llvm.minnum.f32(float %0, float %1) nounwind readnone
  %arrayidx4 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx4, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.maxnum.f32(float, float) nounwind readnone

;CHECK-LABEL: @maxnum_f32(
;CHECK: llvm.maxnum.v4f32
;CHECK: ret void
define void @maxnum_f32(i32 %n, float* noalias %y, float* noalias %x, float* noalias %z) nounwind uwtable {
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, float* %z, i64 %indvars.iv
  %1 = load float, float* %arrayidx2, align 4
  %call = tail call float @llvm.maxnum.f32(float %0, float %1) nounwind readnone
  %arrayidx4 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx4, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.minimum.f32(float, float) nounwind readnone

;CHECK-LABEL: @minimum_f32(
;CHECK: llvm.minimum.v4f32
;CHECK: ret void
define void @minimum_f32(i32 %n, float* noalias %y, float* noalias %x, float* noalias %z) nounwind uwtable {
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, float* %z, i64 %indvars.iv
  %1 = load float, float* %arrayidx2, align 4
  %call = tail call float @llvm.minimum.f32(float %0, float %1) nounwind readnone
  %arrayidx4 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx4, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare float @llvm.maximum.f32(float, float) nounwind readnone

;CHECK-LABEL: @maximum_f32(
;CHECK: llvm.maximum.v4f32
;CHECK: ret void
define void @maximum_f32(i32 %n, float* noalias %y, float* noalias %x, float* noalias %z) nounwind uwtable {
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %y, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, float* %z, i64 %indvars.iv
  %1 = load float, float* %arrayidx2, align 4
  %call = tail call float @llvm.maximum.f32(float %0, float %1) nounwind readnone
  %arrayidx4 = getelementptr inbounds float, float* %x, i64 %indvars.iv
  store float %call, float* %arrayidx4, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}
