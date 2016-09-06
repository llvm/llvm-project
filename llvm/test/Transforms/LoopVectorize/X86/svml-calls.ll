; RUN: opt -vector-library=SVML -loop-vectorize -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @sin_f32
; CHECK: <4 x float> @__svml_sinf4
; CHECK: ret

declare float @sinf(float) #0

define void @sin_f32(float* nocapture %varray) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @sinf(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %indvars.iv
  store float %call, float* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; CHECK-LABEL: @cos_f32
; CHECK: <4 x float> @__svml_cosf4
; CHECK: ret

declare float @cosf(float) #0

define void @cos_f32(float* nocapture %varray) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @cosf(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %indvars.iv
  store float %call, float* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; CHECK-LABEL: @exp_f32
; CHECK: <4 x float> @__svml_expf4
; CHECK: ret

declare float @expf(float) #0

define void @exp_f32(float* nocapture %varray) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @expf(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %indvars.iv
  store float %call, float* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; CHECK-LABEL: @exp_f32_intrin
; CHECK: <4 x float> @__svml_expf4
; CHECK: ret

declare float @llvm.exp.f32(float) #0

define void @exp_f32_intrin(float* nocapture %varray) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @llvm.exp.f32(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %indvars.iv
  store float %call, float* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; CHECK-LABEL: @log_f32
; CHECK: <4 x float> @__svml_logf4
; CHECK: ret

declare float @logf(float) #0

define void @log_f32(float* nocapture %varray) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call fast float @logf(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %indvars.iv
  store float %call, float* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; CHECK-LABEL: @pow_f32
; CHECK: <4 x float> @__svml_powf4
; CHECK: ret

declare float @powf(float, float) #0

define void @pow_f32(float* nocapture %varray, float* nocapture readonly %exp) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %arrayidx = getelementptr inbounds float, float* %exp, i64 %indvars.iv
  %tmp1 = load float, float* %arrayidx, align 4
  %tmp2 = tail call fast float @powf(float %conv, float %tmp1)
  %arrayidx2 = getelementptr inbounds float, float* %varray, i64 %indvars.iv
  store float %tmp2, float* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; CHECK-LABEL: @pow_f32_intrin
; CHECK: <4 x float> @__svml_powf4
; CHECK: ret

declare float @llvm.pow.f32(float, float) #0

define void @pow_f32_intrin(float* nocapture %varray, float* nocapture readonly %exp) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %tmp = trunc i64 %indvars.iv to i32
  %conv = sitofp i32 %tmp to float
  %arrayidx = getelementptr inbounds float, float* %exp, i64 %indvars.iv
  %tmp1 = load float, float* %arrayidx, align 4
  %tmp2 = tail call fast float @llvm.pow.f32(float %conv, float %tmp1)
  %arrayidx2 = getelementptr inbounds float, float* %varray, i64 %indvars.iv
  store float %tmp2, float* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

attributes #0 = { nounwind readnone }
