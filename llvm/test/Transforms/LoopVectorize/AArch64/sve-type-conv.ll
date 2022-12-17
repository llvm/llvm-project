; RUN: opt -passes=loop-vectorize,dce,instcombine < %s -S | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"


define void @f16_to_f32(float* noalias nocapture %dst, half* noalias nocapture readonly %src, i64 %N) #0 {
; CHECK-LABEL: @f16_to_f32(
; CHECK: vector.body
; CHECK:   %{{.*}} = fpext <vscale x 8 x half> %{{.*}} to <vscale x 8 x float>
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.07 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds half, half* %src, i64 %i.07
  %0 = load half, half* %arrayidx, align 2
  %conv = fpext half %0 to float
  %arrayidx1 = getelementptr inbounds float, float* %dst, i64 %i.07
  store float %conv, float* %arrayidx1, align 4
  %inc = add nuw nsw i64 %i.07, 1
  %exitcond.not = icmp eq i64 %inc, %N
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:                                          ; preds = %for.body, %entry
  ret void
}


define void @f64_to_f32(float* noalias nocapture %dst, double* noalias nocapture readonly %src, i64 %N) #0 {
; CHECK-LABEL: @f64_to_f32(
; CHECK: vector.body
; CHECK:   %{{.*}} = fptrunc <vscale x 8 x double> %{{.*}} to <vscale x 8 x float>
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.07 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds double, double* %src, i64 %i.07
  %0 = load double, double* %arrayidx, align 8
  %conv = fptrunc double %0 to float
  %arrayidx1 = getelementptr inbounds float, float* %dst, i64 %i.07
  store float %conv, float* %arrayidx1, align 4
  %inc = add nuw nsw i64 %i.07, 1
  %exitcond.not = icmp eq i64 %inc, %N
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:                                          ; preds = %for.body, %entry
  ret void
}


define void @f16_to_s8(i8* noalias nocapture %dst, half* noalias nocapture readonly %src, i64 %N) #0 {
; CHECK-LABEL: @f16_to_s8(
; CHECK: vector.body
; CHECK:   %{{.*}} = fptosi <vscale x 8 x half> %{{.*}} to <vscale x 8 x i8>
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.08 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds half, half* %src, i64 %i.08
  %0 = load half, half* %arrayidx, align 2
  %conv1 = fptosi half %0 to i8
  %arrayidx2 = getelementptr inbounds i8, i8* %dst, i64 %i.08
  store i8 %conv1, i8* %arrayidx2, align 1
  %inc = add nuw nsw i64 %i.08, 1
  %exitcond.not = icmp eq i64 %inc, %N
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:                                          ; preds = %for.body, %entry
  ret void
}


define void @f32_to_u64(i64* noalias nocapture %dst, float* noalias nocapture readonly %src, i64 %N) #0 {
; CHECK-LABEL: @f32_to_u64(
; CHECK: vector.body
; CHECK:   %{{.*}} = fptoui <vscale x 8 x float> %{{.*}} to <vscale x 8 x i64>
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.07 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %src, i64 %i.07
  %0 = load float, float* %arrayidx, align 4
  %conv = fptoui float %0 to i64
  %arrayidx1 = getelementptr inbounds i64, i64* %dst, i64 %i.07
  store i64 %conv, i64* %arrayidx1, align 8
  %inc = add nuw nsw i64 %i.07, 1
  %exitcond.not = icmp eq i64 %inc, %N
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:                                          ; preds = %for.body, %entry
  ret void
}


define void @s8_to_f32(float* noalias nocapture %dst, i8* noalias nocapture readonly %src, i64 %N) #0 {
; CHECK-LABEL: @s8_to_f32(
; CHECK: vector.body
; CHECK:   %{{.*}} = sitofp <vscale x 8 x i8> %{{.*}} to <vscale x 8 x float>
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.07 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i8, i8* %src, i64 %i.07
  %0 = load i8, i8* %arrayidx, align 1
  %conv = sitofp i8 %0 to float
  %arrayidx1 = getelementptr inbounds float, float* %dst, i64 %i.07
  store float %conv, float* %arrayidx1, align 4
  %inc = add nuw nsw i64 %i.07, 1
  %exitcond.not = icmp eq i64 %inc, %N
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:                                          ; preds = %for.body, %entry
  ret void
}


define void @u16_to_f32(float* noalias nocapture %dst, i16* noalias nocapture readonly %src, i64 %N) #0 {
; CHECK-LABEL: @u16_to_f32(
; CHECK: vector.body
; CHECK:   %{{.*}} = uitofp <vscale x 8 x i16> %{{.*}} to <vscale x 8 x float>
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.07 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i16, i16* %src, i64 %i.07
  %0 = load i16, i16* %arrayidx, align 2
  %conv = uitofp i16 %0 to float
  %arrayidx1 = getelementptr inbounds float, float* %dst, i64 %i.07
  store float %conv, float* %arrayidx1, align 4
  %inc = add nuw nsw i64 %i.07, 1
  %exitcond.not = icmp eq i64 %inc, %N
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:                                          ; preds = %for.body, %entry
  ret void
}


define void @u64_to_f16(half* noalias nocapture %dst, i64* noalias nocapture readonly %src, i64 %N) #0 {
; CHECK-LABEL: @u64_to_f16(
; CHECK:      vector.body
; CHECK:        %{{.*}} = uitofp <vscale x 8 x i64> %{{.*}} to <vscale x 8 x half>
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.08 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i64, i64* %src, i64 %i.08
  %0 = load i64, i64* %arrayidx, align 8
  %conv1 = uitofp i64 %0 to half
  %arrayidx2 = getelementptr inbounds half, half* %dst, i64 %i.08
  store half %conv1, half* %arrayidx2, align 2
  %inc = add nuw nsw i64 %i.08, 1
  %exitcond.not = icmp eq i64 %inc, %N
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:                                          ; preds = %for.body, %entry
  ret void
}


define void @s64_to_f16(half* noalias nocapture %dst, i64* noalias nocapture readonly %src, i64 %N) #0 {
; CHECK-LABEL: @s64_to_f16(
; CHECK:      vector.body
; CHECK:        %{{.*}} = sitofp <vscale x 8 x i64> %{{.*}} to <vscale x 8 x half>
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.08 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i64, i64* %src, i64 %i.08
  %0 = load i64, i64* %arrayidx, align 8
  %conv1 = sitofp i64 %0 to half
  %arrayidx2 = getelementptr inbounds half, half* %dst, i64 %i.08
  store half %conv1, half* %arrayidx2, align 2
  %inc = add nuw nsw i64 %i.08, 1
  %exitcond.not = icmp eq i64 %inc, %N
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:                                          ; preds = %for.body, %entry
  ret void
}


define void @s8_to_s32(i32* noalias nocapture %dst, i8* noalias nocapture readonly %src, i64 %N) #0 {
; CHECK-LABEL: @s8_to_s32(
; CHECK: vector.body
; CHECK:   %{{.*}} = sext <vscale x 8 x i8> %{{.*}} to <vscale x 8 x i32>
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.07 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i8, i8* %src, i64 %i.07
  %0 = load i8, i8* %arrayidx, align 1
  %conv = sext i8 %0 to i32
  %arrayidx1 = getelementptr inbounds i32, i32* %dst, i64 %i.07
  store i32 %conv, i32* %arrayidx1, align 4
  %inc = add nuw nsw i64 %i.07, 1
  %exitcond.not = icmp eq i64 %inc, %N
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:                                          ; preds = %for.body, %entry
  ret void
}


define void @u8_to_u16(i16* noalias nocapture %dst, i8* noalias nocapture readonly %src, i64 %N) #0 {
; CHECK-LABEL: @u8_to_u16(
; CHECK: vector.body
; CHECK:   %{{.*}} = zext <vscale x 8 x i8> %{{.*}} to <vscale x 8 x i16>
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.07 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i8, i8* %src, i64 %i.07
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i16
  %arrayidx1 = getelementptr inbounds i16, i16* %dst, i64 %i.07
  store i16 %conv, i16* %arrayidx1, align 2
  %inc = add nuw nsw i64 %i.07, 1
  %exitcond.not = icmp eq i64 %inc, %N
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:                                          ; preds = %for.body, %entry
  ret void
}


define void @s64_to_s8(i8* noalias nocapture %dst, i64* noalias nocapture readonly %src, i64 %N) #0 {
; CHECK-LABEL: @s64_to_s8(
; CHECK: vector.body
; CHECK:   %{{.*}} = trunc <vscale x 8 x i64> %{{.*}} to <vscale x 8 x i8>
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.07 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i64, i64* %src, i64 %i.07
  %0 = load i64, i64* %arrayidx, align 8
  %conv = trunc i64 %0 to i8
  %arrayidx1 = getelementptr inbounds i8, i8* %dst, i64 %i.07
  store i8 %conv, i8* %arrayidx1, align 1
  %inc = add nuw nsw i64 %i.07, 1
  %exitcond.not = icmp eq i64 %inc, %N
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:                                          ; preds = %for.body, %entry
  ret void
}


attributes #0 = { "target-features"="+sve" }

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.vectorize.width", i32 8}
!3 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
!4 = !{!"llvm.loop.interleave.count", i32 1}
!5 = !{!"llvm.loop.vectorize.enable", i1 true}
