; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf \
; RUN:   -enable-arm-maskedgatscat=false \
; RUN:   -tail-predication=enabled -loop-vectorize -S < %s | \
; RUN:   FileCheck %s -check-prefixes=CHECK,PREFER-FOLDING

; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf -mattr=-mve \
; RUN:   -tail-predication=enabled -loop-vectorize \
; RUN:   -enable-arm-maskedgatscat=false \
; RUN:   -enable-arm-maskedldst=true -S < %s | \
; RUN:   FileCheck %s -check-prefixes=CHECK,NO-FOLDING

; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf -mattr=+mve \
; RUN:   -tail-predication=enabled -loop-vectorize \
; RUN:   -enable-arm-maskedgatscat=false \
; RUN:   -enable-arm-maskedldst=false -S < %s | \
; RUN:   FileCheck %s -check-prefixes=CHECK,NO-FOLDING

; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf -mattr=+mve \
; RUN:   -tail-predication=disabled -loop-vectorize \
; RUN:   -enable-arm-maskedgatscat=false \
; RUN:   -enable-arm-maskedldst=true -S < %s | \
; RUN:   FileCheck %s -check-prefixes=CHECK,NO-FOLDING

; Disabling the low-overhead branch extension will make
; 'isHardwareLoopProfitable' return false, so that we test avoiding folding for
; these cases.
; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf -mattr=+mve,-lob \
; RUN:   -tail-predication=enabled -loop-vectorize \
; RUN:   -enable-arm-maskedgatscat=false \
; RUN:   -enable-arm-maskedldst=true -S < %s | \
; RUN:   FileCheck %s -check-prefixes=CHECK,NO-FOLDING

; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf -mattr=+mve.fp \
; RUN:   -tail-predication=enabled -loop-vectorize \
; RUN:   -enable-arm-maskedgatscat=false \
; RUN:   -enable-arm-maskedldst=true -S < %s | \
; RUN:   FileCheck %s -check-prefixes=CHECK,PREFER-FOLDING

; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf -mattr=+mve.fp \
; RUN:   -prefer-predicate-over-epilogue=scalar-epilogue \
; RUN:   -tail-predication=enabled -loop-vectorize \
; RUN:   -enable-arm-maskedgatscat=false \
; RUN:   -enable-arm-maskedldst=true -S < %s | \
; RUN:   FileCheck %s -check-prefixes=CHECK,NO-FOLDING

; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf -mattr=+mve.fp \
; RUN:   -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN:   -tail-predication=enabled -loop-vectorize \
; RUN:   -enable-arm-maskedgatscat=false \
; RUN:   -enable-arm-maskedldst=true -S < %s | \
; RUN:   FileCheck %s -check-prefixes=CHECK

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

define void @prefer_folding(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C) #0 {
; CHECK-LABEL:    prefer_folding(
; PREFER-FOLDING: vector.body:
; PREFER-FOLDING: %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; PREFER-FOLDING: %[[VIVELEM0:.*]] = add i32 %index, 0
; PREFER-FOLDING: %active.lane.mask = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %[[VIVELEM0]], i32 431)
; PREFER-FOLDING: call <4 x i32> @llvm.masked.load.v4i32.p0v4i32({{.*}}, <4 x i1> %active.lane.mask,
; PREFER-FOLDING: call <4 x i32> @llvm.masked.load.v4i32.p0v4i32({{.*}}, <4 x i1> %active.lane.mask,
; PREFER-FOLDING: call void @llvm.masked.store.v4i32.p0v4i32({{.*}}, <4 x i1> %active.lane.mask
; PREFER-FOLDING: br i1 %{{.*}}, label %{{.*}}, label %vector.body
;
; NO-FOLDING-NOT: call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(
; NO-FOLDING-NOT: call void @llvm.masked.store.v4i32.p0v4i32(
; NO-FOLDING:     br i1 %{{.*}}, label %{{.*}}, label %for.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %i.09
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %C, i32 %i.09
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %1, %0
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i32 %i.09
  store i32 %add, i32* %arrayidx2, align 4
  %add3 = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %add3, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define void @mixed_types(i16* noalias nocapture %A, i16* noalias nocapture readonly %B, i16* noalias nocapture readonly %C, i32* noalias nocapture %D, i32* noalias nocapture readonly %E, i32* noalias nocapture readonly %F) #0 {
; CHECK-LABEL:        mixed_types(
; PREFER-FOLDING:     vector.body:
; PREFER-FOLDING:     call <4 x i16> @llvm.masked.load.v4i16.p0v4i16
; PREFER-FOLDING:     call <4 x i16> @llvm.masked.load.v4i16.p0v4i16
; PREFER-FOLDING:     call void @llvm.masked.store.v4i16.p0v4i16
; PREFER-FOLDING:     call <4 x i32> @llvm.masked.load.v4i32.p0v4i32
; PREFER-FOLDING:     call <4 x i32> @llvm.masked.load.v4i32.p0v4i32
; PREFER-FOLDING:     call void @llvm.masked.store.v4i32.p0v4i32
; PREFER-FOLDING:     br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.018 = phi i32 [ 0, %entry ], [ %add9, %for.body ]
  %arrayidx = getelementptr inbounds i16, i16* %B, i32 %i.018
  %0 = load i16, i16* %arrayidx, align 2
  %arrayidx1 = getelementptr inbounds i16, i16* %C, i32 %i.018
  %1 = load i16, i16* %arrayidx1, align 2
  %add = add i16 %1, %0
  %arrayidx4 = getelementptr inbounds i16, i16* %A, i32 %i.018
  store i16 %add, i16* %arrayidx4, align 2
  %arrayidx5 = getelementptr inbounds i32, i32* %E, i32 %i.018
  %2 = load i32, i32* %arrayidx5, align 4
  %arrayidx6 = getelementptr inbounds i32, i32* %F, i32 %i.018
  %3 = load i32, i32* %arrayidx6, align 4
  %add7 = add nsw i32 %3, %2
  %arrayidx8 = getelementptr inbounds i32, i32* %D, i32 %i.018
  store i32 %add7, i32* %arrayidx8, align 4
  %add9 = add nuw nsw i32 %i.018, 1
  %exitcond = icmp eq i32 %add9, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define void @zero_extending_load_allowed(i32* noalias nocapture %A, i8* noalias nocapture readonly %B, i32* noalias nocapture readonly %C) #0 {
; CHECK-LABEL:    zero_extending_load_allowed(
; PREFER-FOLDING: vector.body:
; PREFER-FOLDING: call <4 x i8> @llvm.masked.load.v4i8.p0v4i8
; PREFER-FOLDING: call <4 x i32> @llvm.masked.load.v4i32.p0v4i32
; PREFER-FOLDING: call void @llvm.masked.store.v4i32.p0v4i32
; PREFER-FOLDING: br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds i8, i8* %B, i32 %i.09
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %arrayidx1 = getelementptr inbounds i32, i32* %C, i32 %i.09
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %1, %conv
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i32 %i.09
  store i32 %add, i32* %arrayidx2, align 4
  %add3 = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %add3, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define void @sign_extending_load_allowed(i32* noalias nocapture %A, i8* noalias nocapture readonly %B, i32* noalias nocapture readonly %C) #0 {
; CHECK-LABEL:    sign_extending_load_allowed(
; PREFER-FOLDING: vector.body:
; PREFER-FOLDING: call <4 x i8> @llvm.masked.load.v4i8.p0v4i8
; PREFER-FOLDING: call <4 x i32> @llvm.masked.load.v4i32.p0v4i32
; PREFER-FOLDING: call void @llvm.masked.store.v4i32.p0v4i32
; PREFER-FOLDING: br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds i8, i8* %B, i32 %i.09
  %0 = load i8, i8* %arrayidx, align 1
  %conv = sext i8 %0 to i32
  %arrayidx1 = getelementptr inbounds i32, i32* %C, i32 %i.09
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %1, %conv
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i32 %i.09
  store i32 %add, i32* %arrayidx2, align 4
  %add3 = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %add3, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define void @narrowing_store_allowed(i8* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C) #0 {
; CHECK-LABEL:    narrowing_store_allowed(
; PREFER-FOLDING: call void @llvm.masked.store.v4i8.p0v4i8
; PREFER-FOLDING: br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %i.09
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %C, i32 %i.09
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %1, %0
  %conv = trunc i32 %add to i8
  %arrayidx2 = getelementptr inbounds i8, i8* %A, i32 %i.09
  store i8 %conv, i8* %arrayidx2, align 1
  %add3 = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %add3, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

@tab = common global [32 x i8] zeroinitializer, align 1

define i32 @icmp_not_allowed() #0 {
; CHECK-LABEL:        icmp_not_allowed(
; PREFER-FOLDING:     vector.body:
; PREFER-FOLDING-NOT: llvm.masked.load
; PREFER-FOLDING-NOT: llvm.masked.store
; PREFER-FOLDING:     br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.body:
  %i.08 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [32 x i8], [32 x i8]* @tab, i32 0, i32 %i.08
  %0 = load i8, i8* %arrayidx, align 1
  %cmp1 = icmp eq i8 %0, 0
  %. = select i1 %cmp1, i8 2, i8 1
  store i8 %., i8* %arrayidx, align 1
  %inc = add nsw i32 %i.08, 1
  %exitcond = icmp slt i32 %inc, 1000
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret i32 0
}

@ftab = common global [32 x float] zeroinitializer, align 1

define float @fcmp_not_allowed() #0 {
; CHECK-LABEL:        fcmp_not_allowed(
; PREFER-FOLDING:     vector.body:
; PREFER-FOLDING-NOT: llvm.masked.load
; PREFER-FOLDING-NOT: llvm.masked.store
; PREFER-FOLDING:     br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.body:
  %i.08 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds [32 x float], [32 x float]* @ftab, i32 0, i32 %i.08
  %0 = load float, float* %arrayidx, align 4
  %cmp1 = fcmp oeq float %0, 0.000000e+00
  %. = select i1 %cmp1, float 2.000000e+00, float 1.000000e+00
  store float %., float* %arrayidx, align 4
  %inc = add nsw i32 %i.08, 1
  %exitcond = icmp slt i32 %inc, 999
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret float 0.000000e+00
}

define void @pragma_vect_predicate_disable(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C) #0 {
; CHECK-LABEL:        pragma_vect_predicate_disable(
; PREFER-FOLDING:     vector.body:
; PREFER-FOLDING-NOT: call <4 x i32> @llvm.masked.load.v4i32.p0v4i32
; PREFER-FOLDING-NOT: call <4 x i32> @llvm.masked.load.v4i32.p0v4i32
; PREFER-FOLDING-NOT: call void @llvm.masked.store.v4i32.p0v4i32
; PREFER-FOLDING:     br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %i.09
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %C, i32 %i.09
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %1, %0
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i32 %i.09
  store i32 %add, i32* %arrayidx2, align 4
  %add3 = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %add3, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !llvm.loop !7
}

define void @stride_4(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C) #0 {
; CHECK-LABEL:        stride_4(
; PREFER-FOLDING:     vector.body:
; PREFER-FOLDING-NOT: llvm.masked.load
; PREFER-FOLDING-NOT: llvm.masked.store
; PREFER-FOLDING:     br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %i.09
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %C, i32 %i.09
  %1 = load i32, i32* %arrayidx1, align 4
  %add = add nsw i32 %1, %0
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i32 %i.09
  store i32 %add, i32* %arrayidx2, align 4
  %add3 = add nuw nsw i32 %i.09, 4
  %cmp = icmp ult i32 %add3, 731
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !5
}

define dso_local void @half(half* noalias nocapture %A, half* noalias nocapture readonly %B, half* noalias nocapture readonly %C) #0 {
; CHECK-LABEL:    half(
; PREFER-FOLDING: vector.body:
; PREFER-FOLDING: call <8 x half> @llvm.masked.load.v8f16.p0v8f16
; PREFER-FOLDING: call <8 x half> @llvm.masked.load.v8f16.p0v8f16
; PREFER-FOLDING: call void @llvm.masked.store.v8f16.p0v8f16
; PREFER-FOLDING: br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds half, half* %B, i32 %i.09
  %0 = load half, half* %arrayidx, align 2
  %arrayidx1 = getelementptr inbounds half, half* %C, i32 %i.09
  %1 = load half, half* %arrayidx1, align 2
  %add = fadd fast half %1, %0
  %arrayidx2 = getelementptr inbounds half, half* %A, i32 %i.09
  store half %add, half* %arrayidx2, align 2
  %add3 = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %add3, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define void @float(float* noalias nocapture %A, float* noalias nocapture readonly %B, float* noalias nocapture readonly %C) #0 {
; CHECK-LABEL:    float(
; PREFER-FOLDING: vector.body:
; PREFER-FOLDING: %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; PREFER-FOLDING: %[[VIVELEM0:.*]] = add i32 %index, 0
; PREFER-FOLDING: %active.lane.mask = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %[[VIVELEM0]], i32 431)
; PREFER-FOLDING: call <4 x float> @llvm.masked.load.v4f32.p0v4f32({{.*}}%active.lane.mask
; PREFER-FOLDING: call <4 x float> @llvm.masked.load.v4f32.p0v4f32({{.*}}%active.lane.mask
; PREFER-FOLDING: call void @llvm.masked.store.v4f32.p0v4f32({{.*}}%active.lane.mask
; PREFER-FOLDING: %index.next = add i32 %index, 4
; PREFER-FOLDING: br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %B, i32 %i.09
  %0 = load float, float* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, float* %C, i32 %i.09
  %1 = load float, float* %arrayidx1, align 4
  %add = fadd fast float %1, %0
  %arrayidx2 = getelementptr inbounds float, float* %A, i32 %i.09
  store float %add, float* %arrayidx2, align 4
  %add3 = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %add3, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !llvm.loop !10
}

define void @fpext_allowed(float* noalias nocapture %A, half* noalias nocapture readonly %B, float* noalias nocapture readonly %C) #0 {
; CHECK-LABEL:        fpext_allowed(
; PREFER-FOLDING:     vector.body:
; PREFER-FOLDING-NOT: llvm.masked.load
; PREFER-FOLDING-NOT: llvm.masked.store
; PREFER-FOLDING:     br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds half, half* %B, i32 %i.09
  %0 = load half, half* %arrayidx, align 2
  %conv = fpext half %0 to float
  %arrayidx1 = getelementptr inbounds float, float* %C, i32 %i.09
  %1 = load float, float* %arrayidx1, align 4
  %add = fadd fast float %1, %conv
  %arrayidx2 = getelementptr inbounds float, float* %A, i32 %i.09
  store float %add, float* %arrayidx2, align 4
  %add3 = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %add3, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

define void @fptrunc_allowed(half* noalias nocapture %A, float* noalias nocapture readonly %B, float* noalias nocapture readonly %C) #0 {
; CHECK-LABEL:        fptrunc_allowed(
; PREFER-FOLDING:     vector.body:
; PREFER-FOLDING-NOT: llvm.masked.load
; PREFER-FOLDING-NOT: llvm.masked.store
; PREFER-FOLDING:     br i1 %{{.*}}, label %{{.*}}, label %vector.body
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %i.09 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %B, i32 %i.09
  %0 = load float, float* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, float* %C, i32 %i.09
  %1 = load float, float* %arrayidx1, align 4
  %add = fadd fast float %1, %0
  %conv = fptrunc float %add to half
  %arrayidx2 = getelementptr inbounds half, half* %A, i32 %i.09
  store half %conv, half* %arrayidx2, align 2
  %add3 = add nuw nsw i32 %i.09, 1
  %exitcond = icmp eq i32 %add3, 431
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

attributes #0 = { nofree norecurse nounwind "target-features"="+armv8.1-m.main,+mve.fp" }

!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.vectorize.enable", i1 true}

!7 = distinct !{!7, !8}
!8 = !{!"llvm.loop.vectorize.predicate.enable", i1 false}

!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.vectorize.width", i32 4}
