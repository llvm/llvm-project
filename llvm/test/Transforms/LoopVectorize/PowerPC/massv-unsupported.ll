; RUN: opt -vector-library=MASSV -passes=inject-tli-mappings,loop-vectorize -force-vector-interleave=1 -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

declare double @ceil(double) #0
declare float @fabsf(float) #0

declare double @llvm.sqrt.f64(double) #0
declare float @llvm.sqrt.f32(float) #0

; Vector counterpart of ceil is unsupported in MASSV library.
define void @ceil_f64(double* nocapture %varray) {
; CHECK-LABEL: @ceil_f64(
; CHECK-NOT: __ceild2_massv{{.*}}<2 x double>
; CHECK-NOT: __ceild2_P8{{.*}}<2 x double>
; CHECK-NOT: __ceild2{{.*}}<2 x double>
; CHECK: ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @ceil(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

; Vector counterpart of fabs is unsupported in MASSV library.
define void @fabs_f32(float* nocapture %varray) {
; CHECK-LABEL: @fabs_f32(
; CHECK-NOT: __fabsf4_massv{{.*}}<4 x float>
; CHECK-NOT: __fabsf4_P8{{.*}}<4 x float>
; CHECK-NOT: __fabsf4{{.*}}<4 x float>
; CHECK: ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @fabsf(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

; sqrt intrinsics are converted to their vector counterpart intrinsics.
; They are not lowered to MASSV entries.
define void @sqrt_f64_intrinsic(double* nocapture %varray) {
; CHECK-LABEL: @sqrt_f64_intrinsic(
; CHECK: llvm.sqrt.v2f64{{.*}}<2 x double>
; CHECK: ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to double
  %call = tail call double @llvm.sqrt.f64(double %conv)
  %arrayidx = getelementptr inbounds double, double* %varray, i64 %iv
  store double %call, double* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

define void @sqrt_f32_intrinsic(float* nocapture %varray) {
; CHECK-LABEL: @sqrt_f32_intrinsic(
; CHECK: llvm.sqrt.v4f32{{.*}}<4 x float>
; CHECK: ret void
;
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %tmp = trunc i64 %iv to i32
  %conv = sitofp i32 %tmp to float
  %call = tail call float @llvm.sqrt.f32(float %conv)
  %arrayidx = getelementptr inbounds float, float* %varray, i64 %iv
  store float %call, float* %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1000
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

attributes #0 = { nounwind }
