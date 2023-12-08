; RUN: llc -mtriple=powerpc64-unknown-linux-gnu -verify-machineinstrs -mcpu=pwr7 < %s | FileCheck %s

declare float @fabsf(float)

declare float @fminf(float, float)
declare double @fmin(double, double)
declare float @llvm.minnum.f32(float, float)
declare double @llvm.minnum.f64(double, double)

declare float @fmaxf(float, float)
declare double @fmax(double, double)
declare float @llvm.maxnum.f32(float, float)
declare double @llvm.maxnum.f64(double, double)

declare <4 x float> @llvm.minnum.v4f32(<4 x float>, <4 x float>)
declare <4 x double> @llvm.minnum.v4f64(<4 x double>, <4 x double>)
declare <4 x float> @llvm.maxnum.v4f32(<4 x float>, <4 x float>)
declare <4 x double> @llvm.maxnum.v4f64(<4 x double>, <4 x double>)

define void @test1(float %f, ptr %fp) {
entry:
  br label %loop_body

loop_body:
  %invar_address.dim.0.01 = phi i64 [ 0, %entry ], [ %1, %loop_body ]
  %0 = call float @llvm.minnum.f32(float %f, float 1.0)
  store float %0, ptr %fp, align 4
  %1 = add i64 %invar_address.dim.0.01, 1
  %2 = icmp eq i64 %1, 2
  br i1 %2, label %loop_exit, label %loop_body

loop_exit:
  ret void
}

; CHECK-LABEL: test1:
; CHECK-NOT: mtctr
; CHECK: xsmindp
; CHECK-NOT: xsmindp
; CHECK-NOT: mtctr
; CHECK: blr

define void @test1v(<4 x float> %f, ptr %fp) {
entry:
  br label %loop_body

loop_body:
  %invar_address.dim.0.01 = phi i64 [ 0, %entry ], [ %1, %loop_body ]
  %0 = call <4 x float> @llvm.minnum.v4f32(<4 x float> %f, <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>)
  store <4 x float> %0, ptr %fp, align 16
  %1 = add i64 %invar_address.dim.0.01, 1
  %2 = icmp eq i64 %1, 4
  br i1 %2, label %loop_exit, label %loop_body

loop_exit:
  ret void
}

; CHECK-LABEL: test1v:
; CHECK: xvminsp
; CHECK-NOT: xsmindp
; CHECK: mtctr
; CHECK-NOT: xsmindp
; CHECK: blr

define void @test1a(float %f, ptr %fp) {
entry:
  br label %loop_body

loop_body:
  %invar_address.dim.0.01 = phi i64 [ 0, %entry ], [ %1, %loop_body ]
  %0 = call float @fminf(float %f, float 1.0) readnone
  store float %0, ptr %fp, align 4
  %1 = add i64 %invar_address.dim.0.01, 1
  %2 = icmp eq i64 %1, 2
  br i1 %2, label %loop_exit, label %loop_body

loop_exit:
  ret void
}

; CHECK-LABEL: test1a:
; CHECK-NOT: mtctr
; CHECK: xsmindp
; CHECK-NOT: xsmindp
; CHECK-NOT: mtctr
; CHECK: blr

define void @test2(float %f, ptr %fp) {
entry:
  br label %loop_body

loop_body:
  %invar_address.dim.0.01 = phi i64 [ 0, %entry ], [ %1, %loop_body ]
  %0 = call float @llvm.maxnum.f32(float %f, float 1.0)
  store float %0, ptr %fp, align 4
  %1 = add i64 %invar_address.dim.0.01, 1
  %2 = icmp eq i64 %1, 2
  br i1 %2, label %loop_exit, label %loop_body

loop_exit:
  ret void
}

; CHECK-LABEL: test2:
; CHECK-NOT: mtctr
; CHECK: xsmaxdp
; CHECK-NOT: xsmaxdp
; CHECK-NOT: mtctr
; CHECK: blr

define void @test2v(<4 x double> %f, ptr %fp) {
entry:
  br label %loop_body

loop_body:
  %invar_address.dim.0.01 = phi i64 [ 0, %entry ], [ %1, %loop_body ]
  %0 = call <4 x double> @llvm.maxnum.v4f64(<4 x double> %f, <4 x double> <double 1.0, double 1.0, double 1.0, double 1.0>)
  store <4 x double> %0, ptr %fp, align 16
  %1 = add i64 %invar_address.dim.0.01, 1
  %2 = icmp eq i64 %1, 4
  br i1 %2, label %loop_exit, label %loop_body

loop_exit:
  ret void
}

; CHECK-LABEL: test2v:
; CHECK: xvmaxdp
; CHECK: xvmaxdp
; CHECK-NOT: xsmaxdp
; CHECK: mtctr
; CHECK-NOT: xsmaxdp
; CHECK: blr

define void @test2a(float %f, ptr %fp) {
entry:
  br label %loop_body

loop_body:
  %invar_address.dim.0.01 = phi i64 [ 0, %entry ], [ %1, %loop_body ]
  %0 = call float @fmaxf(float %f, float 1.0) readnone
  store float %0, ptr %fp, align 4
  %1 = add i64 %invar_address.dim.0.01, 1
  %2 = icmp eq i64 %1, 2
  br i1 %2, label %loop_exit, label %loop_body

loop_exit:
  ret void
}

; CHECK-LABEL: test2a:
; CHECK-NOT: mtctr
; CHECK: xsmaxdp
; CHECK-NOT: xsmaxdp
; CHECK-NOT: mtctr
; CHECK: blr

define void @test3(double %f, ptr %fp) {
entry:
  br label %loop_body

loop_body:
  %invar_address.dim.0.01 = phi i64 [ 0, %entry ], [ %1, %loop_body ]
  %0 = call double @llvm.minnum.f64(double %f, double 1.0)
  store double %0, ptr %fp, align 8
  %1 = add i64 %invar_address.dim.0.01, 1
  %2 = icmp eq i64 %1, 2
  br i1 %2, label %loop_exit, label %loop_body

loop_exit:
  ret void
}

; CHECK-LABEL: test3:
; CHECK-NOT: mtctr
; CHECK: xsmindp
; CHECK-NOT: xsmindp
; CHECK-NOT: mtctr
; CHECK: blr

define void @test3a(double %f, ptr %fp) {
entry:
  br label %loop_body

loop_body:
  %invar_address.dim.0.01 = phi i64 [ 0, %entry ], [ %1, %loop_body ]
  %0 = call double @fmin(double %f, double 1.0) readnone
  store double %0, ptr %fp, align 8
  %1 = add i64 %invar_address.dim.0.01, 1
  %2 = icmp eq i64 %1, 2
  br i1 %2, label %loop_exit, label %loop_body

loop_exit:
  ret void
}

; CHECK-LABEL: test3a:
; CHECK-NOT: mtctr
; CHECK: xsmindp
; CHECK-NOT: xsmindp
; CHECK-NOT: mtctr
; CHECK: blr

define void @test4(double %f, ptr %fp) {
entry:
  br label %loop_body

loop_body:
  %invar_address.dim.0.01 = phi i64 [ 0, %entry ], [ %1, %loop_body ]
  %0 = call double @llvm.maxnum.f64(double %f, double 1.0)
  store double %0, ptr %fp, align 8
  %1 = add i64 %invar_address.dim.0.01, 1
  %2 = icmp eq i64 %1, 2
  br i1 %2, label %loop_exit, label %loop_body

loop_exit:
  ret void
}

; CHECK-LABEL: test4:
; CHECK-NOT: mtctr
; CHECK: xsmaxdp
; CHECK-NOT: xsmaxdp
; CHECK-NOT: mtctr
; CHECK: blr

define void @test4a(double %f, ptr %fp) {
entry:
  br label %loop_body

loop_body:
  %invar_address.dim.0.01 = phi i64 [ 0, %entry ], [ %1, %loop_body ]
  %0 = call double @fmax(double %f, double 1.0) readnone
  store double %0, ptr %fp, align 8
  %1 = add i64 %invar_address.dim.0.01, 1
  %2 = icmp eq i64 %1, 2
  br i1 %2, label %loop_exit, label %loop_body

loop_exit:
  ret void
}

; CHECK-LABEL: test4a:
; CHECK-NOT: mtctr
; CHECK: xsmaxdp
; CHECK-NOT: xsmaxdp
; CHECK-NOT: mtctr
; CHECK: blr

