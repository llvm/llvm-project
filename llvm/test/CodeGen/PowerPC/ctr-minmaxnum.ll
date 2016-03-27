; RUN: llc -mcpu=pwr7 < %s | FileCheck %s
; RUN: llc -mcpu=a2q < %s | FileCheck %s --check-prefix=QPX
target triple = "powerpc64-unknown-linux-gnu"

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

define void @test1(float %f, float* %fp) {
entry:
  br label %loop_body

loop_body:
  %invar_address.dim.0.01 = phi i64 [ 0, %entry ], [ %1, %loop_body ]
  %0 = call float @llvm.minnum.f32(float %f, float 1.0)
  store float %0, float* %fp, align 4
  %1 = add i64 %invar_address.dim.0.01, 1
  %2 = icmp eq i64 %1, 2
  br i1 %2, label %loop_exit, label %loop_body

loop_exit:
  ret void
}

; CHECK-LABEL: test1:
; CHECK-NOT: mtctr
; CHECK: bl fminf

define void @test1v(<4 x float> %f, <4 x float>* %fp) {
entry:
  br label %loop_body

loop_body:
  %invar_address.dim.0.01 = phi i64 [ 0, %entry ], [ %1, %loop_body ]
  %0 = call <4 x float> @llvm.minnum.v4f32(<4 x float> %f, <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>)
  store <4 x float> %0, <4 x float>* %fp, align 16
  %1 = add i64 %invar_address.dim.0.01, 1
  %2 = icmp eq i64 %1, 2
  br i1 %2, label %loop_exit, label %loop_body

loop_exit:
  ret void
}

; CHECK-LABEL: test1v:
; CHECK-NOT: mtctr
; CHECK: bl fminf

; QPX-LABEL: test1v:
; QPX: mtctr
; QPX-NOT: bl fminf
; QPX: blr

define void @test1a(float %f, float* %fp) {
entry:
  br label %loop_body

loop_body:
  %invar_address.dim.0.01 = phi i64 [ 0, %entry ], [ %1, %loop_body ]
  %0 = call float @fminf(float %f, float 1.0) readnone
  store float %0, float* %fp, align 4
  %1 = add i64 %invar_address.dim.0.01, 1
  %2 = icmp eq i64 %1, 2
  br i1 %2, label %loop_exit, label %loop_body

loop_exit:
  ret void
}

; CHECK-LABEL: test1a:
; CHECK-NOT: mtctr
; CHECK: bl fminf

define void @test2(float %f, float* %fp) {
entry:
  br label %loop_body

loop_body:
  %invar_address.dim.0.01 = phi i64 [ 0, %entry ], [ %1, %loop_body ]
  %0 = call float @llvm.maxnum.f32(float %f, float 1.0)
  store float %0, float* %fp, align 4
  %1 = add i64 %invar_address.dim.0.01, 1
  %2 = icmp eq i64 %1, 2
  br i1 %2, label %loop_exit, label %loop_body

loop_exit:
  ret void
}

; CHECK-LABEL: test2:
; CHECK-NOT: mtctr
; CHECK: bl fmaxf

define void @test2v(<4 x double> %f, <4 x double>* %fp) {
entry:
  br label %loop_body

loop_body:
  %invar_address.dim.0.01 = phi i64 [ 0, %entry ], [ %1, %loop_body ]
  %0 = call <4 x double> @llvm.maxnum.v4f64(<4 x double> %f, <4 x double> <double 1.0, double 1.0, double 1.0, double 1.0>)
  store <4 x double> %0, <4 x double>* %fp, align 16
  %1 = add i64 %invar_address.dim.0.01, 1
  %2 = icmp eq i64 %1, 2
  br i1 %2, label %loop_exit, label %loop_body

loop_exit:
  ret void
}

; CHECK-LABEL: test2v:
; CHECK-NOT: mtctr
; CHECK: bl fmax

; QPX-LABEL: test2v:
; QPX: mtctr
; QPX-NOT: bl fmax
; QPX: blr

define void @test2a(float %f, float* %fp) {
entry:
  br label %loop_body

loop_body:
  %invar_address.dim.0.01 = phi i64 [ 0, %entry ], [ %1, %loop_body ]
  %0 = call float @fmaxf(float %f, float 1.0) readnone
  store float %0, float* %fp, align 4
  %1 = add i64 %invar_address.dim.0.01, 1
  %2 = icmp eq i64 %1, 2
  br i1 %2, label %loop_exit, label %loop_body

loop_exit:
  ret void
}

; CHECK-LABEL: test2a:
; CHECK-NOT: mtctr
; CHECK: bl fmaxf

define void @test3(double %f, double* %fp) {
entry:
  br label %loop_body

loop_body:
  %invar_address.dim.0.01 = phi i64 [ 0, %entry ], [ %1, %loop_body ]
  %0 = call double @llvm.minnum.f64(double %f, double 1.0)
  store double %0, double* %fp, align 8
  %1 = add i64 %invar_address.dim.0.01, 1
  %2 = icmp eq i64 %1, 2
  br i1 %2, label %loop_exit, label %loop_body

loop_exit:
  ret void
}

; CHECK-LABEL: test3:
; CHECK-NOT: mtctr
; CHECK: bl fmin

define void @test3a(double %f, double* %fp) {
entry:
  br label %loop_body

loop_body:
  %invar_address.dim.0.01 = phi i64 [ 0, %entry ], [ %1, %loop_body ]
  %0 = call double @fmin(double %f, double 1.0) readnone
  store double %0, double* %fp, align 8
  %1 = add i64 %invar_address.dim.0.01, 1
  %2 = icmp eq i64 %1, 2
  br i1 %2, label %loop_exit, label %loop_body

loop_exit:
  ret void
}

; CHECK-LABEL: test3a:
; CHECK-NOT: mtctr
; CHECK: bl fmin

define void @test4(double %f, double* %fp) {
entry:
  br label %loop_body

loop_body:
  %invar_address.dim.0.01 = phi i64 [ 0, %entry ], [ %1, %loop_body ]
  %0 = call double @llvm.maxnum.f64(double %f, double 1.0)
  store double %0, double* %fp, align 8
  %1 = add i64 %invar_address.dim.0.01, 1
  %2 = icmp eq i64 %1, 2
  br i1 %2, label %loop_exit, label %loop_body

loop_exit:
  ret void
}

; CHECK-LABEL: test4:
; CHECK-NOT: mtctr
; CHECK: bl fmax

define void @test4a(double %f, double* %fp) {
entry:
  br label %loop_body

loop_body:
  %invar_address.dim.0.01 = phi i64 [ 0, %entry ], [ %1, %loop_body ]
  %0 = call double @fmax(double %f, double 1.0) readnone
  store double %0, double* %fp, align 8
  %1 = add i64 %invar_address.dim.0.01, 1
  %2 = icmp eq i64 %1, 2
  br i1 %2, label %loop_exit, label %loop_body

loop_exit:
  ret void
}

; CHECK-LABEL: test4a:
; CHECK-NOT: mtctr
; CHECK: bl fmax

