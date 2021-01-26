; RUN: opt < %s -instsimplify -S | FileCheck %s
; RUN: opt < %s -instsimplify -disable-simplify-libcalls -S | FileCheck %s --check-prefix=FNOBUILTIN

declare double @acos(double) readnone nounwind willreturn
declare double @asin(double) readnone nounwind willreturn
declare double @atan(double) readnone nounwind willreturn
declare double @atan2(double, double) readnone nounwind willreturn
declare double @ceil(double) readnone nounwind willreturn
declare double @cos(double) readnone nounwind willreturn
declare double @cosh(double) readnone nounwind willreturn
declare double @exp(double) readnone nounwind willreturn
declare double @exp2(double) readnone nounwind willreturn
declare double @fabs(double) readnone nounwind willreturn
declare double @floor(double) readnone nounwind willreturn
declare double @fmod(double, double) readnone nounwind willreturn
declare double @log(double) readnone nounwind willreturn
declare double @log10(double) readnone nounwind willreturn
declare double @pow(double, double) readnone nounwind willreturn
declare double @round(double) readnone nounwind willreturn
declare double @sin(double) readnone nounwind willreturn
declare double @sinh(double) readnone nounwind willreturn
declare double @sqrt(double) readnone nounwind willreturn
declare double @tan(double) readnone nounwind willreturn
declare double @tanh(double) readnone nounwind willreturn

declare float @acosf(float) readnone nounwind willreturn
declare float @asinf(float) readnone nounwind willreturn
declare float @atanf(float) readnone nounwind willreturn
declare float @atan2f(float, float) readnone nounwind willreturn
declare float @ceilf(float) readnone nounwind willreturn
declare float @cosf(float) readnone nounwind willreturn
declare float @coshf(float) readnone nounwind willreturn
declare float @expf(float) readnone nounwind willreturn
declare float @exp2f(float) readnone nounwind willreturn
declare float @fabsf(float) readnone nounwind willreturn
declare float @floorf(float) readnone nounwind willreturn
declare float @fmodf(float, float) readnone nounwind willreturn
declare float @logf(float) readnone nounwind willreturn
declare float @log10f(float) readnone nounwind willreturn
declare float @powf(float, float) readnone nounwind willreturn
declare float @roundf(float) readnone nounwind willreturn
declare float @sinf(float) readnone nounwind willreturn
declare float @sinhf(float) readnone nounwind willreturn
declare float @sqrtf(float) readnone nounwind willreturn
declare float @tanf(float) readnone nounwind willreturn
declare float @tanhf(float) readnone nounwind willreturn

define double @T() {
; CHECK-LABEL: @T(
; FNOBUILTIN-LABEL: @T(

; CHECK-NOT: call
; CHECK: ret
  %A = call double @cos(double 0.000000e+00)
  %B = call double @sin(double 0.000000e+00)
  %a = fadd double %A, %B
  %C = call double @tan(double 0.000000e+00)
  %b = fadd double %a, %C
  %D = call double @sqrt(double 4.000000e+00)
  %c = fadd double %b, %D

  %slot = alloca double
  %slotf = alloca float
; FNOBUILTIN: call
  %1 = call double @acos(double 1.000000e+00)
  store double %1, double* %slot
; FNOBUILTIN: call
  %2 = call double @asin(double 1.000000e+00)
  store double %2, double* %slot
; FNOBUILTIN: call
  %3 = call double @atan(double 3.000000e+00)
  store double %3, double* %slot
; FNOBUILTIN: call
  %4 = call double @atan2(double 3.000000e+00, double 4.000000e+00)
  store double %4, double* %slot
; FNOBUILTIN: call
  %5 = call double @ceil(double 3.000000e+00)
  store double %5, double* %slot
; FNOBUILTIN: call
  %6 = call double @cosh(double 3.000000e+00)
  store double %6, double* %slot
; FNOBUILTIN: call
  %7 = call double @exp(double 3.000000e+00)
  store double %7, double* %slot
; FNOBUILTIN: call
  %8 = call double @exp2(double 3.000000e+00)
  store double %8, double* %slot
; FNOBUILTIN: call
  %9 = call double @fabs(double 3.000000e+00)
  store double %9, double* %slot
; FNOBUILTIN: call
  %10 = call double @floor(double 3.000000e+00)
  store double %10, double* %slot
; FNOBUILTIN: call
  %11 = call double @fmod(double 3.000000e+00, double 4.000000e+00)
  store double %11, double* %slot
; FNOBUILTIN: call
  %12 = call double @log(double 3.000000e+00)
  store double %12, double* %slot
; FNOBUILTIN: call
  %13 = call double @log10(double 3.000000e+00)
  store double %13, double* %slot
; FNOBUILTIN: call
  %14 = call double @pow(double 3.000000e+00, double 4.000000e+00)
  store double %14, double* %slot
; FNOBUILTIN: call
  %round_val = call double @round(double 3.000000e+00)
  store double %round_val, double* %slot
; FNOBUILTIN: call
  %15 = call double @sinh(double 3.000000e+00)
  store double %15, double* %slot
; FNOBUILTIN: call
  %16 = call double @tanh(double 3.000000e+00)
  store double %16, double* %slot
; FNOBUILTIN: call
  %17 = call float @acosf(float 1.000000e+00)
  store float %17, float* %slotf
; FNOBUILTIN: call
  %18 = call float @asinf(float 1.000000e+00)
  store float %18, float* %slotf
; FNOBUILTIN: call
  %19 = call float @atanf(float 3.000000e+00)
  store float %19, float* %slotf
; FNOBUILTIN: call
  %20 = call float @atan2f(float 3.000000e+00, float 4.000000e+00)
  store float %20, float* %slotf
; FNOBUILTIN: call
  %21 = call float @ceilf(float 3.000000e+00)
  store float %21, float* %slotf
; FNOBUILTIN: call
  %22 = call float @cosf(float 3.000000e+00)
  store float %22, float* %slotf
; FNOBUILTIN: call
  %23 = call float @coshf(float 3.000000e+00)
  store float %23, float* %slotf
; FNOBUILTIN: call
  %24 = call float @expf(float 3.000000e+00)
  store float %24, float* %slotf
; FNOBUILTIN: call
  %25 = call float @exp2f(float 3.000000e+00)
  store float %25, float* %slotf
; FNOBUILTIN: call
  %26 = call float @fabsf(float 3.000000e+00)
  store float %26, float* %slotf
; FNOBUILTIN: call
  %27 = call float @floorf(float 3.000000e+00)
  store float %27, float* %slotf
; FNOBUILTIN: call
  %28 = call float @fmodf(float 3.000000e+00, float 4.000000e+00)
  store float %28, float* %slotf
; FNOBUILTIN: call
  %29 = call float @logf(float 3.000000e+00)
  store float %29, float* %slotf
; FNOBUILTIN: call
  %30 = call float @log10f(float 3.000000e+00)
  store float %30, float* %slotf
; FNOBUILTIN: call
  %31 = call float @powf(float 3.000000e+00, float 4.000000e+00)
  store float %31, float* %slotf
; FNOBUILTIN: call
  %roundf_val = call float @roundf(float 3.000000e+00)
  store float %roundf_val, float* %slotf
; FNOBUILTIN: call
  %32 = call float @sinf(float 3.000000e+00)
  store float %32, float* %slotf
; FNOBUILTIN: call
  %33 = call float @sinhf(float 3.000000e+00)
  store float %33, float* %slotf
; FNOBUILTIN: call
  %34 = call float @sqrtf(float 3.000000e+00)
  store float %34, float* %slotf
; FNOBUILTIN: call
  %35 = call float @tanf(float 3.000000e+00)
  store float %35, float* %slotf
; FNOBUILTIN: call
  %36 = call float @tanhf(float 3.000000e+00)
  store float %36, float* %slotf

; FNOBUILTIN: ret

  ; PR9315
  %E = call double @exp2(double 4.0)
  %d = fadd double %c, %E 
  ret double %d
}

define double @test_intrinsic_pow() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_intrinsic_pow(
; CHECK-NOT: call
; CHECK: ret
  %0 = call double @llvm.pow.f64(double 1.500000e+00, double 3.000000e+00)
  ret double %0
}

define float @test_intrinsic_pow_f32_overflow() nounwind uwtable ssp {
entry:
; CHECK-LABEL: @test_intrinsic_pow_f32_overflow(
; CHECK-NOT: call
; CHECK: ret float 0x7FF0000000000000
  %0 = call float @llvm.pow.f32(float 40.0, float 50.0)
  ret float %0
}

declare double @llvm.pow.f64(double, double) nounwind readonly
declare float @llvm.pow.f32(float, float) nounwind readonly
