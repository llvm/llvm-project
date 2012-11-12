; RUN: opt < %s -constprop -S | FileCheck %s
; RUN: opt < %s -constprop -disable-simplify-libcalls -S | FileCheck %s --check-prefix=FNOBUILTIN

declare double @cos(double)

declare double @sin(double)

declare double @tan(double)

declare double @sqrt(double)
declare double @exp2(double)

define double @T() {
; CHECK: @T
; CHECK-NOT: call
; CHECK: ret
  %A = call double @cos(double 0.000000e+00)
  %B = call double @sin(double 0.000000e+00)
  %a = fadd double %A, %B
  %C = call double @tan(double 0.000000e+00)
  %b = fadd double %a, %C
  %D = call double @sqrt(double 4.000000e+00)
  %c = fadd double %b, %D

  ; PR9315
  %E = call double @exp2(double 4.0)
  %d = fadd double %c, %E 
  ret double %d
}

define i1 @test_sse_cvt() nounwind readnone {
; CHECK: @test_sse_cvt
; CHECK-NOT: call
; CHECK: ret i1 true
entry:
  %i0 = tail call i32 @llvm.x86.sse.cvtss2si(<4 x float> <float 1.75, float undef, float undef, float undef>) nounwind
  %i1 = tail call i32 @llvm.x86.sse.cvttss2si(<4 x float> <float 1.75, float undef, float undef, float undef>) nounwind
  %i2 = tail call i64 @llvm.x86.sse.cvtss2si64(<4 x float> <float 1.75, float undef, float undef, float undef>) nounwind
  %i3 = tail call i64 @llvm.x86.sse.cvttss2si64(<4 x float> <float 1.75, float undef, float undef, float undef>) nounwind
  %i4 = call i32 @llvm.x86.sse2.cvtsd2si(<2 x double> <double 1.75, double undef>) nounwind
  %i5 = call i32 @llvm.x86.sse2.cvttsd2si(<2 x double> <double 1.75, double undef>) nounwind
  %i6 = call i64 @llvm.x86.sse2.cvtsd2si64(<2 x double> <double 1.75, double undef>) nounwind
  %i7 = call i64 @llvm.x86.sse2.cvttsd2si64(<2 x double> <double 1.75, double undef>) nounwind
  %sum11 = add i32 %i0, %i1
  %sum12 = add i32 %i4, %i5
  %sum1 = add i32 %sum11, %sum12
  %sum21 = add i64 %i2, %i3
  %sum22 = add i64 %i6, %i7
  %sum2 = add i64 %sum21, %sum22
  %sum1.sext = sext i32 %sum1 to i64
  %b = icmp eq i64 %sum1.sext, %sum2
  ret i1 %b
}

declare i32 @llvm.x86.sse.cvtss2si(<4 x float>) nounwind readnone
declare i32 @llvm.x86.sse.cvttss2si(<4 x float>) nounwind readnone
declare i64 @llvm.x86.sse.cvtss2si64(<4 x float>) nounwind readnone
declare i64 @llvm.x86.sse.cvttss2si64(<4 x float>) nounwind readnone
declare i32 @llvm.x86.sse2.cvtsd2si(<2 x double>) nounwind readnone
declare i32 @llvm.x86.sse2.cvttsd2si(<2 x double>) nounwind readnone
declare i64 @llvm.x86.sse2.cvtsd2si64(<2 x double>) nounwind readnone
declare i64 @llvm.x86.sse2.cvttsd2si64(<2 x double>) nounwind readnone

define double @test_intrinsic_pow() nounwind uwtable ssp {
entry:
; CHECK: @test_intrinsic_pow
; CHECK-NOT: call
  %0 = call double @llvm.pow.f64(double 1.500000e+00, double 3.000000e+00)
  ret double %0
}
declare double @llvm.pow.f64(double, double) nounwind readonly

; Shouldn't fold because of -fno-builtin
define double @sin_() nounwind uwtable ssp {
; FNOBUILTIN: @sin_
; FNOBUILTIN: %1 = call double @sin(double 3.000000e+00)
  %1 = call double @sin(double 3.000000e+00)
  ret double %1
}

; Shouldn't fold because of -fno-builtin
define double @sqrt_() nounwind uwtable ssp {
; FNOBUILTIN: @sqrt_
; FNOBUILTIN: %1 = call double @sqrt(double 3.000000e+00)
  %1 = call double @sqrt(double 3.000000e+00)
  ret double %1
}

; Shouldn't fold because of -fno-builtin
define float @sqrtf_() nounwind uwtable ssp {
; FNOBUILTIN: @sqrtf_
; FNOBUILTIN: %1 = call float @sqrtf(float 3.000000e+00)
  %1 = call float @sqrtf(float 3.000000e+00)
  ret float %1
}
declare float @sqrtf(float)

; Shouldn't fold because of -fno-builtin
define float @sinf_() nounwind uwtable ssp {
; FNOBUILTIN: @sinf_
; FNOBUILTIN: %1 = call float @sinf(float 3.000000e+00)
  %1 = call float @sinf(float 3.000000e+00)
  ret float %1
}
declare float @sinf(float)

; Shouldn't fold because of -fno-builtin
define double @tan_() nounwind uwtable ssp {
; FNOBUILTIN: @tan_
; FNOBUILTIN: %1 = call double @tan(double 3.000000e+00)
  %1 = call double @tan(double 3.000000e+00)
  ret double %1
}

; Shouldn't fold because of -fno-builtin
define double @tanh_() nounwind uwtable ssp {
; FNOBUILTIN: @tanh_
; FNOBUILTIN: %1 = call double @tanh(double 3.000000e+00)
  %1 = call double @tanh(double 3.000000e+00)
  ret double %1
}
declare double @tanh(double)

; Shouldn't fold because of -fno-builtin
define double @pow_() nounwind uwtable ssp {
; FNOBUILTIN: @pow_
; FNOBUILTIN: %1 = call double @pow(double 3.000000e+00, double 3.000000e+00)
  %1 = call double @pow(double 3.000000e+00, double 3.000000e+00)
  ret double %1
}
declare double @pow(double, double)

; Shouldn't fold because of -fno-builtin
define double @fmod_() nounwind uwtable ssp {
; FNOBUILTIN: @fmod_
; FNOBUILTIN: %1 = call double @fmod(double 3.000000e+00, double 3.000000e+00)
  %1 = call double @fmod(double 3.000000e+00, double 3.000000e+00)
  ret double %1
}
declare double @fmod(double, double)

; Shouldn't fold because of -fno-builtin
define double @atan2_() nounwind uwtable ssp {
; FNOBUILTIN: @atan2_
; FNOBUILTIN: %1 = call double @atan2(double 3.000000e+00, double 3.000000e+00)
  %1 = call double @atan2(double 3.000000e+00, double 3.000000e+00)
  ret double %1
}
declare double @atan2(double, double)
