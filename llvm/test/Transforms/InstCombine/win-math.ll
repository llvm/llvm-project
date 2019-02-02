; RUN: opt -O2 -S -mtriple=i386-pc-win32 < %s     | FileCheck %s --check-prefixes=CHECK,WIN32
; RUN: opt -O2 -S -mtriple=x86_64-pc-win32 < %s   | FileCheck %s --check-prefixes=CHECK,WIN64
; RUN: opt -O2 -S -mtriple=i386-pc-mingw32 < %s   | FileCheck %s --check-prefixes=CHECK,MINGW32
; RUN: opt -O2 -S -mtriple=x86_64-pc-mingw32 < %s | FileCheck %s --check-prefixes=CHECK,MINGW64

; x86 win32 msvcrt does not provide entry points for single-precision libm.
; x86-64 win32 msvcrt does (except for fabsf)
; msvcrt does not provide C99 math, but mingw32 does.

declare double @acos(double %x)
define float @float_acos(float %x) nounwind readnone {
; CHECK-LABEL: @float_acos(
; WIN32-NOT: float @acosf
; WIN32: double @acos
; WIN64-NOT: float @acosf
; WIN64: double @acos
    %1 = fpext float %x to double
    %2 = call double @acos(double %1)
    %3 = fptrunc double %2 to float
    ret float %3
}

declare double @asin(double %x)
define float @float_asin(float %x) nounwind readnone {
; CHECK-LABEL: @float_asin(
; WIN32-NOT: float @asinf
; WIN32: double @asin
; WIN64-NOT: float @asinf
; WIN64: double @asin
    %1 = fpext float %x to double
    %2 = call double @asin(double %1)
    %3 = fptrunc double %2 to float
    ret float %3
}

declare double @atan(double %x)
define float @float_atan(float %x) nounwind readnone {
; CHECK-LABEL: @float_atan(
; WIN32-NOT: float @atanf
; WIN32: double @atan
; WIN64-NOT: float @atanf
; WIN64: double @atan
    %1 = fpext float %x to double
    %2 = call double @atan(double %1)
    %3 = fptrunc double %2 to float
    ret float %3
}

declare double @atan2(double %x, double %y)
define float @float_atan2(float %x, float %y) nounwind readnone {
; CHECK-LABEL: @float_atan2(
; WIN32-NOT: float @atan2f
; WIN32: double @atan2
; WIN64-NOT: float @atan2f
; WIN64: double @atan2
    %1 = fpext float %x to double
    %2 = fpext float %y to double
    %3 = call double @atan2(double %1, double %2)
    %4 = fptrunc double %3 to float
    ret float %4
}

declare double @ceil(double %x)
define float @float_ceil(float %x) nounwind readnone {
; CHECK-LABEL: @float_ceil(
; WIN32-NOT: float @ceilf
; WIN32: float @llvm.ceil.f32
; WIN64-NOT: double @ceil
; WIN64: float @llvm.ceil.f32
; MINGW32-NOT: double @ceil
; MINGW32: float @llvm.ceil.f32
; MINGW64-NOT: double @ceil
; MINGW64: float @llvm.ceil.f32
    %1 = fpext float %x to double
    %2 = call double @ceil(double %1)
    %3 = fptrunc double %2 to float
    ret float %3
}

declare double @_copysign(double %x)
define float @float_copysign(float %x) nounwind readnone {
; CHECK-LABEL: @float_copysign(
; WIN32-NOT: float @_copysignf
; WIN32: double @_copysign
; WIN64-NOT: float @_copysignf
; WIN64: double @_copysign
    %1 = fpext float %x to double
    %2 = call double @_copysign(double %1)
    %3 = fptrunc double %2 to float
    ret float %3
}

declare double @cos(double %x)
define float @float_cos(float %x) nounwind readnone {
; CHECK-LABEL: @float_cos(
; WIN32-NOT: float @cosf
; WIN32: double @cos
; WIN64-NOT: float @cosf
; WIN64: double @cos
    %1 = fpext float %x to double
    %2 = call double @cos(double %1)
    %3 = fptrunc double %2 to float
    ret float %3
}

declare double @cosh(double %x)
define float @float_cosh(float %x) nounwind readnone {
; CHECK-LABEL: @float_cosh(
; WIN32-NOT: float @coshf
; WIN32: double @cosh
; WIN64-NOT: float @coshf
; WIN64: double @cosh
    %1 = fpext float %x to double
    %2 = call double @cosh(double %1)
    %3 = fptrunc double %2 to float
    ret float %3
}

declare double @exp(double %x, double %y)
define float @float_exp(float %x, float %y) nounwind readnone {
; CHECK-LABEL: @float_exp(
; WIN32-NOT: float @expf
; WIN32: double @exp
; WIN64-NOT: float @expf
; WIN64: double @exp
    %1 = fpext float %x to double
    %2 = fpext float %y to double
    %3 = call double @exp(double %1, double %2)
    %4 = fptrunc double %3 to float
    ret float %4
}

declare double @fabs(double %x, double %y)
define float @float_fabs(float %x, float %y) nounwind readnone {
; CHECK-LABEL: @float_fabs(
; WIN32-NOT: float @fabsf
; WIN32: double @fabs
; WIN64-NOT: float @fabsf
; WIN64: double @fabs
    %1 = fpext float %x to double
    %2 = fpext float %y to double
    %3 = call double @fabs(double %1, double %2)
    %4 = fptrunc double %3 to float
    ret float %4
}

declare double @floor(double %x)
define float @float_floor(float %x) nounwind readnone {
; CHECK-LABEL: @float_floor(
; WIN32-NOT: float @floorf
; WIN32: float @llvm.floor.f32
; WIN64-NOT: double @floor
; WIN64: float @llvm.floor.f32
; MINGW32-NOT: double @floor
; MINGW32: float @llvm.floor.f32
; MINGW64-NOT: double @floor
; MINGW64: float @llvm.floor.f32
    %1 = fpext float %x to double
    %2 = call double @floor(double %1)
    %3 = fptrunc double %2 to float
    ret float %3
}

declare double @fmod(double %x, double %y)
define float @float_fmod(float %x, float %y) nounwind readnone {
; WIN32-LABEL: @float_fmod(
; WIN32-NOT: float @fmodf
; WIN32: double @fmod
; WIN64-NOT: float @fmodf
; WIN64: double @fmod
    %1 = fpext float %x to double
    %2 = fpext float %y to double
    %3 = call double @fmod(double %1, double %2)
    %4 = fptrunc double %3 to float
    ret float %4
}

declare double @log(double %x)
define float @float_log(float %x) nounwind readnone {
; CHECK-LABEL: @float_log(
; WIN32-NOT: float @logf
; WIN32: double @log
; WIN64-NOT: float @logf
; WIN64: double @log
    %1 = fpext float %x to double
    %2 = call double @log(double %1)
    %3 = fptrunc double %2 to float
    ret float %3
}

declare double @pow(double %x, double %y)
define float @float_pow(float %x, float %y) nounwind readnone {
; CHECK-LABEL: @float_pow(
; WIN32-NOT: float @powf
; WIN32: double @pow
; WIN64-NOT: float @powf
; WIN64: double @pow
    %1 = fpext float %x to double
    %2 = fpext float %y to double
    %3 = call double @pow(double %1, double %2)
    %4 = fptrunc double %3 to float
    ret float %4
}

declare double @sin(double %x)
define float @float_sin(float %x) nounwind readnone {
; CHECK-LABEL: @float_sin(
; WIN32-NOT: float @sinf
; WIN32: double @sin
; WIN64-NOT: float @sinf
; WIN64: double @sin
    %1 = fpext float %x to double
    %2 = call double @sin(double %1)
    %3 = fptrunc double %2 to float
    ret float %3
}

declare double @sinh(double %x)
define float @float_sinh(float %x) nounwind readnone {
; CHECK-LABEL: @float_sinh(
; WIN32-NOT: float @sinhf
; WIN32: double @sinh
; WIN64-NOT: float @sinhf
; WIN64: double @sinh
    %1 = fpext float %x to double
    %2 = call double @sinh(double %1)
    %3 = fptrunc double %2 to float
    ret float %3
}

declare double @sqrt(double %x)
define float @float_sqrt(float %x) nounwind readnone {
; CHECK-LABEL: @float_sqrt(
; WIN32-NOT: float @sqrtf
; WIN32: double @sqrt
; WIN64-NOT: double @sqrt
; WIN64: float @sqrtf
; MINGW32-NOT: double @sqrt
; MINGW32: float @sqrtf
; MINGW64-NOT: double @sqrt
; MINGW64: float @sqrtf
    %1 = fpext float %x to double
    %2 = call double @sqrt(double %1)
    %3 = fptrunc double %2 to float
    ret float %3
}

declare double @tan(double %x)
define float @float_tan(float %x) nounwind readnone {
; CHECK-LABEL: @float_tan(
; WIN32-NOT: float @tanf
; WIN32: double @tan
; WIN64-NOT: float @tanf
; WIN64: double @tan
    %1 = fpext float %x to double
    %2 = call double @tan(double %1)
    %3 = fptrunc double %2 to float
    ret float %3
}

declare double @tanh(double %x)
define float @float_tanh(float %x) nounwind readnone {
; CHECK-LABEL: @float_tanh(
; WIN32-NOT: float @tanhf
; WIN32: double @tanh
; WIN64-NOT: float @tanhf
; WIN64: double @tanh
    %1 = fpext float %x to double
    %2 = call double @tanh(double %1)
    %3 = fptrunc double %2 to float
    ret float %3
}

; win32 does not have round; mingw32 does
declare double @round(double %x)
define float @float_round(float %x) nounwind readnone {
; CHECK-LABEL: @float_round(
; WIN32-NOT: float @roundf
; WIN32: double @round
; WIN64-NOT: float @roundf
; WIN64: double @round
; MINGW32-NOT: double @round
; MINGW32: float @llvm.round.f32
; MINGW64-NOT: double @round
; MINGW64: float @llvm.round.f32
    %1 = fpext float %x to double
    %2 = call double @round(double %1)
    %3 = fptrunc double %2 to float
    ret float %3
}

declare float @powf(float, float)

; win32 lacks sqrtf&fabsf, win64 lacks fabsf, but
; calls to the intrinsics can be emitted instead.
define float @float_powsqrt(float %x) nounwind readnone {
; CHECK-LABEL: @float_powsqrt(
; WIN32-NOT: float @sqrtf
; WIN32: float @powf
; WIN64-NOT: float @powf
; WIN64: float @sqrtf
; WIN64: float @llvm.fabs.f32(
; MINGW32-NOT: float @powf
; MINGW32: float @sqrtf
; MINGW32: float @llvm.fabs.f32
; MINGW64-NOT: float @powf
; MINGW64: float @sqrtf
; MINGW64: float @llvm.fabs.f32(
    %1 = call float @powf(float %x, float 0.5)
    ret float %1
}
