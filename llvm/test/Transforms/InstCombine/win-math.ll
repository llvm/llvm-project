; RUN: opt < %s -O2 -S -mtriple=i386-pc-windows-msvc18   | FileCheck %s --check-prefixes=CHECK,MSVCXX,MSVC32
; RUN: opt < %s -O2 -S -mtriple=i386-pc-windows-msvc     | FileCheck %s --check-prefixes=CHECK,MSVC19,MSVC51
; RUN: opt < %s -O2 -S -mtriple=x86_64-pc-windows-msvc17 | FileCheck %s --check-prefixes=CHECK,MSVCXX,MSVC64
; RUN: opt < %s -O2 -S -mtriple=x86_64-pc-win32          | FileCheck %s --check-prefixes=CHECK,MSVC19,MSVC83
; RUN: opt < %s -O2 -S -mtriple=i386-pc-mingw32          | FileCheck %s --check-prefixes=CHECK,MINGW32
; RUN: opt < %s -O2 -S -mtriple=x86_64-pc-mingw32        | FileCheck %s --check-prefixes=CHECK,MINGW64

; x86 win32 msvcrt does not provide entry points for single-precision libm.
; x86-64 win32 msvcrt does, but with exceptions
; msvcrt does not provide all of C99 math, but mingw32 does.

declare double @acos(double %x)
define float @float_acos(float %x) nounwind readnone {
; CHECK-LABEL: @float_acos(
; MSVCXX-NOT: float @acosf
; MSVCXX: double @acos
; MSVC19-NOT: float @acosf
; MSVC19: double @acos
    %1 = fpext float %x to double
    %2 = call double @acos(double %1)
    %3 = fptrunc double %2 to float
    ret float %3
}

declare double @asin(double %x)
define float @float_asin(float %x) nounwind readnone {
; CHECK-LABEL: @float_asin(
; MSVCXX-NOT: float @asinf
; MSVCXX: double @asin
; MSVC19-NOT: float @asinf
; MSVC19: double @asin
    %1 = fpext float %x to double
    %2 = call double @asin(double %1)
    %3 = fptrunc double %2 to float
    ret float %3
}

declare double @atan(double %x)
define float @float_atan(float %x) nounwind readnone {
; CHECK-LABEL: @float_atan(
; MSVCXX-NOT: float @atanf
; MSVCXX: double @atan
; MSVC19-NOT: float @atanf
; MSVC19: double @atan
    %1 = fpext float %x to double
    %2 = call double @atan(double %1)
    %3 = fptrunc double %2 to float
    ret float %3
}

declare double @atan2(double %x, double %y)
define float @float_atan2(float %x, float %y) nounwind readnone {
; CHECK-LABEL: @float_atan2(
; MSVCXX-NOT: float @atan2f
; MSVCXX: double @atan2
; MSVC19-NOT: float @atan2f
; MSVC19: double @atan2
    %1 = fpext float %x to double
    %2 = fpext float %y to double
    %3 = call double @atan2(double %1, double %2)
    %4 = fptrunc double %3 to float
    ret float %4
}

declare double @ceil(double %x)
define float @float_ceil(float %x) nounwind readnone {
; CHECK-LABEL: @float_ceil(
; MSVCXX-NOT: float @ceilf
; MSVCXX: float @llvm.ceil.f32
; MSVC19-NOT: double @ceil
; MSVC19: float @llvm.ceil.f32
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
; MSVCXX-NOT: float @_copysignf
; MSVCXX: double @_copysign
; MSVC19-NOT: float @_copysignf
; MSVC19: double @_copysign
    %1 = fpext float %x to double
    %2 = call double @_copysign(double %1)
    %3 = fptrunc double %2 to float
    ret float %3
}

declare double @cos(double %x)
define float @float_cos(float %x) nounwind readnone {
; CHECK-LABEL: @float_cos(
; MSVCXX-NOT: float @cosf
; MSVCXX: double @cos
; MSVC19-NOT: float @cosf
; MSVC19: double @cos
    %1 = fpext float %x to double
    %2 = call double @cos(double %1)
    %3 = fptrunc double %2 to float
    ret float %3
}

declare double @cosh(double %x)
define float @float_cosh(float %x) nounwind readnone {
; CHECK-LABEL: @float_cosh(
; MSVCXX-NOT: float @coshf
; MSVCXX: double @cosh
; MSVC19-NOT: float @coshf
; MSVC19: double @cosh
    %1 = fpext float %x to double
    %2 = call double @cosh(double %1)
    %3 = fptrunc double %2 to float
    ret float %3
}

declare double @exp(double %x, double %y)
define float @float_exp(float %x, float %y) nounwind readnone {
; CHECK-LABEL: @float_exp(
; MSVCXX-NOT: float @expf
; MSVCXX: double @exp
; MSVC19-NOT: float @expf
; MSVC19: double @exp
    %1 = fpext float %x to double
    %2 = fpext float %y to double
    %3 = call double @exp(double %1, double %2)
    %4 = fptrunc double %3 to float
    ret float %4
}

declare double @fabs(double %x, double %y)
define float @float_fabs(float %x, float %y) nounwind readnone {
; CHECK-LABEL: @float_fabs(
; MSVCXX-NOT: float @fabsf
; MSVCXX: double @fabs
; MSVC19-NOT: float @fabsf
; MSVC19: double @fabs
    %1 = fpext float %x to double
    %2 = fpext float %y to double
    %3 = call double @fabs(double %1, double %2)
    %4 = fptrunc double %3 to float
    ret float %4
}

declare double @floor(double %x)
define float @float_floor(float %x) nounwind readnone {
; CHECK-LABEL: @float_floor(
; MSVCXX-NOT: float @floorf
; MSVCXX: float @llvm.floor.f32
; MSVC19-NOT: double @floor
; MSVC19: float @llvm.floor.f32
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
; MSVCXX-LABEL: @float_fmod(
; MSVCXX-NOT: float @fmodf
; MSVCXX: double @fmod
; MSVC19-NOT: float @fmodf
; MSVC19: double @fmod
    %1 = fpext float %x to double
    %2 = fpext float %y to double
    %3 = call double @fmod(double %1, double %2)
    %4 = fptrunc double %3 to float
    ret float %4
}

declare double @log(double %x)
define float @float_log(float %x) nounwind readnone {
; CHECK-LABEL: @float_log(
; MSVCXX-NOT: float @logf
; MSVCXX: double @log
; MSVC19-NOT: float @logf
; MSVC19: double @log
    %1 = fpext float %x to double
    %2 = call double @log(double %1)
    %3 = fptrunc double %2 to float
    ret float %3
}

declare double @logb(double %x)
define float @float_logb(float %x) nounwind readnone {
; CHECK-LABEL: @float_logb(
; MSVCXX-NOT: float @logbf
; MSVCXX: double @logb
; MSVC19-NOT: float @logbf
; MSVC19: double @logb
    %1 = fpext float %x to double
    %2 = call double @logb(double %1)
    %3 = fptrunc double %2 to float
    ret float %3
}

declare double @pow(double %x, double %y)
define float @float_pow(float %x, float %y) nounwind readnone {
; CHECK-LABEL: @float_pow(
; MSVCXX-NOT: float @powf
; MSVCXX: double @pow
; MSVC19-NOT: float @powf
; MSVC19: double @pow
    %1 = fpext float %x to double
    %2 = fpext float %y to double
    %3 = call double @pow(double %1, double %2)
    %4 = fptrunc double %3 to float
    ret float %4
}

declare double @sin(double %x)
define float @float_sin(float %x) nounwind readnone {
; CHECK-LABEL: @float_sin(
; MSVCXX-NOT: float @sinf
; MSVCXX: double @sin
; MSVC19-NOT: float @sinf
; MSVC19: double @sin
    %1 = fpext float %x to double
    %2 = call double @sin(double %1)
    %3 = fptrunc double %2 to float
    ret float %3
}

declare double @sinh(double %x)
define float @float_sinh(float %x) nounwind readnone {
; CHECK-LABEL: @float_sinh(
; MSVCXX-NOT: float @sinhf
; MSVCXX: double @sinh
; MSVC19-NOT: float @sinhf
; MSVC19: double @sinh
    %1 = fpext float %x to double
    %2 = call double @sinh(double %1)
    %3 = fptrunc double %2 to float
    ret float %3
}

declare double @sqrt(double %x)
define float @float_sqrt(float %x) nounwind readnone {
; CHECK-LABEL: @float_sqrt(
; MSVC32-NOT: float @sqrtf
; MSVC32: double @sqrt
; MSVC51-NOT: float @sqrtf
; MSVC51: double @sqrt
; MSVC64-NOT: double @sqrt
; MSVC64: float @sqrtf
; MSVC83-NOT: double @sqrt
; MSVC83: float @sqrtf
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
; MSVCXX-NOT: float @tanf
; MSVCXX: double @tan
; MSVC19-NOT: float @tanf
; MSVC19: double @tan
    %1 = fpext float %x to double
    %2 = call double @tan(double %1)
    %3 = fptrunc double %2 to float
    ret float %3
}

declare double @tanh(double %x)
define float @float_tanh(float %x) nounwind readnone {
; CHECK-LABEL: @float_tanh(
; MSVCXX-NOT: float @tanhf
; MSVCXX: double @tanh
; MSVC19-NOT: float @tanhf
; MSVC19: double @tanh
    %1 = fpext float %x to double
    %2 = call double @tanh(double %1)
    %3 = fptrunc double %2 to float
    ret float %3
}

; win32 does not have roundf; mingw32 does
declare double @round(double %x)
define float @float_round(float %x) nounwind readnone {
; CHECK-LABEL: @float_round(
; MSVCXX-NOT: double @roundf
; MSVCXX: double @round
; MSVC19-NOT: double @round
; MSVC19: float @llvm.round.f32
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

; win32 lacks sqrtf & fabsf, win64 lacks fabsf, but
; calls to the intrinsics can be emitted instead.
define float @float_powsqrt(float %x) nounwind readnone {
; CHECK-LABEL: @float_powsqrt(
; MSVC32-NOT: float @sqrtf
; MSVC32: float @powf
; MSVC51-NOT: float @sqrtf
; MSVC51: float @powf
; MSVC64-NOT: float @powf
; MSVC64: float @sqrtf
; MSVC64: float @llvm.fabs.f32(
; MSVC83-NOT: float @powf
; MSVC83: float @sqrtf
; MSVC83: float @llvm.fabs.f32(
; MINGW32-NOT: float @powf
; MINGW32: float @sqrtf
; MINGW32: float @llvm.fabs.f32
; MINGW64-NOT: float @powf
; MINGW64: float @sqrtf
; MINGW64: float @llvm.fabs.f32(
    %1 = call float @powf(float %x, float 0.5)
    ret float %1
}
