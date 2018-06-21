; RUN: opt < %s -instcombine -S -mtriple "i386-pc-linux" | FileCheck -check-prefix=DO-SIMPLIFY -check-prefix=ALL %s
; RUN: opt < %s -instcombine -S -mtriple "i386-pc-win32" | FileCheck -check-prefix=DONT-SIMPLIFY -check-prefix=ALL %s
; RUN: opt < %s -instcombine -S -mtriple "x86_64-pc-win32" | FileCheck -check-prefix=C89-SIMPLIFY -check-prefix=ALL %s
; RUN: opt < %s -instcombine -S -mtriple "i386-pc-mingw32" | FileCheck -check-prefix=DO-SIMPLIFY -check-prefix=ALL %s
; RUN: opt < %s -instcombine -S -mtriple "x86_64-pc-mingw32" | FileCheck -check-prefix=DO-SIMPLIFY -check-prefix=ALL %s
; RUN: opt < %s -instcombine -S -mtriple "sparc-sun-solaris" | FileCheck -check-prefix=DO-SIMPLIFY -check-prefix=ALL %s

declare double @floor(double)
declare double @ceil(double)
declare double @round(double)
declare double @nearbyint(double)
declare double @trunc(double)
declare double @fabs(double)

declare double @llvm.floor.f64(double)
declare double @llvm.ceil.f64(double)
declare double @llvm.round.f64(double)
declare double @llvm.nearbyint.f64(double)
declare double @llvm.trunc.f64(double)
declare double @llvm.fabs.f64(double)

; ALL-LABEL: @test_shrink_libcall_floor(
; DO-SIMPLIFY: call float @llvm.floor.f32(
; C89-SIMPLIFY: call float @llvm.floor.f32(
; DONT-SIMPLIFY: call float @llvm.floor.f32(
define float @test_shrink_libcall_floor(float %C) {
  %D = fpext float %C to double
  ; --> floorf
  %E = call double @floor(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_shrink_libcall_ceil(
; DO-SIMPLIFY: call float @llvm.ceil.f32(
; C89-SIMPLIFY: call float @llvm.ceil.f32(
; DONT-SIMPLIFY: call float @llvm.ceil.f32(
define float @test_shrink_libcall_ceil(float %C) {
  %D = fpext float %C to double
  ; --> ceilf
  %E = call double @ceil(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_shrink_libcall_round(
; DO-SIMPLIFY: call float @llvm.round.f32(
; C89-SIMPLIFY: call double @round(
; DONT-SIMPLIFY: call double @round(
define float @test_shrink_libcall_round(float %C) {
  %D = fpext float %C to double
  ; --> roundf
  %E = call double @round(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_shrink_libcall_nearbyint(
; DO-SIMPLIFY: call float @llvm.nearbyint.f32(
; C89-SIMPLIFY: call double @nearbyint(
; DONT-SIMPLIFY: call double @nearbyint(
define float @test_shrink_libcall_nearbyint(float %C) {
  %D = fpext float %C to double
  ; --> nearbyintf
  %E = call double @nearbyint(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_shrink_libcall_trunc(
; DO-SIMPLIFY: call float @llvm.trunc.f32(
; DONT-SIMPLIFY: call double @trunc(
define float @test_shrink_libcall_trunc(float %C) {
  %D = fpext float %C to double
  ; --> truncf
  %E = call double @trunc(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_shrink_libcall_fabs(
; DO-SIMPLIFY: call float @llvm.fabs.f32(

; This is replaced with the intrinsic, which does the right thing on
; all platforms.
; DONT-SIMPLIFY: call float @llvm.fabs.f32(
define float @test_shrink_libcall_fabs(float %C) {
  %D = fpext float %C to double
  ; --> fabsf
  %E = call double @fabs(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; Make sure fast math flags are preserved
; ALL-LABEL: @test_shrink_libcall_fabs_fast(
; DO-SIMPLIFY: call fast float @llvm.fabs.f32(
define float @test_shrink_libcall_fabs_fast(float %C) {
  %D = fpext float %C to double
  ; --> fabsf
  %E = call fast double @fabs(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_shrink_intrin_floor(
; ALL: call float @llvm.floor.f32(
define float @test_shrink_intrin_floor(float %C) {
  %D = fpext float %C to double
  ; --> floorf
  %E = call double @llvm.floor.f64(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_shrink_intrin_ceil(
; ALL: call float @llvm.ceil.f32(
define float @test_shrink_intrin_ceil(float %C) {
  %D = fpext float %C to double
  ; --> ceilf
  %E = call double @llvm.ceil.f64(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_shrink_intrin_round(
; ALL: call float @llvm.round.f32(
define float @test_shrink_intrin_round(float %C) {
  %D = fpext float %C to double
  ; --> roundf
  %E = call double @llvm.round.f64(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_shrink_intrin_nearbyint(
; ALL: call float @llvm.nearbyint.f32(
define float @test_shrink_intrin_nearbyint(float %C) {
  %D = fpext float %C to double
  ; --> nearbyintf
  %E = call double @llvm.nearbyint.f64(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_shrink_intrin_trunc(
; ALL-SIMPLIFY: call float @llvm.trunc.f32(
define float @test_shrink_intrin_trunc(float %C) {
  %D = fpext float %C to double
  %E = call double @llvm.trunc.f64(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_shrink_intrin_fabs(
; ALL: call float @llvm.fabs.f32(
define float @test_shrink_intrin_fabs(float %C) {
  %D = fpext float %C to double
  %E = call double @llvm.fabs.f64(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; Make sure fast math flags are preserved
; ALL-LABEL: @test_shrink_intrin_fabs_fast(
; ALL: call fast float @llvm.fabs.f32(
define float @test_shrink_intrin_fabs_fast(float %C) {
  %D = fpext float %C to double
  %E = call fast double @llvm.fabs.f64(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_no_shrink_intrin_floor(
; ALL: call double @llvm.floor.f64(
define float @test_no_shrink_intrin_floor(double %D) {
  %E = call double @llvm.floor.f64(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_no_shrink_intrin_ceil(
; ALL: call double @llvm.ceil.f64(
define float @test_no_shrink_intrin_ceil(double %D) {
  %E = call double @llvm.ceil.f64(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_no_shrink_intrin_round(
; ALL: call double @llvm.round.f64(
define float @test_no_shrink_intrin_round(double %D) {
  %E = call double @llvm.round.f64(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_no_shrink_intrin_nearbyint(
; ALL: call double @llvm.nearbyint.f64(
define float @test_no_shrink_intrin_nearbyint(double %D) {
  %E = call double @llvm.nearbyint.f64(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_no_shrink_intrin_trunc(
; ALL-SIMPLIFY: call double @llvm.trunc.f64(
define float @test_no_shrink_intrin_trunc(double %D) {
  %E = call double @llvm.trunc.f64(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_shrink_intrin_fabs_double_src(
; ALL: call float @llvm.fabs.f32(
define float @test_shrink_intrin_fabs_double_src(double %D) {
  %E = call double @llvm.fabs.f64(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; Make sure fast math flags are preserved
; ALL-LABEL: @test_shrink_intrin_fabs_fast_double_src(
; ALL: call fast float @llvm.fabs.f32(
define float @test_shrink_intrin_fabs_fast_double_src(double %D) {
  %E = call fast double @llvm.fabs.f64(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_shrink_float_convertible_constant_intrin_floor(
; ALL: ret float 2.000000e+00
define float @test_shrink_float_convertible_constant_intrin_floor() {
  %E = call double @llvm.floor.f64(double 2.1)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_shrink_float_convertible_constant_intrin_ceil(
; ALL: ret float 3.000000e+00
define float @test_shrink_float_convertible_constant_intrin_ceil() {
  %E = call double @llvm.ceil.f64(double 2.1)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_shrink_float_convertible_constant_intrin_round(
; ALL: ret float 2.000000e+00
define float @test_shrink_float_convertible_constant_intrin_round() {
  %E = call double @llvm.round.f64(double 2.1)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_shrink_float_convertible_constant_intrin_nearbyint(
; ALL: ret float 2.000000e+00
define float @test_shrink_float_convertible_constant_intrin_nearbyint() {
  %E = call double @llvm.nearbyint.f64(double 2.1)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_shrink_float_convertible_constant_intrin_trunc(
; ALL: ret float 2.000000e+00
define float @test_shrink_float_convertible_constant_intrin_trunc() {
  %E = call double @llvm.trunc.f64(double 2.1)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_shrink_float_convertible_constant_intrin_fabs(
; ALL: ret float 0x4000CCCCC0000000
define float @test_shrink_float_convertible_constant_intrin_fabs() {
  %E = call double @llvm.fabs.f64(double 2.1)
  %F = fptrunc double %E to float
  ret float %F
}

; Make sure fast math flags are preserved
; ALL-LABEL: @test_shrink_float_convertible_constant_intrin_fabs_fast(
; ALL: ret float 0x4000CCCCC0000000
define float @test_shrink_float_convertible_constant_intrin_fabs_fast() {
  %E = call fast double @llvm.fabs.f64(double 2.1)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_no_shrink_mismatched_type_intrin_floor(
; ALL-NEXT: %E = call double @llvm.floor.f64(double %D)
; ALL-NEXT: %F = fptrunc double %E to half
; ALL-NEXT: ret half %F
define half @test_no_shrink_mismatched_type_intrin_floor(double %D) {
  %E = call double @llvm.floor.f64(double %D)
  %F = fptrunc double %E to half
  ret half %F
}

; ALL-LABEL: @test_no_shrink_mismatched_type_intrin_ceil(
; ALL-NEXT: %E = call double @llvm.ceil.f64(double %D)
; ALL-NEXT: %F = fptrunc double %E to half
; ALL-NEXT: ret half %F
define half @test_no_shrink_mismatched_type_intrin_ceil(double %D) {
  %E = call double @llvm.ceil.f64(double %D)
  %F = fptrunc double %E to half
  ret half %F
}

; ALL-LABEL: @test_no_shrink_mismatched_type_intrin_round(
; ALL-NEXT: %E = call double @llvm.round.f64(double %D)
; ALL-NEXT: %F = fptrunc double %E to half
; ALL-NEXT: ret half %F
define half @test_no_shrink_mismatched_type_intrin_round(double %D) {
  %E = call double @llvm.round.f64(double %D)
  %F = fptrunc double %E to half
  ret half %F
}

; ALL-LABEL: @test_no_shrink_mismatched_type_intrin_nearbyint(
; ALL-NEXT: %E = call double @llvm.nearbyint.f64(double %D)
; ALL-NEXT: %F = fptrunc double %E to half
; ALL-NEXT: ret half %F
define half @test_no_shrink_mismatched_type_intrin_nearbyint(double %D) {
  %E = call double @llvm.nearbyint.f64(double %D)
  %F = fptrunc double %E to half
  ret half %F
}

; ALL-LABEL: @test_no_shrink_mismatched_type_intrin_trunc(
; ALL-NEXT: %E = call double @llvm.trunc.f64(double %D)
; ALL-NEXT: %F = fptrunc double %E to half
; ALL-NEXT: ret half %F
define half @test_no_shrink_mismatched_type_intrin_trunc(double %D) {
  %E = call double @llvm.trunc.f64(double %D)
  %F = fptrunc double %E to half
  ret half %F
}

; ALL-LABEL: @test_shrink_mismatched_type_intrin_fabs_double_src(
; ALL-NEXT: %1 = fptrunc double %D to half
; ALL-NEXT: %F = call half @llvm.fabs.f16(half %1)
; ALL-NEXT: ret half %F
define half @test_shrink_mismatched_type_intrin_fabs_double_src(double %D) {
  %E = call double @llvm.fabs.f64(double %D)
  %F = fptrunc double %E to half
  ret half %F
}

; Make sure fast math flags are preserved
; ALL-LABEL: @test_mismatched_type_intrin_fabs_fast_double_src(
; ALL-NEXT: %1 = fptrunc double %D to half
; ALL-NEXT: %F = call fast half @llvm.fabs.f16(half %1)
; ALL-NEXT: ret half %F
define half @test_mismatched_type_intrin_fabs_fast_double_src(double %D) {
  %E = call fast double @llvm.fabs.f64(double %D)
  %F = fptrunc double %E to half
  ret half %F
}

; ALL-LABEL: @test_shrink_intrin_floor_fp16_src(
; ALL-NEXT: %E = call half @llvm.floor.f16(half %C)
; ALL-NEXT: %1 = fpext half %E to double
; ALL-NEXT: %F = fptrunc double %1 to float
define float @test_shrink_intrin_floor_fp16_src(half %C) {
  %D = fpext half %C to double
  %E = call double @llvm.floor.f64(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_shrink_intrin_ceil_fp16_src(
; ALL-NEXT: %E = call half @llvm.ceil.f16(half %C)
; ALL-NEXT: %1 = fpext half %E to double
; ALL-NEXT: %F = fptrunc double %1 to float
; ALL-NEXT: ret float %F
define float @test_shrink_intrin_ceil_fp16_src(half %C) {
  %D = fpext half %C to double
  %E = call double @llvm.ceil.f64(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_shrink_intrin_round_fp16_src(
; ALL-NEXT: %E = call half @llvm.round.f16(half %C)
; ALL-NEXT: %1 = fpext half %E to double
; ALL-NEXT: %F = fptrunc double %1 to float
; ALL-NEXT: ret float %F
define float @test_shrink_intrin_round_fp16_src(half %C) {
  %D = fpext half %C to double
  %E = call double @llvm.round.f64(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_shrink_intrin_nearbyint_fp16_src(
; ALL-NEXT: %E = call half @llvm.nearbyint.f16(half %C)
; ALL-NEXT: %1 = fpext half %E to double
; ALL-NEXT: %F = fptrunc double %1 to float
; ALL-NEXT: ret float %F
define float @test_shrink_intrin_nearbyint_fp16_src(half %C) {
  %D = fpext half %C to double
  %E = call double @llvm.nearbyint.f64(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_shrink_intrin_trunc_fp16_src(
; ALL-NEXT: %E = call half @llvm.trunc.f16(half %C)
; ALL-NEXT: %1 = fpext half %E to double
; ALL-NEXT: %F = fptrunc double %1 to float
; ALL-NEXT: ret float %F
define float @test_shrink_intrin_trunc_fp16_src(half %C) {
  %D = fpext half %C to double
  %E = call double @llvm.trunc.f64(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_shrink_intrin_fabs_fp16_src(
; ALL-NEXT: %E = call half @llvm.fabs.f16(half %C)
; ALL-NEXT: %1 = fpext half %E to double
; ALL-NEXT: %F = fptrunc double %1 to float
; ALL-NEXT: ret float %F
define float @test_shrink_intrin_fabs_fp16_src(half %C) {
  %D = fpext half %C to double
  %E = call double @llvm.fabs.f64(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; Make sure fast math flags are preserved
; ALL-LABEL: @test_shrink_intrin_fabs_fast_fp16_src(
; ALL-NEXT: %E = call fast half @llvm.fabs.f16(half %C)
; ALL-NEXT: %1 = fpext half %E to double
; ALL-NEXT: %F = fptrunc double %1 to float
; ALL-NEXT: ret float %F
define float @test_shrink_intrin_fabs_fast_fp16_src(half %C) {
  %D = fpext half %C to double
  %E = call fast double @llvm.fabs.f64(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_no_shrink_intrin_floor_multi_use_fpext(
; ALL: %D = fpext half %C to double
; ALL: call double @llvm.floor.f64
define float @test_no_shrink_intrin_floor_multi_use_fpext(half %C) {
  %D = fpext half %C to double
  store volatile double %D, double* undef
  %E = call double @llvm.floor.f64(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; ALL-LABEL: @test_no_shrink_intrin_fabs_multi_use_fpext(
; ALL: %D = fpext half %C to double
; ALL: call double @llvm.fabs.f64
define float @test_no_shrink_intrin_fabs_multi_use_fpext(half %C) {
  %D = fpext half %C to double
  store volatile double %D, double* undef
  %E = call double @llvm.fabs.f64(double %D)
  %F = fptrunc double %E to float
  ret float %F
}
