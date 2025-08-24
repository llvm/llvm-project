//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains platform-specific definitions and forward declarations
/// for device-side APIs used by the kernels.
///
//===----------------------------------------------------------------------===//

#ifndef CONFORMANCE_DEVICE_CODE_DEVICEAPIS_HPP
#define CONFORMANCE_DEVICE_CODE_DEVICEAPIS_HPP

#include <stdint.h>

typedef _Float16 float16;

#ifdef __AMDGPU__

// The ROCm device library uses control globals to alter codegen for the
// different targets. To avoid needing to link them in manually, we simply
// define them here.
extern "C" {
extern const inline uint8_t __oclc_unsafe_math_opt = 0;
extern const inline uint8_t __oclc_daz_opt = 0;
extern const inline uint8_t __oclc_correctly_rounded_sqrt32 = 1;
extern const inline uint8_t __oclc_finite_only_opt = 0;
extern const inline uint32_t __oclc_ISA_version = 9000;
}

// These aliases cause Clang to emit the control constants with ODR linkage.
// This allows us to link against the symbols without preventing them from being
// optimized out or causing symbol collisions.
[[gnu::alias("__oclc_unsafe_math_opt")]] const uint8_t __oclc_unsafe_math_opt__;
[[gnu::alias("__oclc_daz_opt")]] const uint8_t __oclc_daz_opt__;
[[gnu::alias("__oclc_correctly_rounded_sqrt32")]] const uint8_t
    __oclc_correctly_rounded_sqrt32__;
[[gnu::alias("__oclc_finite_only_opt")]] const uint8_t __oclc_finite_only_opt__;
[[gnu::alias("__oclc_ISA_version")]] const uint32_t __oclc_ISA_version__;

#endif // __AMDGPU__

#ifdef CUDA_MATH_FOUND

extern "C" {

double __nv_acos(double);
float __nv_acosf(float);
float __nv_acoshf(float);
double __nv_asin(double);
float __nv_asinf(float);
float __nv_asinhf(float);
float __nv_atanf(float);
float __nv_atan2f(float, float);
float __nv_atanhf(float);
double __nv_cbrt(double);
float __nv_cbrtf(float);
double __nv_cos(double);
float __nv_cosf(float);
float __nv_coshf(float);
float __nv_cospif(float);
float __nv_erff(float);
double __nv_exp(double);
float __nv_expf(float);
double __nv_exp10(double);
float __nv_exp10f(float);
double __nv_exp2(double);
float __nv_exp2f(float);
double __nv_expm1(double);
float __nv_expm1f(float);
double __nv_hypot(double, double);
float __nv_hypotf(float, float);
double __nv_log(double);
float __nv_logf(float);
double __nv_log10(double);
float __nv_log10f(float);
double __nv_log1p(double);
float __nv_log1pf(float);
double __nv_log2(double);
float __nv_log2f(float);
float __nv_powf(float, float);
float __nv_roundf(float);
double __nv_sin(double);
float __nv_sinf(float);
void __nv_sincos(double, double *, double *);
void __nv_sincosf(float, float *, float *);
float __nv_sinhf(float);
float __nv_sinpif(float);
double __nv_tan(double);
float __nv_tanf(float);
float __nv_tanhf(float);
} // extern "C"

#endif // CUDA_MATH_FOUND

#ifdef HIP_MATH_FOUND

extern "C" {

double __ocml_acos_f64(double);
float __ocml_acos_f32(float);
float __ocml_acosh_f32(float);
double __ocml_asin_f64(double);
float __ocml_asin_f32(float);
float __ocml_asinh_f32(float);
float __ocml_atan_f32(float);
float __ocml_atan2_f32(float, float);
float __ocml_atanh_f32(float);
double __ocml_cbrt_f64(double);
float __ocml_cbrt_f32(float);
double __ocml_cos_f64(double);
float __ocml_cos_f32(float);
float __ocml_cosh_f32(float);
float __ocml_cospi_f32(float);
float __ocml_erf_f32(float);
double __ocml_exp_f64(double);
float __ocml_exp_f32(float);
double __ocml_exp10_f64(double);
float __ocml_exp10_f32(float);
double __ocml_exp2_f64(double);
float __ocml_exp2_f32(float);
double __ocml_expm1_f64(double);
float __ocml_expm1_f32(float);
double __ocml_hypot_f64(double, double);
float __ocml_hypot_f32(float, float);
double __ocml_log_f64(double);
float __ocml_log_f32(float);
double __ocml_log10_f64(double);
float __ocml_log10_f32(float);
double __ocml_log1p_f64(double);
float __ocml_log1p_f32(float);
double __ocml_log2_f64(double);
float __ocml_log2_f32(float);
float __ocml_pow_f32(float, float);
float __ocml_round_f32(float);
double __ocml_sin_f64(double);
float __ocml_sin_f32(float);
double __ocml_sincos_f64(double, double *);
float __ocml_sincos_f32(float, float *);
float __ocml_sinh_f32(float);
float __ocml_sinpi_f32(float);
double __ocml_tan_f64(double);
float __ocml_tan_f32(float);
float __ocml_tanh_f32(float);
} // extern "C"

#endif // HIP_MATH_FOUND

#endif // CONFORMANCE_DEVICE_CODE_DEVICEAPIS_HPP
