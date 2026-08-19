//===-- LanguageMath.h - Kernel language math declarations ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OFFLOAD_LANGUAGES_INCLUDE_KERNEL_LANGUAGE_MATH_H
#define LLVM_OFFLOAD_LANGUAGES_INCLUDE_KERNEL_LANGUAGE_MATH_H

#if defined(__CUDA__) || defined(__HIP__)

extern "C" {
__device__ float acosf(float);
__device__ double acos(double);
__device__ float acoshf(float);
__device__ double acosh(double);
__device__ float asinf(float);
__device__ double asin(double);
__device__ float asinhf(float);
__device__ double asinh(double);
__device__ float atanf(float);
__device__ double atan(double);
__device__ float atan2f(float, float);
__device__ double atan2(double, double);
__device__ float atanhf(float);
__device__ double atanh(double);
__device__ float cbrtf(float);
__device__ double cbrt(double);
__device__ float ceilf(float);
__device__ double ceil(double);
__device__ float copysignf(float, float);
__device__ double copysign(double, double);
__device__ float cosf(float);
__device__ double cos(double);
__device__ float coshf(float);
__device__ double cosh(double);
__device__ float erff(float);
__device__ double erf(double);
__device__ float erfcf(float);
__device__ double erfc(double);
__device__ float expf(float);
__device__ double exp(double);
__device__ float exp2f(float);
__device__ double exp2(double);
__device__ float exp10f(float);
__device__ double exp10(double);
__device__ float expm1f(float);
__device__ double expm1(double);
__device__ float fabsf(float);
__device__ double fabs(double);
__device__ float fdimf(float, float);
__device__ double fdim(double, double);
__device__ float floorf(float);
__device__ double floor(double);
__device__ float fmaf(float, float, float);
__device__ double fma(double, double, double);
__device__ float fmaxf(float, float);
__device__ double fmax(double, double);
__device__ float fminf(float, float);
__device__ double fmin(double, double);
__device__ float fmodf(float, float);
__device__ double fmod(double, double);
__device__ float frexpf(float, int *);
__device__ double frexp(double, int *);
__device__ float hypotf(float, float);
__device__ double hypot(double, double);
__device__ int ilogbf(float);
__device__ int ilogb(double);
__device__ float ldexpf(float, int);
__device__ double ldexp(double, int);
__device__ float lgammaf(float);
__device__ double lgamma(double);
__device__ long long llrintf(float);
__device__ long long llrint(double);
__device__ long long llroundf(float);
__device__ long long llround(double);
__device__ float logf(float);
__device__ double log(double);
__device__ float log10f(float);
__device__ double log10(double);
__device__ float log1pf(float);
__device__ double log1p(double);
__device__ float log2f(float);
__device__ double log2(double);
__device__ float logbf(float);
__device__ double logb(double);
__device__ long lrintf(float);
__device__ long lrint(double);
__device__ long lroundf(float);
__device__ long lround(double);
__device__ float modff(float, float *);
__device__ double modf(double, double *);
__device__ float nearbyintf(float);
__device__ double nearbyint(double);
__device__ float nextafterf(float, float);
__device__ double nextafter(double, double);
__device__ float powf(float, float);
__device__ double pow(double, double);
__device__ float remainderf(float, float);
__device__ double remainder(double, double);
__device__ float remquof(float, float, int *);
__device__ double remquo(double, double, int *);
__device__ float rintf(float);
__device__ double rint(double);
__device__ float roundf(float);
__device__ double round(double);
__device__ float roundevenf(float);
__device__ double roundeven(double);
__device__ float scalblnf(float, long);
__device__ double scalbln(double, long);
__device__ float scalbnf(float, int);
__device__ double scalbn(double, int);
__device__ float sinf(float);
__device__ double sin(double);
__device__ void sincosf(float, float *, float *);
__device__ void sincos(double, double *, double *);
__device__ void sincospif(float, float *, float *);
__device__ void sincospi(double, double *, double *);
__device__ float sinhf(float);
__device__ double sinh(double);
__device__ float sqrtf(float);
__device__ double sqrt(double);
__device__ float tanf(float);
__device__ double tan(double);
__device__ float tanhf(float);
__device__ double tanh(double);
__device__ float tgammaf(float);
__device__ double tgamma(double);
__device__ float truncf(float);
__device__ double trunc(double);
}

#endif // defined(__CUDA__) || defined(__HIP__)

#endif // LLVM_OFFLOAD_LANGUAGES_INCLUDE_KERNEL_LANGUAGE_MATH_H
