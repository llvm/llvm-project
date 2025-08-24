//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation of the device kernels that wrap the
/// math functions from the llvm-libm provider.
///
//===----------------------------------------------------------------------===//

#include "Conformance/device_code/DeviceAPIs.hpp"
#include "Conformance/device_code/KernelRunner.hpp"

#include <gpuintrin.h>
#include <math.h>
#include <stddef.h>

using namespace kernels;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static inline float powfRoundedExponent(float Base, float Exponent) {
  return powf(Base, roundf(Exponent));
}

static inline double sincosSin(double X) {
  double SinX, CosX;
  sincos(X, &SinX, &CosX);
  return SinX;
}

static inline double sincosCos(double X) {
  double SinX, CosX;
  sincos(X, &SinX, &CosX);
  return CosX;
}

static inline float sincosfSin(float X) {
  float SinX, CosX;
  sincosf(X, &SinX, &CosX);
  return SinX;
}

static inline float sincosfCos(float X) {
  float SinX, CosX;
  sincosf(X, &SinX, &CosX);
  return CosX;
}

//===----------------------------------------------------------------------===//
// Kernels
//===----------------------------------------------------------------------===//

extern "C" {

__gpu_kernel void acosKernel(const double *X, double *Out,
                             size_t NumElements) noexcept {
  runKernelBody<acos>(NumElements, Out, X);
}

__gpu_kernel void acosfKernel(const float *X, float *Out,
                              size_t NumElements) noexcept {
  runKernelBody<acosf>(NumElements, Out, X);
}

__gpu_kernel void acoshfKernel(const float *X, float *Out,
                               size_t NumElements) noexcept {
  runKernelBody<acoshf>(NumElements, Out, X);
}

__gpu_kernel void asinKernel(const double *X, double *Out,
                             size_t NumElements) noexcept {
  runKernelBody<asin>(NumElements, Out, X);
}

__gpu_kernel void asinfKernel(const float *X, float *Out,
                              size_t NumElements) noexcept {
  runKernelBody<asinf>(NumElements, Out, X);
}

__gpu_kernel void asinhfKernel(const float *X, float *Out,
                               size_t NumElements) noexcept {
  runKernelBody<asinhf>(NumElements, Out, X);
}

__gpu_kernel void atanfKernel(const float *X, float *Out,
                              size_t NumElements) noexcept {
  runKernelBody<atanf>(NumElements, Out, X);
}

__gpu_kernel void atan2fKernel(const float *X, const float *Y, float *Out,
                               size_t NumElements) noexcept {
  runKernelBody<atan2f>(NumElements, Out, X, Y);
}

__gpu_kernel void atanhfKernel(const float *X, float *Out,
                               size_t NumElements) noexcept {
  runKernelBody<atanhf>(NumElements, Out, X);
}

__gpu_kernel void cbrtKernel(const double *X, double *Out,
                             size_t NumElements) noexcept {
  runKernelBody<cbrt>(NumElements, Out, X);
}

__gpu_kernel void cbrtfKernel(const float *X, float *Out,
                              size_t NumElements) noexcept {
  runKernelBody<cbrtf>(NumElements, Out, X);
}

__gpu_kernel void cosKernel(const double *X, double *Out,
                            size_t NumElements) noexcept {
  runKernelBody<cos>(NumElements, Out, X);
}

__gpu_kernel void cosfKernel(const float *X, float *Out,
                             size_t NumElements) noexcept {
  runKernelBody<cosf>(NumElements, Out, X);
}

__gpu_kernel void coshfKernel(const float *X, float *Out,
                              size_t NumElements) noexcept {
  runKernelBody<coshf>(NumElements, Out, X);
}

__gpu_kernel void cospifKernel(const float *X, float *Out,
                               size_t NumElements) noexcept {
  runKernelBody<cospif>(NumElements, Out, X);
}

__gpu_kernel void erffKernel(const float *X, float *Out,
                             size_t NumElements) noexcept {
  runKernelBody<erff>(NumElements, Out, X);
}

__gpu_kernel void expKernel(const double *X, double *Out,
                            size_t NumElements) noexcept {
  runKernelBody<exp>(NumElements, Out, X);
}

__gpu_kernel void expfKernel(const float *X, float *Out,
                             size_t NumElements) noexcept {
  runKernelBody<expf>(NumElements, Out, X);
}

__gpu_kernel void exp10Kernel(const double *X, double *Out,
                              size_t NumElements) noexcept {
  runKernelBody<exp10>(NumElements, Out, X);
}

__gpu_kernel void exp10fKernel(const float *X, float *Out,
                               size_t NumElements) noexcept {
  runKernelBody<exp10f>(NumElements, Out, X);
}

__gpu_kernel void exp2Kernel(const double *X, double *Out,
                             size_t NumElements) noexcept {
  runKernelBody<exp2>(NumElements, Out, X);
}

__gpu_kernel void exp2fKernel(const float *X, float *Out,
                              size_t NumElements) noexcept {
  runKernelBody<exp2f>(NumElements, Out, X);
}

__gpu_kernel void expm1Kernel(const double *X, double *Out,
                              size_t NumElements) noexcept {
  runKernelBody<expm1>(NumElements, Out, X);
}

__gpu_kernel void expm1fKernel(const float *X, float *Out,
                               size_t NumElements) noexcept {
  runKernelBody<expm1f>(NumElements, Out, X);
}

__gpu_kernel void hypotKernel(const double *X, const double *Y, double *Out,
                              size_t NumElements) noexcept {
  runKernelBody<hypot>(NumElements, Out, X, Y);
}

__gpu_kernel void hypotfKernel(const float *X, const float *Y, float *Out,
                               size_t NumElements) noexcept {
  runKernelBody<hypotf>(NumElements, Out, X, Y);
}

__gpu_kernel void hypotf16Kernel(const float16 *X, const float16 *Y,
                                 float16 *Out, size_t NumElements) noexcept {
  runKernelBody<hypotf16>(NumElements, Out, X, Y);
}

__gpu_kernel void logKernel(const double *X, double *Out,
                            size_t NumElements) noexcept {
  runKernelBody<log>(NumElements, Out, X);
}

__gpu_kernel void logfKernel(const float *X, float *Out,
                             size_t NumElements) noexcept {
  runKernelBody<logf>(NumElements, Out, X);
}

__gpu_kernel void log10Kernel(const double *X, double *Out,
                              size_t NumElements) noexcept {
  runKernelBody<log10>(NumElements, Out, X);
}

__gpu_kernel void log10fKernel(const float *X, float *Out,
                               size_t NumElements) noexcept {
  runKernelBody<log10f>(NumElements, Out, X);
}

__gpu_kernel void log1pKernel(const double *X, double *Out,
                              size_t NumElements) noexcept {
  runKernelBody<log1p>(NumElements, Out, X);
}

__gpu_kernel void log1pfKernel(const float *X, float *Out,
                               size_t NumElements) noexcept {
  runKernelBody<log1pf>(NumElements, Out, X);
}

__gpu_kernel void log2Kernel(const double *X, double *Out,
                             size_t NumElements) noexcept {
  runKernelBody<log2>(NumElements, Out, X);
}

__gpu_kernel void log2fKernel(const float *X, float *Out,
                              size_t NumElements) noexcept {
  runKernelBody<log2f>(NumElements, Out, X);
}

__gpu_kernel void powfKernel(const float *X, float *Y, float *Out,
                             size_t NumElements) noexcept {
  runKernelBody<powf>(NumElements, Out, X, Y);
}

__gpu_kernel void powfRoundedExponentKernel(const float *X, float *Y,
                                            float *Out,
                                            size_t NumElements) noexcept {
  runKernelBody<powfRoundedExponent>(NumElements, Out, X, Y);
}

__gpu_kernel void sinKernel(const double *X, double *Out,
                            size_t NumElements) noexcept {
  runKernelBody<sin>(NumElements, Out, X);
}

__gpu_kernel void sinfKernel(const float *X, float *Out,
                             size_t NumElements) noexcept {
  runKernelBody<sinf>(NumElements, Out, X);
}

__gpu_kernel void sincosSinKernel(const double *X, double *Out,
                                  size_t NumElements) noexcept {
  runKernelBody<sincosSin>(NumElements, Out, X);
}

__gpu_kernel void sincosCosKernel(const double *X, double *Out,
                                  size_t NumElements) noexcept {
  runKernelBody<sincosCos>(NumElements, Out, X);
}

__gpu_kernel void sincosfSinKernel(const float *X, float *Out,
                                   size_t NumElements) noexcept {
  runKernelBody<sincosfSin>(NumElements, Out, X);
}

__gpu_kernel void sincosfCosKernel(const float *X, float *Out,
                                   size_t NumElements) noexcept {
  runKernelBody<sincosfCos>(NumElements, Out, X);
}

__gpu_kernel void sinhfKernel(const float *X, float *Out,
                              size_t NumElements) noexcept {
  runKernelBody<sinhf>(NumElements, Out, X);
}

__gpu_kernel void sinpifKernel(const float *X, float *Out,
                               size_t NumElements) noexcept {
  runKernelBody<sinpif>(NumElements, Out, X);
}

__gpu_kernel void tanKernel(const double *X, double *Out,
                            size_t NumElements) noexcept {
  runKernelBody<tan>(NumElements, Out, X);
}

__gpu_kernel void tanfKernel(const float *X, float *Out,
                             size_t NumElements) noexcept {
  runKernelBody<tanf>(NumElements, Out, X);
}

__gpu_kernel void tanhfKernel(const float *X, float *Out,
                              size_t NumElements) noexcept {
  runKernelBody<tanhf>(NumElements, Out, X);
}

__gpu_kernel void tanpifKernel(const float *X, float *Out,
                               size_t NumElements) noexcept {
  runKernelBody<tanpif>(NumElements, Out, X);
}
} // extern "C"
