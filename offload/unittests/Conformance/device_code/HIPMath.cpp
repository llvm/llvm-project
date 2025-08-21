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
/// math functions from the hip-math provider.
///
//===----------------------------------------------------------------------===//

#ifdef HIP_MATH_FOUND

#include "Conformance/device_code/DeviceAPIs.hpp"
#include "Conformance/device_code/KernelRunner.hpp"

#include <gpuintrin.h>
#include <stddef.h>

using namespace kernels;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static inline float sincosfSin(float X) {
  float CosX;
  float SinX = __ocml_sincos_f32(X, &CosX);
  return SinX;
}

static inline float sincosfCos(float X) {
  float CosX;
  float SinX = __ocml_sincos_f32(X, &CosX);
  return CosX;
}

//===----------------------------------------------------------------------===//
// Kernels
//===----------------------------------------------------------------------===//

extern "C" {

__gpu_kernel void acosfKernel(const float *X, float *Out,
                              size_t NumElements) noexcept {
  runKernelBody<__ocml_acos_f32>(NumElements, Out, X);
}

__gpu_kernel void acoshfKernel(const float *X, float *Out,
                               size_t NumElements) noexcept {
  runKernelBody<__ocml_acosh_f32>(NumElements, Out, X);
}

__gpu_kernel void asinfKernel(const float *X, float *Out,
                              size_t NumElements) noexcept {
  runKernelBody<__ocml_asin_f32>(NumElements, Out, X);
}

__gpu_kernel void asinhfKernel(const float *X, float *Out,
                               size_t NumElements) noexcept {
  runKernelBody<__ocml_asinh_f32>(NumElements, Out, X);
}

__gpu_kernel void atanfKernel(const float *X, float *Out,
                              size_t NumElements) noexcept {
  runKernelBody<__ocml_atan_f32>(NumElements, Out, X);
}

__gpu_kernel void atanhfKernel(const float *X, float *Out,
                               size_t NumElements) noexcept {
  runKernelBody<__ocml_atanh_f32>(NumElements, Out, X);
}

__gpu_kernel void cbrtfKernel(const float *X, float *Out,
                              size_t NumElements) noexcept {
  runKernelBody<__ocml_cbrt_f32>(NumElements, Out, X);
}

__gpu_kernel void cosfKernel(const float *X, float *Out,
                             size_t NumElements) noexcept {
  runKernelBody<__ocml_cos_f32>(NumElements, Out, X);
}

__gpu_kernel void coshfKernel(const float *X, float *Out,
                              size_t NumElements) noexcept {
  runKernelBody<__ocml_cosh_f32>(NumElements, Out, X);
}

__gpu_kernel void cospifKernel(const float *X, float *Out,
                               size_t NumElements) noexcept {
  runKernelBody<__ocml_cospi_f32>(NumElements, Out, X);
}

__gpu_kernel void erffKernel(const float *X, float *Out,
                             size_t NumElements) noexcept {
  runKernelBody<__ocml_erf_f32>(NumElements, Out, X);
}

__gpu_kernel void expfKernel(const float *X, float *Out,
                             size_t NumElements) noexcept {
  runKernelBody<__ocml_exp_f32>(NumElements, Out, X);
}

__gpu_kernel void exp10fKernel(const float *X, float *Out,
                               size_t NumElements) noexcept {
  runKernelBody<__ocml_exp10_f32>(NumElements, Out, X);
}

__gpu_kernel void exp2fKernel(const float *X, float *Out,
                              size_t NumElements) noexcept {
  runKernelBody<__ocml_exp2_f32>(NumElements, Out, X);
}

__gpu_kernel void expm1fKernel(const float *X, float *Out,
                               size_t NumElements) noexcept {
  runKernelBody<__ocml_expm1_f32>(NumElements, Out, X);
}

__gpu_kernel void logKernel(const double *X, double *Out,
                            size_t NumElements) noexcept {
  runKernelBody<__ocml_log_f64>(NumElements, Out, X);
}

__gpu_kernel void logfKernel(const float *X, float *Out,
                             size_t NumElements) noexcept {
  runKernelBody<__ocml_log_f32>(NumElements, Out, X);
}

__gpu_kernel void log10fKernel(const float *X, float *Out,
                               size_t NumElements) noexcept {
  runKernelBody<__ocml_log10_f32>(NumElements, Out, X);
}

__gpu_kernel void log1pfKernel(const float *X, float *Out,
                               size_t NumElements) noexcept {
  runKernelBody<__ocml_log1p_f32>(NumElements, Out, X);
}

__gpu_kernel void log2fKernel(const float *X, float *Out,
                              size_t NumElements) noexcept {
  runKernelBody<__ocml_log2_f32>(NumElements, Out, X);
}

__gpu_kernel void sinfKernel(const float *X, float *Out,
                             size_t NumElements) noexcept {
  runKernelBody<__ocml_sin_f32>(NumElements, Out, X);
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
  runKernelBody<__ocml_sinh_f32>(NumElements, Out, X);
}

__gpu_kernel void sinpifKernel(const float *X, float *Out,
                               size_t NumElements) noexcept {
  runKernelBody<__ocml_sinpi_f32>(NumElements, Out, X);
}

__gpu_kernel void tanfKernel(const float *X, float *Out,
                             size_t NumElements) noexcept {
  runKernelBody<__ocml_tan_f32>(NumElements, Out, X);
}

__gpu_kernel void tanhfKernel(const float *X, float *Out,
                              size_t NumElements) noexcept {
  runKernelBody<__ocml_tanh_f32>(NumElements, Out, X);
}
} // extern "C"

#endif // HIP_MATH_FOUND
