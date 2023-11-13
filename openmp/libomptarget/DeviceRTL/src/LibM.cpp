//===------- LibC.cpp - Simple implementation of libc functions --- C++ ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma omp begin declare target device_type(nohost)

#define __BUILD_MATH_BUILTINS_LIB__

#include "Platform.h"

using size_t = decltype(sizeof(char));

// We cannot use variants as we need the "C" symbol names to be exported.
#ifdef __AMDGPU__

#define __OPENMP_AMDGCN__

#pragma push_macro("__device__")
#define __device__

#include <__clang_hip_libdevice_declares.h>

#pragma pop_macro("__device__")

#include <__clang_cuda_complex_builtins.h>
#include <__clang_hip_math.h>

#undef __OPENMP_AMDGCN__

extern "C" {

#ifndef FORTRAN_NO_LONGER_NEEDS
// Attach Fortran runtimes which are used by Classic Flang
double __f90_dmodulov(double a, double p) {
  double d;
  d = a - floor(a/p) * p;
  return d;
}

float __f90_amodulov(float a, float p) { return __f90_dmodulov(a, p); }

int32_t __f90_modulov(int32_t a, int32_t p) {
  int32_t q, r;

  q = a / p;
  r = a - q * p;
  if (r != 0 && (a ^ p) < 0) { /* signs differ */
    r += p;
  }
  return r;
}

int64_t __f90_i8modulov_i8(int64_t a, int64_t p) {
  int64_t q, r;

  q = a / p;
  r = a - q * p;
  if (r != 0 && (a ^ p) < 0) { /* signs differ */
    r += (p);
  }
  return r;
}

int16_t __f90_imodulov(int16_t a, int16_t p) {
  int32_t q, r;

  q = a / p;
  r = a - q * p;
  if (r != 0 && (a ^ p) < 0) { /* signs differ */
    r += p;
  }
  return r;
}
}

#endif

#endif // __AMDGPU__

#ifdef __NVPTX__

#define __CUDA__
#define __OPENMP_NVPTX__

#pragma push_macro("__device__")
#define __device__

#include <__clang_cuda_libdevice_declares.h>

#include <__clang_cuda_device_functions.h>

#pragma pop_macro("__device__")

#include <__clang_cuda_complex_builtins.h>
#include <__clang_cuda_math.h>

#undef __OPENMP_NVPTX__
#undef __CUDA__

#endif // __NVPTX__

#pragma omp end declare target
