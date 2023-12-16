//===--------- libm/libm.c ------------------------------------------------===//
//
//                The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// If the library needs to destinguish betwen different targets,
// target specific macros for that GPU can be used.
// For nvptx use __NVPTX__.  For amdgcn, use __AMDGCN__.
// Example:
//   #ifdef __AMDGCN__ && (__AMDGCN__ == 1000)
//     double fast_sqrt(double __a) { ... }
//   #endif

#define __BUILD_MATH_BUILTINS_LIB__

#ifdef __AMDGCN__
#pragma omp declare target
#define __OPENMP_AMDGCN__
#include <__clang_cuda_complex_builtins.h>
#include <__clang_hip_math.h>

#ifndef FORTRAN_NO_LONGER_NEEDS
// Attach Fortran runtimes which are used by Classic Flang
double __f90_dmodulov(double a, double p) {
  double d;
  d = fmod(a, p);
  if (d != 0 && ((a < 0 && p > 0) || (a > 0 && p < 0)))
    d += p;
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

#endif

#pragma omp end declare target
#endif

#ifdef __NVPTX__
#pragma omp declare target
#define __CUDA__
#define __OPENMP_NVPTX__
#include <__clang_cuda_math.h>
#pragma omp end declare target
#endif
