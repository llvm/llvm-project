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
#include <__clang_hip_math.h>
#include <__clang_cuda_complex_builtins.h>
#pragma omp end declare target
#endif

#ifdef __NVPTX__
#pragma omp declare target
#define __CUDA__
#define __OPENMP_NVPTX__
#include <__clang_cuda_math.h>
#pragma omp end declare target
#endif
