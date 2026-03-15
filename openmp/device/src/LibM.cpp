//===------- LibC.cpp - Simple implementation of libc functions --- C++ ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __BUILD_MATH_BUILTINS_LIB__

#include "DeviceTypes.h"
#include "Platform.h"

using size_t = decltype(sizeof(char));

// We cannot use variants as we need the "C" symbol names to be exported.
#ifdef __AMDGPU__

#define __OPENMP_SKIP_INCLUDE__
#define __OPENMP_AMDGCN__

#pragma push_macro("__device__")
#define __device__

#include <__clang_hip_libdevice_declares.h>

#pragma pop_macro("__device__")

#include <__clang_cuda_complex_builtins.h>
#include <__clang_hip_math.h>

#undef __OPENMP_AMDGCN__

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
