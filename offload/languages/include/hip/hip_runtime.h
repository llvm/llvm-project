/*===---- hip_runtime.h - HIP runtime api declarations ---------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#pragma once

#define LANGUAGE hip

#include "../kernel/DefineLanguageNames.inc"

#include "../kernel/LanguageRuntime.h"

#include "../kernel/UndefineLanguageNames.inc"

#undef LANGUAGE

enum hipHostMallocFlag_t { hipHostMallocNonCoherent = 0x80000000 };

hipError_t hipHostMalloc(void **Ptr, size_t Size, unsigned int Flags) {
  return hipHostAlloc(Ptr, Size, Flags);
}

template <class T>
static inline hipError_t hipHostMalloc(T **Ptr, size_t Size,
                                       unsigned int Flags) {
  return ::hipHostMalloc((void **)Ptr, Size, Flags);
}

#if defined(__AMDGPU__) || defined(__NVPTX__)
#define HIP_KERNEL_NAME(...) __VA_ARGS__

extern "C" hipError_t hipLaunchKernel(const char *Kernel, dim3 GridDim,
                                      dim3 BlockDim, void **KernelArgs,
                                      size_t DynamicSharedMem, void *Stream);

template <typename... AT, typename FT = void (*)(AT...)>
static inline void hipLaunchKernelGGL(FT Kernel, dim3 GridDim, dim3 BlockDim,
                                      size_t DynamicSharedMem, void *Stream,
                                      AT... KernelArgs) {
  Kernel<<<GridDim, BlockDim, DynamicSharedMem, Stream>>>(KernelArgs...);
}
#endif
