//===-- hip_runtime.h - HIP runtime API declarations ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OFFLOAD_LANGUAGES_INCLUDE_HIP_HIP_RUNTIME_H
#define LLVM_OFFLOAD_LANGUAGES_INCLUDE_HIP_HIP_RUNTIME_H

#define LANGUAGE hip

#include "../kernel/DefineLanguageNames.inc"

#include "../kernel/LanguageRuntime.h"

#include "../kernel/UndefineLanguageNames.inc"

#undef LANGUAGE

#define hipHostMallocDefault hipHostAllocDefault
#define hipHostMallocPortable hipHostAllocPortable
#define hipHostMallocMapped hipHostAllocMapped
#define hipHostMallocWriteCombined hipHostAllocWriteCombined
#define hipHostMallocNonCoherent 0x80000000

inline hipError_t hipHostMalloc(void **Ptr, size_t Size, unsigned int Flags) {
  return hipHostAlloc(Ptr, Size, Flags);
}

inline hipError_t hipHostFree(void *Ptr) { return ::hipFreeHost(Ptr); }

#ifdef __cplusplus
template <class T>
static inline hipError_t hipHostMalloc(T **Ptr, size_t Size,
                                       unsigned int Flags) {
  return ::hipHostMalloc((void **)Ptr, Size, Flags);
}

template <class T> static inline hipError_t hipHostFree(T *Ptr) {
  return ::hipHostFree((void *)Ptr);
}
#endif

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

#endif // LLVM_OFFLOAD_LANGUAGES_INCLUDE_HIP_HIP_RUNTIME_H
