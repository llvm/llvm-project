//===------ LanguageLaunch.h - Header for LanguageLaunch.cpp ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LANGUAGE_LAUNCH_H
#define LLVM_LANGUAGE_LAUNCH_H

#include "OffloadAPI.h"
#include "Types.h"

#include <algorithm> // for std::max
#include <cstddef>
#include <cstdint>

extern "C" {

/// Push call configuration for kernel launch
unsigned __llvmPushCallConfiguration(dim3 __grid_size, dim3 __block_size,
                                     size_t __shared_memory, void *__stream);

/// Pop call configuration for kernel launch
unsigned __llvmPopCallConfiguration(dim3 *__grid_size, dim3 *__block_size,
                                    size_t *__shared_memory, void *__stream);

/// Internal kernel launch implementation
ol_result_t __llvmLaunchKernelImpl(const char *KernelID, dim3 GridDim,
                                   dim3 BlockDim, void *KernelArgsPtr,
                                   size_t DynamicSharedMem, void *Stream);

/// LLVM-style kernel launch entry points
unsigned __llvmLaunchKernel(const char *KernelID, dim3 GridDim, dim3 BlockDim,
                            void *KernelArgsPtr, size_t DynamicSharedMem,
                            void *Stream);

unsigned __llvmLaunchKernel_spt(const char *KernelID, dim3 GridDim,
                                dim3 BlockDim, void *KernelArgsPtr,
                                size_t DynamicSharedMem, void *Stream);

unsigned __llvmLaunchKernel_ptsz(const char *KernelID, dim3 GridDim,
                                 dim3 BlockDim, void *KernelArgsPtr,
                                 size_t DynamicSharedMem, void *Stream);
}

#endif // LLVM_LANGUAGE_LAUNCH_H
