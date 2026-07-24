//===------ LanguageLaunch.cpp - Language (CUDA/HIP) launch api -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "LanguageLaunch.h"
#include "RuntimeAPI.h"

#include <cstdio>

namespace language_launch = llvm::offload::kernel;

extern "C" {

/// Push call configuration for kernel launch
unsigned __llvmPushCallConfiguration(dim3 __grid_size, dim3 __block_size,
                                     size_t __shared_memory, void *__stream) {
  CallConfigurationTy &CC = *language_launch::getCallConfiguration();

  CC.GridSize = __grid_size;
  CC.BlockSize = __block_size;
  CC.SharedMemory = __shared_memory;
  CC.Stream = __stream;
  return 0;
}

/// Pop call configuration for kernel launch
unsigned __llvmPopCallConfiguration(dim3 *__grid_size, dim3 *__block_size,
                                    size_t *__shared_memory, void *__stream) {
  CallConfigurationTy &CC = *language_launch::getCallConfiguration();
  *__grid_size = CC.GridSize;
  *__block_size = CC.BlockSize;
  *__shared_memory = CC.SharedMemory;
  *((void **)__stream) = CC.Stream;
  return 0;
}

/// Internal kernel launch implementation
ol_result_t __llvmLaunchKernelImpl(const char *KernelID, dim3 GridDim,
                                   dim3 BlockDim, void *KernelArgsPtr,
                                   size_t DynamicSharedMem, void *Stream) {
  ol_device_handle_t Device = language_launch::getDefaultDevice();
  ol_symbol_handle_t Kernel = language_launch::getKernel(KernelID);

  ol_dimensions_t GridDimensions, BlockDimensions;
  ol_kernel_launch_size_args_t LaunchSizeArgs;
  LaunchSizeArgs.Dimensions =
      1 + !!(GridDim.y * BlockDim.y > 1) + !!(GridDim.z * BlockDim.z > 1);
  GridDimensions.x = GridDim.x;
  GridDimensions.y = std::max(GridDim.y, 1u);
  GridDimensions.z = std::max(GridDim.z, 1u);
  LaunchSizeArgs.NumGroups = GridDimensions;
  BlockDimensions.x = BlockDim.x;
  BlockDimensions.y = std::max(BlockDim.y, 1u);
  BlockDimensions.z = std::max(BlockDim.z, 1u);
  LaunchSizeArgs.GroupSize = BlockDimensions;
  LaunchSizeArgs.DynSharedMemory = DynamicSharedMem;

  ol_queue_handle_t Queue = Stream ? reinterpret_cast<ol_queue_handle_t>(Stream)
                                   : language_launch::getDefaultQueue();

  ol_kernel_launch_prop_t Properties = {.type = OL_KERNEL_LAUNCH_PROP_TYPE_NONE,
                                        .data = nullptr};

  struct OffloadKernelArgs {
    void **Args;
    size_t NumArgs;
    size_t *ArgSizes;
  };
  OffloadKernelArgs *OKA = reinterpret_cast<OffloadKernelArgs *>(KernelArgsPtr);

  ol_result_t Result;
  Result = olLaunchKernel(Queue, Device, Kernel, &LaunchSizeArgs, &Properties,
                          OKA->NumArgs, OKA->Args, OKA->ArgSizes);
  return Result;
}

#define LLVM_STYLE_LAUNCH(SUFFIX, PER_THREAD_STREAM)                           \
  unsigned __llvmLaunchKernel##SUFFIX(const char *KernelID, dim3 GridDim,      \
                                      dim3 BlockDim, void *KernelArgsPtr,      \
                                      size_t DynamicSharedMem, void *Stream) { \
    ol_result_t Result = __llvmLaunchKernelImpl(                               \
        KernelID, GridDim, BlockDim, KernelArgsPtr, DynamicSharedMem, Stream); \
    return Result ? Result->Code : 0;                                          \
  }

LLVM_STYLE_LAUNCH(, false);
LLVM_STYLE_LAUNCH(_spt, true);
LLVM_STYLE_LAUNCH(_ptsz, true);

} // extern "C"
