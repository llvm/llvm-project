//===-- LanguageLaunch.cpp - Language launch API --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LanguageLaunch.h"
#include "State.h"

#include <algorithm>
#include <cstdio>

using RuntimeState = llvm::offload::StateTy;
using ThreadState = llvm::offload::ThreadStateTy;

/// Internal kernel launch implementation
ol_result_t __llvmLaunchKernelImpl(const char *KernelID, dim3 GridDim,
                                   dim3 BlockDim, void *KernelArgsPtr,
                                   size_t DynamicSharedMem, void *Stream) {
  ol_device_handle_t Device = ThreadState::getDefaultDevice();
  ol_symbol_handle_t Kernel = RuntimeState::getKernel(KernelID);

  ol_kernel_launch_size_args_t LaunchSizeArgs;
  LaunchSizeArgs.Dimensions =
      1 + (GridDim.y > 1 || BlockDim.y > 1) + (GridDim.z > 1 || BlockDim.z > 1);
  LaunchSizeArgs.NumGroups.x = GridDim.x;
  LaunchSizeArgs.NumGroups.y = std::max(GridDim.y, 1u);
  LaunchSizeArgs.NumGroups.z = std::max(GridDim.z, 1u);
  LaunchSizeArgs.GroupSize.x = BlockDim.x;
  LaunchSizeArgs.GroupSize.y = std::max(BlockDim.y, 1u);
  LaunchSizeArgs.GroupSize.z = std::max(BlockDim.z, 1u);
  LaunchSizeArgs.DynSharedMemory = DynamicSharedMem;

  ol_queue_handle_t Queue = Stream ? reinterpret_cast<ol_queue_handle_t>(Stream)
                                   : ThreadState::getDefaultQueue();

  struct OffloadKernelArgs {
    void **Args;
    size_t NumArgs;
    size_t *ArgSizes;
  };
  OffloadKernelArgs *OKA = static_cast<OffloadKernelArgs *>(KernelArgsPtr);

  return olLaunchKernel(Queue, Device, Kernel, &LaunchSizeArgs,
                        /*Properties=*/nullptr, OKA->NumArgs, OKA->Args,
                        OKA->ArgSizes);
}

extern "C" {

/// Push call configuration for kernel launch
unsigned __llvmPushCallConfiguration(dim3 GridSize, dim3 BlockSize,
                                     size_t SharedMemory, void *Stream) {
  CallConfigurationTy &CC = ThreadState::getCallConfiguration();

  CC.GridSize = GridSize;
  CC.BlockSize = BlockSize;
  CC.SharedMemory = SharedMemory;
  CC.Stream = Stream;
  return 0;
}

/// Pop call configuration for kernel launch
unsigned __llvmPopCallConfiguration(dim3 *GridSize, dim3 *BlockSize,
                                    size_t *SharedMemory, void **Stream) {
  CallConfigurationTy &CC = ThreadState::getCallConfiguration();
  *GridSize = CC.GridSize;
  *BlockSize = CC.BlockSize;
  *SharedMemory = CC.SharedMemory;
  *Stream = CC.Stream;
  return 0;
}

unsigned llvmLaunchKernel(const char *KernelID, dim3 GridDim, dim3 BlockDim,
                          void *KernelArgsPtr, size_t DynamicSharedMem,
                          void *Stream) {
  ol_result_t Result = __llvmLaunchKernelImpl(
      KernelID, GridDim, BlockDim, KernelArgsPtr, DynamicSharedMem, Stream);
  return Result ? Result->Code : 0;
}

} // extern "C"
