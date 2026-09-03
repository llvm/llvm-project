//===-- LanguageLaunch.cpp - Language launch API --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LanguageLaunch.h"
#include "LanguageUtils.h"
#include "OffloadErrors.h"
#include "State.h"
#include "Stream.h"
#include <cstdio>

using namespace llvm::offload;

/// Internal kernel launch implementation
ol_result_t __llvmLaunchKernelImpl(const char *KernelID, dim3 GridDim,
                                   dim3 BlockDim, void *KernelArgsPtr,
                                   size_t DynamicSharedMem, void *Stream) {
  StateTy &State = StateTy::get();
  ThreadStateTy &ThreadState = ThreadStateTy::get();
  ol_device_handle_t Device = ThreadState.getDefaultDevice();
  ol_symbol_handle_t Kernel = State.getKernel(KernelID);
  if (!Device)
    return &InvalidDeviceError;
  if (!KernelID || !KernelArgsPtr)
    return &InvalidArgumentError;
  if (!Kernel)
    return &InvalidKernelError;

  if (GridDim.x == 0 || GridDim.y == 0 || GridDim.z == 0 || BlockDim.x == 0 ||
      BlockDim.y == 0 || BlockDim.z == 0)
    return &InvalidConfigurationError;

  ol_kernel_launch_size_args_t LaunchSizeArgs;
  LaunchSizeArgs.Dimensions =
      1 + (GridDim.y > 1 || BlockDim.y > 1) + (GridDim.z > 1 || BlockDim.z > 1);
  LaunchSizeArgs.NumGroups.x = GridDim.x;
  LaunchSizeArgs.NumGroups.y = GridDim.y;
  LaunchSizeArgs.NumGroups.z = GridDim.z;
  LaunchSizeArgs.GroupSize.x = BlockDim.x;
  LaunchSizeArgs.GroupSize.y = BlockDim.y;
  LaunchSizeArgs.GroupSize.z = BlockDim.z;
  LaunchSizeArgs.DynSharedMemory = DynamicSharedMem;

  ol_queue_handle_t Queue = Stream ? reinterpret_cast<StreamTy *>(Stream)->Queue
                                   : ThreadState.getDefaultQueue();

  struct OffloadKernelArgs {
    void **Args;
    size_t NumArgs;
    size_t *ArgSizes;
  };
  OffloadKernelArgs *OKA = static_cast<OffloadKernelArgs *>(KernelArgsPtr);
  if ((!OKA->Args) != (!OKA->ArgSizes))
    return &InvalidArgumentError;
  if (OKA->NumArgs > 0 && !OKA->Args)
    return &InvalidArgumentError;
  for (size_t I = 0; I < OKA->NumArgs; ++I)
    if (!OKA->Args[I] || OKA->ArgSizes[I] == 0)
      return &InvalidArgumentError;

  return olLaunchKernel(Queue, Device, Kernel, &LaunchSizeArgs,
                        /*Properties=*/nullptr, OKA->NumArgs, OKA->Args,
                        OKA->ArgSizes);
}

extern "C" {

/// Push call configuration for kernel launch
unsigned __llvmPushCallConfiguration(dim3 GridSize, dim3 BlockSize,
                                     size_t SharedMemory, void *Stream) {
  ThreadStateTy &ThreadState = ThreadStateTy::get();
  CallConfigurationTy &CC = ThreadState.getCallConfiguration();

  CC.GridSize = GridSize;
  CC.BlockSize = BlockSize;
  CC.SharedMemory = SharedMemory;
  CC.Stream = Stream;
  return 0;
}

/// Pop call configuration for kernel launch
unsigned __llvmPopCallConfiguration(dim3 *GridSize, dim3 *BlockSize,
                                    size_t *SharedMemory, void **Stream) {
  ThreadStateTy &ThreadState = ThreadStateTy::get();
  CallConfigurationTy &CC = ThreadState.getCallConfiguration();
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
  convertAndSetLastError(Result);
  return Result ? Result->Code : 0;
}

} // extern "C"
