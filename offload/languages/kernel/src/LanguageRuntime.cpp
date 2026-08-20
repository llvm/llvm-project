//===-- LanguageRuntime.cpp - Kernel language runtime API implementation --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE
#error This file should be included, or used, with a LANGUAGE macro set.
#endif

// Rename the generic runtime API before declaring or defining language symbols.
// clang-format off
#include "DefineLanguageNames.inc"
#include "LanguageErrors.h"
#include "LanguageRuntime.h"
// clang-format on

#include "LanguageUtils.h"
#include "State.h"
#include "Types.h"

#include "OffloadAPI.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using RuntimeState = llvm::offload::StateTy;
using ThreadState = llvm::offload::ThreadStateTy;

Error_t Malloc(void **DevPtr, size_t Size) {
  ol_device_handle_t Device = ThreadState::getDefaultDevice();
  ol_result_t Result = olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, Size, DevPtr);
  return convertAndSetLastError(Result);
}

Error_t Free(void *DevPtr) {
  ol_result_t Result = olMemFree(DevPtr);
  return convertAndSetLastError(Result);
}

Error_t Memcpy(void *Dst, const void *Src, size_t Size, MemcpyKind Kind) {
  ol_queue_handle_t Queue = ThreadState::getDefaultQueue();

  ol_result_t Result;
  switch (Kind) {
  case MemcpyHostToHost: {
    ol_device_handle_t Host = RuntimeState::getHostDevice();
    Result = olMemcpy(nullptr, Dst, Host, const_cast<void *>(Src), Host, Size);
    break;
  }
  case MemcpyHostToDevice: {
    ol_device_handle_t Device = ThreadState::getDefaultDevice();
    ol_device_handle_t Host = RuntimeState::getHostDevice();
    Result = olMemcpy(Queue, Dst, Device, const_cast<void *>(Src), Host, Size);
    break;
  }
  case MemcpyDeviceToHost: {
    ol_device_handle_t Device = ThreadState::getDefaultDevice();
    ol_device_handle_t Host = RuntimeState::getHostDevice();

    Result = olMemcpy(Queue, Dst, Host, const_cast<void *>(Src), Device, Size);
    break;
  }
  case MemcpyDeviceToDevice: {
    ol_device_handle_t Device = ThreadState::getDefaultDevice();

    Result =
        olMemcpy(Queue, Dst, Device, const_cast<void *>(Src), Device, Size);
    break;
  }
  case MemcpyDefault:
    FATAL_UNIMPLEMENTED("MemcpyDefault is not implemented yet");
  };

  if (Result != OL_SUCCESS)
    return convertAndSetLastError(Result);

  Result = olSyncQueue(Queue);
  return convertAndSetLastError(Result);
}

Error_t DeviceSynchronize() {
  // TODO: This is not correct. We likely want to pipe this through to the
  // plugins.
  ol_queue_handle_t Queue = ThreadState::getDefaultQueue();
  ol_result_t Result = olSyncQueue(Queue);
  return convertAndSetLastError(Result);
}

Error_t GetDevice(int *DeviceNo) {
  ol_device_handle_t Device = ThreadState::getDevice(DeviceNo);
  if (!Device)
    return setLastError(ErrorInvalidDevice);
  return setLastError(Success);
}

Error_t GetDeviceCount(int *Count) {
  *Count = RuntimeState::getDeviceCount();
  return setLastError(Success);
}

Error_t SetDevice(int DeviceNo) {
  ol_device_handle_t Device = ThreadState::setDefaultDevice(DeviceNo);
  if (!Device)
    return setLastError(ErrorInvalidDevice);
  assert(Device == ThreadState::getDefaultDevice() &&
         "Set Device is not Default Device");
  return setLastError(Success);
}

Error_t HostAlloc(void **Ptr, size_t Size, unsigned int Flags) {
  ol_device_handle_t Device = ThreadState::getDefaultDevice();
  ol_result_t Result = olMemAllocHost(Device, Size, Ptr);
  return convertAndSetLastError(Result);
}

Error_t MallocHost(void **Ptr, size_t Size) {
  return HostAlloc(Ptr, Size, /* HostAllocDefault */ 0);
}

Error_t FreeHost(void *Ptr) {
  ol_result_t Result = olMemFree(Ptr);
  return convertAndSetLastError(Result);
}

Error_t GetDeviceProperties(DeviceProp_t *DeviceProp, int DeviceNo) {
  ol_device_handle_t Device = ThreadState::getDefaultDevice();
  size_t NameSize = 0;
  olGetDeviceInfoSize(Device, OL_DEVICE_INFO_NAME, &NameSize);
  assert(NameSize <= sizeof(DeviceProp->name) &&
         "Device name is too long for DeviceProp_t");
  olGetDeviceInfo(Device, OL_DEVICE_INFO_NAME, NameSize, &DeviceProp->name[0]);
  olGetDeviceInfo(Device, OL_DEVICE_INFO_GLOBAL_MEM_SIZE, sizeof(size_t),
                  &DeviceProp->totalGlobalMem);
  olGetDeviceInfo(Device, OL_DEVICE_INFO_NUM_COMPUTE_UNITS, sizeof(uint32_t),
                  &DeviceProp->multiProcessorCount);
  olGetDeviceInfo(Device, OL_DEVICE_INFO_NUM_LANES, sizeof(uint32_t),
                  &DeviceProp->warpSize);
  return setLastError(Success);
}

Error_t StreamCreate(Stream_t *Stream) {
  ol_queue_handle_t Queue;
  ol_result_t Result = olCreateQueue(RuntimeState::getContext(),
                                     ThreadState::getDefaultDevice(), &Queue);
  if (Result == OL_SUCCESS)
    *Stream = reinterpret_cast<Stream_t>(Queue);
  return convertAndSetLastError(Result);
}

Error_t StreamDestroy(Stream_t Stream) {
  ol_queue_handle_t Queue;
  Error_t Err = getQueueFromStream(Stream, &Queue);
  if (Err != Success)
    return setLastError(Err);
  ol_result_t Result = olDestroyQueue(Queue);
  return convertAndSetLastError(Result);
}

Error_t StreamSynchronize(Stream_t Stream) {
  ol_queue_handle_t Queue;
  Error_t Err = getQueueFromStream(Stream, &Queue);
  if (Err != Success)
    return setLastError(Err);
  ol_result_t Result = olSyncQueue(Queue);
  return convertAndSetLastError(Result);
}

#include "UndefineLanguageNames.inc"
