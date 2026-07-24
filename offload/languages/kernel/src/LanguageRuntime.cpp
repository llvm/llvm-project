//===-- LanguageRuntime.cpp - Kernel Language runtime API implementation --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "LanguageRuntime.h"
#include <cassert>

#ifndef LANGUAGE
#error This file should be included, or used, with a LANGUAGE macro set.
#endif

#include "RuntimeAPI.h"
#include "Types.h"

#include "OffloadAPI.h"

#include "DefineLanguageNames.inc"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define STR(X) #X
#define LANGUAGE_STR STR(LANGUAGE)

namespace language_runtime = llvm::offload::kernel;

static Error_t convertResult(ol_result_t Result) {
  if (Result == OL_SUCCESS)
    return Success;
  switch (Result->Code) {
  case OL_ERRC_INVALID_VALUE:
    return ErrorInvalidValue;
  default:
    return ErrorInvalidValue;
  }
}

Error_t Malloc(void **DevPtr, size_t Size) {
  ol_device_handle_t Device = language_runtime::getDefaultDevice();
  ol_result_t Result = olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, Size, DevPtr);
  return convertResult(Result);
}

Error_t Free(void *DevPtr) {
  ol_result_t Result = olMemFree(DevPtr);
  return convertResult(Result);
}

Error_t Memcpy(void *Dst, const void *Src, size_t Size, MemcpyKind Kind) {
  ol_queue_handle_t Queue = language_runtime::getDefaultQueue();

  ol_result_t Result;
  switch (Kind) {
  case MemcpyHostToHost: {
    ol_device_handle_t Host = language_runtime::getHostDevice();
    Result = olMemcpy(Queue, Dst, Host, const_cast<void *>(Src), Host, Size);
    break;
  }
  case MemcpyHostToDevice: {
    ol_device_handle_t Device = language_runtime::getDefaultDevice();
    ol_device_handle_t Host = language_runtime::getHostDevice();
    Result = olMemcpy(Queue, Dst, Device, const_cast<void *>(Src), Host, Size);
    break;
  }
  case MemcpyDeviceToHost: {
    ol_device_handle_t Device = language_runtime::getDefaultDevice();
    ol_device_handle_t Host = language_runtime::getHostDevice();

    Result = olMemcpy(Queue, Dst, Host, const_cast<void *>(Src), Device, Size);
    break;
  }
  case MemcpyDeviceToDevice: {
    ol_device_handle_t Device = language_runtime::getDefaultDevice();

    Result =
        olMemcpy(Queue, Dst, Device, const_cast<void *>(Src), Device, Size);
    break;
  }
  case MemcpyDefault:
    fprintf(stderr, LANGUAGE_STR "MemcpyDefault is not implemented yet");
    abort();
  };

  Result = olSyncQueue(Queue);

  return convertResult(Result);
}

Error_t DeviceSynchronize() {
  // TODO: This is not correct. We likely want to pipe this through to the
  // plugins.
  ol_queue_handle_t Queue = language_runtime::getDefaultQueue();
  ol_result_t Result = olSyncQueue(Queue);
  return convertResult(Result);
}

Error_t GetLastError() {
  // TODO:
  return Success;
}

Error_t PeekAtLastError() {
  // TODO:
  return Success;
}

const char *GetErrorName(Error_t Error) {
  // TODO:
  return "";
}

const char *GetErrorString(Error_t Error) {
  // TODO:
  return "";
}

Error_t GetDevice(int *DeviceNo) {
  ol_device_handle_t Device = language_runtime::getDevice(DeviceNo);
  if (!Device)
    return ErrorInvalidValue;
  return Success;
}

Error_t GetDeviceCount(int *Count) {
  *Count = language_runtime::getDeviceCount();
  return Success;
}

Error_t SetDevice(int DeviceNo) {
  ol_device_handle_t Device = language_runtime::setDefaultDevice(DeviceNo);
  assert(Device == language_runtime::getDefaultDevice() &&
         "Set Device is not Default Device");
  return Device ? Success : ErrorInvalidValue;
}

Error_t HostAlloc(void **Ptr, size_t Size, unsigned int Flags) {
  // TODO:
  ol_device_handle_t Device = language_runtime::getDefaultDevice();
  ol_result_t Result = olMemAlloc(Device, OL_ALLOC_TYPE_HOST, Size, Ptr);
  return convertResult(Result);
}

Error_t MallocHost(void **Ptr, size_t Size) {
  return HostAlloc(Ptr, Size, /* HostAllocDefault */ 0);
}

Error_t FreeHost(void *Ptr) {
  ol_result_t Result = olMemFree(Ptr);
  return convertResult(Result);
}

Error_t DriverGetVersion(int *Version) {
  // TODO:
  *Version = 42;
  return Success;
}

Error_t GetDeviceProperties(DeviceProp_t *DeviceProp, int DeviceNo) {
  // TODO: [h15] add remaining pci/mem fields
  ol_device_handle_t Device = language_runtime::getDefaultDevice();
  size_t nameSize = 0;
  olGetDeviceInfoSize(Device, OL_DEVICE_INFO_NAME, &nameSize);
  olGetDeviceInfo(Device, OL_DEVICE_INFO_NAME, nameSize, &DeviceProp->name[0]);
  olGetDeviceInfo(Device, OL_DEVICE_INFO_GLOBAL_MEM_SIZE, sizeof(size_t),
                  &DeviceProp->totalGlobalMem);
  olGetDeviceInfo(Device, OL_DEVICE_INFO_NUM_COMPUTE_UNITS, sizeof(uint32_t),
                  &DeviceProp->multiProcessorCount);
  olGetDeviceInfo(Device, OL_DEVICE_INFO_NUM_LANES, sizeof(uint32_t),
                  &DeviceProp->warpSize);
  return Success;
}

static Error_t getQueueFromStream(Stream_t Stream, ol_queue_handle_t *Queue) {
  if (!Stream)
    return ErrorInvalidValue;
  *Queue = reinterpret_cast<ol_queue_handle_t>(Stream);
  return Success;
}

Error_t StreamCreate(Stream_t *Stream) {
  ol_queue_handle_t Queue;
  olCreateQueue(language_runtime::getDefaultDevice(), &Queue);
  *Stream = reinterpret_cast<Stream_t>(Queue);
  return Success;
}

Error_t StreamCreateWithFlags(Stream_t *Stream, unsigned int Flags) {
  if (Flags == StreamCreateWithFlagsFlags::StreamDefault)
    // FIXME: [h15] offload streams are non-blocking by default
    return StreamCreate(Stream);
  if (Flags == StreamCreateWithFlagsFlags::StreamNonBlocking) {
    return StreamCreate(Stream);
  }
  return ErrorInvalidValue;
}

Error_t StreamDestroy(Stream_t Stream) {
  ol_queue_handle_t Queue;
  Error_t Err = getQueueFromStream(Stream, &Queue);
  if (Err != Success)
    return Err;
  ol_result_t Result = olDestroyQueue(Queue);
  return convertResult(Result);
}

Error_t StreamSynchronize(Stream_t Stream) {
  ol_queue_handle_t Queue;
  Error_t Err = getQueueFromStream(Stream, &Queue);
  if (Err != Success)
    return Err;
  ol_result_t Result = olSyncQueue(Queue);
  return convertResult(Result);
}
