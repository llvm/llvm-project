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

#ifndef LANGUAGE
#error This file should be included, or used, with a LANGUAGE macro set.
#endif

#include "ExportedAPI.h"
#include "Types.h"

#include "OffloadAPI.h"

#include "DefineLanguageNames.inc"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define STR(X) #X
#define LANGUAGE_STR STR(LANGUAGE)

Error_t olKConvertResult(ol_result_t Result) { return Success; }

Error_t Malloc(void **DevPtr, size_t Size) {
  ol_device_handle_t Device = olKGetDefaultDevice();
  ol_result_t Result = olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, Size, DevPtr);
  return olKConvertResult(Result);
}

Error_t Free(void *DevPtr) {
  ol_result_t Result = olMemFree(DevPtr);
  return olKConvertResult(Result);
}

Error_t Memcpy(void *Dst, const void *Src, size_t Size, MemcpyKind Kind) {
  ol_queue_handle_t Queue = olKGetDefaultQueue();

  ol_result_t Result;
  switch (Kind) {
  case MemcpyHostToHost: {
    ol_device_handle_t Host = olKGetHostDevice();
    Result = olMemcpy(Queue, Dst, Host, const_cast<void *>(Src), Host, Size);
    break;
  }
  case MemcpyHostToDevice: {
    ol_device_handle_t Device = olKGetDefaultDevice();
    ol_device_handle_t Host = olKGetHostDevice();
    Result = olMemcpy(Queue, Dst, Device, const_cast<void *>(Src), Host, Size);
    break;
  }
  case MemcpyDeviceToHost: {
    ol_device_handle_t Device = olKGetDefaultDevice();
    ol_device_handle_t Host = olKGetHostDevice();

    Result = olMemcpy(Queue, Dst, Host, const_cast<void *>(Src), Device, Size);
    break;
  }
  case MemcpyDeviceToDevice: {
    ol_device_handle_t Device = olKGetDefaultDevice();

    Result =
        olMemcpy(Queue, Dst, Device, const_cast<void *>(Src), Device, Size);
    break;
  }
  case MemcpyDefault:
    fprintf(stderr, LANGUAGE_STR "MemcpyDefault is not implemented yet");
    abort();
  };

  Result = olSyncQueue(Queue);

  return olKConvertResult(Result);
}

Error_t DeviceSynchronize() {
  // TODO: This is not correct. We likely want to pipe this through to the
  // plugins.
  ol_queue_handle_t Queue = olKGetDefaultQueue();
  ol_result_t Result = olSyncQueue(Queue);
  return olKConvertResult(Result);
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

Error_t GetDeviceCount(int *Count) {
  // TODO:
  *Count = 1;
  return Success;
}

Error_t SetDevice(int DeviceNo) {
  // TODO:
  return Success;
}

Error_t HostAlloc(void **Ptr, size_t Size, unsigned int Flags) {
  // TODO:
  ol_device_handle_t Device = olKGetDefaultDevice();
  ol_result_t Result = olMemAlloc(Device, OL_ALLOC_TYPE_HOST, Size, Ptr);
  return olKConvertResult(Result);
}

Error_t MallocHost(void **Ptr, size_t Size) {
  return HostAlloc(Ptr, Size, /* HostAllocDefault */ 0);
}

Error_t FreeHost(void *Ptr) {
  ol_result_t Result = olMemFree(Ptr);
  return olKConvertResult(Result);
}

Error_t DriverGetVersion(int *Version) {
  // TODO:
  *Version = 42;
  return Success;
}

Error_t GetDeviceProperties(DeviceProp_t *DeviceProp, int DeviceNo) {
  // TODO: [h15] add remaining pci/mem fields
  ol_device_handle_t Device = olKGetDefaultDevice();
  size_t name_size = 0;
  olGetDeviceInfoSize(Device, OL_DEVICE_INFO_NAME, &name_size);
  olGetDeviceInfo(Device, OL_DEVICE_INFO_NAME, name_size, &DeviceProp->name[0]);
  olGetDeviceInfo(Device, OL_DEVICE_INFO_GLOBAL_MEM_SIZE, sizeof(size_t),
                  &DeviceProp->totalGlobalMem);
  olGetDeviceInfo(Device, OL_DEVICE_INFO_WARP_SIZE, sizeof(uint32_t),
                  &DeviceProp->warpSize);
  DeviceProp->multiProcessorCount = 110;
  DeviceProp->major = 47;
  DeviceProp->minor = 11;
  return Success;
}

Error_t StreamCreate(Stream_t *Stream) {
  // TODO: [h15] implement
  fprintf(stderr, LANGUAGE_STR "StreamCreate is not implemented yet");
  abort();
}

Error_t StreamCreateWithFlags(Stream_t *Stream, unsigned int Flags) {
  // TODO: [h15] implement
  fprintf(stderr, LANGUAGE_STR "StreamCreateWithFlags is not implemented yet");
  abort();
}

Error_t StreamDestroy(Stream_t Stream) {
  // TODO: [h15] implement
  fprintf(stderr, LANGUAGE_STR "StreamDestroy is not implemented yet");
  abort();
}

Error_t StreamSynchronize(Stream_t Stream) {
  // TODO: [h15] implement
  fprintf(stderr, LANGUAGE_STR "StreamSynchronize is not implemented yet");
  abort();
}
