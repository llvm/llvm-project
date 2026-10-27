//===-- DefineLanguageNames.inc - Kernel language API name definitions ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define COMBINE2(X, Y) X##Y
#define COMBINE(X, Y) COMBINE2(X, Y)

#define Error_t COMBINE(LANGUAGE, Error_t)
#define DeviceProp_t COMBINE(LANGUAGE, DeviceProp_t)
#define Malloc COMBINE(LANGUAGE, Malloc)
#define Free COMBINE(LANGUAGE, Free)
#define Memcpy COMBINE(LANGUAGE, Memcpy)
#define DeviceSynchronize COMBINE(LANGUAGE, DeviceSynchronize)
#define Success COMBINE(LANGUAGE, Success)
#define ErrorInvalidValue COMBINE(LANGUAGE, ErrorInvalidValue)
#define ErrorInvalidDevice COMBINE(LANGUAGE, ErrorInvalidDevice)
#define ErrorUnknown COMBINE(LANGUAGE, ErrorUnknown)
#define ErrorInvalidResourceHandle COMBINE(LANGUAGE, ErrorInvalidResourceHandle)
#define ErrorInvalidConfiguration COMBINE(LANGUAGE, ErrorInvalidConfiguration)
#define GetErrorName COMBINE(LANGUAGE, GetErrorName)
#define GetErrorString COMBINE(LANGUAGE, GetErrorString)
#define GetLastError COMBINE(LANGUAGE, GetLastError)
#define PeekAtLastError COMBINE(LANGUAGE, PeekAtLastError)
#define MemcpyKind COMBINE(LANGUAGE, MemcpyKind)
#define MemcpyHostToHost COMBINE(LANGUAGE, MemcpyHostToHost)
#define MemcpyHostToDevice COMBINE(LANGUAGE, MemcpyHostToDevice)
#define MemcpyDeviceToHost COMBINE(LANGUAGE, MemcpyDeviceToHost)
#define MemcpyDeviceToDevice COMBINE(LANGUAGE, MemcpyDeviceToDevice)
#define MemcpyDefault COMBINE(LANGUAGE, MemcpyDefault)
#define GetDevice COMBINE(LANGUAGE, GetDevice)
#define GetDeviceCount COMBINE(LANGUAGE, GetDeviceCount)
#define SetDevice COMBINE(LANGUAGE, SetDevice)
#define HostAlloc COMBINE(LANGUAGE, HostAlloc)
#define HostAllocDefault COMBINE(LANGUAGE, HostAllocDefault)
#define HostAllocPortable COMBINE(LANGUAGE, HostAllocPortable)
#define HostAllocMapped COMBINE(LANGUAGE, HostAllocMapped)
#define HostAllocWriteCombined COMBINE(LANGUAGE, HostAllocWriteCombined)
#define MallocHost COMBINE(LANGUAGE, MallocHost)
#define FreeHost COMBINE(LANGUAGE, FreeHost)
#define GetDeviceProperties COMBINE(LANGUAGE, GetDeviceProperties)
#define Stream_t COMBINE(LANGUAGE, Stream_t)
#define StreamDefault COMBINE(LANGUAGE, StreamDefault)
#define StreamNonBlocking COMBINE(LANGUAGE, StreamNonBlocking)
#define StreamCreate COMBINE(LANGUAGE, StreamCreate)
#define StreamCreateWithFlags COMBINE(LANGUAGE, StreamCreateWithFlags)
#define StreamDestroy COMBINE(LANGUAGE, StreamDestroy)
#define StreamSynchronize COMBINE(LANGUAGE, StreamSynchronize)
