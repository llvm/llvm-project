//===-- LanguageNames.inc - Kernel Language runtime API names -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
#define MemcpyKind COMBINE(LANGUAGE, MemcpyKind)
#define MemcpyHostToHost COMBINE(LANGUAGE, MemcpyHostToHost)
#define MemcpyHostToDevice COMBINE(LANGUAGE, MemcpyHostToDevice)
#define MemcpyDeviceToHost COMBINE(LANGUAGE, MemcpyDeviceToHost)
#define MemcpyDeviceToDevice COMBINE(LANGUAGE, MemcpyDeviceToDevice)
#define MemcpyDefault COMBINE(LANGUAGE, MemcpyDefault)
#define GetLastError COMBINE(LANGUAGE, GetLastError)
#define PeekAtLastError COMBINE(LANGUAGE, PeekAtLastError)
#define GetErrorName COMBINE(LANGUAGE, GetErrorName)
#define GetErrorString COMBINE(LANGUAGE, GetErrorString)
#define GetDeviceCount COMBINE(LANGUAGE, GetDeviceCount)
#define SetDevice COMBINE(LANGUAGE, SetDevice)
#define HostAlloc COMBINE(LANGUAGE, HostAlloc)
#define MallocHost COMBINE(LANGUAGE, MallocHost)
#define HostFree COMBINE(LANGUAGE, HostFree)
#define DriverGetVersion COMBINE(LANGUAGE, DriverGetVersion)
#define GetDeviceProperties COMBINE(LANGUAGE, GetDeviceProperties)
