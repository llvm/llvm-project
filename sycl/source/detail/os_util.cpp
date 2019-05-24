//===-- os_util.cpp - OS utilities implementation---------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/exception.hpp>

#if defined(SYCL_RT_OS_LINUX)

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif // _GNU_SOURCE

#include <link.h>
#include <stdio.h>
#include <sys/sysinfo.h>

#elif defined(SYCL_RT_OS_WINDOWS)

#include <Windows.h>
#include <malloc.h>
#endif

namespace cl {
namespace sycl {
namespace detail {

const OSModuleHandle OSUtil::ExeModuleHandle =
    reinterpret_cast<OSModuleHandle>(-1);

#if defined(SYCL_RT_OS_LINUX)

struct ModuleInfo {
  const void *VirtAddr; // in
  void *Handle;         // out
  const char *Name;     // out
};

static int callback(struct dl_phdr_info *Info, size_t Size, void *Data) {
  auto Base = reinterpret_cast<unsigned char *>(Info->dlpi_addr);
  auto MI = reinterpret_cast<ModuleInfo *>(Data);
  auto TestAddr = reinterpret_cast<const unsigned char *>(MI->VirtAddr);

  for (int i = 0; i < Info->dlpi_phnum; ++i) {
    unsigned char *SegStart = Base + Info->dlpi_phdr[i].p_vaddr;
    unsigned char *SegEnd = SegStart + Info->dlpi_phdr[i].p_memsz;

    // check if the tested address is within current segment
    if (TestAddr >= SegStart && TestAddr < SegEnd) {
      // ... it is - belongs to the module then
      // dlpi_addr is zero for the executable, replace it
      auto H = reinterpret_cast<void *>(Info->dlpi_addr);
      MI->Handle = H ? H : OSUtil::ExeModuleHandle;
      MI->Name = Info->dlpi_name;
      return 1; // non-zero tells to finish iteration via modules
    }
  }
  return 0;
}

OSModuleHandle OSUtil::getOSModuleHandle(const void *VirtAddr) {
  ModuleInfo Res = {VirtAddr, nullptr, nullptr};
  dl_iterate_phdr(callback, &Res);

  return reinterpret_cast<OSModuleHandle>(Res.Handle);
}

#elif defined(SYCL_RT_OS_WINDOWS)
// TODO: implement this function for Windows probably by using
// GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,...)
OSModuleHandle OSUtil::getOSModuleHandle(const void *VirtAddr) {
  throw runtime_error("OSUtil::getOSModuleHandle() is not implemented yet");
}
#endif // SYCL_RT_OS_WINDOWS

size_t OSUtil::getOSMemSize() {
#if defined(SYCL_RT_OS_LINUX)
  struct sysinfo MemInfo;
  sysinfo(&MemInfo);
  return static_cast<size_t>(MemInfo.totalram * MemInfo.mem_unit);
#elif defined(SYCL_RT_OS_WINDOWS)
  MEMORYSTATUSEX MemInfo;
  MemInfo.dwLength = sizeof(MemInfo);
  GlobalMemoryStatusEx(&MemInfo);
  return static_cast<size_t>(MemInfo.ullTotalPhys);
#endif
}

void *OSUtil::alignedAlloc(size_t Alignment, size_t NumBytes) {
#if defined(SYCL_RT_OS_LINUX)
  return aligned_alloc(Alignment, NumBytes);
#elif defined(SYCL_RT_OS_WINDOWS)
  return _aligned_malloc(NumBytes, Alignment);
#endif
}

void OSUtil::alignedFree(void *Ptr) {
#if defined(SYCL_RT_OS_LINUX)
  free(Ptr);
#elif defined(SYCL_RT_OS_WINDOWS)
  _aligned_free(Ptr);
#endif
}

} // namespace detail
} // namespace sycl
} // namespace cl
