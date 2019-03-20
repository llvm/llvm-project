//===-- os_util.cpp - OS utilities implementation---------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/os_util.hpp>

#if defined(SYCL_RT_OS_LINUX)

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif // _GNU_SOURCE

#include <link.h>
#include <stdio.h>
#include <stdlib.h>

#endif // SYCL_RT_OS_LINUX

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

static int callback(struct dl_phdr_info *info, size_t size, void *data) {
  unsigned char *Base = reinterpret_cast<unsigned char *>(info->dlpi_addr);
  ModuleInfo *MI = (ModuleInfo *)data;

  for (int i = 0; i < info->dlpi_phnum; ++i) {
    unsigned char *SegStart = Base + info->dlpi_phdr[i].p_vaddr;
    unsigned char *SegEnd = SegStart + info->dlpi_phdr[i].p_memsz;
    const unsigned char *TestAddr =
        reinterpret_cast<const unsigned char *>(MI->VirtAddr);

    // check if the tested address is within current segment
    if (TestAddr >= SegStart && TestAddr < SegEnd) {
      // ... it is - belongs to the module then
      // dlpi_addr is zero for the executable, replace it
      void *H = (void *)info->dlpi_addr;
      MI->Handle = H ? H : OSUtil::ExeModuleHandle;
      MI->Name = info->dlpi_name;
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
// GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,...)
// to implement getOSModuleHandle
#endif // SYCL_RT_OS_LINUX

} // namespace detail
} // namespace sycl
} // namespace cl
