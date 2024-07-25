//===- Unix/MemoryLinux.h - Linux specific Helper Fuctions ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines Linux specific helper functions for memory management.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_SUPPORT_UNIX_MEMORYLINUX_H
#define LLVM_LIB_SUPPORT_UNIX_MEMORYLINUX_H

#ifndef __linux__
#error Linux only support header!
#endif

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"

#include <sys/mman.h>
#include <sys/syscall.h>

#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif
#ifndef MFD_EXEC
#define MFD_EXEC 0x0010U
#endif

namespace llvm {
namespace sys {
namespace {

static inline bool isPermissionError(int err) {
  // PaX uses EPERM, SELinux uses EACCES
  return err == EPERM || err == EACCES;
}

static inline bool execProtChangeNeedsNewMapping() {
  static int status = -1;

  if (status != -1)
    return status;

  // Try to get the status from /proc/self/status, looking for PaX flags.
  if (auto file = MemoryBuffer::getFileAsStream("/proc/self/status")) {
    auto pax_flags = (*file)->getBuffer().rsplit("PaX:").second.trim();
    if (!pax_flags.empty())
      // 'M' indicates MPROTECT is enabled
      return (status = pax_flags.find('M') != StringRef::npos);
  }

  // Create a temporary writable mapping and try to make it executable.  If
  // this fails, test 'errno' to ensure it failed because we were not allowed
  // to create such a mapping and not because of some transient error.
  size_t size = Process::getPageSizeEstimate();
  void *addr = ::mmap(NULL, size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (addr == MAP_FAILED) {
    // Must be low on memory or have too many mappings already, not much we can
    // do here.
    status = 0;
  } else {
    if (::mprotect(addr, size, PROT_READ | PROT_EXEC) < 0)
      status = isPermissionError(errno);
    else
      status = 0;
    ::munmap(addr, size);
  }

  return status;
}

static inline int memfd_create(const char *name, int flags) {
#ifdef SYS_memfd_create
  return syscall(SYS_memfd_create, name, flags);
#else
  errno = ENOSYS;
  return -1;
#endif
}

} // anonymous namespace
} // namespace sys
} // namespace llvm

#endif
