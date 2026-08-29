//===- Memory.cpp - POSIX system memory operations -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of orc-rt-internal/support/sys/Memory.h on POSIX
// systems, in terms of mmap / munmap / mprotect.
//
//===----------------------------------------------------------------------===//

#include "orc-rt-internal/support/sys/Memory.h"

#include "orc-rt-internal/support/sys/CacheControl.h"

#include <fcntl.h>
#include <string.h>
#include <sys/errno.h>
#include <sys/mman.h>

namespace orc_rt::sys {

namespace {

int toNativeProtFlags(MemProt MP) {
  int Prot = PROT_NONE;
  if ((MP & MemProt::Read) != MemProt::None)
    Prot |= PROT_READ;
  if ((MP & MemProt::Write) != MemProt::None)
    Prot |= PROT_WRITE;
  if ((MP & MemProt::Exec) != MemProt::None)
    Prot |= PROT_EXEC;
  return Prot;
}

} // namespace

Expected<void *> reserveMemory(size_t Size) {
  if (Size == 0)
    return nullptr;

  int FD = 0;
  int MapFlags = MAP_PRIVATE;

#if defined(MAP_ANON)
  // If MAP_ANON is available then use it.
  FD = -1;
  MapFlags |= MAP_ANON;
#else // !defined(MAP_ANON)
  // Fall back to /dev/zero for strict POSIX.
  FD = open("/dev/zero", O_RDWR);
  if (FD == -1) {
    auto ErrNum = errno;
    return make_error<StringError>(
        std::string("Could not open /dev/zero for memory reserve: ") +
        strerror(ErrNum));
  }
#endif

  void *Addr = mmap(nullptr, Size, PROT_READ | PROT_WRITE, MapFlags, FD, 0);
  if (Addr == MAP_FAILED) {
    auto ErrNum = errno;
    return make_error<StringError>(
        std::string("mmap for memory reserve failed: ") + strerror(ErrNum));
  }

  return Addr;
}

Error releaseMemory(void *Base, size_t Size) {
  if (munmap(Base, Size) != 0) {
    auto ErrNum = errno;
    return make_error<StringError>(
        std::string("munmap for memory release failed: ") + strerror(ErrNum));
  }
  return Error::success();
}

Error protectMemory(void *Base, size_t Size, MemProt MP) {
  if (mprotect(Base, Size, toNativeProtFlags(MP)) != 0) {
    auto ErrNum = errno;
    return make_error<StringError>(
        std::string("mprotect for memory finalize failed: ") +
        strerror(ErrNum));
  }

  if ((MP & MemProt::Exec) != MemProt::None)
    clear_icache(Base, Size);

  return Error::success();
}

} // namespace orc_rt::sys
