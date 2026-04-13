//===- NativeMemoryAPIs.inc -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic wrappers for unix-style memory APIs (mmap, mprotect, etc.).
//
//===----------------------------------------------------------------------===//

#include "orc-rt/Error.h"
#include "orc-rt/MemoryFlags.h"

#include <fcntl.h>
#include <string.h>
#include <sys/errno.h>
#include <sys/mman.h>

namespace {

int toNativeProtFlags(orc_rt::MemProt MP) {
  int Prot = PROT_NONE;
  if ((MP & orc_rt::MemProt::Read) != orc_rt::MemProt::None)
    Prot |= PROT_READ;
  if ((MP & orc_rt::MemProt::Write) != orc_rt::MemProt::None)
    Prot |= PROT_WRITE;
  if ((MP & orc_rt::MemProt::Exec) != orc_rt::MemProt::None)
    Prot |= PROT_EXEC;
  return Prot;
}

#if defined(__APPLE__)
extern "C" void sys_icache_invalidate(const void *Addr, size_t Size);
#else
extern "C" void __clear_cache(void *Start, void *End);
#endif

static void invalidateInstructionCache(void *Addr, size_t Size) {
#if defined(__APPLE__)
  sys_icache_invalidate(Addr, Size);
#else
  __clear_cache(Addr, reinterpret_cast<char *>(Addr) + Size);
#endif
}

orc_rt::Expected<void *> hostOSMemoryReserve(size_t Size) {
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
  fd = open("/dev/zero", O_RDWR);
  if (fd == -1) {
    auto ErrNum = errno;
    return make_error<orc_rt::StringError>(
        std::string("Could not open /dev/zero for memory reserve: ") +
        strerror(ErrNum));
  }
#endif

  void *Addr = mmap(nullptr, Size, PROT_READ | PROT_WRITE, MapFlags, FD, 0);
  if (Addr == MAP_FAILED) {
    auto ErrNum = errno;
    return orc_rt::make_error<orc_rt::StringError>(
        std::string("mmap for memory reserve failed: ") + strerror(ErrNum));
  }

  return Addr;
}

orc_rt::Error hostOSMemoryRelease(void *Base, size_t Size) {
  if (munmap(Base, Size) != 0) {
    auto ErrNum = errno;
    return orc_rt::make_error<orc_rt::StringError>(
        std::string("munmap for memory release failed: ") + strerror(ErrNum));
  }
  return orc_rt::Error::success();
}

orc_rt::Error hostOSMemoryProtect(void *Base, size_t Size, orc_rt::MemProt MP) {
  if (mprotect(Base, Size, toNativeProtFlags(MP)) != 0) {
    auto ErrNum = errno;
    return orc_rt::make_error<orc_rt::StringError>(
        std::string("mprotect for memory finalize failed: ") +
        strerror(ErrNum));
  }

  if ((MP & orc_rt::MemProt::Exec) != orc_rt::MemProt::None)
    invalidateInstructionCache(Base, Size);

  return orc_rt::Error::success();
}

} // anonymous namespace
