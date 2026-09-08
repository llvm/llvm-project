//===- Memory.cpp - POSIX system memory operations -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of orc-rt-internal/bedrock/sys/Memory.h on POSIX
// systems, in terms of mmap / munmap / mprotect.
//
//===----------------------------------------------------------------------===//

#include "orc-rt-internal/bedrock/sys/Memory.h"

#include "orc-rt-internal/support/sys/CacheControl.h"
#include "orc-rt-internal/support/sys/Errno.h"

#include <errno.h>
#include <sys/mman.h>

namespace orc_rt::sys {

namespace {

#if defined(MAP_ANON)
constexpr int MapAnonFlag = MAP_ANON;
#elif defined(MAP_ANONYMOUS)
constexpr int MapAnonFlag = MAP_ANONYMOUS;
#else
#error "orc-rt requires anonymous mmap (MAP_ANON / MAP_ANONYMOUS)"
#endif

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

  void *Addr = mmap(nullptr, Size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MapAnonFlag, -1, 0);
  if (Addr == MAP_FAILED) {
    auto ErrNum = errno;
    return make_error<StringError>(
        std::string("mmap for memory reserve failed: ") + strError(ErrNum));
  }

  return Addr;
}

Error releaseMemory(void *Base, size_t Size) {
  if (munmap(Base, Size) != 0) {
    auto ErrNum = errno;
    return make_error<StringError>(
        std::string("munmap for memory release failed: ") + strError(ErrNum));
  }
  return Error::success();
}

Error protectMemory(void *Base, size_t Size, MemProt MP) {
  if (mprotect(Base, Size, toNativeProtFlags(MP)) != 0) {
    auto ErrNum = errno;
    return make_error<StringError>(
        std::string("mprotect for memory finalize failed: ") +
        strError(ErrNum));
  }

  if ((MP & MemProt::Exec) != MemProt::None)
    clear_icache(Base, Size);

  return Error::success();
}

} // namespace orc_rt::sys
