//===--- Memory.cpp - Windows memory operations ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "orc-rt-internal/bedrock/sys/Memory.h"
#include "orc-rt-internal/support/sys/WinErrorToORCError.h"

#include <windows.h>

namespace orc_rt::sys {

namespace {

DWORD getWindowsProtection(MemProt Prot) {
  const bool Read = (Prot & MemProt::Read) != MemProt::None;
  const bool Write = (Prot & MemProt::Write) != MemProt::None;
  const bool Exec = (Prot & MemProt::Exec) != MemProt::None;

  if (Exec) {
    if (Write)
      return PAGE_EXECUTE_READWRITE;
    if (Read)
      return PAGE_EXECUTE_READ;
    return PAGE_EXECUTE;
  }

  if (Write)
    return PAGE_READWRITE;
  if (Read)
    return PAGE_READONLY;

  return PAGE_NOACCESS;
}

} // namespace

Expected<void *> reserveMemory(uint64_t Size) {
  void *Addr = VirtualAlloc(nullptr, static_cast<SIZE_T>(Size),
                            MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);

  if (!Addr)
    return generateErrorFromGetLastError("VirtualAlloc failed");

  return Addr;
}

Error releaseMemory(void *Base, uint64_t) {
  // When MEM_RELEASE is specified, dwSize must be zero and Base must be the
  // address returned by VirtualAlloc.
  if (!VirtualFree(Base, 0, MEM_RELEASE))
    return generateErrorFromGetLastError("VirtualFree failed");

  return Error::success();
}

Error protectMemory(void *Base, uint64_t Size, MemProt Prot) {
  DWORD OldProtect = 0;

  if (!VirtualProtect(Base, static_cast<SIZE_T>(Size),
                      getWindowsProtection(Prot), &OldProtect))
    return generateErrorFromGetLastError("VirtualProtect failed");

  if ((Prot & MemProt::Exec) != MemProt::None &&
      !FlushInstructionCache(GetCurrentProcess(), Base,
                             static_cast<SIZE_T>(Size)))
    return generateErrorFromGetLastError("FlushInstructionCache failed");

  return Error::success();
}

} // namespace orc_rt::sys
