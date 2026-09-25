//===- DynamicLibrary.cpp - Windows dynamic library operations ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "orc-rt-internal/bedrock/sys/DynamicLibrary.h"
#include "orc-rt-internal/support/sys/WinErrorToORCError.h"

#include <cassert>
#include <cstdint>

#include <psapi.h>
#include <windows.h>

namespace orc_rt::sys {

namespace {

// Window has no equivalent of RTLD_DEFAULT. nullptr means "current exe",
// so to avoid confusiion use a sentinal address.
// Keep the sentinel in an anonymous namespace so that it has internal linkage
char GlobalLookupSentinel;
bool isGlobalLookupHandle(void *Handle) {
  return Handle == &GlobalLookupSentinel;
}

std::optional<void *> lookupSymbol(HMODULE Handle, const std::string &Name) {
  if (auto Addr = GetProcAddress(Handle, Name.c_str()))
    return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(Addr));

  return std::nullopt;
}

bool getProcessModules(std::vector<HMODULE> &Modules) {
  HANDLE Process = GetCurrentProcess();

  DWORD BytesNeeded = 0;
  if (!EnumProcessModulesEx(Process, nullptr, 0, &BytesNeeded,
                            LIST_MODULES_64BIT))
    return false;

  for (;;) {
    assert(BytesNeeded % sizeof(HMODULE) == 0);

    Modules.resize(BytesNeeded / sizeof(HMODULE));

    DWORD NewBytesNeeded = 0;
    if (!EnumProcessModulesEx(
            Process, Modules.data(),
            static_cast<DWORD>(Modules.size() * sizeof(HMODULE)),
            &NewBytesNeeded, LIST_MODULES_64BIT))
      return false;

    if (NewBytesNeeded <= Modules.size() * sizeof(HMODULE)) {
      Modules.resize(NewBytesNeeded / sizeof(HMODULE));
      return true;
    }

    // The module list changed between the size query and enumeration.
    BytesNeeded = NewBytesNeeded;
  }
}

} // namespace

void *globalLookupHandle() { return &GlobalLookupSentinel; }
Expected<void *> loadLibrary(const std::string &Path) {
  assert(!Path.empty() && "loadLibrary doesn't support empty paths");

  HMODULE Handle = LoadLibraryA(Path.c_str());
  if (!Handle) {
    std::string Prefix = "error loading \"" + Path + "\"";
    return generateErrorFromGetLastError(Prefix);
  }

  return reinterpret_cast<void *>(Handle);
}

Error unloadLibrary(void *Handle) {
  assert(Handle && "invalid library handle");
  assert(!isGlobalLookupHandle(Handle) &&
         "global lookup handle must not be unloaded");

  if (!FreeLibrary(static_cast<HMODULE>(Handle)))
    return generateErrorFromGetLastError("error unloading library");

  return Error::success();
}

std::vector<std::optional<void *>>
lookupLibrarySymbols(void *Handle, const std::vector<std::string> &Names) {
  std::vector<std::optional<void *>> Result;
  Result.reserve(Names.size());

  if (isGlobalLookupHandle(Handle)) {
    std::vector<HMODULE> Modules;

    if (!getProcessModules(Modules)) {
      Result.resize(Names.size(), std::nullopt);
      return Result;
    }

    for (const auto &Name : Names) {
      std::optional<void *> Addr;

      // Search the executable first.
      if (!Modules.empty())
        Addr = lookupSymbol(Modules.front(), Name);

      // Match LLVM's existing Windows DynamicLibrary behavior by searching
      // loaded DLLs in reverse order.
      if (!Addr && Modules.size() > 1) {
        for (auto I = Modules.rbegin(), E = Modules.rend() - 1; I != E; ++I) {
          Addr = lookupSymbol(*I, Name);
          if (Addr)
            break;
        }
      }

      Result.push_back(Addr);
    }

    return Result;
  }

  assert(Handle && "invalid library handle");

  HMODULE Library = static_cast<HMODULE>(Handle);
  for (const auto &Name : Names)
    Result.push_back(lookupSymbol(Library, Name));

  return Result;
}

} // namespace orc_rt::sys
