//===- DynamicLibrary.cpp - POSIX dynamic library operations ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of orc-rt-internal/bedrock/sys/DynamicLibrary.h on POSIX
// systems, in terms of dlfcn.h.
//
//===----------------------------------------------------------------------===//

#include "orc-rt-internal/bedrock/sys/DynamicLibrary.h"

#include "orc-rt-internal/support/StringExtras.h"

#include <cassert>
#include <dlfcn.h>

namespace orc_rt::sys {

namespace {

/// Map an orc-rt symbol name to the name dlsym expects, or nullopt if the name
/// cannot name a symbol on this system.
///
/// This is the whole of the OS difference in lookupLibrarySymbols, so it is
/// forked here rather than in its caller.
std::optional<const char *> toDLSymName(const std::string &Name) {
#if defined(__APPLE__)
  // Mach-O prefixes global symbols with '_', but dlsym takes the unprefixed
  // form, so a name that lacks the prefix cannot name a global symbol.
  if (Name.empty() || Name[0] != '_')
    return std::nullopt;
  return Name.c_str() + 1;
#else
  return Name.c_str();
#endif
}

} // namespace

using DylibHandle = orc_rt::NativeDylibManager::DylibHandle;
using SymbolLookupResult =
        orc_rt::NativeDylibManager::SymbolLookupResult;

orc_rt::Expected<orc_rt::NativeDylibManager::DylibHandle>
hostOSLoadLibrary(const std::string &Path) {
  assert(!Path.empty() && "hostOSLoadLibrary doesn't support empty paths");

  void *Handle = dlopen(Path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (!Handle) {
    std::ostringstream ErrMsg;
    ErrMsg << "error loading \"" << Path << "\": " << dlerror();
    return orc_rt::make_error<orc_rt::StringError>(ErrMsg.str());
  }

  return orc_rt::NativeDylibManager::DylibHandle{
      orc_rt::NativeDylibManager::DylibHandle::Kind::Library, Handle};
}

orc_rt::Error
unloadLibrary(const orc_rt::NativeDylibManager::DylibHandle &Handle) {
  assert(Handle.K == orc_rt::NativeDylibManager::DylibHandle::Kind::Library &&
         "global dylib handle must not be unloaded");
  assert(Handle.LibraryHandle && "invalid library handle");

  if (dlclose(Handle.LibraryHandle) != 0)
    return orc_rt::make_error<orc_rt::StringError>(
        (std::ostringstream()
         << "error unloading " << Handle.LibraryHandle << ": " << dlerror())
            .str());

  return orc_rt::Error::success();
}

NativeDylibManager::SymbolLookupResult
hostOSLookup(void *Handle, const std::vector<std::string> &Names) {
  NativeDylibManager::SymbolLookupResult Result;
  Result.reserve(Names.size());

  for (const auto &Name : Names) {
    auto LookupName = toDLSymName(Name);
    if (!LookupName) {
      Result.push_back(std::nullopt);
      continue;
    }

    if (void *Addr = dlsym(Handle, *LookupName))
      Result.push_back(Addr);
    else if (dlerror() == nullptr)
      Result.push_back(nullptr);
    else
      Result.push_back(std::nullopt);
  }

  return Result;
}

NativeDylibManager::SymbolLookupResult
hostOSLibraryLookup(const NativeDylibManager::DylibHandle &Handle,
                    const std::vector<std::string> &Names) {
  assert(Handle.K == NativeDylibManager::DylibHandle::Kind::Library &&
         "expected library dylib handle");
  assert(Handle.LibraryHandle && "invalid library handle");

  return hostOSLookup(Handle.LibraryHandle, Names);
}

NativeDylibManager::SymbolLookupResult
hostOSGlobalLookup(const std::vector<std::string> &Names) {
  return hostOSLookup(RTLD_DEFAULT, Names);
}
} // namespace
