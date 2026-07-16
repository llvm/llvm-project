//===- NativeDylibAPIs.inc --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic wrappers for POSIX dlfcn.h APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/Error.h"

#include <dlfcn.h>

namespace {

inline void *hostOSGetGlobalLookupHandle() { return RTLD_DEFAULT; }

orc_rt::Expected<void *> hostOSLoadLibrary(const std::string &Path) {
  assert(!Path.empty() && "hostOSLoadLibray doesn't support empty paths");
  void *H = dlopen(Path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (H == nullptr) {
    std::ostringstream ErrMsg;
    ErrMsg << "error loading \"" << Path << "\": " << dlerror();
    return orc_rt::make_error<orc_rt::StringError>(ErrMsg.str());
  }

  return H;
}

orc_rt::Error hostOSUnloadLibrary(void *Handle) {
  if (dlclose(Handle) != 0)
    return orc_rt::make_error<orc_rt::StringError>(
        (std::ostringstream()
         << "error unloading " << Handle << ": " << dlerror())
            .str());
  return orc_rt::Error::success();
}

std::vector<std::optional<void *>>
hostOSLibraryLookup(void *Handle, const std::vector<std::string> &Names) {
  std::vector<std::optional<void *>> Result;
  Result.reserve(Names.size());
  // Reset dlerror so we can distinguish "dlsym returned null because the
  // symbol is present at address 0" from "dlsym returned null because the
  // symbol isn't in the library" via per-iteration dlerror() checks.
  dlerror();
  for (const auto &Name : Names) {
#if defined(__APPLE__)
    if (Name.empty() || Name[0] != '_') {
      Result.push_back(std::nullopt);
      continue;
    }
    const char *LookupName = Name.c_str() + 1;
#else
    const char *LookupName = Name.c_str();
#endif // defined(__APPLE__)

    if (void *Addr = dlsym(Handle, LookupName))
      Result.push_back(Addr);
    else if (dlerror() == nullptr)
      Result.push_back(nullptr);
    else
      Result.push_back(std::nullopt);
  }
  return Result;
}

} // namespace
