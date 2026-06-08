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

orc_rt::Expected<void *> hostOSLoadLibrary(const std::string &Path) {
  void *H = dlopen(Path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (H == nullptr) {
    std::ostringstream ErrMsg;
    ErrMsg << "error loading ";
    if (!Path.empty())
      ErrMsg << "\"" << Path << "\"";
    else
      ErrMsg << "process symbols";
    ErrMsg << ": " << dlerror();
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
    if (void *Addr = dlsym(Handle, Name.c_str()))
      Result.push_back(Addr);
    else if (dlerror() == nullptr)
      Result.push_back(nullptr);
    else
      Result.push_back(std::nullopt);
  }
  return Result;
}

} // anonymous namespace
