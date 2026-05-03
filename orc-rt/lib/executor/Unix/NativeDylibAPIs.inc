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

std::vector<void *> hostOSLibraryLookup(void *Handle,
                                        const std::vector<std::string> &Names) {
  std::vector<void *> Result;
  Result.reserve(Names.size());
  for (const auto &Name : Names)
    Result.push_back(dlsym(Handle, Name.c_str()));
  return Result;
}

} // anonymous namespace
