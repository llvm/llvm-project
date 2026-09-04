//===- DynamicLibrary.cpp - POSIX dynamic library operations ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of orc-rt-internal/support/sys/DynamicLibrary.h on POSIX
// systems, in terms of dlfcn.h.
//
//===----------------------------------------------------------------------===//

#include "orc-rt-internal/support/sys/DynamicLibrary.h"

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

void *globalLookupHandle() { return RTLD_DEFAULT; }

Expected<void *> loadLibrary(const std::string &Path) {
  assert(!Path.empty() && "loadLibrary doesn't support empty paths");
  void *H = dlopen(Path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (H == nullptr)
    return make_error<StringError>((StringOutputStream()
                                    << "error loading \"" << Path
                                    << "\": " << dlerror())
                                       .str());

  return H;
}

Error unloadLibrary(void *Handle) {
  if (dlclose(Handle) != 0)
    return make_error<StringError>((StringOutputStream()
                                    << "error unloading " << Handle << ": "
                                    << dlerror())
                                       .str());
  return Error::success();
}

std::vector<std::optional<void *>>
lookupLibrarySymbols(void *Handle, const std::vector<std::string> &Names) {
  std::vector<std::optional<void *>> Result;
  Result.reserve(Names.size());
  // Reset dlerror so we can distinguish "dlsym returned null because the
  // symbol is present at address 0" from "dlsym returned null because the
  // symbol isn't in the library" via per-iteration dlerror() checks.
  dlerror();
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

} // namespace orc_rt::sys
