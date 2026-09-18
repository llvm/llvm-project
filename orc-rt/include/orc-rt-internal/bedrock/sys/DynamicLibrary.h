//===--- DynamicLibrary.h - System dynamic library operations ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The host dynamic-library operations that NativeDylibManager is built on.
//
// Exactly one implementation is compiled into the runtime, chosen by the build:
// see lib/bedrock/sys/posix/DynamicLibrary.cpp and its siblings. A target with
// no dynamic loader has no native dylib manager.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_INTERNAL_BEDROCK_SYS_DYNAMICLIBRARY_H
#define ORC_RT_INTERNAL_BEDROCK_SYS_DYNAMICLIBRARY_H

#include "orc-rt/support/Error.h"

#include <optional>
#include <string>
#include <vector>

namespace orc_rt::sys {

using DylibHandle = void *;
using SymbolLookupResult = std::vector<std::optional<void *>>;

/// Load the library at the given path. Path must not be empty.
Expected<DylibHandle> loadLibrary(const std::string &Path);

/// Unload a library previously returned by loadLibrary.
Error unloadLibrary(DylibHandle Handle);

/// Look Names up in Handle, returning one result per name in order.
SymbolLookupResult lookupLibrarySymbols(DylibHandle Handle,
                                        const std::vector<std::string> &Names);

/// Look Names up across libraries loaded into the process.
SymbolLookupResult lookupGlobalSymbols(const std::vector<std::string> &Names);

} // namespace orc_rt::sys

#endif // ORC_RT_INTERNAL_BEDROCK_SYS_DYNAMICLIBRARY_H
