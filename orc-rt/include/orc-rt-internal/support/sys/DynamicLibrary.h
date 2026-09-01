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
// see lib/bedrock/posix/DynamicLibrary.cpp and its siblings. A target with no
// dynamic loader has no native dylib manager.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_INTERNAL_SUPPORT_SYS_DYNAMICLIBRARY_H
#define ORC_RT_INTERNAL_SUPPORT_SYS_DYNAMICLIBRARY_H

#include "orc-rt/support/Error.h"

#include <optional>
#include <string>
#include <vector>

namespace orc_rt::sys {

/// Returns a handle that looks up symbols in every library loaded into the
/// process.
void *globalLookupHandle();

/// Load the library at the given path. Path must not be empty.
Expected<void *> loadLibrary(const std::string &Path);

/// Unload a library previously returned by loadLibrary.
Error unloadLibrary(void *Handle);

/// Look Names up in Handle, returning one result per name in order.
///
/// A result is nullopt if the name is not present in the library, and a
/// (possibly null) address if it is: a symbol genuinely located at address zero
/// is reported as null rather than as missing.
std::vector<std::optional<void *>>
lookupLibrarySymbols(void *Handle, const std::vector<std::string> &Names);

} // namespace orc_rt::sys

#endif // ORC_RT_INTERNAL_SUPPORT_SYS_DYNAMICLIBRARY_H
