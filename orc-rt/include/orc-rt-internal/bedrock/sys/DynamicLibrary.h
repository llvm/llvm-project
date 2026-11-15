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
#include "orc-rt/support/SymbolLookupSet.h"

#include <string>

namespace orc_rt::sys {

/// Returns a handle that looks up symbols in every library loaded into the
/// process.
void *globalLookupHandle();

/// Load the library at the given path. Path must not be empty.
Expected<void *> loadLibrary(const std::string &Path);

/// Unload a library previously returned by loadLibrary.
Error unloadLibrary(void *Handle);

/// Look the names in Symbols up in Handle, returning one result per name in
/// order. The lookup flags are ignored.
///
/// A result is nullopt if the name is not present in the library, and a
/// (possibly null) address if it is: a symbol genuinely located at address zero
/// is reported as null rather than as missing.
SymbolLookupResult lookupLibrarySymbols(void *Handle,
                                        const SymbolLookupSet &Symbols);

} // namespace orc_rt::sys

#endif // ORC_RT_INTERNAL_BEDROCK_SYS_DYNAMICLIBRARY_H
