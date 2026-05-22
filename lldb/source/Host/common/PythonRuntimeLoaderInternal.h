//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_HOST_COMMON_PYTHONRUNTIMELOADERINTERNAL_H
#define LLDB_SOURCE_HOST_COMMON_PYTHONRUNTIMELOADERINTERNAL_H

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

namespace lldb_private {

/// Platform-specific candidate enumeration. Calls \p callback once per
/// candidate path in priority order; stops at the first call that returns
/// true. Implemented in PythonRuntimeLoaderDarwin.cpp /
/// PythonRuntimeLoaderLinux.cpp / PythonRuntimeLoaderWindows.cpp.
///
/// Using a callback (rather than returning a vector) lets the caller
/// short-circuit on the first candidate that loads cleanly, so platforms
/// that synthesize candidates lazily (e.g. Darwin invokes `xcrun` only when
/// hardcoded paths miss) don't pay for the more expensive ones up front.
void ForEachPythonRuntimeCandidate(
    llvm::function_ref<bool(llvm::StringRef)> callback);

} // namespace lldb_private

#endif // LLDB_SOURCE_HOST_COMMON_PYTHONRUNTIMELOADERINTERNAL_H
