//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Config.h"

#if LLDB_ENABLE_PYTHON

#include "../common/PythonRuntimeLoaderInternal.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"

namespace lldb_private {

void ForEachPythonRuntimeCandidate(
    llvm::function_ref<bool(llvm::StringRef)> callback) {
  // Bare names rely on the dynamic linker's search (LD_LIBRARY_PATH,
  // ldconfig cache, default lib paths). Stable ABI guarantees any of these
  // is sufficient for a stable-ABI plugin.
  //
  // libpython3.so is generally only present in -dev packages, so the
  // versioned SONAMEs are tried as a fallback. The supported range is
  // 3.8+ (the lower bound is the Python Stable ABI baseline LLDB already
  // requires).
  static constexpr llvm::StringLiteral kCandidates[] = {
      "libpython3.so",        "libpython3.13.so.1.0", "libpython3.12.so.1.0",
      "libpython3.11.so.1.0", "libpython3.10.so.1.0", "libpython3.9.so.1.0",
      "libpython3.8.so.1.0",
  };
  for (llvm::StringRef candidate : kCandidates) {
    if (callback(candidate))
      return;
  }
}

} // namespace lldb_private

#endif // LLDB_ENABLE_PYTHON
