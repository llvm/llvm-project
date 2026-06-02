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

namespace lldb_private {

void ForEachPythonRuntimeCandidate(
    llvm::function_ref<bool(const char *)> callback) {
  // Bare names rely on the dynamic linker's search (LD_LIBRARY_PATH,
  // ldconfig cache, default paths). libpython3.so usually requires a -dev
  // package; the versioned SONAMEs cover stripped runtime installs. The
  // 3.8 floor matches Python's Stable ABI baseline.
  static constexpr const char *kCandidates[] = {
      "libpython3.so",        "libpython3.13.so.1.0", "libpython3.12.so.1.0",
      "libpython3.11.so.1.0", "libpython3.10.so.1.0", "libpython3.9.so.1.0",
      "libpython3.8.so.1.0",
  };
  for (const char *candidate : kCandidates) {
    if (callback(candidate))
      return;
  }
}

} // namespace lldb_private

#endif // LLDB_ENABLE_PYTHON
