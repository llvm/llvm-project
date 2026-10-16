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
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <dlfcn.h>

namespace lldb_private {

namespace {

// Xcode and the Command Line Tools ship Python3.framework. Toolchains and
// python.org ship Python.framework.
constexpr llvm::StringLiteral kFrameworks[] = {
    "Python3.framework/Versions/Current/Python3",
    "Python.framework/Versions/Current/Python",
};

// The directories relative to LLDB.framework where Python(3).framework could be
// present.
constexpr llvm::StringLiteral kLLDBRelativeDirs[] = {
    "../../../../../Developer/Library/Frameworks",
    "../../../../Developer/Library/Frameworks",
    "../../../../Frameworks",
    "../../..",
    "../../../../../../AppleInternal/Library/Frameworks",
};

/// Checks each Python framework in \p dir until one is found and \p callback
/// returns true.
bool TryFrameworks(llvm::function_ref<bool(const char *)> callback,
                   const llvm::Twine &dir) {
  for (llvm::StringRef framework : kFrameworks) {
    llvm::SmallString<256> path;
    llvm::sys::path::append(path, dir, framework);
    if (llvm::sys::fs::exists(path) && callback(path.c_str()))
      return true;
  }
  return false;
}

} // namespace

void ForEachPythonRuntimeCandidate(
    llvm::function_ref<bool(const char *)> callback) {

  // Prefer the Python relative to LLDB. Locate ourself with dladdr because the
  // alterantive is HostInfo::GetShlibDir which must not run before plugins
  // register their shared library directory helper.
  Dl_info info;
  if (::dladdr(reinterpret_cast<const void *>(&ForEachPythonRuntimeCandidate),
               &info)) {
    llvm::StringRef lldb_dir = llvm::sys::path::parent_path(info.dli_fname);
    for (llvm::StringRef dir : kLLDBRelativeDirs)
      if (TryFrameworks(callback, lldb_dir + "/" + dir))
        return;
  }

  // Fall back to a Python from Python.org.
  if (TryFrameworks(callback, "/Library/Frameworks"))
    return;

  // Finally, use a bare dlopen to load it from the dyld shared cache.
  callback("libpython3.dylib");
}

} // namespace lldb_private

#endif // LLDB_ENABLE_PYTHON
