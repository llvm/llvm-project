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
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>

namespace lldb_private {

namespace {

// Apple ships Python3.framework; python.org and Homebrew ship Python.framework.
constexpr llvm::StringLiteral kAppleFrameworkSuffix =
    "Library/Frameworks/Python3.framework/Versions/Current/Python3";
constexpr llvm::StringLiteral kPythonOrgFrameworkSuffix =
    "Library/Frameworks/Python.framework/Versions/Current/Python";

bool TryIfExists(llvm::function_ref<bool(const char *)> callback,
                 llvm::StringRef base, llvm::StringRef relative) {
  llvm::SmallString<256> path(base);
  llvm::sys::path::append(path, relative);
  if (!llvm::sys::fs::exists(path))
    return false;
  return callback(path.c_str());
}

/// Locates the Python3.framework that `xcrun -f python3` points at. Returns
/// an empty string on any failure or unexpected layout.
std::string FindPythonViaXcrun() {
  llvm::ErrorOr<std::string> xcrun = llvm::sys::findProgramByName("xcrun");
  if (!xcrun)
    return {};

  llvm::SmallString<128> stdout_path;
  if (llvm::sys::fs::createTemporaryFile("xcrun-python3", "txt", stdout_path))
    return {};
  auto remove_temp =
      llvm::scope_exit([&] { llvm::sys::fs::remove(stdout_path.str()); });

  std::optional<llvm::StringRef> redirects[3] = {
      llvm::StringRef(""), llvm::StringRef(stdout_path), llvm::StringRef("")};
  llvm::StringRef args[] = {*xcrun, "-f", "python3"};
  int rc =
      llvm::sys::ExecuteAndWait(*xcrun, args, /*env=*/std::nullopt, redirects);
  if (rc != 0)
    return {};

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buf =
      llvm::MemoryBuffer::getFile(stdout_path.str());
  if (!buf)
    return {};

  llvm::StringRef python = (*buf)->getBuffer().rtrim();
  if (python.empty())
    return {};

  // xcrun is not guaranteed to return an Xcode-layout interpreter, so verify
  // the .../usr/bin/python3 structure before deriving a developer dir.
  if (llvm::sys::path::filename(python) != "python3")
    return {};
  llvm::StringRef parent = llvm::sys::path::parent_path(python);
  if (llvm::sys::path::filename(parent) != "bin")
    return {};
  llvm::StringRef grandparent = llvm::sys::path::parent_path(parent);
  if (llvm::sys::path::filename(grandparent) != "usr")
    return {};
  llvm::StringRef developer = llvm::sys::path::parent_path(grandparent);

  llvm::SmallString<256> framework(developer);
  llvm::sys::path::append(framework, kAppleFrameworkSuffix);
  if (llvm::sys::fs::exists(framework))
    return std::string(framework);

  return {};
}

} // namespace

void ForEachPythonRuntimeCandidate(
    llvm::function_ref<bool(const char *)> callback) {
  if (const char *developer_dir = std::getenv("DEVELOPER_DIR");
      developer_dir && *developer_dir)
    if (TryIfExists(callback, developer_dir, kAppleFrameworkSuffix))
      return;

  if (TryIfExists(callback, "/Applications/Xcode.app/Contents/Developer",
                  kAppleFrameworkSuffix))
    return;
  if (TryIfExists(callback, "/Library/Developer/CommandLineTools",
                  kAppleFrameworkSuffix))
    return;
  if (TryIfExists(callback, "/", kPythonOrgFrameworkSuffix))
    return;
  if (TryIfExists(callback, "/opt/homebrew", kPythonOrgFrameworkSuffix))
    return;
  if (TryIfExists(callback, "/usr/local", kPythonOrgFrameworkSuffix))
    return;

  // xcrun is a subprocess; only run it when the well-known paths miss.
  if (std::string xcrun_path = FindPythonViaXcrun(); !xcrun_path.empty())
    if (callback(xcrun_path.c_str()))
      return;

  // Bare-name dlopen is the only way to hit the dyld shared cache.
  callback("libpython3.dylib");
}

} // namespace lldb_private

#endif // LLDB_ENABLE_PYTHON
