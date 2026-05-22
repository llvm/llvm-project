//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Config.h"

#if LLDB_ENABLE_PYTHON

#include "lldb/Host/PythonRuntimeLoader.h"

#include "PythonRuntimeLoaderInternal.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/ErrorExtras.h"
#include "llvm/Support/Threading.h"

#include <cstdlib>
#include <string>

#if defined(_WIN32)
#include "lldb/Host/windows/windows.h"
#else
#include <dlfcn.h>
#endif

namespace lldb_private {

namespace {

/// Process-wide state of the Python runtime load attempt. Populated once via
/// GetState and observed read-only thereafter.
struct PythonRuntimeState {
  /// Path of the Python runtime that was loaded, empty if Python was already
  /// in the process or no load was attempted.
  std::string path;
  /// Aggregated error message, empty if no attempt has been made or the load
  /// succeeded.
  std::string error_message;
};

/// True if libpython is already in the current process. This happens when
/// LLDB is imported into a Python interpreter, in which case we must not
/// dlopen another libpython on top of it.
///
/// Probe two stable-ABI symbols. Py_IsInitialized has existed forever
/// (including incompatible Python 2), but Py_InitializeFromConfig is 3.8+.
/// Requiring both rejects a stale Python 2 libpython, a stub, or another
/// tool's vendored runtime that happens to export Py_IsInitialized.
bool IsPythonAlreadyLoaded() {
#if defined(_WIN32)
  HMODULE main = ::GetModuleHandleW(nullptr);
  return ::GetProcAddress(main, "Py_IsInitialized") != nullptr &&
         ::GetProcAddress(main, "Py_InitializeFromConfig") != nullptr;
#else
  return ::dlsym(RTLD_DEFAULT, "Py_IsInitialized") != nullptr &&
         ::dlsym(RTLD_DEFAULT, "Py_InitializeFromConfig") != nullptr;
#endif
}

/// Returns success only if the load succeeded and the resulting library
/// exposes Py_IsInitialized (a stable-ABI symbol).
llvm::Error TryLoad(llvm::StringRef path) {
  std::string err_msg;
  llvm::sys::DynamicLibrary lib =
      llvm::sys::DynamicLibrary::getPermanentLibrary(path.str().c_str(),
                                                     &err_msg);
  if (!lib.isValid())
    return llvm::createStringErrorV("could not load '{0}': {1}", path, err_msg);

  if (!lib.getAddressOfSymbol("Py_IsInitialized"))
    return llvm::createStringErrorV(
        "'{0}' does not export Py_IsInitialized; not a Python runtime", path);

  return llvm::Error::success();
}

llvm::Error LoadPythonRuntimeImpl(std::string &out_path) {
  if (IsPythonAlreadyLoaded())
    return llvm::Error::success();

  llvm::Error failures =
      llvm::createStringError("could not locate the Python runtime library");

  auto try_path = [&](llvm::StringRef path) -> bool {
    if (llvm::Error err = TryLoad(path)) {
      failures = llvm::joinErrors(std::move(failures), std::move(err));
      return false;
    }
    out_path = std::string(path);
    return true;
  };

  // Honor an explicit override via LLDB_PYTHON_LIBRARY. Empty values are
  // ignored so an exported-but-unset variable doesn't trigger dlopen("").
  if (const char *override_path = std::getenv("LLDB_PYTHON_LIBRARY");
      override_path && *override_path) {
    if (try_path(override_path)) {
      consumeError(std::move(failures));
      return llvm::Error::success();
    }
  }

#ifdef LLDB_PYTHON_RUNTIME_LIBRARY_BUILD_PATH
  // Try the Python lldb was linked against at build time. The stable-ABI
  // guarantees most closely match what the plugin was built for, so prefer
  // it over the platform search list when an env override hasn't been set
  // (or has been set but didn't load).
  if (try_path(LLDB_PYTHON_RUNTIME_LIBRARY_BUILD_PATH)) {
    consumeError(std::move(failures));
    return llvm::Error::success();
  }
#endif

  // Walk platform-specific well-known locations. The first candidate that
  // loads cleanly wins; the rest of the list is not enumerated.
  bool found = false;
  ForEachPythonRuntimeCandidate(
      [&](llvm::StringRef candidate) { return found = try_path(candidate); });

  if (found) {
    consumeError(std::move(failures));
    return llvm::Error::success();
  }
  return failures;
}

const PythonRuntimeState &GetState() {
  static PythonRuntimeState g_state;
  static llvm::once_flag g_once;
  llvm::call_once(g_once, [] {
    if (llvm::Error err = LoadPythonRuntimeImpl(g_state.path)) {
      g_state.error_message = llvm::toString(std::move(err));
      LLDB_LOG(GetLog(LLDBLog::Host), "Python runtime load failed: {0}",
               g_state.error_message);
    }
  });
  return g_state;
}

} // namespace

llvm::Error PythonRuntimeLoader::Load() {
  const PythonRuntimeState &state = GetState();
  if (state.error_message.empty())
    return llvm::Error::success();
  return llvm::createStringError(state.error_message);
}

llvm::StringRef PythonRuntimeLoader::GetLoadedPath() { return GetState().path; }

} // namespace lldb_private

#else // !LLDB_ENABLE_PYTHON

namespace lldb_private {
llvm::Error PythonRuntimeLoader::Load() { return llvm::Error::success(); }
llvm::StringRef PythonRuntimeLoader::GetLoadedPath() { return {}; }
} // namespace lldb_private

#endif // LLDB_ENABLE_PYTHON
