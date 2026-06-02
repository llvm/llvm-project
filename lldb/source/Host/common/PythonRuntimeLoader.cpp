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

/// Captured once by GetState; observers must treat it as read-only.
struct PythonRuntimeState {
  std::string path;
  std::string error_message;
};

llvm::Error TryLoad(const char *path) {
  std::string err_msg;
  llvm::sys::DynamicLibrary lib =
      llvm::sys::DynamicLibrary::getPermanentLibrary(path, &err_msg);
  if (!lib.isValid())
    return llvm::createStringErrorV("could not load '{0}': {1}", path, err_msg);

  // A successful load does not imply the library is actually libpython,
  // so reject anything that doesn't expose a stable-ABI symbol.
  if (!lib.getAddressOfSymbol("Py_IsInitialized"))
    return llvm::createStringErrorV(
        "'{0}' does not export Py_IsInitialized; not a Python runtime", path);

  return llvm::Error::success();
}

llvm::Error LoadPythonRuntimeImpl(std::string &out_path) {
  if (PythonRuntimeLoader::IsLoaded())
    return llvm::Error::success();

  llvm::Error failures =
      llvm::createStringError("could not locate the Python runtime library");

  auto try_path = [&](const char *path) -> bool {
    if (llvm::Error err = TryLoad(path)) {
      failures = llvm::joinErrors(std::move(failures), std::move(err));
      return false;
    }
    out_path = path;
    return true;
  };

  if (const char *override_path = std::getenv("LLDB_PYTHON_LIBRARY");
      override_path && *override_path) {
    if (try_path(override_path)) {
      consumeError(std::move(failures));
      return llvm::Error::success();
    }
  }

#ifdef LLDB_PYTHON_RUNTIME_LIBRARY_BUILD_PATH
  // The build-time Python is the closest stable-ABI match for what the
  // plugin was compiled against, so prefer it over the platform fallbacks.
  if (try_path(LLDB_PYTHON_RUNTIME_LIBRARY_BUILD_PATH)) {
    consumeError(std::move(failures));
    return llvm::Error::success();
  }
#endif

  bool found = false;
  ForEachPythonRuntimeCandidate(
      [&](const char *candidate) { return found = try_path(candidate); });

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

bool PythonRuntimeLoader::IsLoaded() {
  // A single-symbol probe can match an incompatible runtime that happens
  // to export it, so pair an old symbol with one introduced in 3.8 to
  // bound the version on both ends.
#if defined(_WIN32)
  HMODULE main = ::GetModuleHandleW(nullptr);
  return ::GetProcAddress(main, "Py_IsInitialized") != nullptr &&
         ::GetProcAddress(main, "Py_InitializeFromConfig") != nullptr;
#else
  return ::dlsym(RTLD_DEFAULT, "Py_IsInitialized") != nullptr &&
         ::dlsym(RTLD_DEFAULT, "Py_InitializeFromConfig") != nullptr;
#endif
}

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
bool PythonRuntimeLoader::IsLoaded() { return false; }
} // namespace lldb_private

#endif // LLDB_ENABLE_PYTHON
