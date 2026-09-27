//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Config.h"
#include "lldb/Host/ScriptInterpreterRuntimeLoader.h"

#include "PythonRuntimeLoaderInternal.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/ErrorExtras.h"
#include "llvm/Support/Path.h"
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

/// True if libpython is currently mapped into the process. A single-symbol
/// probe can match an incompatible runtime that happens to export it, so
/// pair an old symbol with one introduced in 3.8 to bound the version on
/// both ends.
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

llvm::Expected<std::string> LoadPythonRuntime() {
  if (IsPythonAlreadyLoaded())
    return std::string();

  llvm::Error failures =
      llvm::createStringError("could not locate the Python runtime library");
  std::string loaded;

  auto try_path = [&](const char *path) -> bool {
    if (llvm::Error err = TryLoad(path)) {
      failures = llvm::joinErrors(std::move(failures), std::move(err));
      return false;
    }
    loaded = path;
    return true;
  };

  if (const char *override_path = std::getenv("LLDB_PYTHON_LIBRARY");
      override_path && *override_path) {
    if (try_path(override_path)) {
      consumeError(std::move(failures));
      return loaded;
    }
  }

#ifdef LLDB_PYTHON_RUNTIME_LIBRARY_BUILD_PATH
  // The build-time Python is the closest stable-ABI match for what the
  // plugin was compiled against, so prefer it over the platform fallbacks.
  if (try_path(LLDB_PYTHON_RUNTIME_LIBRARY_BUILD_PATH)) {
    consumeError(std::move(failures));
    return loaded;
  }
#endif

  bool found = false;
  ForEachPythonRuntimeCandidate(
      [&](const char *candidate) { return found = try_path(candidate); });

  if (found) {
    consumeError(std::move(failures));
    return loaded;
  }
  return std::move(failures);
}

class PythonRuntimeLoader : public ScriptInterpreterRuntimeLoader {
public:
  llvm::Error Load() override {
    EnsureLoaded();
    if (m_error_message.empty())
      return llvm::Error::success();
    return llvm::createStringError(m_error_message);
  }

  llvm::Expected<llvm::StringRef> GetLoadedPath() override {
    EnsureLoaded();
    if (!m_error_message.empty())
      return llvm::createStringError(m_error_message);
    return llvm::StringRef(m_path);
  }

  bool IsLoaded() override { return IsPythonAlreadyLoaded(); }

private:
  void EnsureLoaded() {
    llvm::call_once(m_once, [this] {
      if (llvm::Expected<std::string> result = LoadPythonRuntime()) {
        m_path = std::move(*result);
        if (m_path.empty())
          return;
#if defined(_WIN32)
        // liblldb.dll may link against (lib)python3.dll (stable ABI). Ensure
        // it is loaded from the same directory as the version-specific runtime
        // so the delay-load resolver can find it.
        llvm::SmallString<256> stable_abi_path(m_path);
        llvm::sys::path::remove_filename(stable_abi_path);
#if defined(__MINGW32__)
        llvm::sys::path::append(stable_abi_path, "libpython3.dll");
#else
        llvm::sys::path::append(stable_abi_path, "python3.dll");
#endif
        std::string err;
        llvm::sys::DynamicLibrary::getPermanentLibrary(stable_abi_path.c_str(),
                                                       &err);
#endif
      } else {
        m_error_message = llvm::toString(result.takeError());
        LLDB_LOG(GetLog(LLDBLog::Host), "Python runtime load failed: {0}",
                 m_error_message);
      }
    });
  }

  llvm::once_flag m_once;
  std::string m_path;
  std::string m_error_message;
};

} // namespace

llvm::Expected<ScriptInterpreterRuntimeLoader &>
ScriptInterpreterRuntimeLoader::Get(lldb::ScriptLanguage language) {
  switch (language) {
  case lldb::eScriptLanguagePython: {
    static PythonRuntimeLoader instance;
    return instance;
  }
  case lldb::eScriptLanguageLua:
  case lldb::eScriptLanguageNone:
  case lldb::eScriptLanguageUnknown:
    return llvm::createStringError(
        "no runtime loader for the requested script language");
  }
  llvm_unreachable("unhandled ScriptLanguage");
}

} // namespace lldb_private
