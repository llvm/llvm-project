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
#include "lldb/Host/windows/windows.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Windows/WindowsSupport.h"

#include <pathcch.h>
#include <string>

namespace lldb_private {

namespace {

/// Absolute path of \p module, or the running executable when null.
std::string GetModulePath(HMODULE module) {
  std::vector<WCHAR> buffer(MAX_PATH);
  while (buffer.size() <= PATHCCH_MAX_CCH) {
    DWORD len = ::GetModuleFileNameW(module, buffer.data(), buffer.size());
    if (len == 0)
      return "";
    if (len < buffer.size()) {
      std::string utf8;
      if (llvm::convertWideToUTF8(std::wstring(buffer.data(), len), utf8))
        return utf8;
      return "";
    }
    if (::GetLastError() == ERROR_INSUFFICIENT_BUFFER)
      buffer.resize(buffer.size() * 2);
  }
  return "";
}

#ifdef LLDB_PYTHON_DLL_RELATIVE_PATH
std::string ExeRelativeCandidate() {
#ifdef LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME
  std::string exe = GetModulePath(nullptr);
  if (exe.empty())
    return "";
  llvm::SmallString<MAX_PATH> path(exe);
  llvm::sys::path::remove_filename(path);
  llvm::sys::path::append(path, LLDB_PYTHON_DLL_RELATIVE_PATH);
  llvm::sys::path::append(path, LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME);
  llvm::sys::fs::make_absolute(path);
  return std::string(path);
#else
  return "";
#endif
}
#endif

} // namespace

void ForEachPythonRuntimeCandidate(
    llvm::function_ref<bool(const char *)> callback) {
  // Mapping the DLL by base name first lets the plugin's delay-load thunks
  // (which LoadLibrary by base name) resolve against the already-mapped
  // module on first use, bypassing DLL search rules.
#ifdef LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME
  if (callback(LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME))
    return;
#endif

#ifdef LLDB_PYTHON_DLL_RELATIVE_PATH
  std::string exe_relative = ExeRelativeCandidate();
  if (!exe_relative.empty() && callback(exe_relative.c_str()))
    return;
#endif
}

} // namespace lldb_private

#endif // LLDB_ENABLE_PYTHON
