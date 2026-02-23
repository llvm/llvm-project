//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/windows/PythonPathSetup/PythonPathSetup.h"

#include "lldb/Host/windows/windows.h"
#include "llvm/Support/Windows/WindowsSupport.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <pathcch.h>

using namespace llvm;

static std::string GetModulePath(HMODULE module) {
  std::vector<WCHAR> buffer(MAX_PATH);
  while (buffer.size() <= PATHCCH_MAX_CCH) {
    DWORD len = GetModuleFileNameW(module, buffer.data(), buffer.size());
    if (len == 0)
      return "";
    if (len < buffer.size()) {
      std::string buffer_utf8;
      if (convertWideToUTF8(std::wstring(buffer.data(), len), buffer_utf8))
        return buffer_utf8;
      return "";
    }
    if (::GetLastError() == ERROR_INSUFFICIENT_BUFFER)
      buffer.resize(buffer.size() * 2);
  }
  return "";
}

/// Returns the full path to the lldb.exe executable.
static std::string GetPathToExecutable() { return GetModulePath(NULL); }

#ifdef LLDB_PYTHON_DLL_RELATIVE_PATH
bool AddPythonDLLToSearchPath() {
  std::string path_str = GetPathToExecutable();
  if (path_str.empty())
    return false;

  SmallVector<char, MAX_PATH> path(path_str.begin(), path_str.end());
  sys::path::remove_filename(path);
  sys::path::append(path, LLDB_PYTHON_DLL_RELATIVE_PATH);
  sys::fs::make_absolute(path);

  SmallVector<wchar_t, 1> path_wide;
  if (sys::windows::widenPath(path.data(), path_wide))
    return false;

  if (sys::fs::exists(path))
    return SetDllDirectoryW(path_wide.data());
  return false;
}
#endif

#ifdef LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME
std::optional<std::string> GetPythonDLLPath() {
#define WIDEN2(x) L##x
#define WIDEN(x) WIDEN2(x)
  HMODULE h = LoadLibraryW(WIDEN(LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME));
  if (!h)
    return std::nullopt;

  std::string path = GetModulePath(h);
  FreeLibrary(h);

  return path;
#undef WIDEN2
#undef WIDEN
}
#endif

llvm::Expected<std::string> SetupPythonRuntimeLibrary() {
#ifdef LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME
  if (std::optional<std::string> python_path = GetPythonDLLPath())
    return *python_path;
#ifdef LLDB_PYTHON_DLL_RELATIVE_PATH
  if (AddPythonDLLToSearchPath()) {
    if (std::optional<std::string> python_path = GetPythonDLLPath())
      return *python_path;
  }
#endif
  return createStringError(
      inconvertibleErrorCode(),
      "unable to find '" LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME "'");
#elif defined(LLDB_PYTHON_DLL_RELATIVE_PATH)
  if (!AddPythonDLLToSearchPath())
    return createStringError(inconvertibleErrorCode(),
                             "unable to find the Python runtime library");
#endif
  return "";
}
