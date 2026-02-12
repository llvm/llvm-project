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

#ifdef LLDB_PYTHON_DLL_RELATIVE_PATH
/// Returns the full path to the lldb.exe executable.
static std::string GetPathToExecutable() {
  std::vector<WCHAR> buffer(MAX_PATH);
  while (buffer.size() <= PATHCCH_MAX_CCH) {
    DWORD len = GetModuleFileNameW(NULL, buffer.data(), buffer.size());
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
bool IsPythonDLLInPath() {
#define WIDEN2(x) L##x
#define WIDEN(x) WIDEN2(x)
  HMODULE h = LoadLibraryW(WIDEN(LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME));
  if (!h)
    return false;
  FreeLibrary(h);
  return true;
#undef WIDEN2
#undef WIDEN
}
#endif

llvm::Error SetupPythonRuntimeLibrary() {
#ifdef LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME
  if (IsPythonDLLInPath())
    return Error::success();
#ifdef LLDB_PYTHON_DLL_RELATIVE_PATH
  if (AddPythonDLLToSearchPath() && IsPythonDLLInPath())
    return Error::success();
#endif
  return createStringError(
      inconvertibleErrorCode(),
      "unable to find '" LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME "'");
#elif defined(LLDB_PYTHON_DLL_RELATIVE_PATH)
  if (!AddPythonDLLToSearchPath())
    return createStringError(inconvertibleErrorCode(),
                             "unable to find the Python runtime library");
#endif
  return Error::success();
}
