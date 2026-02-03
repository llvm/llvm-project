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
static llvm::Expected<std::wstring> GetPathToExecutableW() {
  std::vector<WCHAR> buffer(MAX_PATH);
  while (buffer.size() <= PATHCCH_MAX_CCH) {
    DWORD len = GetModuleFileNameW(NULL, buffer.data(), buffer.size());
    if (len == 0)
      return L"";
    if (len < buffer.size())
      return std::wstring(buffer.data(), len);
    if (::GetLastError() == ERROR_INSUFFICIENT_BUFFER)
      buffer.resize(buffer.size() * 2);
  }
  return L"";
}

bool AddPythonDLLToSearchPath() {
  std::wstring modulePath = GetPathToExecutableW();
  if (modulePath.empty())
    return false;

  SmallVector<char, MAX_PATH> utf8Path;
  if (sys::windows::UTF16ToUTF8(modulePath.c_str(), modulePath.length(),
                                utf8Path))
    return false;
  sys::path::remove_filename(utf8Path);
  sys::path::append(utf8Path, LLDB_PYTHON_DLL_RELATIVE_PATH);
  sys::fs::make_absolute(utf8Path);

  SmallVector<wchar_t, 1> widePath;
  if (sys::windows::widenPath(utf8Path.data(), widePath))
    return false;

  if (sys::fs::exists(utf8Path))
    return SetDllDirectoryW(widePath.data());
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
