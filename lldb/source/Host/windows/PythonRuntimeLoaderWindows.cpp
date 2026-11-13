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

#include <algorithm>
#include <cwchar>
#include <iterator>
#include <pathcch.h>
#include <string>
#include <tuple>
#include <vector>

namespace lldb_private {

namespace {

#ifdef LLDB_PYTHON_DLL_RELATIVE_PATH
#ifdef LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME
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
#endif // ifdef LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME

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
#endif // ifdef LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME
}
#endif // ifdef LLDB_PYTHON_DLL_RELATIVE_PATH

#ifdef LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME
/// PEP 514 tag suffix for installs matching this process' architecture.
const wchar_t *RegistryTagSuffix() {
#if defined(_M_ARM64) || defined(__aarch64__)
  return L"-arm64";
#elif defined(_M_X64) || defined(__x86_64__)
  return L"";
#elif defined(_M_IX86) || defined(__i386__)
  return L"-32";
#else
  return nullptr;
#endif
}

struct RegisteredInstall {
  unsigned long major;
  unsigned long minor;
  std::wstring install_path;
};

void CollectRegisteredInstalls(HKEY root, REGSAM view, const wchar_t *suffix,
                               std::vector<RegisteredInstall> &installs) {
  HKEY core;
  if (::RegOpenKeyExW(root, L"Software\\Python\\PythonCore", 0, KEY_READ | view,
                      &core) != ERROR_SUCCESS)
    return;

  wchar_t tag[256];
  for (DWORD i = 0;; ++i) {
    DWORD tag_len = std::size(tag);
    LONG status = ::RegEnumKeyExW(core, i, tag, &tag_len, nullptr, nullptr,
                                  nullptr, nullptr);
    if (status == ERROR_NO_MORE_ITEMS)
      break;
    if (status != ERROR_SUCCESS)
      continue;

    wchar_t *end;
    unsigned long major = wcstoul(tag, &end, 10);
    if (end == tag || *end != L'.')
      continue;
    wchar_t *minor_start = end + 1;
    unsigned long minor = wcstoul(minor_start, &end, 10);
    if (end == minor_start || wcscmp(end, suffix) != 0)
      continue;

    std::wstring tag_key = std::wstring(tag) + L"\\InstallPath";
    DWORD size = 0;
    if (::RegGetValueW(core, tag_key.c_str(), nullptr, RRF_RT_REG_SZ, nullptr,
                       nullptr, &size) != ERROR_SUCCESS ||
        size == 0)
      continue;
    std::wstring path(size / sizeof(wchar_t), L'\0');
    if (::RegGetValueW(core, tag_key.c_str(), nullptr, RRF_RT_REG_SZ, nullptr,
                       path.data(), &size) != ERROR_SUCCESS)
      continue;
    path.resize(wcslen(path.c_str()));
    if (!path.empty())
      installs.push_back({major, minor, std::move(path)});
  }
  ::RegCloseKey(core);
}

/// Runtime DLLs of the Python installs registered per PEP 514, which is
/// where the py launcher finds interpreters that are not on PATH. Newest
/// version first.
std::vector<std::string> RegisteredInstallCandidates() {
  std::vector<std::string> candidates;
  const wchar_t *suffix = RegistryTagSuffix();
  if (!suffix)
    return candidates;

  std::vector<RegisteredInstall> installs;
  CollectRegisteredInstalls(HKEY_CURRENT_USER, 0, suffix, installs);
  CollectRegisteredInstalls(HKEY_LOCAL_MACHINE, KEY_WOW64_64KEY, suffix,
                            installs);
  CollectRegisteredInstalls(HKEY_LOCAL_MACHINE, KEY_WOW64_32KEY, suffix,
                            installs);
  std::stable_sort(installs.begin(), installs.end(),
                   [](const RegisteredInstall &a, const RegisteredInstall &b) {
                     return std::tie(a.major, a.minor) >
                            std::tie(b.major, b.minor);
                   });

  for (const RegisteredInstall &install : installs) {
    std::string dir;
    if (!llvm::convertWideToUTF8(install.install_path, dir))
      continue;
    llvm::SmallString<MAX_PATH> path(dir);
    llvm::sys::path::append(path, LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME);
    if (!llvm::sys::fs::exists(path))
      continue;
    if (std::find(candidates.begin(), candidates.end(), path.str()) ==
        candidates.end())
      candidates.emplace_back(path.str());
  }
  return candidates;
}
#endif // ifdef LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME

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

#ifdef LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME
  for (const std::string &candidate : RegisteredInstallCandidates())
    if (callback(candidate.c_str()))
      return;
#endif
}

} // namespace lldb_private

#endif // LLDB_ENABLE_PYTHON
