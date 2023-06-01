//===-- HostInfoWindowsSwift.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Config.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/common/HostInfoSwift.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ConvertUTF.h>

#include <cstdlib>
#include <string>
#include <vector>

using namespace lldb_private;

bool HostInfoWindows::ComputeSwiftResourceDirectory(FileSpec &lldb_shlib_spec,
                                                    FileSpec &file_spec,
                                                    bool verify) {
  return DefaultComputeSwiftResourceDirectory(lldb_shlib_spec, file_spec,
                                              verify);
}

FileSpec HostInfoWindows::GetSwiftResourceDir() {
  static std::once_flag g_once_flag;
  static FileSpec g_swift_resource_dir;
  std::call_once(g_once_flag, []() {
    FileSpec lldb_file_spec = HostInfoBase::GetShlibDir();
    HostInfoWindows::ComputeSwiftResourceDirectory(lldb_file_spec,
                                                   g_swift_resource_dir, true);
    Log *log = GetLog(LLDBLog::Host);
    LLDB_LOG(log, "swift dir -> '{0}'", g_swift_resource_dir);
  });
  return g_swift_resource_dir;
}

std::string
HostInfoWindows::GetSwiftResourceDir(llvm::Triple triple,
                                     llvm::StringRef platform_sdk_path) {
  return GetSwiftResourceDir().GetPath();
}

llvm::Expected<llvm::StringRef> HostInfoWindows::GetSDKRoot(SDKOptions options) {
  static std::once_flag g_flag;
  static std::string g_sdkroot;

  std::call_once(g_flag, []() {
    if (wchar_t *path = _wgetenv(L"SDKROOT"))
      llvm::convertUTF16ToUTF8String(
          llvm::ArrayRef{reinterpret_cast<llvm::UTF16 *>(path), wcslen(path)},
          g_sdkroot);
  });

  if (!g_sdkroot.empty())
    return g_sdkroot;
  return llvm::make_error<HostInfoError>("`SDKROOT` is unset");
}

std::vector<std::string> HostInfoWindows::GetSwiftLibrarySearchPaths() {
  static std::once_flag g_flag;
  static std::vector<std::string> g_library_paths;

  std::call_once(g_flag, []() {
    llvm::for_each(llvm::split(std::getenv("Path"), ";"),
                   [](llvm::StringRef path) {
      g_library_paths.emplace_back(path);
    });
  });

  return g_library_paths;
}
