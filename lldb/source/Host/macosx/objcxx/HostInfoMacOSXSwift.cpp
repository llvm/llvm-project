//===-- HostInfoMacOSSwift.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "lldb/Host/Config.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/posix/HostInfoPosix.h"
#include "lldb/Host/common/HostInfoSwift.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

#include <string>

using namespace lldb_private;

bool HostInfoMacOSX::ComputeSwiftResourceDirectory(FileSpec &lldb_shlib_spec,
                                                   FileSpec &file_spec,
                                                   bool verify) {
  if (!lldb_shlib_spec)
    return false;

  std::string raw_path = lldb_shlib_spec.GetPath();
  size_t framework_pos = raw_path.find("LLDB.framework");
  if (framework_pos == std::string::npos)
    return HostInfoPosix::ComputeSwiftResourceDirectory(lldb_shlib_spec,
                                                           file_spec, verify);

  framework_pos += strlen("LLDB.framework");
  raw_path.resize(framework_pos);
  raw_path.append("/Resources/Swift");
  if (!verify || VerifySwiftPath(raw_path)) {
    file_spec.GetDirectory().SetString(raw_path);
    FileSystem::Instance().Resolve(file_spec);
    return true;
  }
  return true;
}

FileSpec HostInfoMacOSX::GetSwiftResourceDir() {
  static std::once_flag g_once_flag;
  static FileSpec g_swift_resource_dir;
  std::call_once(g_once_flag, []() {
    FileSpec lldb_file_spec = HostInfo::GetShlibDir();
    ComputeSwiftResourceDirectory(lldb_file_spec, g_swift_resource_dir, true);
    Log *log = GetLog(LLDBLog::Host);
    LLDB_LOG(log, "swift dir -> '{0}'", g_swift_resource_dir);
  });
  return g_swift_resource_dir;
}
