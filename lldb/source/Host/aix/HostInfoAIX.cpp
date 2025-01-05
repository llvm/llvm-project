//===-- HostInfoAIX.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/aix/HostInfoAIX.h"
#include "lldb/Host/Config.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/Support/Threading.h"
#include <climits>
#include <cstdio>
#include <cstring>
#include <sys/utsname.h>
#include <unistd.h>
#include <algorithm>
#include <mutex>

using namespace lldb_private;

void HostInfoAIX::Initialize(SharedLibraryDirectoryHelper *helper) {
  HostInfoPosix::Initialize(helper);
}

void HostInfoAIX::Terminate() {
  HostInfoBase::Terminate();
}

FileSpec HostInfoAIX::GetProgramFileSpec() {
  static FileSpec g_program_filespec;

  if (!g_program_filespec) {
    char exe_path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len > 0) {
      exe_path[len] = 0;
      g_program_filespec.SetFile(exe_path, FileSpec::Style::native);
    }
  }

  return g_program_filespec;
}
