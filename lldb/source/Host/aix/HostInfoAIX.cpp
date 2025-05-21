//===-- HostInfoAIX.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/aix/HostInfoAIX.h"
#include "lldb/Host/posix/Support.h"
#include <sys/procfs.h>

using namespace lldb_private;

void HostInfoAIX::Initialize(SharedLibraryDirectoryHelper *helper) {
  HostInfoPosix::Initialize(helper);
}

void HostInfoAIX::Terminate() { HostInfoBase::Terminate(); }

FileSpec HostInfoAIX::GetProgramFileSpec() {
  static FileSpec g_program_filespec;
  struct psinfo psinfoData;
  auto BufferOrError = getProcFile(getpid(), "psinfo");
  if (BufferOrError) {
    std::unique_ptr<llvm::MemoryBuffer> PsinfoBuffer =
        std::move(*BufferOrError);
    memcpy(&psinfoData, PsinfoBuffer->getBufferStart(), sizeof(psinfoData));
    llvm::StringRef exe_path(
        psinfoData.pr_psargs,
        strnlen(psinfoData.pr_psargs, sizeof(psinfoData.pr_psargs)));
    if (!exe_path.empty())
      g_program_filespec.SetFile(exe_path, FileSpec::Style::native);
  }
  return g_program_filespec;
}
