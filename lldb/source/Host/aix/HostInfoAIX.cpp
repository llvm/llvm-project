//===-- HostInfoAIX.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/aix/HostInfoAIX.h"

using namespace lldb_private;

void HostInfoAIX::Initialize(SharedLibraryDirectoryHelper *helper) {
  HostInfoPosix::Initialize(helper);
}

void HostInfoAIX::Terminate() { HostInfoBase::Terminate(); }

FileSpec HostInfoAIX::GetProgramFileSpec() {
  static FileSpec g_program_filespec;
  return g_program_filespec;
}
