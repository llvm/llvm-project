//===-- Host.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Host.h"
#include "lldb/Utility/Status.h"

using namespace lldb_private;

uint32_t Host::FindProcessesImpl(const ProcessInstanceInfoMatch &,
                                 ProcessInstanceInfoList &) {
  return 0;
}

bool Host::GetProcessInfo(lldb::pid_t, ProcessInstanceInfo &) { return false; }

Status Host::LaunchProcess(ProcessLaunchInfo &) {
  return Status::FromErrorString(
      "launching a host process is not supported under Emscripten");
}

Status Host::ShellExpandArguments(ProcessLaunchInfo &) {
  return Status::FromErrorString(
      "shell expansion is not supported under Emscripten");
}
