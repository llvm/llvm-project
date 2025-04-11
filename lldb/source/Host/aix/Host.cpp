//===-- source/Host/aix/Host.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Host.h"
#include "lldb/Utility/Status.h"

using namespace lldb_private;

uint32_t Host::FindProcessesImpl(const ProcessInstanceInfoMatch &match_info,
                                 ProcessInstanceInfoList &process_infos) {
  return 0;
}

bool Host::GetProcessInfo(lldb::pid_t pid, ProcessInstanceInfo &process_info) {
  return false;
}

Status Host::ShellExpandArguments(ProcessLaunchInfo &launch_info) {
  return Status("unimplemented");
}
