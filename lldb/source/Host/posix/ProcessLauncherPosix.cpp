//===-- ProcessLauncherPosix.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/posix/ProcessLauncherPosix.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostProcess.h"

#include "lldb/Target/ProcessLaunchInfo.h"

#include <limits.h>

using namespace lldb;
using namespace lldb_private;

HostProcess
ProcessLauncherPosix::LaunchProcess(const ProcessLaunchInfo &launch_info,
                                    Error &error) {
  lldb::pid_t pid;
  char exe_path[PATH_MAX];

  launch_info.GetExecutableFile().GetPath(exe_path, sizeof(exe_path));

  // TODO(zturner): Move the code from LaunchProcessPosixSpawn to here, and make
  // MacOSX re-use this
  // ProcessLauncher when it wants a posix_spawn launch.
  error = Host::LaunchProcessPosixSpawn(exe_path, launch_info, pid);
  return HostProcess(pid);
}
