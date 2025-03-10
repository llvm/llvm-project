//===-- source/Host/aix/Host.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Host.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/ProcessInfo.h"
#include "lldb/Utility/Status.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/ScopedPrinter.h"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <unistd.h>

using namespace llvm;
using namespace lldb;
using namespace lldb_private;

namespace {
enum class ProcessState {
  Unknown,
  Dead,
  DiskSleep,
  Idle,
  Paging,
  Parked,
  Running,
  Sleeping,
  TracedOrStopped,
  Zombie,
};
}

namespace lldb_private {
class ProcessLaunchInfo;
}

static bool GetStatusInfo(::pid_t Pid, ProcessInstanceInfo &ProcessInfo,
                          ProcessState &State, ::pid_t &TracerPid,
                          ::pid_t &Tgid) {
  return false;
}

static void GetProcessArgs(::pid_t pid, ProcessInstanceInfo &process_info) {}

static bool GetProcessAndStatInfo(::pid_t pid,
                                  ProcessInstanceInfo &process_info,
                                  ProcessState &State, ::pid_t &tracerpid) {
  return false;
}

uint32_t Host::FindProcessesImpl(const ProcessInstanceInfoMatch &match_info,
                                 ProcessInstanceInfoList &process_infos) {
  static const char procdir[] = "/proc/";

  DIR *dirproc = opendir(procdir);
  if (dirproc) {
    struct dirent *direntry = nullptr;
    const uid_t our_uid = getuid();
    const lldb::pid_t our_pid = getpid();
    bool all_users = match_info.GetMatchAllUsers();

    while ((direntry = readdir(dirproc)) != nullptr) {

      lldb::pid_t pid = atoi(direntry->d_name);

      // Skip this process.
      if (pid == our_pid)
        continue;

      ::pid_t tracerpid;
      ProcessState State;
      ProcessInstanceInfo process_info;

      if (!GetProcessAndStatInfo(pid, process_info, State, tracerpid))
        continue;

      // Skip if process is being debugged.
      if (tracerpid != 0)
        continue;

      if (State == ProcessState::Zombie)
        continue;

      // Check for user match if we're not matching all users and not running
      // as root.
      if (!all_users && (our_uid != 0) && (process_info.GetUserID() != our_uid))
        continue;

      if (match_info.Matches(process_info)) {
        process_infos.push_back(process_info);
      }
    }

    closedir(dirproc);
  }

  return process_infos.size();
}

bool Host::GetProcessInfo(lldb::pid_t pid, ProcessInstanceInfo &process_info) {
  ::pid_t tracerpid;
  ProcessState State;
  return GetProcessAndStatInfo(pid, process_info, State, tracerpid);
}

Environment Host::GetEnvironment() { return Environment(environ); }

Status Host::ShellExpandArguments(ProcessLaunchInfo &launch_info) {
  return Status("unimplemented");
}
