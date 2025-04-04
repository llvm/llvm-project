//===-- source/Host/aix/Host.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <fcntl.h>
#include <sstream>
#include <sys/procfs.h>

#include "lldb/Host/Host.h"
#include "lldb/Host/linux/Support.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/ProcessInfo.h"
#include "lldb/Utility/Status.h"
#include "llvm/BinaryFormat/XCOFF.h"

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

static bool GetStatusInfo(::pid_t Pid, ProcessInstanceInfo &ProcessInfo,
                          ProcessState &State, ::pid_t &TracerPid,
                          ::pid_t &Tgid) {
  Log *log = GetLog(LLDBLog::Host);

  auto BufferOrError = getProcFile(Pid, "status");
  if (!BufferOrError)
    return false;

  llvm::StringRef Rest = BufferOrError.get()->getBuffer();
  while (!Rest.empty()) {
    llvm::StringRef Line;
    std::tie(Line, Rest) = Rest.split('\n');

    if (Line.consume_front("Gid:")) {
      // Real, effective, saved set, and file system GIDs. Read the first two.
      Line = Line.ltrim();
      uint32_t RGid, EGid;
      Line.consumeInteger(10, RGid);
      Line = Line.ltrim();
      Line.consumeInteger(10, EGid);

      ProcessInfo.SetGroupID(RGid);
      ProcessInfo.SetEffectiveGroupID(EGid);
    } else if (Line.consume_front("Uid:")) {
      // Real, effective, saved set, and file system UIDs. Read the first two.
      Line = Line.ltrim();
      uint32_t RUid, EUid;
      Line.consumeInteger(10, RUid);
      Line = Line.ltrim();
      Line.consumeInteger(10, EUid);

      ProcessInfo.SetUserID(RUid);
      ProcessInfo.SetEffectiveUserID(EUid);
    } else if (Line.consume_front("PPid:")) {
      ::pid_t PPid;
      Line.ltrim().consumeInteger(10, PPid);
      ProcessInfo.SetParentProcessID(PPid);
    } else if (Line.consume_front("State:")) {
      State = llvm::StringSwitch<ProcessState>(Line.ltrim().take_front(1))
                  .Case("D", ProcessState::DiskSleep)
                  .Case("I", ProcessState::Idle)
                  .Case("R", ProcessState::Running)
                  .Case("S", ProcessState::Sleeping)
                  .CaseLower("T", ProcessState::TracedOrStopped)
                  .Case("W", ProcessState::Paging)
                  .Case("P", ProcessState::Parked)
                  .Case("X", ProcessState::Dead)
                  .Case("Z", ProcessState::Zombie)
                  .Default(ProcessState::Unknown);
      if (State == ProcessState::Unknown) {
        LLDB_LOG(log, "Unknown process state {0}", Line);
      }
    } else if (Line.consume_front("TracerPid:")) {
      Line = Line.ltrim();
      Line.consumeInteger(10, TracerPid);
    } else if (Line.consume_front("Tgid:")) {
      Line = Line.ltrim();
      Line.consumeInteger(10, Tgid);
    }
  }
  return true;
}

static void GetExePathAndArch(::pid_t pid, ProcessInstanceInfo &process_info) {
  Log *log = GetLog(LLDBLog::Process);
  std::string ExePath(PATH_MAX, '\0');
  struct psinfo psinfoData;

  // We can't use getProcFile here because proc/[pid]/exe is a symbolic link.
  llvm::SmallString<64> ProcExe;
  (llvm::Twine("/proc/") + llvm::Twine(pid) + "/cwd").toVector(ProcExe);

  ssize_t len = readlink(ProcExe.c_str(), &ExePath[0], PATH_MAX);
  if (len > 0) {
    ExePath.resize(len);

    struct stat statData;

    std::ostringstream oss;

    oss << "/proc/" << std::dec << pid << "/psinfo";
    assert(stat(oss.str().c_str(), &statData) == 0);

    const int fd = open(oss.str().c_str(), O_RDONLY);
    assert(fd >= 0);

    ssize_t readNum = read(fd, &psinfoData, sizeof(psinfoData));
    assert(readNum >= 0);

    close(fd);
  } else {
    LLDB_LOG(log, "failed to read link exe link for {0}: {1}", pid,
             Status(errno, eErrorTypePOSIX));
    ExePath.resize(0);
  }

  llvm::StringRef PathRef(&(psinfoData.pr_psargs[0]));

  if (!PathRef.empty()) {
    process_info.GetExecutableFile().SetFile(PathRef, FileSpec::Style::native);
    ArchSpec arch_spec = ArchSpec();
    arch_spec.SetArchitecture(eArchTypeXCOFF, XCOFF::TCPU_PPC64,
                              LLDB_INVALID_CPUTYPE, llvm::Triple::AIX);
    process_info.SetArchitecture(arch_spec);
  }
}

static bool GetProcessAndStatInfo(::pid_t pid,
                                  ProcessInstanceInfo &process_info,
                                  ProcessState &State, ::pid_t &tracerpid) {
  ::pid_t tgid;
  tracerpid = 0;
  process_info.Clear();

  process_info.SetProcessID(pid);

  GetExePathAndArch(pid, process_info);

  // Get User and Group IDs and get tracer pid.
  if (!GetStatusInfo(pid, process_info, State, tracerpid, tgid))
    return false;

  return true;
}

uint32_t Host::FindProcessesImpl(const ProcessInstanceInfoMatch &match_info,
                                 ProcessInstanceInfoList &process_infos) {
  return 0;
}

bool Host::GetProcessInfo(lldb::pid_t pid, ProcessInstanceInfo &process_info) {
  ::pid_t tracerpid;
  ProcessState State;
  return GetProcessAndStatInfo(pid, process_info, State, tracerpid);
}

Status Host::ShellExpandArguments(ProcessLaunchInfo &launch_info) {
  return Status("unimplemented");
}
