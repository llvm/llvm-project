//===-- source/Host/aix/Host.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Host.h"
#include "lldb/Host/posix/Support.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/ProcessInfo.h"
#include "lldb/Utility/Status.h"
#include "llvm/BinaryFormat/XCOFF.h"
#include <sys/proc.h>
#include <sys/procfs.h>

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

static ProcessInstanceInfo::timespec convert(pr_timestruc64_t t) {
  ProcessInstanceInfo::timespec ts;
  ts.tv_sec = t.tv_sec;
  ts.tv_usec = t.tv_nsec / 1000; // nanos to micros
  return ts;
}

static bool GetStatusInfo(::pid_t pid, ProcessInstanceInfo &processInfo,
                          ProcessState &State) {
  struct pstatus pstatusData;
  auto BufferOrError = getProcFile(pid, "status");
  if (!BufferOrError)
    return false;

  std::unique_ptr<llvm::MemoryBuffer> StatusBuffer = std::move(*BufferOrError);
  // Ensure there's enough data for psinfoData
  if (StatusBuffer->getBufferSize() < sizeof(pstatusData))
    return false;

  std::memcpy(&pstatusData, StatusBuffer->getBufferStart(),
              sizeof(pstatusData));
  switch (pstatusData.pr_stat) {
  case SIDL:
    State = ProcessState::Idle;
    break;
  case SACTIVE:
    State = ProcessState::Running;
    break;
  case SSTOP:
    State = ProcessState::TracedOrStopped;
    break;
  case SZOMB:
    State = ProcessState::Zombie;
    break;
  default:
    State = ProcessState::Unknown;
    break;
  }
  processInfo.SetIsZombie(State == ProcessState::Zombie);
  processInfo.SetUserTime(convert(pstatusData.pr_utime));
  processInfo.SetSystemTime(convert(pstatusData.pr_stime));
  processInfo.SetCumulativeUserTime(convert(pstatusData.pr_cutime));
  processInfo.SetCumulativeSystemTime(convert(pstatusData.pr_cstime));
  return true;
}

static bool GetExePathAndIds(::pid_t pid, ProcessInstanceInfo &process_info) {
  struct psinfo psinfoData;
  auto BufferOrError = getProcFile(pid, "psinfo");
  if (!BufferOrError)
    return false;

  std::unique_ptr<llvm::MemoryBuffer> PsinfoBuffer = std::move(*BufferOrError);
  // Ensure there's enough data for psinfoData
  if (PsinfoBuffer->getBufferSize() < sizeof(psinfoData))
    return false;

  std::memcpy(&psinfoData, PsinfoBuffer->getBufferStart(), sizeof(psinfoData));
  llvm::StringRef PathRef(
      psinfoData.pr_psargs,
      strnlen(psinfoData.pr_psargs, sizeof(psinfoData.pr_psargs)));
  if (PathRef.empty())
    return false;

  process_info.GetExecutableFile().SetFile(PathRef, FileSpec::Style::native);
  ArchSpec arch_spec = ArchSpec();
  arch_spec.SetArchitecture(eArchTypeXCOFF, llvm::XCOFF::TCPU_PPC64,
                            LLDB_INVALID_CPUTYPE, llvm::Triple::AIX);
  process_info.SetArchitecture(arch_spec);
  process_info.SetParentProcessID(psinfoData.pr_ppid);
  process_info.SetGroupID(psinfoData.pr_gid);
  process_info.SetEffectiveGroupID(psinfoData.pr_egid);
  process_info.SetUserID(psinfoData.pr_uid);
  process_info.SetEffectiveUserID(psinfoData.pr_euid);
  process_info.SetProcessGroupID(psinfoData.pr_pgid);
  process_info.SetProcessSessionID(psinfoData.pr_sid);
  return true;
}

static bool GetProcessAndStatInfo(::pid_t pid,
                                  ProcessInstanceInfo &process_info,
                                  ProcessState &State) {
  process_info.Clear();
  process_info.SetProcessID(pid);

  if (pid == LLDB_INVALID_PROCESS_ID)
    return false;
  // Get Executable path/Arch and Get User and Group IDs.
  if (!GetExePathAndIds(pid, process_info))
    return false;
  // Get process status and timing info.
  if (!GetStatusInfo(pid, process_info, State))
    return false;

  return true;
}

uint32_t Host::FindProcessesImpl(const ProcessInstanceInfoMatch &match_info,
                                 ProcessInstanceInfoList &process_infos) {
  return 0;
}

bool Host::GetProcessInfo(lldb::pid_t pid, ProcessInstanceInfo &process_info) {
  ProcessState State;
  return GetProcessAndStatInfo(pid, process_info, State);
}

Status Host::ShellExpandArguments(ProcessLaunchInfo &launch_info) {
  return Status("unimplemented");
}
