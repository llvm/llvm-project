//===-- NativeProcessAIX.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NativeProcessAIX.h"
#include "NativeThreadAIX.h"
#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostProcess.h"
#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/State.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/StringExtractor.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unistd.h>
#include <unordered_map>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_aix;
using namespace llvm;

static constexpr unsigned k_ptrace_word_size = sizeof(void *);
static_assert(sizeof(long) >= k_ptrace_word_size,
              "Size of long must be larger than ptrace word size");

// Simple helper function to ensure flags are enabled on the given file
// descriptor.
static Status EnsureFDFlags(int fd, int flags) {
  Status error;

  int status = fcntl(fd, F_GETFL);
  if (status == -1) {
    error = Status::FromErrno();
    return error;
  }

  if (fcntl(fd, F_SETFL, status | flags) == -1) {
    error = Status::FromErrno();
    return error;
  }

  return error;
}

NativeProcessAIX::Manager::Manager(MainLoop &mainloop)
    : NativeProcessProtocol::Manager(mainloop) {
  Status status;
  m_sigchld_handle = mainloop.RegisterSignal(
      SIGCHLD, [this](MainLoopBase &) { SigchldHandler(); }, status);
  assert(m_sigchld_handle && status.Success());
}

// Public Static Methods

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
NativeProcessAIX::Manager::Launch(ProcessLaunchInfo &launch_info,
                                  NativeDelegate &native_delegate) {
  Log *log = GetLog(POSIXLog::Process);

  Status status;
  ::pid_t pid = ProcessLauncherPosixFork()
                    .LaunchProcess(launch_info, status)
                    .GetProcessId();
  LLDB_LOG(log, "pid = {0:x}", pid);
  if (status.Fail()) {
    LLDB_LOG(log, "failed to launch process: {0}", status);
    return status.ToError();
  }

  // Wait for the child process to trap on its call to execve.
  int wstatus = 0;
  ::pid_t wpid = llvm::sys::RetryAfterSignal(-1, ::waitpid, pid, &wstatus, 0);
  assert(wpid == pid);
  UNUSED_IF_ASSERT_DISABLED(wpid);
  if (!WIFSTOPPED(wstatus)) {
    LLDB_LOG(log, "Could not sync with inferior process: wstatus={1}",
             WaitStatus::Decode(wstatus));
    return llvm::make_error<StringError>("Could not sync with inferior process",
                                         llvm::inconvertibleErrorCode());
  }
  LLDB_LOG(log, "inferior started, now in stopped state");

  ProcessInstanceInfo Info;
  if (!Host::GetProcessInfo(pid, Info)) {
    return llvm::make_error<StringError>("Cannot get process architectrue",
                                         llvm::inconvertibleErrorCode());
  }

  // Set the architecture to the exe architecture.
  LLDB_LOG(log, "pid = {0}, detected architecture {1}", pid,
           Info.GetArchitecture().GetArchitectureName());

  return std::unique_ptr<NativeProcessAIX>(new NativeProcessAIX(
      pid, launch_info.GetPTY().ReleasePrimaryFileDescriptor(), native_delegate,
      Info.GetArchitecture(), *this, {pid}));
}

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
NativeProcessAIX::Manager::Attach(
    lldb::pid_t pid, NativeProcessProtocol::NativeDelegate &native_delegate) {
  Log *log = GetLog(POSIXLog::Process);
  LLDB_LOG(log, "pid = {0:x}", pid);

  ProcessInstanceInfo Info;
  if (!Host::GetProcessInfo(pid, Info)) {
    return llvm::make_error<StringError>("Cannot get process architectrue",
                                         llvm::inconvertibleErrorCode());
  }
  auto tids_or = NativeProcessAIX::Attach(pid);
  if (!tids_or)
    return tids_or.takeError();

  return std::unique_ptr<NativeProcessAIX>(new NativeProcessAIX(
      pid, -1, native_delegate, Info.GetArchitecture(), *this, *tids_or));
}

NativeProcessAIX::Extension
NativeProcessAIX::Manager::GetSupportedExtensions() const {
  NativeProcessAIX::Extension supported =
      Extension::multiprocess | Extension::fork | Extension::vfork |
      Extension::pass_signals | Extension::auxv | Extension::libraries_svr4 |
      Extension::siginfo_read;

  return supported;
}

void NativeProcessAIX::Manager::SigchldHandler() {}

void NativeProcessAIX::Manager::CollectThread(::pid_t tid) {}

// Public Instance Methods

NativeProcessAIX::NativeProcessAIX(::pid_t pid, int terminal_fd,
                                   NativeDelegate &delegate,
                                   const ArchSpec &arch, Manager &manager,
                                   llvm::ArrayRef<::pid_t> tids)
    : NativeProcessProtocol(pid, terminal_fd, delegate), m_manager(manager),
      m_arch(arch) {
  manager.AddProcess(*this);
  if (m_terminal_fd != -1) {
    Status status = EnsureFDFlags(m_terminal_fd, O_NONBLOCK);
    assert(status.Success());
  }

  // Let our process instance know the thread has stopped.
  SetCurrentThreadID(tids[0]);
  SetState(StateType::eStateStopped, false);
}

llvm::Expected<std::vector<::pid_t>> NativeProcessAIX::Attach(::pid_t pid) {

  Status status;
  if ((status = PtraceWrapper(PT_ATTACH, pid)).Fail()) {
    return status.ToError();
  }

  std::vector<::pid_t> tids;
  tids.push_back(pid);
  return std::move(tids);
}

void NativeProcessAIX::MonitorSIGTRAP(const WaitStatus status,
                                      NativeThreadAIX &thread) {}

void NativeProcessAIX::MonitorBreakpoint(NativeThreadAIX &thread) {}

bool NativeProcessAIX::SupportHardwareSingleStepping() const { return false; }

Status NativeProcessAIX::Resume(const ResumeActionList &resume_actions) {
  return Status();
}

Status NativeProcessAIX::Halt() {
  Status error;
  return error;
}

Status NativeProcessAIX::Detach() {
  Status error;
  return error;
}

Status NativeProcessAIX::Signal(int signo) {
  Status error;
  return error;
}

Status NativeProcessAIX::Interrupt() { return Status(); }

Status NativeProcessAIX::Kill() {
  Status error;
  return error;
}

Status NativeProcessAIX::SetBreakpoint(lldb::addr_t addr, uint32_t size,
                                       bool hardware) {
  if (hardware)
    return SetHardwareBreakpoint(addr, size);
  else
    return SetSoftwareBreakpoint(addr, size);
}

Status NativeProcessAIX::RemoveBreakpoint(lldb::addr_t addr, bool hardware) {
  if (hardware)
    return RemoveHardwareBreakpoint(addr);
  else
    return NativeProcessProtocol::RemoveBreakpoint(addr);
}

int8_t NativeProcessAIX::GetSignalInfo(WaitStatus wstatus) const {
  return wstatus.status;
}

Status NativeProcessAIX::Detach(lldb::tid_t tid) {
  if (tid == LLDB_INVALID_THREAD_ID)
    return Status();

  return PtraceWrapper(PT_DETACH, tid);
}

// Wrapper for ptrace to catch errors and log calls. Note that ptrace sets
// errno on error because -1 can be a valid result (i.e. for PTRACE_PEEK*)
Status NativeProcessAIX::PtraceWrapper(int req, lldb::pid_t pid, void *addr,
                                       void *data, size_t data_size,
                                       long *result) {
  Status error;
  return error;
}
