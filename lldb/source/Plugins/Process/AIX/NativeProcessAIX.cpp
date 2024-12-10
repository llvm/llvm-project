//===-- NativeProcessAIX.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NativeProcessAIX.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostProcess.h"
#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/State.h"
#include "lldb/Utility/Status.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/Error.h"
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <unistd.h>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_aix;
using namespace llvm;

static constexpr unsigned k_ptrace_word_size = sizeof(void *);
static_assert(sizeof(long) >= k_ptrace_word_size,
              "Size of long must be larger than ptrace word size");

// Simple helper function to ensure flags are enabled on the given file
// descriptor.
static llvm::Error EnsureFDFlags(int fd, int flags) {
  Error error;

  int status = fcntl(fd, F_GETFL);
  if (status == -1) {
    error = errorCodeToError(errnoAsErrorCode());
    return error;
  }

  if (fcntl(fd, F_SETFL, status | flags) == -1) {
    error = errorCodeToError(errnoAsErrorCode());
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
  NativeProcessAIX::Extension supported = {};

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

  Error status;
  if ((status = PtraceWrapper(PT_ATTACH, pid)).Fail()) {
    return errorCodeToError(errnoAsErrorCode());
  }

  int wpid = llvm::sys::RetryAfterSignal(-1, ::waitpid, pid, nullptr, WNOHANG);
  if (wpid <= 0) {
    return llvm::errorCodeToError(
        std::error_code(errno, std::generic_category()));
  }
  LLDB_LOG(log, "adding pid = {0}", pid);

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

Error NativeProcessAIX::Halt() {
  Error error;
  return error;
}

Error NativeProcessAIX::Detach() {
  Error error;
  return error;
}

Error NativeProcessAIX::Signal(int signo) {
  Error error;
  return error;
}

Error NativeProcessAIX::Interrupt() { return Status(); }

Error NativeProcessAIX::Kill() {
  Error error;
  return error;
}

Error NativeProcessAIX::SetBreakpoint(lldb::addr_t addr, uint32_t size,
                                      bool hardware) {
  if (hardware)
    return SetHardwareBreakpoint(addr, size);
  else
    return SetSoftwareBreakpoint(addr, size);
}

Error NativeProcessAIX::RemoveBreakpoint(lldb::addr_t addr, bool hardware) {
  if (hardware)
    return RemoveHardwareBreakpoint(addr);
  else
    return NativeProcessProtocol::RemoveBreakpoint(addr);
}

int8_t NativeProcessAIX::GetSignalInfo(WaitStatus wstatus) const {
  return wstatus.status;
}

Error NativeProcessAIX::Detach(lldb::tid_t tid) {
  if (tid == LLDB_INVALID_THREAD_ID)
    return Status();

  return PtraceWrapper(PT_DETACH, tid);
}

// Wrapper for ptrace to catch errors and log calls. Note that ptrace sets
// errno on error because -1 can be a valid result (i.e. for PTRACE_PEEK*)
Error NativeProcessAIX::PtraceWrapper(int req, lldb::pid_t pid, void *addr,
                                      void *data, size_t data_size,
                                      long *result) {
  Error error;
  long int ret;

  Log *log = GetLog(POSIXLog::Ptrace);
  errno = 0;
  if (req < PT_COMMAND_MAX) {
    if (req == PT_ATTACH) {
      ptrace64(req, pid, 0, 0, nullptr);
    } else if (req == PT_DETACH) {
      ptrace64(req, pid, 0, 0, nullptr);
    }
  } else {
    assert(0 && "Not supported yet.");
  }

  if (errno) {
    error = errorCodeToError(errnoAsErrorCode());
    ret = -1;
  }

  LLDB_LOG(log, "ptrace({0}, {1}, {2}, {3}, {4})={5:x}", req, pid, addr, data,
           data_size, ret);
  if (error.Fail())
    LLDB_LOG(log, "ptrace() failed: {0}", error);

  return error;
}
