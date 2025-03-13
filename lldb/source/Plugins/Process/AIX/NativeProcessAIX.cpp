//===-- NativeProcessAIX.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NativeProcessAIX.h"
#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/HostProcess.h"
#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Host/posix/ProcessLauncherPosixFork.h"
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
#include <sys/ptrace.h>
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
static llvm::Error SetFDFlags(int fd, int flags) {
  int status = fcntl(fd, F_GETFL);
  if (status == -1)
    return errorCodeToError(errnoAsErrorCode());
  if (fcntl(fd, F_SETFL, status | flags) == -1)
    return errorCodeToError(errnoAsErrorCode());
  return Error::success();
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
  LLDB_LOG(
      log, "pid = {0}, detected architecture {1}", pid,
      HostInfo::GetArchitecture(HostInfo::eArchKind64).GetArchitectureName());

  return std::unique_ptr<NativeProcessAIX>(new NativeProcessAIX(
      pid, launch_info.GetPTY().ReleasePrimaryFileDescriptor(), native_delegate,
      HostInfo::GetArchitecture(HostInfo::eArchKind64), *this, {pid}));
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
      pid, -1, native_delegate,
      HostInfo::GetArchitecture(HostInfo::eArchKind64), *this, *tids_or));
}

lldb::addr_t NativeProcessAIX::GetSharedLibraryInfoAddress() {
  return LLDB_INVALID_ADDRESS;
}

static std::optional<std::pair<lldb::pid_t, WaitStatus>> WaitPid() {
  Log *log = GetLog(POSIXLog::Process);

  int status;
  ::pid_t wait_pid =
      llvm::sys::RetryAfterSignal(-1, ::waitpid, -1, &status, WNOHANG);

  if (wait_pid == 0)
    return std::nullopt;

  if (wait_pid == -1) {
    Status error(errno, eErrorTypePOSIX);
    LLDB_LOG(log, "waitpid(-1, &status, _) failed: {0}", error);
    return std::nullopt;
  }

  WaitStatus wait_status = WaitStatus::Decode(status);

  LLDB_LOG(log, "waitpid(-1, &status, _) = {0}, status = {1}", wait_pid,
           wait_status);
  return std::make_pair(wait_pid, wait_status);
}

void NativeProcessAIX::Manager::SigchldHandler() {
  Log *log = GetLog(POSIXLog::Process);
  while (true) {
    auto wait_result = WaitPid();
    if (!wait_result)
      return;
    lldb::pid_t pid = wait_result->first;
    WaitStatus status = wait_result->second;

    // Ask each process whether it wants to handle the event. Each event should
    // be handled by exactly one process, but thread creation events require
    // special handling.
    // Thread creation consists of two events (one on the parent and one on the
    // child thread) and they can arrive in any order nondeterministically. The
    // parent event carries the information about the child thread, but not
    // vice-versa. This means that if the child event arrives first, it may not
    // be handled by any process (because it doesn't know the thread belongs to
    // it).
    bool handled = llvm::any_of(m_processes, [&](NativeProcessAIX *process) {
      return process->TryHandleWaitStatus(pid, status);
    });
    if (!handled) {
      if (status.type == WaitStatus::Stop && status.status == SIGSTOP) {
        // Store the thread creation event for later collection.
        m_unowned_threads.insert(pid);
      } else {
        LLDB_LOG(log, "Ignoring waitpid event {0} for pid {1}", status, pid);
      }
    }
  }
}

void NativeProcessAIX::Manager::CollectThread(::pid_t tid) {
  Log *log = GetLog(POSIXLog::Process);

  if (m_unowned_threads.erase(tid))
    return; // We've encountered this thread already.

  // The TID is not tracked yet, let's wait for it to appear.
  int status = -1;
  LLDB_LOG(log,
           "received clone event for tid {0}. tid not tracked yet, "
           "waiting for it to appear...",
           tid);
  ::pid_t wait_pid =
      llvm::sys::RetryAfterSignal(-1, ::waitpid, tid, &status, P_ALL);

  // It's theoretically possible to get other events if the entire process was
  // SIGKILLed before we got a chance to check this. In that case, we'll just
  // clean everything up when we get the process exit event.

  LLDB_LOG(log,
           "waitpid({0}, &status, __WALL) => {1} (errno: {2}, status = {3})",
           tid, wait_pid, errno, WaitStatus::Decode(status));
}

// Public Instance Methods

NativeProcessAIX::NativeProcessAIX(::pid_t pid, int terminal_fd,
                                   NativeDelegate &delegate,
                                   const ArchSpec &arch, Manager &manager,
                                   llvm::ArrayRef<::pid_t> tids)
    : NativeProcessProtocol(pid, terminal_fd, delegate), m_manager(manager),
      m_arch(arch) {
  manager.AddProcess(*this);
  if (m_terminal_fd != -1)
    cantFail(SetFDFlags(m_terminal_fd, O_NONBLOCK));

  // Let our process instance know the thread has stopped.
  SetCurrentThreadID(tids[0]);
  SetState(StateType::eStateStopped, false);
}

llvm::Expected<std::vector<::pid_t>> NativeProcessAIX::Attach(::pid_t pid) {
  Log *log = GetLog(POSIXLog::Process);
  Status status;
  if (PtraceWrapper(PT_ATTACH, pid))
    return errorCodeToError(errnoAsErrorCode());

  int wpid = llvm::sys::RetryAfterSignal(-1, ::waitpid, pid, nullptr, WNOHANG);
  if (wpid <= 0)
    return llvm::errorCodeToError(errnoAsErrorCode());
  LLDB_LOG(log, "adding pid = {0}", pid);

  return std::vector<::pid_t>{pid};
}

bool NativeProcessAIX::TryHandleWaitStatus(lldb::pid_t pid, WaitStatus status) {
  if (pid == GetID() &&
      (status.type == WaitStatus::Exit || status.type == WaitStatus::Signal)) {
    // The process exited.  We're done monitoring.  Report to delegate.
    SetExitStatus(status, true);
    return true;
  }
  return false;
}

bool NativeProcessAIX::SupportHardwareSingleStepping() const { return false; }

Status NativeProcessAIX::Resume(const ResumeActionList &resume_actions) {
  return Status("unsupported");
}

Status NativeProcessAIX::Halt() { return Status("unsupported"); }

Status NativeProcessAIX::Detach() { return Status("unsupported"); }

Status NativeProcessAIX::Signal(int signo) { return Status("unsupported"); }

Status NativeProcessAIX::Interrupt() { return Status("unsupported"); }

Status NativeProcessAIX::Kill() { return Status("unsupported"); }

Status NativeProcessAIX::ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                                    size_t &bytes_read) {
  return Status("unsupported");
}

Status NativeProcessAIX::WriteMemory(lldb::addr_t addr, const void *buf,
                                     size_t size, size_t &bytes_written) {
  return Status("unsupported");
}

size_t NativeProcessAIX::UpdateThreads() {
  // The NativeProcessAIX monitoring threads are always up to date with
  // respect to thread state and they keep the thread list populated properly.
  // All this method needs to do is return the thread count.
  return m_threads.size();
}

Status NativeProcessAIX::GetLoadedModuleFileSpec(const char *module_path,
                                                 FileSpec &file_spec) {
  return Status("unsupported");
}

Status NativeProcessAIX::SetBreakpoint(lldb::addr_t addr, uint32_t size,
                                       bool hardware) {
  if (hardware)
    return SetHardwareBreakpoint(addr, size);
  return SetSoftwareBreakpoint(addr, size);
}

Status NativeProcessAIX::RemoveBreakpoint(lldb::addr_t addr, bool hardware) {
  if (hardware)
    return RemoveHardwareBreakpoint(addr);
  return NativeProcessProtocol::RemoveBreakpoint(addr);
}

llvm::Error NativeProcessAIX::Detach(lldb::tid_t tid) {
  if (tid == LLDB_INVALID_THREAD_ID)
    return llvm::Error::success();

  return PtraceWrapper(PT_DETACH, tid);
}

llvm::Error NativeProcessAIX::PtraceWrapper(int req, lldb::pid_t pid,
                                            void *addr, void *data,
                                            size_t data_size, long *result) {
  long int ret;

  Log *log = GetLog(POSIXLog::Ptrace);
  errno = 0;
  switch (req) {
  case PT_ATTACH:
  case PT_DETACH:
    ret = ptrace64(req, pid, 0, 0, nullptr);
    break;
  default:
    llvm_unreachable("PT_ request not supported yet.");
  }

  if (errno) {
    *result = -1;
    LLDB_LOG(log, "ptrace() failed");
  }
  LLDB_LOG(log, "ptrace({0}, {1}, {2}, {3}, {4})={5:x}", req, pid, addr, data,
           data_size, ret);

  return Error::success();
}
