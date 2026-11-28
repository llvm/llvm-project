//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NativeProcessOpenBSD.h"

#include <errno.h>
#include <fcntl.h>
#define _DYN_LOADER
#include <elf.h>
#include <util.h>

#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"
#include "lldb/Host/HostProcess.h"
#include "lldb/Host/common/NativeRegisterContext.h"
#include "lldb/Host/posix/ProcessLauncherPosixFork.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/State.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errno.h"

#include <cstring>

// System includes - They have to be included after framework includes because
// they define some macros which collide with variable names in other modules
// clang-format off
#include <sys/types.h>
#include <sys/ptrace.h>
#include <sys/sysctl.h>
#include <sys/wait.h>
#include <unistd.h>
// clang-format on

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_openbsd;
using namespace llvm;

// Simple helper function to ensure flags are enabled on the given file
// descriptor.
static Status EnsureFDFlags(int fd, int flags) {
  int status = fcntl(fd, F_GETFL);
  if (status == -1)
    return Status::FromErrno();

  if (fcntl(fd, F_SETFL, status | flags) == -1)
    return Status::FromErrno();

  return Status();
}

// No architecture has a register context yet.
static llvm::Error CheckArchitectureSupported() {
  return llvm::createStringError(
      "debugging is not supported on this architecture");
}

// Whether kern.global_ptrace allows attaching to processes that are not
// children of the debugger.
static bool IsGlobalPtraceEnabled() {
  int mib[2] = {CTL_KERN, KERN_GLOBAL_PTRACE};
  int value = 0;
  size_t len = sizeof(value);
  return ::sysctl(mib, 2, &value, &len, nullptr, 0) == 0 && value != 0;
}

// Public Static Methods

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
NativeProcessOpenBSD::Manager::Launch(ProcessLaunchInfo &launch_info,
                                      NativeDelegate &native_delegate) {
  Log *log = GetLog(POSIXLog::Process);

  if (llvm::Error error = CheckArchitectureSupported())
    return std::move(error);

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
  int wstatus;
  ::pid_t wpid = llvm::sys::RetryAfterSignal(-1, ::waitpid, pid, &wstatus, 0);
  assert(wpid == pid);
  UNUSED_IF_ASSERT_DISABLED(wpid);
  if (!WIFSTOPPED(wstatus)) {
    LLDB_LOG(log, "Could not sync with inferior process: wstatus={0}",
             WaitStatus::Decode(wstatus));
    return llvm::createStringError("could not sync with inferior process");
  }
  LLDB_LOG(log, "inferior started, now in stopped state");

  ProcessInstanceInfo Info;
  if (!Host::GetProcessInfo(pid, Info)) {
    return llvm::createStringError("cannot get process architecture");
  }

  // Set the architecture to the exe architecture.
  LLDB_LOG(log, "pid = {0:x}, detected architecture {1}", pid,
           Info.GetArchitecture().GetArchitectureName());

  std::unique_ptr<NativeProcessOpenBSD> process_up(new NativeProcessOpenBSD(
      pid, launch_info.GetPTY().ReleasePrimaryFileDescriptor(), native_delegate,
      Info.GetArchitecture(), m_mainloop));

  status = process_up->RefreshThreads();
  if (status.Fail())
    return status.ToError();

  process_up->m_trapped_tid = process_up->GetTrappedThreadID();
  for (const auto &thread : process_up->m_threads)
    static_cast<NativeThreadOpenBSD &>(*thread).SetStoppedBySignal(SIGSTOP);
  process_up->SetState(StateType::eStateStopped, false);

  return std::move(process_up);
}

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
NativeProcessOpenBSD::Manager::Attach(
    lldb::pid_t pid, NativeProcessProtocol::NativeDelegate &native_delegate) {

  Log *log = GetLog(POSIXLog::Process);
  LLDB_LOG(log, "pid = {0:x}", pid);

  if (llvm::Error error = CheckArchitectureSupported())
    return std::move(error);

  // Retrieve the architecture for the running process.
  ProcessInstanceInfo Info;
  if (!Host::GetProcessInfo(pid, Info)) {
    return llvm::createStringError("cannot get process architecture");
  }

  std::unique_ptr<NativeProcessOpenBSD> process_up(new NativeProcessOpenBSD(
      pid, -1, native_delegate, Info.GetArchitecture(), m_mainloop));

  Status status = process_up->Attach();
  if (!status.Success())
    return status.ToError();

  return std::move(process_up);
}

NativeProcessOpenBSD::Extension
NativeProcessOpenBSD::Manager::GetSupportedExtensions() const {
  Extension supported =
      Extension::pass_signals | Extension::auxv | Extension::libraries_svr4;
  return supported;
}

// Public Instance Methods

NativeProcessOpenBSD::NativeProcessOpenBSD(::pid_t pid, int terminal_fd,
                                           NativeDelegate &delegate,
                                           const ArchSpec &arch,
                                           MainLoop &mainloop)
    : NativeProcessELF(pid, terminal_fd, delegate), m_arch(arch) {
  if (m_terminal_fd != -1) {
    Status status = EnsureFDFlags(m_terminal_fd, O_NONBLOCK);
    assert(status.Success());
  }

  Status status;
  m_sigchld_handle = mainloop.RegisterSignal(
      SIGCHLD, [this](MainLoopBase &) { SigchldHandler(); }, status);
  assert(m_sigchld_handle && status.Success());
}

// Handles all waitpid events from the inferior process.
void NativeProcessOpenBSD::MonitorCallback(lldb::pid_t pid, int signal) {
  // OpenBSD does not report thread creation or exit, so resynchronize the
  // thread list every time the process stops.
  Status error = RefreshThreads();
  if (error.Fail()) {
    Log *log = GetLog(POSIXLog::Thread);
    LLDB_LOG(log, "failed to refresh threads for pid {0}: {1}", pid, error);
  }

  switch (signal) {
  case SIGTRAP:
    return MonitorSIGTRAP(pid);
  default:
    return MonitorSignal(pid, signal);
  }
}

void NativeProcessOpenBSD::MonitorExited(lldb::pid_t pid, WaitStatus status) {
  Log *log = GetLog(POSIXLog::Process);

  LLDB_LOG(log, "got exit signal({0}) , pid = {1}", status, pid);

  /* Stop Tracking All Threads attached to Process */
  m_threads.clear();

  SetExitStatus(status, true);

  // Notify delegate that our process has exited.
  SetState(StateType::eStateExited, true);
}

void NativeProcessOpenBSD::MonitorSIGTRAP(lldb::pid_t pid) {
  // OpenBSD reports a plain SIGTRAP for both a completed PT_STEP and a
  // software breakpoint, so use what the last resume requested to tell them
  // apart: it is a step only if the thread that trapped is the one that was
  // being stepped.
  m_trapped_tid = GetTrappedThreadID();

  const bool stepped =
      m_step_tid != LLDB_INVALID_THREAD_ID && m_step_tid == m_trapped_tid;
  m_step_tid = LLDB_INVALID_THREAD_ID;

  for (const auto &thread : m_threads) {
    auto &t = static_cast<NativeThreadOpenBSD &>(*thread);
    if (t.GetID() != m_trapped_tid) {
      t.SetStoppedWithNoReason();
    } else if (stepped) {
      t.SetStoppedByTrace();
    } else {
      t.SetStoppedByBreakpoint();
      FixupBreakpointPCAsNeeded(t);
    }
  }
  if (m_trapped_tid != LLDB_INVALID_THREAD_ID)
    SetCurrentThreadID(m_trapped_tid);
  SetState(StateType::eStateStopped, true);
}

void NativeProcessOpenBSD::MonitorSignal(lldb::pid_t pid, int signal) {
  m_trapped_tid = GetTrappedThreadID();
  m_step_tid = LLDB_INVALID_THREAD_ID;

  for (const auto &thread : m_threads) {
    auto &t = static_cast<NativeThreadOpenBSD &>(*thread);
    if (t.GetID() == m_trapped_tid)
      t.SetStoppedBySignal(signal);
    else
      t.SetStoppedWithNoReason();
  }
  if (m_trapped_tid != LLDB_INVALID_THREAD_ID)
    SetCurrentThreadID(m_trapped_tid);
  SetState(StateType::eStateStopped, true);
}

Status NativeProcessOpenBSD::PtraceWrapper(int req, lldb::pid_t pid, void *addr,
                                           int data, int *result) {
  Log *log = GetLog(POSIXLog::Ptrace);
  Status error;
  int ret;

  errno = 0;
  ret =
      ptrace(req, static_cast<::pid_t>(pid), static_cast<caddr_t>(addr), data);

  if (ret == -1)
    error = Status::FromErrno();

  if (result)
    *result = ret;

  LLDB_LOG(log, "ptrace({0}, {1}, {2}, {3})={4:x}", req, pid, addr, data, ret);

  if (error.Fail())
    LLDB_LOG(log, "ptrace() failed: {0}", error);

  return error;
}

Status NativeProcessOpenBSD::Resume(const ResumeActionList &resume_actions) {
  Log *log = GetLog(POSIXLog::Process);
  LLDB_LOG(log, "pid {0}", GetID());

  // OpenBSD's ptrace(2) can resume either the whole process or, when given a
  // thread ID, only the thread that reported the last stop (the others stay
  // suspended).  It cannot suspend individual threads, so map lldb's
  // per-thread actions onto those two modes.
  NativeThreadOpenBSD *step_thread = nullptr;
  std::vector<NativeThreadOpenBSD *> run_threads;
  bool any_suspended = false;
  int signal = 0;

  for (const auto &abs_thread : m_threads) {
    auto &thread = static_cast<NativeThreadOpenBSD &>(*abs_thread);
    const ResumeAction *action =
        resume_actions.GetActionForThread(thread.GetID(), true);
    if (action == nullptr) {
      any_suspended = true;
      continue;
    }

    switch (action->state) {
    case eStateRunning:
      run_threads.push_back(&thread);
      break;
    case eStateStepping:
      if (step_thread)
        return Status::FromErrorString(
            "OpenBSD can only single-step one thread at a time");
      step_thread = &thread;
      break;
    case eStateSuspended:
    case eStateStopped:
      any_suspended = true;
      break;
    default:
      return Status::FromErrorStringWithFormat(
          "NativeProcessOpenBSD::%s (): unexpected state %s specified "
          "for pid %" PRIu64 ", tid %" PRIu64,
          __FUNCTION__, StateAsCString(action->state), GetID(), thread.GetID());
    }

    if (action->signal != LLDB_INVALID_SIGNAL_NUMBER)
      signal = action->signal;
  }

  lldb::pid_t target = GetID();
  int req;
  if (step_thread) {
#ifdef PT_STEP
    if (step_thread->GetID() != m_trapped_tid)
      return Status::FromErrorStringWithFormat(
          "OpenBSD can only single-step the thread that reported the last "
          "stop (tid %" PRIu64 ")",
          m_trapped_tid);
    req = PT_STEP;
    // With the thread ID only the stepping thread runs, which is what lldb
    // wants when the other threads are suspended (e.g. stepping over a
    // breakpoint).  With the process ID the other threads run as well.
    if (run_threads.empty())
      target = step_thread->GetID();
#else
    return Status::FromErrorString(
        "NativeProcessOpenBSD stepping not supported on this platform");
#endif
  } else if (run_threads.empty()) {
    LLDB_LOG(log, "no threads to resume for pid {0}", GetID());
    return Status();
  } else {
    req = PT_CONTINUE;
    if (any_suspended) {
      if (run_threads.size() == 1 &&
          run_threads.front()->GetID() == m_trapped_tid)
        target = m_trapped_tid;
      else
        LLDB_LOG(log,
                 "cannot suspend individual threads, resuming all threads "
                 "of pid {0}",
                 GetID());
    }
  }

  Status error = PtraceWrapper(req, target, (void *)1, signal);
  if (error.Fail())
    return error;

  const bool whole_process = target == GetID();
  for (const auto &abs_thread : m_threads) {
    auto &thread = static_cast<NativeThreadOpenBSD &>(*abs_thread);
    if (&thread == step_thread)
      thread.SetStepping();
    else if (whole_process || thread.GetID() == target)
      thread.SetRunning();
  }
  m_step_tid = step_thread ? step_thread->GetID() : LLDB_INVALID_THREAD_ID;
  SetState(step_thread ? eStateStepping : eStateRunning, true);

  return Status();
}

Status NativeProcessOpenBSD::Halt() {
  // Do not try to stop a process that's already stopped, this may cause
  // the SIGSTOP to get queued and stop the process again once resumed.
  if (StateIsStoppedState(m_state, false))
    return Status();
  if (kill(GetID(), SIGSTOP) != 0)
    return Status::FromErrno();
  return Status();
}

Status NativeProcessOpenBSD::Detach() {
  Status error;

  // Stop monitoring the inferior.
  m_sigchld_handle.reset();

  // Tell ptrace to detach from the process.
  if (GetID() == LLDB_INVALID_PROCESS_ID)
    return error;

  return PtraceWrapper(PT_DETACH, GetID());
}

Status NativeProcessOpenBSD::Signal(int signo) {
  if (kill(GetID(), signo) != 0)
    return Status::FromErrno();
  return Status();
}

Status NativeProcessOpenBSD::Kill() {
  Log *log = GetLog(POSIXLog::Process);
  LLDB_LOG(log, "pid {0}", GetID());

  Status error;

  switch (m_state) {
  case StateType::eStateInvalid:
  case StateType::eStateExited:
  case StateType::eStateCrashed:
  case StateType::eStateDetached:
  case StateType::eStateUnloaded:
    // Nothing to do - the process is already dead.
    LLDB_LOG(log, "ignored for PID {0} due to current state: {1}", GetID(),
             StateAsCString(m_state));
    return error;

  case StateType::eStateConnected:
  case StateType::eStateAttaching:
  case StateType::eStateLaunching:
  case StateType::eStateStopped:
  case StateType::eStateRunning:
  case StateType::eStateStepping:
  case StateType::eStateSuspended:
    // We can try to kill a process in these states.
    break;
  }

  // A process stopped by ptrace(2) does not act on SIGKILL until it is
  // resumed, so use PT_KILL, which resumes it with SIGKILL. PT_KILL fails on
  // a running process, so signal that one directly.
  if (StateIsStoppedState(m_state, false))
    return PtraceWrapper(PT_KILL, GetID());

  if (kill(GetID(), SIGKILL) != 0)
    return Status::FromErrno();

  return error;
}

llvm::Expected<lldb::addr_t>
NativeProcessOpenBSD::AllocateMemory(size_t size, uint32_t permissions) {
  return llvm::make_error<UnimplementedError>();
}

llvm::Error NativeProcessOpenBSD::DeallocateMemory(lldb::addr_t addr) {
  return llvm::make_error<UnimplementedError>();
}

size_t NativeProcessOpenBSD::UpdateThreads() { return m_threads.size(); }

Status NativeProcessOpenBSD::SetBreakpoint(lldb::addr_t addr, uint32_t size,
                                           bool hardware) {
  if (hardware)
    return Status::FromErrorString(
        "NativeProcessOpenBSD does not support hardware breakpoints");
  else
    return SetSoftwareBreakpoint(addr, size);
}

Status NativeProcessOpenBSD::GetLoadedModuleFileSpec(const char *module_path,
                                                     FileSpec &file_spec) {
  return Status::FromErrorString("Unimplemented");
}

Status
NativeProcessOpenBSD::GetFileLoadAddress(const llvm::StringRef &file_name,
                                         lldb::addr_t &load_addr) {
  return Status::FromErrorString("Unimplemented");
}

void NativeProcessOpenBSD::SigchldHandler() {
  Log *log = GetLog(POSIXLog::Process);
  // Process all pending waitpid notifications.
  int status;
  ::pid_t wait_pid =
      llvm::sys::RetryAfterSignal(-1, waitpid, GetID(), &status, WNOHANG);

  if (wait_pid == 0)
    return; // We are done.

  if (wait_pid == -1) {
    Status error(errno, eErrorTypePOSIX);
    LLDB_LOG(log, "waitpid ({0}, &status, _) failed: {1}", GetID(), error);
    return;
  }

  WaitStatus wait_status = WaitStatus::Decode(status);
  bool exited = wait_status.type == WaitStatus::Exit ||
                (wait_status.type == WaitStatus::Signal &&
                 wait_pid == static_cast<::pid_t>(GetID()));

  LLDB_LOG(log,
           "waitpid ({0}, &status, _) => pid = {1}, status = {2}, exited = {3}",
           GetID(), wait_pid, status, exited);

  if (exited)
    MonitorExited(wait_pid, wait_status);
  else {
    assert(wait_status.type == WaitStatus::Stop);
    MonitorCallback(wait_pid, wait_status.status);
  }
}

bool NativeProcessOpenBSD::HasThreadNoLock(lldb::tid_t thread_id) {
  for (const auto &thread : m_threads) {
    assert(thread && "thread list should not contain NULL threads");
    if (thread->GetID() == thread_id) {
      // We have this thread.
      return true;
    }
  }

  // We don't have this thread.
  return false;
}

NativeThreadOpenBSD &NativeProcessOpenBSD::AddThread(lldb::tid_t thread_id) {

  Log *log = GetLog(POSIXLog::Thread);
  LLDB_LOG(log, "pid {0} adding thread with tid {1}", GetID(), thread_id);

  assert(!HasThreadNoLock(thread_id) &&
         "attempted to add a thread by id that already exists");

  // If this is the first thread, save it as the current thread
  if (m_threads.empty())
    SetCurrentThreadID(thread_id);

  m_threads.push_back(std::make_unique<NativeThreadOpenBSD>(*this, thread_id));
  return static_cast<NativeThreadOpenBSD &>(*m_threads.back());
}

Status NativeProcessOpenBSD::Attach() {
  // Attach to the requested process.
  // An attach will cause the thread to stop with a SIGSTOP.
  Status status = PtraceWrapper(PT_ATTACH, m_pid);
  if (status.Fail()) {
    if (status.GetError() == EPERM && ::geteuid() != 0 &&
        !IsGlobalPtraceEnabled())
      return Status::FromErrorString(
          "operation not permitted: attaching to a process that is not a "
          "child of the debugger requires root or kern.global_ptrace=1");
    return status;
  }

  int wstatus;
  // At this point we should have a thread stopped if waitpid succeeds.
  if ((wstatus = llvm::sys::RetryAfterSignal(-1, waitpid, m_pid, nullptr, 0)) <
      0)
    return Status(errno, eErrorTypePOSIX);

  // Initialize threads.
  status = RefreshThreads();
  if (status.Fail())
    return status;

  m_trapped_tid = GetTrappedThreadID();

  for (const auto &thread : m_threads)
    static_cast<NativeThreadOpenBSD &>(*thread).SetStoppedBySignal(SIGSTOP);

  // Let our process instance know the thread has stopped.
  SetCurrentThreadID(m_threads.front()->GetID());
  SetState(StateType::eStateStopped, false);
  return Status();
}

Status NativeProcessOpenBSD::ReadMemory(const ProcessAddress &process_addr,
                                        void *buf, size_t size,
                                        size_t &bytes_read) {
  lldb::addr_t addr = process_addr.GetValue();
  unsigned char *dst = static_cast<unsigned char *>(buf);
  struct ptrace_io_desc io;

  Log *log = GetLog(POSIXLog::Memory);
  LLDB_LOG(log, "addr = {0}, buf = {1}, size = {2}", addr, buf, size);

  bytes_read = 0;
  io.piod_op = PIOD_READ_D;
  io.piod_len = size;

  do {
    io.piod_offs = (void *)(addr + bytes_read);
    io.piod_addr = dst + bytes_read;

    Status error = NativeProcessOpenBSD::PtraceWrapper(PT_IO, GetID(), &io);
    if (error.Fail() || io.piod_len == 0)
      return error;

    bytes_read += io.piod_len;
    io.piod_len = size - bytes_read;
  } while (bytes_read < size);

  return Status();
}

Status NativeProcessOpenBSD::DoWriteMemory(lldb::addr_t addr, const void *buf,
                                           size_t size, size_t &bytes_written) {
  const unsigned char *src = static_cast<const unsigned char *>(buf);
  struct ptrace_io_desc io;

  Log *log = GetLog(POSIXLog::Memory);
  LLDB_LOG(log, "addr = {0}, buf = {1}, size = {2}", addr, buf, size);

  bytes_written = 0;
  io.piod_op = PIOD_WRITE_D;
  io.piod_len = size;

  do {
    io.piod_addr =
        const_cast<void *>(static_cast<const void *>(src + bytes_written));
    io.piod_offs = (void *)(addr + bytes_written);

    Status error = NativeProcessOpenBSD::PtraceWrapper(PT_IO, GetID(), &io);
    if (error.Fail() || io.piod_len == 0)
      return error;

    bytes_written += io.piod_len;
    io.piod_len = size - bytes_written;
  } while (bytes_written < size);

  return Status();
}

llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
NativeProcessOpenBSD::GetAuxvData() const {
  size_t auxv_size = 100 * sizeof(AuxInfo);

  std::unique_ptr<WritableMemoryBuffer> buf =
      llvm::WritableMemoryBuffer::getNewMemBuffer(auxv_size);
  if (!buf)
    return std::make_error_code(std::errc::not_enough_memory);

  struct ptrace_io_desc io;
  io.piod_op = PIOD_READ_AUXV;
  io.piod_offs = 0;
  io.piod_addr = buf->getBufferStart();
  io.piod_len = auxv_size;

  Status error = NativeProcessOpenBSD::PtraceWrapper(PT_IO, GetID(), &io);
  if (error.Fail())
    return std::error_code(error.GetError(), std::generic_category());

  if (io.piod_len < 1)
    return std::error_code(ECANCELED, std::generic_category());

  // Return only what the kernel actually filled in.
  return llvm::MemoryBuffer::getMemBufferCopy(
      llvm::StringRef(buf->getBufferStart(), io.piod_len));
}

Status NativeProcessOpenBSD::RefreshThreads() {
  Log *log = GetLog(POSIXLog::Thread);

  // Read the kernel's current list of threads.
  std::vector<std::pair<lldb::tid_t, std::string>> kernel_threads;
  struct ptrace_thread_state info = {};
  Status error =
      PtraceWrapper(PT_GET_THREAD_FIRST, GetID(), &info, sizeof(info));
  if (error.Fail())
    return error;
  while (info.pts_tid > 0) {
    kernel_threads.emplace_back(
        info.pts_tid,
        std::string(info.pts_name,
                    strnlen(info.pts_name, sizeof(info.pts_name))));
    error = PtraceWrapper(PT_GET_THREAD_NEXT, GetID(), &info, sizeof(info));
    if (error.Fail()) {
      LLDB_LOG(log, "PT_GET_THREAD_NEXT failed for pid {0}: {1}", GetID(),
               error);
      break;
    }
  }

  // Drop threads that have exited since the last stop.
  llvm::erase_if(
      m_threads, [&](const std::unique_ptr<NativeThreadProtocol> &t) {
        bool alive = llvm::any_of(kernel_threads, [&](const auto &kt) {
          return kt.first == t->GetID();
        });
        if (!alive)
          LLDB_LOG(log, "pid {0} removing exited thread with tid {1}", GetID(),
                   t->GetID());
        return !alive;
      });

  // Add threads created since the last stop, and refresh thread names.
  for (const auto &[tid, name] : kernel_threads) {
    auto *thread = static_cast<NativeThreadOpenBSD *>(GetThreadByID(tid));
    if (!thread)
      thread = &AddThread(tid);
    thread->SetName(name);
  }

  if (!m_threads.empty() && !HasThreadNoLock(GetCurrentThreadID()))
    SetCurrentThreadID(m_threads.front()->GetID());

  return Status();
}

lldb::tid_t NativeProcessOpenBSD::GetTrappedThreadID() {
  Log *log = GetLog(POSIXLog::Thread);

  struct ptrace_state state = {};
  Status error =
      PtraceWrapper(PT_GET_PROCESS_STATE, GetID(), &state, sizeof(state));
  if (error.Success() && state.pe_tid > 0 &&
      HasThreadNoLock(static_cast<lldb::tid_t>(state.pe_tid)))
    return state.pe_tid;

  LLDB_LOG(log, "could not determine the stopped thread of pid {0}: {1}",
           GetID(), error);
  return m_threads.empty() ? LLDB_INVALID_THREAD_ID
                           : m_threads.front()->GetID();
}
