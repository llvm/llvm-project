//===-- NativeProcessOpenBSD.cpp ------------------------------------------===//
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
#include "lldb/Host/FileSystem.h"
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

// Register contexts are only implemented for these architectures.
static llvm::Error CheckArchitectureSupported() {
#if defined(__aarch64__) || defined(__i386__) || defined(__x86_64__)
  return llvm::Error::success();
#else
  return llvm::createStringError(
      "debugging is not supported on this architecture");
#endif
}

// Whether kern.global_ptrace allows attaching to processes that are not
// children of the debugger.
static bool IsGlobalPtraceEnabled() {
#ifdef KERN_GLOBAL_PTRACE
  int mib[2] = {CTL_KERN, KERN_GLOBAL_PTRACE};
  int value = 0;
  size_t len = sizeof(value);
  return ::sysctl(mib, 2, &value, &len, nullptr, 0) == 0 && value != 0;
#else
  return false;
#endif
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
#ifdef PTRACE_FORK
  // OpenBSD reports fork() but not vfork().
  supported |= Extension::multiprocess | Extension::fork;
#endif
  return supported;
}

// Public Instance Methods

NativeProcessOpenBSD::NativeProcessOpenBSD(::pid_t pid, int terminal_fd,
                                           NativeDelegate &delegate,
                                           const ArchSpec &arch,
                                           MainLoop &mainloop)
    : NativeProcessELF(pid, terminal_fd, delegate), m_arch(arch),
      m_main_loop(mainloop) {
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

  if (::pid_t child_pid = GetForkedChildPID()) {
    m_step_tid = LLDB_INVALID_THREAD_ID;
    auto *parent_thread =
        static_cast<NativeThreadOpenBSD *>(GetThreadByID(m_trapped_tid));
    if (parent_thread)
      return MonitorFork(child_pid, *parent_thread);
    LLDB_LOG(GetLog(POSIXLog::Process),
             "fork reported by pid {0} without a trapped thread", GetID());
  }

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

void NativeProcessOpenBSD::MonitorFork(::pid_t child_pid,
                                       NativeThreadOpenBSD &parent_thread) {
  Log *log = GetLog(POSIXLog::Process);
  LLDB_LOG(log, "fork, child_pid={0}", child_pid);

  auto resume_parent = [&]() {
    Status error =
        PtraceWrapper(PT_CONTINUE, GetID(), reinterpret_cast<void *>(1), 0);
    if (error.Fail()) {
      LLDB_LOG(log, "unable to resume parent process {0}: {1}", GetID(), error);
      SetState(StateType::eStateInvalid);
    }
  };

  // The child is traced by us from the start, and stops with a SIGTRAP as
  // soon as it runs.
  int status;
  ::pid_t wait_pid =
      llvm::sys::RetryAfterSignal(-1, ::waitpid, child_pid, &status, 0);
  if (wait_pid != child_pid || !WIFSTOPPED(status)) {
    LLDB_LOG(log, "fork child {0} did not stop, not tracking it", child_pid);
    return resume_parent();
  }

  std::unique_ptr<NativeProcessOpenBSD> child_process{new NativeProcessOpenBSD(
      child_pid, m_terminal_fd, m_delegate, m_arch, m_main_loop)};
  // The child's ptrace state names this process until the child forks.
  child_process->m_last_fork_pid = GetID();
  // The child has a copy of our memory, breakpoints included.
  child_process->m_software_breakpoints = m_software_breakpoints;

  Status error = child_process->RefreshThreads();
  if (error.Fail() || child_process->m_threads.empty() ||
      !bool(m_enabled_extensions & Extension::fork)) {
    LLDB_LOG(log, "not following fork child {0}: {1}", child_pid, error);
    for (const auto &[addr, bp] : m_software_breakpoints) {
      size_t bytes_written;
      child_process->DoWriteMemory(addr, bp.saved_opcodes.data(),
                                   bp.saved_opcodes.size(), bytes_written);
    }
    child_process->Detach();
    return resume_parent();
  }

  child_process->SetEnabledExtensions(m_enabled_extensions);
  child_process->m_trapped_tid = child_process->GetTrappedThreadID();
  const lldb::tid_t child_tid = child_process->m_trapped_tid;
  for (const auto &thread : child_process->m_threads)
    static_cast<NativeThreadOpenBSD &>(*thread).SetStoppedBySignal(SIGSTOP);
  child_process->SetState(StateType::eStateStopped, false);

  m_delegate.NewSubprocess(this, std::move(child_process));

  for (const auto &thread : m_threads) {
    auto &t = static_cast<NativeThreadOpenBSD &>(*thread);
    if (&t == &parent_thread)
      t.SetStoppedByFork(child_pid, child_tid);
    else
      t.SetStoppedWithNoReason();
  }
  SetCurrentThreadID(parent_thread.GetID());
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

Status NativeProcessOpenBSD::GetMemoryRegionInfo(lldb::addr_t load_addr,
                                                 MemoryRegionInfo &range_info) {
  if (m_supports_mem_region == eLazyBoolNo)
    return Status::FromErrorString("unsupported");

  Status error = PopulateMemoryRegionCache();
  if (error.Fail())
    return error;

  // The cache is sorted by address.
  for (const auto &entry : m_mem_region_cache) {
    const MemoryRegionInfo &proc_entry_info = entry.first;
    // If the target address comes before this entry, indicate distance to
    // next region.
    if (load_addr < proc_entry_info.GetRange().GetRangeBase()) {
      range_info.GetRange().SetRangeBase(load_addr);
      range_info.GetRange().SetByteSize(
          proc_entry_info.GetRange().GetRangeBase() - load_addr);
      range_info.SetReadable(eLazyBoolNo);
      range_info.SetWritable(eLazyBoolNo);
      range_info.SetExecutable(eLazyBoolNo);
      range_info.SetMapped(eLazyBoolNo);
      return Status();
    }
    if (proc_entry_info.GetRange().Contains(load_addr)) {
      range_info = proc_entry_info;
      return Status();
    }
  }

  // The address is past the last mapping.
  range_info.GetRange().SetRangeBase(load_addr);
  range_info.GetRange().SetRangeEnd(LLDB_INVALID_ADDRESS);
  range_info.SetReadable(eLazyBoolNo);
  range_info.SetWritable(eLazyBoolNo);
  range_info.SetExecutable(eLazyBoolNo);
  range_info.SetMapped(eLazyBoolNo);
  return Status();
}

Status NativeProcessOpenBSD::PopulateMemoryRegionCache() {
  Log *log = GetLog(POSIXLog::Process);

  // The cache is cleared every time the process stops.
  if (!m_mem_region_cache.empty())
    return Status();

  // KERN_PROC_VMMAP returns the entries starting at the address passed in the
  // first entry's kve_start, and ENOMEM if they did not all fit; continue from
  // the end of the last entry returned.  Only root may read the map of
  // another process.
  int mib[3] = {CTL_KERN, KERN_PROC_VMMAP, static_cast<int>(GetID())};
  std::vector<struct kinfo_vmentry> entries(256);
  u_long next_start = 0;
  while (true) {
    entries[0].kve_start = next_start;
    size_t len = entries.size() * sizeof(struct kinfo_vmentry);
    int ret = ::sysctl(mib, 3, entries.data(), &len, nullptr, 0);
    int saved_errno = errno;
    if (ret != 0 && saved_errno != ENOMEM) {
      m_mem_region_cache.clear();
      m_supports_mem_region = eLazyBoolNo;
      LLDB_LOG(log, "sysctl(KERN_PROC_VMMAP) failed for pid {0}: {1}", GetID(),
               llvm::sys::StrError(saved_errno));
      return Status(saved_errno, eErrorTypePOSIX);
    }

    size_t count = len / sizeof(struct kinfo_vmentry);
    for (size_t i = 0; i < count; ++i) {
      const struct kinfo_vmentry &kve = entries[i];
#ifdef KVE_ET_HOLE
      if (kve.kve_etype & KVE_ET_HOLE)
        continue;
#endif
      MemoryRegionInfo info;
      info.Clear();
      info.GetRange().SetRangeBase(kve.kve_start);
      info.GetRange().SetRangeEnd(kve.kve_end);
      info.SetMapped(eLazyBoolYes);
      info.SetReadable(kve.kve_protection & KVE_PROT_READ ? eLazyBoolYes
                                                          : eLazyBoolNo);
      info.SetWritable(kve.kve_protection & KVE_PROT_WRITE ? eLazyBoolYes
                                                           : eLazyBoolNo);
      info.SetExecutable(kve.kve_protection & KVE_PROT_EXEC ? eLazyBoolYes
                                                            : eLazyBoolNo);
      m_mem_region_cache.emplace_back(info, FileSpec());
    }

    if (ret == 0 || count == 0)
      break;
    next_start = entries[count - 1].kve_end;
  }

  if (m_mem_region_cache.empty()) {
    m_supports_mem_region = eLazyBoolNo;
    return Status::FromErrorString("no memory map entries found");
  }

  LLDB_LOG(log, "read {0} memory region entries from process {1}",
           m_mem_region_cache.size(), GetID());
  m_supports_mem_region = eLazyBoolYes;
  return Status();
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

// OpenBSD's memory map (KERN_PROC_VMMAP) carries no file names, so look
// modules up in the dynamic loader's list of loaded objects instead.
Status NativeProcessOpenBSD::GetLoadedModuleFileSpec(const char *module_path,
                                                     FileSpec &file_spec) {
  file_spec.Clear();

  llvm::Expected<std::vector<SVR4LibraryInfo>> libraries =
      GetLoadedSVR4Libraries();
  if (!libraries)
    return Status::FromError(libraries.takeError());

  FileSpec module_file_spec(module_path);
  FileSystem::Instance().Resolve(module_file_spec);

  for (const SVR4LibraryInfo &library : *libraries) {
    FileSpec library_file_spec(library.name);
    if (library_file_spec.GetFilename() == module_file_spec.GetFilename()) {
      file_spec = library_file_spec;
      return Status();
    }
  }
  return Status::FromErrorStringWithFormatv(
      "Module file ({0}) not found in the loaded object list",
      module_file_spec.GetFilename());
}

Status
NativeProcessOpenBSD::GetFileLoadAddress(const llvm::StringRef &file_name,
                                         lldb::addr_t &load_addr) {
  load_addr = LLDB_INVALID_ADDRESS;

  llvm::Expected<std::vector<SVR4LibraryInfo>> libraries =
      GetLoadedSVR4Libraries();
  if (!libraries)
    return Status::FromError(libraries.takeError());

  FileSpec file(file_name);
  for (const SVR4LibraryInfo &library : *libraries) {
    if (FileSpec(library.name) == file) {
      load_addr = library.base_addr;
      return Status();
    }
  }
  return Status::FromErrorStringWithFormatv(
      "No load address found for file {0}", file_name);
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
  if (!process_addr.IsInDefaultAddressSpace())
    return Status::FromErrorString("address spaces are not supported");
  lldb::addr_t addr = process_addr.GetValue();
  unsigned char *dst = static_cast<unsigned char *>(buf);
  struct ptrace_io_desc io;

  Log *log = GetLog(POSIXLog::Memory);
  LLDB_LOG(log, "addr = {0}, buf = {1}, size = {2}", addr, buf, size);

  bytes_read = 0;
  io.piod_op = PIOD_READ_D;

  do {
    io.piod_offs = (void *)(addr + bytes_read);
    io.piod_addr = dst + bytes_read;
    io.piod_len = size - bytes_read;

    Status error = NativeProcessOpenBSD::PtraceWrapper(PT_IO, GetID(), &io);
    if (error.Fail() || io.piod_len == 0)
      return error;

    bytes_read += io.piod_len;
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

  do {
    io.piod_addr =
        const_cast<void *>(static_cast<const void *>(src + bytes_written));
    io.piod_offs = (void *)(addr + bytes_written);
    io.piod_len = size - bytes_written;

    Status error = NativeProcessOpenBSD::PtraceWrapper(PT_IO, GetID(), &io);
    if (error.Fail() || io.piod_len == 0)
      return error;

    bytes_written += io.piod_len;
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

// Returns the child of the fork() that this stop reports, or 0 if it does not
// report one.
::pid_t NativeProcessOpenBSD::GetForkedChildPID() {
#ifdef PTRACE_FORK
  struct ptrace_state state = {};
  if (PtraceWrapper(PT_GET_PROCESS_STATE, GetID(), &state, sizeof(state))
          .Fail())
    return 0;
  // The kernel does not clear the report once it has been read, so a fork is
  // only new if it names another process than the last one.
  if (state.pe_report_event != PTRACE_FORK ||
      state.pe_other_pid == m_last_fork_pid)
    return 0;
  m_last_fork_pid = state.pe_other_pid;
  return state.pe_other_pid;
#else
  return 0;
#endif
}

Status NativeProcessOpenBSD::SetForkReporting(bool enable) {
#ifdef PTRACE_FORK
  ptrace_event_t events = {};
  Status error =
      PtraceWrapper(PT_GET_EVENT_MASK, GetID(), &events, sizeof(events));
  if (error.Fail())
    return error;
  if (enable)
    events.pe_set_event |= PTRACE_FORK;
  else
    events.pe_set_event &= ~PTRACE_FORK;
  return PtraceWrapper(PT_SET_EVENT_MASK, GetID(), &events, sizeof(events));
#else
  if (enable)
    return Status::FromErrorString("fork reporting is not supported");
  return Status();
#endif
}

void NativeProcessOpenBSD::SetEnabledExtensions(Extension flags) {
  NativeProcessELF::SetEnabledExtensions(flags);

  // Only have the kernel trace fork children when the client follows them.
  Status error = SetForkReporting(bool(flags & Extension::fork));
  if (error.Fail())
    LLDB_LOG(GetLog(POSIXLog::Process),
             "could not change fork reporting for pid {0}: {1}", GetID(),
             error);
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
