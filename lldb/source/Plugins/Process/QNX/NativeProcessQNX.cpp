//===-- NativeProcessQNX.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NativeProcessQNX.h"

#include <fcntl.h>
#include <sys/auxv.h>
#include <sys/mman.h>
#include <sys/neutrino.h>

#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"

#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostProcess.h"
#include "lldb/Host/posix/ProcessLauncherPosixFork.h"
#include "lldb/Host/qnx/Support.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/State.h"

#include "llvm/Support/Errno.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_qnx;
using namespace llvm;

// Simple helper function to ensure flags are enabled on the given file
// descriptor.
static Status EnsureFDFlags(int fd, int flags) {
  Status error;

  int status = fcntl(fd, F_GETFL);
  if (status == -1) {
    error.SetErrorToErrno();
    return error;
  }

  if (fcntl(fd, F_SETFL, status | flags) == -1) {
    error.SetErrorToErrno();
    return error;
  }

  return error;
}

NativeProcessQNX::Manager::Manager(MainLoop &mainloop)
    : NativeProcessProtocol::Manager(mainloop) {
  Status error;
  m_sigchld_handle = mainloop.RegisterSignal(
      SIGCHLD, [this](MainLoopBase &) { SigchldHandler(); }, error);
  if (!m_sigchld_handle || error.Fail())
    return;
}

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
NativeProcessQNX::Manager::Launch(ProcessLaunchInfo &launch_info,
                                  NativeDelegate &native_delegate) {
  Log *log = GetLog(POSIXLog::Process);

  Status error;
  ::pid_t pid = ProcessLauncherPosixFork()
                    .LaunchProcess(launch_info, error)
                    .GetProcessId();
  LLDB_LOG(log, "pid = {0:x}", pid);
  if (error.Fail()) {
    LLDB_LOG(log, "failed to launch process: {0}", error);
    return error.ToError();
  }

  LLDB_LOG(log, "Inferior started; in stopped state now");

  ProcessInstanceInfo info;
  if (!Host::GetProcessInfo(pid, info)) {
    return llvm::make_error<StringError>("cannot get process architecture",
                                         llvm::inconvertibleErrorCode());
  }

  // Set the architecture to the executable's architecture.
  LLDB_LOG(log, "pid = {0:x}, detected architecture {1}", pid,
           info.GetArchitecture().GetArchitectureName());

  std::unique_ptr<NativeProcessQNX> process_up(new NativeProcessQNX(
      pid, launch_info.GetPTY().ReleasePrimaryFileDescriptor(), native_delegate,
      *this, info.GetArchitecture(), error));

  if (error.Fail())
    return error.ToError();

  error = process_up->SetupTrace();
  if (error.Fail())
    return error.ToError();

  for (const auto &thread : process_up->m_threads)
    static_cast<NativeThreadQNX &>(*thread).SetStoppedBySignal(SIGSTOP);
  process_up->SetState(StateType::eStateStopped, false);

  return std::move(process_up);
}

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
NativeProcessQNX::Manager::Attach(
    lldb::pid_t pid, NativeProcessProtocol::NativeDelegate &native_delegate) {
  Log *log = GetLog(POSIXLog::Process);
  LLDB_LOG(log, "pid = {0:x}", pid);

  // Retrieve the architecture for the running process.
  ProcessInstanceInfo info;
  if (!Host::GetProcessInfo(pid, info)) {
    return llvm::make_error<StringError>("cannot get process architecture",
                                         llvm::inconvertibleErrorCode());
  }

  Status error;

  std::unique_ptr<NativeProcessQNX> process_up(new NativeProcessQNX(
      pid, -1, native_delegate, *this, info.GetArchitecture(), error));

  if (!error.Success()) {
    return error.ToError();
  }

  error.Clear();

  error = process_up->Attach();
  if (!error.Success())
    return error.ToError();

  return std::move(process_up);
}

NativeProcessQNX::Extension
NativeProcessQNX::Manager::GetSupportedExtensions() const {
  // TODO: Enable the multiprocess, fork-events, and vfork-events extensions.
  NativeProcessQNX::Extension supported =
      Extension::pass_signals | Extension::auxv | Extension::libraries_svr4 |
      Extension::siginfo_read;

  return supported;
}

void NativeProcessQNX::Manager::SigchldHandler() {
  // We can identify a debuggee that has stopped/terminated via siginfo_t.si_pid
  // or a call to wait*. We have the server deliver a SIGCHLD to the debugger
  // whenever a debuggee reaches a point of interest, though. So, siginfo_t.
  // si_pid doesn't point to the debuggee anymore. Moreover, wait* doesn't wait
  // for waitable debuggees that aren't children on QNX. So, we have siginfo_t.
  // si_value.sival_int set to the debuggee's PID, MainLoopPosix::RegisterSignal
  // provision a mechanism to save the signal information, and MainLoopPosix::
  // SignalHandle expose the saved signal information.
  ::pid_t pid;

  if (m_sigchld_handle && !m_sigchld_handle.get()->GetSiginfo().expired())
    pid = m_sigchld_handle.get()->GetSiginfo().lock()->si_value.sival_int;
  else
    return;

  auto process_it =
      std::find_if(m_processes.begin(), m_processes.end(),
                   [pid](auto process) { return process->GetID() == pid; });

  if (process_it == m_processes.end()) {
    return;
  }

  procfs_status proc_status;
  Status error =
      DevctlWrapper((*process_it)->GetFileDescriptor(), DCMD_PROC_STATUS,
                    &proc_status, sizeof(procfs_status), nullptr);

  if (error.Fail())
    return;

  switch (proc_status.why) {
  // TODO: Monitor fork, vfork, and spawn.
  case _DEBUG_WHY_REQUESTED:
    (*process_it)->MonitorInterrupt();
    break;
  case _DEBUG_WHY_SIGNALLED:
    (*process_it)->MonitorCallback(proc_status);
    break;
  case _DEBUG_WHY_TERMINATED:
    (*process_it)->MonitorExited(proc_status);
    break;
  case _DEBUG_WHY_THREAD:
    (*process_it)->MonitorThread(proc_status);
    break;
  default:
    break;
  }
}

Status NativeProcessQNX::Resume(const ResumeActionList &resume_actions) {
  Log *log = GetLog(POSIXLog::Process);

  Status error;
  procfs_run proc_run;

  memset(&proc_run, 0, sizeof(procfs_run));
  proc_run.flags =
      _DEBUG_RUN_CLRSIG | _DEBUG_RUN_TRACE | _DEBUG_RUN_ARM | _DEBUG_RUN_THREAD;
  sigfillset(&(proc_run.trace));

  for (const auto &abs_thread : m_threads) {
    if (!abs_thread) {
      error.SetErrorString("thread list should not contain NULL threads");
      return error;
    }
    NativeThreadQNX &thread = static_cast<NativeThreadQNX &>(*abs_thread);

    const ResumeAction *action =
        resume_actions.GetActionForThread(thread.GetID(), true);

    if (action == nullptr) {
      LLDB_LOG(log, "No action specified for thread {0} of process {1}",
               thread.GetID(), GetID());
      continue;
    }

    LLDB_LOG(
        log,
        "Processing resume action (state {0}, signal {1}) for thread {2} of "
        "process {3}",
        action->state, action->signal, thread.GetID(), GetID());

    switch (action->state) {
    case eStateRunning:
      error = thread.Resume();
      break;

    case eStateStepping:
      proc_run.flags |= _DEBUG_RUN_STEP | _DEBUG_RUN_CURTID;
      proc_run.tid = thread.GetID();
      error = thread.SingleStep();
      break;

    default:
      error.SetErrorStringWithFormatv(
          "NativeProcessQNX::%s (): Unexpected state %s specified for pid "
          "%" PRIu64 ", tid %" PRIu64,
          __FUNCTION__, StateAsCString(action->state), GetID(), thread.GetID());
      return error;
    }

    if (error.Fail())
      return error;

    if (action->signal != LLDB_INVALID_SIGNAL_NUMBER) {
      sigdelset(&(proc_run.trace), action->signal);
      error = Signal(action->signal);

      if (error.Fail())
        return error;
    }
  }

  error = DevctlWrapper(m_fd, DCMD_PROC_RUN, &proc_run, sizeof(procfs_run),
                        nullptr);

  if (error.Success())
    SetState(eStateRunning, true);

  return error;
}

Status NativeProcessQNX::Halt() {
  procfs_status status;

  return DevctlWrapper(m_fd, DCMD_PROC_STOP, &status, sizeof(procfs_status),
                       nullptr);
}

Status NativeProcessQNX::Detach() {
  Status error = m_file_up.get()->Close();

  if (error.Fail())
    return error;

  m_fd = -1;

  return error;
}

Status NativeProcessQNX::Signal(int signo) {
  procfs_signal sig;

  sig.tid = 0;
  sig.signo = signo;
  sig.code = 0;
  sig.value = 0;

  return DevctlWrapper(m_fd, DCMD_PROC_SIGNAL, &sig, sizeof(procfs_signal),
                       nullptr);
}

Status NativeProcessQNX::Interrupt() { return Halt(); }

Status NativeProcessQNX::Kill() {
  Log *log = GetLog(POSIXLog::Process);

  Status error;

  switch (m_state) {
  case StateType::eStateInvalid:
  case StateType::eStateExited:
  case StateType::eStateCrashed:
  case StateType::eStateDetached:
  case StateType::eStateUnloaded:
    // Nothing to do; the process is already dead.
    LLDB_LOG(log, "Current state of process {0}: {1}; ignoring", GetID(),
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

  error = Signal(SIGKILL);
  if (error.Fail())
    return error;

  // Resume the debuggee.
  procfs_run proc_run;
  memset(&proc_run, 0, sizeof(procfs_run));
  proc_run.flags =
      _DEBUG_RUN_CLRSIG | _DEBUG_RUN_TRACE | _DEBUG_RUN_ARM | _DEBUG_RUN_THREAD;
  sigfillset(&(proc_run.trace));
  sigdelset(&(proc_run.trace), SIGKILL);

  return DevctlWrapper(m_fd, DCMD_PROC_RUN, &proc_run, sizeof(procfs_run),
                       nullptr);
}

Status NativeProcessQNX::GetMemoryRegionInfo(lldb::addr_t load_addr,
                                             MemoryRegionInfo &range_info) {
  Status error;

  if (m_supports_mem_region == LazyBool::eLazyBoolNo) {
    // We're done.
    error.SetErrorString("unsupported");
    return error;
  }

  error = PopulateMemoryRegionCache();
  if (error.Fail())
    return error;

  lldb::addr_t prev_base_address = 0;
  // TODO: There can be a ton of memory regions in case of numerous threads. The
  // memory regions are expected to be sorted, though. So, identify the last
  // region that is <= target memory address via binary search.
  for (auto it = m_mem_region_cache.begin(); it != m_mem_region_cache.end();
       ++it) {
    MemoryRegionInfo &proc_entry_info = it->first;
    // Sanity check the assumption that memory map entries are ascending.
    if (proc_entry_info.GetRange().GetRangeBase() < prev_base_address) {
      error.SetErrorString("unexpectedly detected descending memory map "
                           "entries");
      return error;
    }
    prev_base_address = proc_entry_info.GetRange().GetRangeBase();
    // If the target memory address comes before this entry, then indicate
    // distance to the next region.
    if (load_addr < proc_entry_info.GetRange().GetRangeBase()) {
      range_info.GetRange().SetRangeBase(load_addr);
      range_info.GetRange().SetByteSize(
          proc_entry_info.GetRange().GetRangeBase() - load_addr);
      range_info.SetReadable(MemoryRegionInfo::OptionalBool::eNo);
      range_info.SetWritable(MemoryRegionInfo::OptionalBool::eNo);
      range_info.SetExecutable(MemoryRegionInfo::OptionalBool::eNo);
      range_info.SetMapped(MemoryRegionInfo::OptionalBool::eNo);
      return error;
    } else if (proc_entry_info.GetRange().Contains(load_addr)) {
      // The target memory address lies within the memory region that we're
      // processing here.
      range_info = proc_entry_info;
      return error;
    }
    // The target memory address lies beyond the region that we've just parsed.
  }
  // If we've made it so far, then we didn't find any entry that contained the
  // given address. So, set the load_addr as the base address and the amount of
  // bytes between the load address and the end of the memory as the size.
  range_info.GetRange().SetRangeBase(load_addr);
  range_info.GetRange().SetRangeEnd(LLDB_INVALID_ADDRESS);
  range_info.SetReadable(MemoryRegionInfo::OptionalBool::eNo);
  range_info.SetWritable(MemoryRegionInfo::OptionalBool::eNo);
  range_info.SetExecutable(MemoryRegionInfo::OptionalBool::eNo);
  range_info.SetMapped(MemoryRegionInfo::OptionalBool::eNo);
  return error;
}

Status NativeProcessQNX::ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                                    size_t &bytes_read) {
  Log *log = GetLog(POSIXLog::Memory);

  LLDB_LOG(log, "addr = {0}, buf = {1}, size = {2}", addr, buf, size);

  bytes_read = size;
  off_t offset = static_cast<off_t>(addr);

  return m_file_up->Read(buf, bytes_read, offset);
}

Status NativeProcessQNX::WriteMemory(lldb::addr_t addr, const void *buf,
                                     size_t size, size_t &bytes_written) {
  Log *log = GetLog(POSIXLog::Memory);

  LLDB_LOG(log, "addr = {0}, buf = {1}, size = {2}", addr, buf, size);

  bytes_written = size;
  off_t offset = static_cast<off_t>(addr);

  // TODO: Determine whether or not we need to lock m_file.
  return m_file_up->Write(buf, bytes_written, offset);
}

size_t NativeProcessQNX::UpdateThreads() {
  // The list of a debuggee's threads, and its threads' states are expected to
  // always be up to date. So, just return the thread count.
  return m_threads.size();
}

Status NativeProcessQNX::SetBreakpoint(lldb::addr_t addr, uint32_t size,
                                       bool hardware) {
  // TODO: Provision support for setting hardware breakpoints.

  (void)hardware;

  procfs_break bkpt;
  memset(&bkpt, 0x0, sizeof(procfs_break));
  bkpt.type = _DEBUG_BREAK_EXEC;
  bkpt.addr = static_cast<_Uintptr64t>(addr);
  bkpt.size = 0;

  return DevctlWrapper(m_fd, DCMD_PROC_BREAK, &bkpt, sizeof(procfs_break),
                       nullptr);
}

Status NativeProcessQNX::RemoveBreakpoint(lldb::addr_t addr, bool hardware) {
  // TODO: Provision support for removing hardware breakpoints.

  (void)hardware;

  procfs_break bkpt;
  memset(&bkpt, 0x0, sizeof(procfs_break));
  bkpt.type = _DEBUG_BREAK_EXEC;
  bkpt.addr = static_cast<_Uintptr64t>(addr);
  bkpt.size = -1;

  return DevctlWrapper(m_fd, DCMD_PROC_BREAK, &bkpt, sizeof(procfs_break),
                       nullptr);
}

Status NativeProcessQNX::GetLoadedModuleFileSpec(const char *module_path,
                                                 FileSpec &file_spec) {
  return Status("not implemented");
}

Status NativeProcessQNX::GetFileLoadAddress(const llvm::StringRef &file_name,
                                            lldb::addr_t &load_addr) {
  return Status("not implemented");
}

Status NativeProcessQNX::DevctlWrapper(int fd, int dcmd, void *dev_data_ptr,
                                       size_t n_bytes, int *dev_info_ptr) {
  Log *log = GetLog(POSIXLog::Ptrace);
  Status error;
  int ret;

  ret = devctl(fd, dcmd, static_cast<caddr_t>(dev_data_ptr),
               static_cast<size_t>(n_bytes), static_cast<int *>(dev_info_ptr));

  if (ret != EOK)
    error.SetError(ret, lldb::eErrorTypePOSIX);

  LLDB_LOG(log, "devctl({0}, {1}, {2}, {3}, {4}) = {5:x}", fd, dcmd,
           dev_data_ptr, n_bytes, dev_info_ptr, ret);

  if (error.Fail())
    LLDB_LOG(log, "devctl failed: {0}", error);

  return error;
}

NativeProcessQNX::NativeProcessQNX(::pid_t pid, int terminal_fd,
                                   NativeDelegate &delegate, Manager &manager,
                                   const ArchSpec &arch, Status &error)
    : NativeProcessELF(pid, terminal_fd, delegate), m_manager(manager),
      m_arch(arch), m_file_up(nullptr), m_fd(-1) {
  manager.AddProcess(*this);

  if (m_terminal_fd != -1) {
    error = EnsureFDFlags(m_terminal_fd, O_NONBLOCK);
    if (!error.Success()) {
      return;
    }
  }

  llvm::Expected<FileUP> file_up =
      openProcFile(m_pid, "as", File::eOpenOptionReadWrite);

  if (!file_up) {
    error = file_up.takeError();
    return;
  }

  m_file_up = std::move(file_up.get());
  m_fd = m_file_up.get()->GetDescriptor();
}

NativeThreadQNX &NativeProcessQNX::AddThread(lldb::tid_t thread_id) {
  Log *log = GetLog(POSIXLog::Thread);

  LLDB_LOG(log, "Adding thread {0} for process {1}", thread_id, GetID());

  NativeThreadQNX *thread =
      static_cast<NativeThreadQNX *>(GetThreadByIDUnlocked(thread_id));

  if (!thread) {
    // If this is the first thread, then save it as the current thread.
    if (m_threads.empty())
      SetCurrentThreadID(thread_id);

    m_threads.push_back(std::make_unique<NativeThreadQNX>(*this, thread_id));
    return static_cast<NativeThreadQNX &>(*m_threads.back());
  }

  return static_cast<NativeThreadQNX &>(*thread);
}

void NativeProcessQNX::RemoveThread(lldb::tid_t thread_id) {
  Log *log = GetLog(POSIXLog::Thread);

  LLDB_LOG(log, "Removing thread {0} for process {1}", thread_id, GetID());

  if (!GetThreadByIDUnlocked(thread_id)) {
    LLDB_LOG(log, "Attempting to remove a thread that doesn't exist");
    return;
  }

  for (auto it = m_threads.begin(); it != m_threads.end(); ++it) {
    if ((*it)->GetID() == thread_id) {
      m_threads.erase(it);
      break;
    }
  }

  if (GetCurrentThreadID() == thread_id)
    SetCurrentThreadID(m_threads.front()->GetID());
}

void NativeProcessQNX::MonitorInterrupt() {
  // Interrupt all threads attached to the process.
  for (const auto &thread : m_threads)
    static_cast<NativeThreadQNX &>(*thread).SetStoppedBySignal(SIGINT, nullptr);

  SetState(StateType::eStateStopped, true);
}

void NativeProcessQNX::MonitorCallback(procfs_status &proc_status) {
  switch (proc_status.what) {
  case SIGTRAP:
    MonitorSIGTRAP(proc_status);
  case SIGSTOP:
    MonitorSIGSTOP();
  default:
    MonitorSignal(proc_status);
  }
}

void NativeProcessQNX::MonitorSIGTRAP(procfs_status &proc_status) {
  Log *log = GetLog(POSIXLog::Process);

  NativeThreadQNX *thread = nullptr;
  lldb::tid_t tid = static_cast<lldb::tid_t>(proc_status.tid);

  for (const auto &t : m_threads) {
    if (t->GetID() == tid)
      thread = static_cast<NativeThreadQNX *>(t.get());
    static_cast<NativeThreadQNX *>(t.get())->SetStoppedWithNoReason();
  }

  if (!thread)
    LLDB_LOG(log, "Couldn't find thread {0} for pid {1}", tid, GetID());

  switch (proc_status.info.si_code) {
  case TRAP_BRKPT:
    if (thread) {
      thread->SetStoppedByBreakpoint();
      FixupBreakpointPCAsNeeded(*thread);
      SetCurrentThreadID(thread->GetID());
    }

    SetState(StateType::eStateStopped, true);
    return;

  case TRAP_TRACE:
    if (thread) {
      thread->SetStoppedByTrace();
      SetCurrentThreadID(thread->GetID());
    }

    SetState(StateType::eStateStopped, true);
    return;

  default:
    break;
  }

  // Encounterd either a user-generated SIGTRAP or an unknown event that would
  // otherwise leave the debugger hanging.
  LLDB_LOG(log, "Unknown SIGTRAP; passing to generic handler");
  MonitorSignal(proc_status);
}

void NativeProcessQNX::MonitorSIGSTOP() {
  // Stop all threads attached to the process.
  for (const auto &thread : m_threads)
    static_cast<NativeThreadQNX &>(*thread).SetStoppedBySignal(SIGSTOP,
                                                               nullptr);

  SetState(StateType::eStateStopped, true);
}

void NativeProcessQNX::MonitorSignal(procfs_status &proc_status) {
  Log *log = GetLog(POSIXLog::Process);

  int signo = proc_status.what;
  LLDB_LOG(log, "Received signal {0} ({1})", Host::GetSignalAsCString(signo),
           signo);

  // Check if the debugger should just ignore this signal and resume the
  // debuggee.
  if (m_signals_to_ignore.contains(signo)) {
    LLDB_LOG(log, "Ignoring signal {0} ({1})", Host::GetSignalAsCString(signo),
             signo);

    procfs_run proc_run;
    memset(&proc_run, 0, sizeof(procfs_run));
    proc_run.flags = _DEBUG_RUN_CLRSIG | _DEBUG_RUN_TRACE | _DEBUG_RUN_ARM |
                     _DEBUG_RUN_THREAD;
    sigfillset(&(proc_run.trace));

    Status error = DevctlWrapper(m_fd, DCMD_PROC_RUN, &proc_run,
                                 sizeof(procfs_run), nullptr);

    if (error.Fail())
      SetState(StateType::eStateInvalid);

    return;
  }

  // Stop all threads attached to the process.
  for (const auto &thread : m_threads)
    static_cast<NativeThreadQNX &>(*thread).SetStoppedBySignal(signo, nullptr);

  SetState(StateType::eStateStopped, true);
}

void NativeProcessQNX::MonitorExited(procfs_status &proc_status) {
  Log *log = GetLog(POSIXLog::Process);

  WaitStatus status(WaitStatus::Exit, proc_status.what);

  LLDB_LOG(log, "Got exit signal({0}) , pid = {1}", status, proc_status.pid);

  // Stop tracking all threads attached to the process.
  m_threads.clear();

  SetExitStatus(status, true);

  // Notify delegate that our process has exited.
  SetState(StateType::eStateExited, true);
}

void NativeProcessQNX::MonitorThread(procfs_status &proc_status) {
  switch (proc_status.what) {
  case _DEBUG_WHAT_DESTROYED:
    RemoveThread(proc_status.blocked.thread_event.tid);
    break;
  case _DEBUG_WHAT_CREATED:
    AddThread(proc_status.blocked.thread_event.tid);
    break;
  }

  // Resume the debuggee.
  procfs_run proc_run;
  memset(&proc_run, 0, sizeof(procfs_run));
  proc_run.flags =
      _DEBUG_RUN_CLRSIG | _DEBUG_RUN_TRACE | _DEBUG_RUN_ARM | _DEBUG_RUN_THREAD;
  sigfillset(&(proc_run.trace));

  Status error = DevctlWrapper(m_fd, DCMD_PROC_RUN, &proc_run,
                               sizeof(procfs_run), nullptr);

  if (error.Fail())
    SetState(StateType::eStateInvalid);
}

Status NativeProcessQNX::PopulateMemoryRegionCache() {
  Log *log = GetLog(POSIXLog::Process);

  Status error;

  if (!m_mem_region_cache.empty()) {
    LLDB_LOG(log, "Reusing {0} cached memory region entries",
             m_mem_region_cache.size());
    return error;
  }

  // If our cache is empty, then pull the latest memory regions. There should
  // always be at least one memory region if memory region handling is
  // supported.

  int proc_map_ents;
  error = DevctlWrapper(m_fd, DCMD_PROC_MAPINFO, NULL, 0, &proc_map_ents);

  if (error.Fail())
    return error;

  procfs_mapinfo *proc_map_info =
      (procfs_mapinfo *)malloc(sizeof(procfs_mapinfo) * proc_map_ents);

  if (!proc_map_info) {
    error.SetErrorToErrno();
    return error;
  }

  error = DevctlWrapper(m_fd, DCMD_PROC_MAPINFO, proc_map_info,
                        sizeof(procfs_mapinfo) * proc_map_ents, &proc_map_ents);

  if (error.Fail())
    return error;

  for (int idx = 0; idx < proc_map_ents; idx++) {
    MemoryRegionInfo info;
    info.Clear();

    info.GetRange().SetRangeBase(proc_map_info[idx].vaddr);
    info.GetRange().SetRangeEnd(proc_map_info[idx].vaddr +
                                proc_map_info[idx].size);
    info.SetMapped(MemoryRegionInfo::OptionalBool::eYes);

    if (proc_map_info[idx].flags & PROT_READ)
      info.SetReadable(MemoryRegionInfo::OptionalBool::eYes);
    else
      info.SetReadable(MemoryRegionInfo::OptionalBool::eNo);

    if (proc_map_info[idx].flags & PROT_WRITE)
      info.SetWritable(MemoryRegionInfo::OptionalBool::eYes);
    else
      info.SetWritable(MemoryRegionInfo::OptionalBool::eNo);

    if (proc_map_info[idx].flags & PROT_EXEC)
      info.SetExecutable(MemoryRegionInfo::OptionalBool::eYes);
    else
      info.SetExecutable(MemoryRegionInfo::OptionalBool::eNo);

    // 'path' in 'procfs_debuginfo' is a one-byte array. If you want to get the
    // path, then you need to allocate more space for it.
    struct {
      procfs_debuginfo debug_info;
      char buffer[_POSIX_PATH_MAX];
    } map;

    map.debug_info.vaddr = proc_map_info[idx].vaddr;
    error = DevctlWrapper(m_fd, DCMD_PROC_MAPDEBUG, &map, sizeof(map), NULL);

    if (error.Fail())
      return error;

    if (map.debug_info.path)
      info.SetName(map.debug_info.path);

    m_mem_region_cache.emplace_back(info,
                                    FileSpec(info.GetName().GetCString()));
  }

  if (m_mem_region_cache.empty()) {
    // Couldn't find any entries. This shouldn't happen. Assume that we don't
    // support map entries.
    LLDB_LOG(log, "Failed to find any vmmap entries, assuming no support "
                  "for memory region metadata retrieval");
    m_supports_mem_region = LazyBool::eLazyBoolNo;
    error.SetErrorString("Not supported");
    return error;
  }

  LLDB_LOG(log, "Read {0} memory region entries for process {1}",
           m_mem_region_cache.size(), GetID());
  // We support memory retrieval; remember that.
  m_supports_mem_region = LazyBool::eLazyBoolYes;

  return error;
}

Status NativeProcessQNX::Attach() {
  // Attach to the requested process. An attach will cause the process to stop
  // with a SIGSTOP.
  procfs_status proc_status;
  Status error = DevctlWrapper(m_fd, DCMD_PROC_STOP, &proc_status,
                               sizeof(proc_status), nullptr);
  if (error.Fail())
    return error;

  error = DevctlWrapper(m_fd, DCMD_PROC_WAITSTOP, &proc_status,
                        sizeof(proc_status), nullptr);
  if (error.Fail())
    return error;

  // Initialize threads and tracing status.
  // NB: This needs to be called before we set the threads' states.
  error = SetupTrace();
  if (error.Fail())
    return error;

  for (const auto &thread : m_threads)
    static_cast<NativeThreadQNX &>(*thread).SetStoppedBySignal(SIGSTOP);

  // Let our process instance know that the thread has stopped.
  SetCurrentThreadID(m_threads.front()->GetID());
  SetState(StateType::eStateStopped, false);
  return error;
}

llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
NativeProcessQNX::GetAuxvData() const {
  // This is what a process's stack looks like after initialization on QNX -
  //
  //          |________________|
  //          |                |
  // addr ->  |      argc      |  _Uint64t      (Number of arguments)
  //          |________________|
  //          |                |
  //          |    argv[0]     |  _Uint64t*     (Program Name)
  //          |________________|
  //          |                |
  //          |    argv[1]     |  _Uint64t*
  //          |________________|
  //          |                |
  //          |    argv[..]    |  _Uint64t*
  //          |________________|
  //          |                |
  //          | argv[argc - 1] |  _Uint64t*
  //          |________________|
  //          |                |
  //          |   argv[argc]   |  _Uint64t*     (NULL)
  //          |________________|
  //          |                |
  //          |    envp[0]     |  _Uint64t*
  //          |________________|
  //          |                |
  //          |    envp[1]     |  _Uint64t*
  //          |________________|
  //          |                |
  //          |    envp[..]    |  _Uint64t*
  //          |________________|
  //          |                |
  //          |   envp[term]   |  _Uint64t*     (NULL)
  //          |________________|
  //          |                |
  //          |    auxv[0]     |  auxv_t
  //          |________________|
  //          |                |
  //          |    auxv[1]     |  auxv_t
  //          |________________|
  //          |                |
  //          |    auxv[..]    |  auxv_t
  //          |________________|
  //          |                |
  //          |   auxv[term]   |  auxv_t        (AT_NULL)
  //          |________________|
  //          |                |
  //
  // So, seek past argc, argv, and envp to read auxv.

  procfs_info proc_info;
  Status error = DevctlWrapper(m_fd, DCMD_PROC_INFO, &proc_info,
                               sizeof(proc_info), nullptr);
  if (error.Fail())
    return std::error_code(error.GetError(), std::generic_category());

  // Get a pointer to the initial stack.
  _Uint8t *addr = reinterpret_cast<_Uint8t *>(proc_info.initial_stack);

  // Read argc, and seek past it.
  _Uint64t argc;
  uint32_t addr_size = GetAddressByteSize();
  size_t bytes_read;

  error = ReadMemory(reinterpret_cast<lldb::addr_t>(addr),
                     reinterpret_cast<void *>(&argc), sizeof(argc), bytes_read);

  if (error.Fail())
    return std::error_code(error.GetError(), std::generic_category());

  if (bytes_read != sizeof(argc))
    return errorToErrorCode(llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Attempted to read {0} bytes at {1:x} but only read {2}", sizeof(argc),
        addr, bytes_read));

  addr += addr_size;

  // Seek past argv.
  addr += ((argc + 1) * addr_size);

  // Seek past envp.
  _Uint64t envp;
  do {
    error =
        ReadMemory(reinterpret_cast<lldb::addr_t>(addr),
                   reinterpret_cast<void *>(&envp), sizeof(envp), bytes_read);

    if (error.Fail())
      return std::error_code(error.GetError(), std::generic_category());

    if (bytes_read != sizeof(envp))
      return errorToErrorCode(llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "Attempted to read {0} bytes at {1:x} but only read {2}",
          sizeof(envp), addr, bytes_read));

    addr += addr_size;
  } while (envp != 0);

  // Read auxv.
  auxv_t entry;
  DataBufferHeap auxv_data(static_cast<lldb::offset_t>(0),
                           static_cast<uint8_t>(0));
  do {
    error =
        ReadMemory(reinterpret_cast<lldb::addr_t>(addr),
                   reinterpret_cast<void *>(&entry), sizeof(entry), bytes_read);

    if (error.Fail())
      return std::error_code(error.GetError(), std::generic_category());

    if (bytes_read != sizeof(entry))
      return errorToErrorCode(llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "Attempted to read {0} bytes at {1:x} but only read {2}",
          sizeof(entry), addr, bytes_read));

    auxv_data.AppendData(&entry, bytes_read);
    addr += sizeof(entry);
  } while (entry.a_type != AT_NULL);

  std::unique_ptr<WritableMemoryBuffer> auxv =
      llvm::WritableMemoryBuffer::getNewMemBuffer(auxv_data.GetByteSize());

  std::memcpy(auxv->getBufferStart(), auxv_data.GetBytes(),
              auxv->getBufferSize());

  return auxv;
}

Status NativeProcessQNX::SetupTrace() {
  // The debugger no longer receives SIGCHLD from the debuggee upon establishing
  // a file descriptor to /proc/$pid/as. So, have the server deliver a SIGCHLD
  // to the debugger whenever the debuggee reaches a point of interest. Also,
  // have siginfo_t.si_value.sival_int set to the debuggee's PID so that we can
  // identify the debuggee in NativeProcessQNX::Manager::SigchldHandler.
  struct sigevent event;
  Status error;
  SIGEV_SIGNAL_VALUE_INT_INIT(&event, SIGCHLD, m_pid);
  if (MsgRegisterEvent(&event, m_fd) != EOK) {
    error.SetErrorToErrno();
    return error;
  }
  error = DevctlWrapper(m_fd, DCMD_PROC_EVENT, &event, sizeof(struct sigevent),
                        nullptr);
  if (error.Fail())
    return error;

  // Let the debuggee run on detach (closure of last file descriptor to
  // /proc/$pid/as).
  int flags = _DEBUG_FLAG_RLC;
  error =
      DevctlWrapper(m_fd, DCMD_PROC_CLEAR_FLAG, &flags, sizeof(flags), nullptr);
  if (error.Fail())
    return error;

  // TODO: Indicate interest in getting notified when the debuggee forks,
  // vforks, spawns, or execs.

  // The child process is held as if it had received a SIGSTOP as soon as it had
  // been spawned. To resume this process, we must send it a SIGCONT. However,
  // we don't want it to resume execution here.
  Halt();
  Signal(SIGCONT);

  return ReinitializeThreads();
}

Status NativeProcessQNX::ReinitializeThreads() {
  // Clear old threads.
  m_threads.clear();

  procfs_info proc_info;
  Status error = DevctlWrapper(m_fd, DCMD_PROC_INFO, &proc_info,
                               sizeof(proc_info), nullptr);
  if (error.Fail())
    return error;

  _Uint32t num_threads = proc_info.num_threads;

  // QNX returns information about the thread specified in the tid member of
  // status if it has it; otherwise, it will either return information on the
  // next available thread ID, or return something other than EOK to indicate
  // that it's done.

  procfs_status proc_status;
  proc_status.tid = 0;

  std::vector<pthread_t> tids;

  for (int idx = 0; idx < num_threads; ++idx) {
    // QNX starts numbering threads from 1.
    proc_status.tid++;

    error = DevctlWrapper(m_fd, DCMD_PROC_TIDSTATUS, &proc_status,
                          sizeof(proc_status), nullptr);

    if (error.Fail())
      return error;

    tids.push_back(proc_status.tid);
  }

  // Reinitialize threads from scratch and register them with the process.
  for (pthread_t tid : tids) {
    if (tid < 1) {
      error.SetErrorString("tid < 1");
      return error;
    }
    AddThread(tid);
  }

  return error;
}

Status NativeProcessQNX::ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                                    size_t &bytes_read) const {
  Log *log = GetLog(POSIXLog::Memory);

  LLDB_LOG(log, "addr = {0}, buf = {1}, size = {2}", addr, buf, size);

  bytes_read = size;
  off_t offset = static_cast<off_t>(addr);

  return m_file_up->Read(buf, bytes_read, offset);
}
