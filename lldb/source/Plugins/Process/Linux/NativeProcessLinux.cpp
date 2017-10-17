//===-- NativeProcessLinux.cpp -------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NativeProcessLinux.h"

// C Includes
#include <errno.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

// C++ Includes
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

// Other libraries and framework includes
#include "lldb/Core/EmulateInstruction.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/State.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostProcess.h"
#include "lldb/Host/PseudoTerminal.h"
#include "lldb/Host/ThreadLauncher.h"
#include "lldb/Host/common/NativeBreakpoint.h"
#include "lldb/Host/common/NativeRegisterContext.h"
#include "lldb/Host/linux/Ptrace.h"
#include "lldb/Host/linux/Uio.h"
#include "lldb/Host/posix/ProcessLauncherPosixFork.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/ProcessLaunchInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/StringExtractor.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Threading.h"

#include "NativeThreadLinux.h"
#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"
#include "Procfs.h"

#include <linux/unistd.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/user.h>
#include <sys/wait.h>

// Support hardware breakpoints in case it has not been defined
#ifndef TRAP_HWBKPT
#define TRAP_HWBKPT 4
#endif

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_linux;
using namespace llvm;

// Private bits we only need internally.

static bool ProcessVmReadvSupported() {
  static bool is_supported;
  static llvm::once_flag flag;

  llvm::call_once(flag, [] {
    Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));

    uint32_t source = 0x47424742;
    uint32_t dest = 0;

    struct iovec local, remote;
    remote.iov_base = &source;
    local.iov_base = &dest;
    remote.iov_len = local.iov_len = sizeof source;

    // We shall try if cross-process-memory reads work by attempting to read a
    // value from our own process.
    ssize_t res = process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
    is_supported = (res == sizeof(source) && source == dest);
    if (is_supported)
      LLDB_LOG(log,
               "Detected kernel support for process_vm_readv syscall. "
               "Fast memory reads enabled.");
    else
      LLDB_LOG(log,
               "syscall process_vm_readv failed (error: {0}). Fast memory "
               "reads disabled.",
               llvm::sys::StrError());
  });

  return is_supported;
}

namespace {
void MaybeLogLaunchInfo(const ProcessLaunchInfo &info) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));
  if (!log)
    return;

  if (const FileAction *action = info.GetFileActionForFD(STDIN_FILENO))
    LLDB_LOG(log, "setting STDIN to '{0}'", action->GetFileSpec());
  else
    LLDB_LOG(log, "leaving STDIN as is");

  if (const FileAction *action = info.GetFileActionForFD(STDOUT_FILENO))
    LLDB_LOG(log, "setting STDOUT to '{0}'", action->GetFileSpec());
  else
    LLDB_LOG(log, "leaving STDOUT as is");

  if (const FileAction *action = info.GetFileActionForFD(STDERR_FILENO))
    LLDB_LOG(log, "setting STDERR to '{0}'", action->GetFileSpec());
  else
    LLDB_LOG(log, "leaving STDERR as is");

  int i = 0;
  for (const char **args = info.GetArguments().GetConstArgumentVector(); *args;
       ++args, ++i)
    LLDB_LOG(log, "arg {0}: '{1}'", i, *args);
}

void DisplayBytes(StreamString &s, void *bytes, uint32_t count) {
  uint8_t *ptr = (uint8_t *)bytes;
  const uint32_t loop_count = std::min<uint32_t>(DEBUG_PTRACE_MAXBYTES, count);
  for (uint32_t i = 0; i < loop_count; i++) {
    s.Printf("[%x]", *ptr);
    ptr++;
  }
}

void PtraceDisplayBytes(int &req, void *data, size_t data_size) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PTRACE));
  if (!log)
    return;
  StreamString buf;

  switch (req) {
  case PTRACE_POKETEXT: {
    DisplayBytes(buf, &data, 8);
    LLDB_LOGV(log, "PTRACE_POKETEXT {0}", buf.GetData());
    break;
  }
  case PTRACE_POKEDATA: {
    DisplayBytes(buf, &data, 8);
    LLDB_LOGV(log, "PTRACE_POKEDATA {0}", buf.GetData());
    break;
  }
  case PTRACE_POKEUSER: {
    DisplayBytes(buf, &data, 8);
    LLDB_LOGV(log, "PTRACE_POKEUSER {0}", buf.GetData());
    break;
  }
  case PTRACE_SETREGS: {
    DisplayBytes(buf, data, data_size);
    LLDB_LOGV(log, "PTRACE_SETREGS {0}", buf.GetData());
    break;
  }
  case PTRACE_SETFPREGS: {
    DisplayBytes(buf, data, data_size);
    LLDB_LOGV(log, "PTRACE_SETFPREGS {0}", buf.GetData());
    break;
  }
  case PTRACE_SETSIGINFO: {
    DisplayBytes(buf, data, sizeof(siginfo_t));
    LLDB_LOGV(log, "PTRACE_SETSIGINFO {0}", buf.GetData());
    break;
  }
  case PTRACE_SETREGSET: {
    // Extract iov_base from data, which is a pointer to the struct IOVEC
    DisplayBytes(buf, *(void **)data, data_size);
    LLDB_LOGV(log, "PTRACE_SETREGSET {0}", buf.GetData());
    break;
  }
  default: {}
  }
}

static constexpr unsigned k_ptrace_word_size = sizeof(void *);
static_assert(sizeof(long) >= k_ptrace_word_size,
              "Size of long must be larger than ptrace word size");
} // end of anonymous namespace

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

// -----------------------------------------------------------------------------
// Public Static Methods
// -----------------------------------------------------------------------------

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
NativeProcessLinux::Factory::Launch(ProcessLaunchInfo &launch_info,
                                    NativeDelegate &native_delegate,
                                    MainLoop &mainloop) const {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));

  MaybeLogLaunchInfo(launch_info);

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
  (void)wpid;
  if (!WIFSTOPPED(wstatus)) {
    LLDB_LOG(log, "Could not sync with inferior process: wstatus={1}",
             WaitStatus::Decode(wstatus));
    return llvm::make_error<StringError>("Could not sync with inferior process",
                                         llvm::inconvertibleErrorCode());
  }
  LLDB_LOG(log, "inferior started, now in stopped state");

  ArchSpec arch;
  if ((status = ResolveProcessArchitecture(pid, arch)).Fail())
    return status.ToError();

  // Set the architecture to the exe architecture.
  LLDB_LOG(log, "pid = {0:x}, detected architecture {1}", pid,
           arch.GetArchitectureName());

  status = SetDefaultPtraceOpts(pid);
  if (status.Fail()) {
    LLDB_LOG(log, "failed to set default ptrace options: {0}", status);
    return status.ToError();
  }

  return std::unique_ptr<NativeProcessLinux>(new NativeProcessLinux(
      pid, launch_info.GetPTY().ReleaseMasterFileDescriptor(), native_delegate,
      arch, mainloop, {pid}));
}

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
NativeProcessLinux::Factory::Attach(
    lldb::pid_t pid, NativeProcessProtocol::NativeDelegate &native_delegate,
    MainLoop &mainloop) const {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));
  LLDB_LOG(log, "pid = {0:x}", pid);

  // Retrieve the architecture for the running process.
  ArchSpec arch;
  Status status = ResolveProcessArchitecture(pid, arch);
  if (!status.Success())
    return status.ToError();

  auto tids_or = NativeProcessLinux::Attach(pid);
  if (!tids_or)
    return tids_or.takeError();

  return std::unique_ptr<NativeProcessLinux>(new NativeProcessLinux(
      pid, -1, native_delegate, arch, mainloop, *tids_or));
}

// -----------------------------------------------------------------------------
// Public Instance Methods
// -----------------------------------------------------------------------------

NativeProcessLinux::NativeProcessLinux(::pid_t pid, int terminal_fd,
                                       NativeDelegate &delegate,
                                       const ArchSpec &arch, MainLoop &mainloop,
                                       llvm::ArrayRef<::pid_t> tids)
    : NativeProcessProtocol(pid, terminal_fd, delegate), m_arch(arch) {
  if (m_terminal_fd != -1) {
    Status status = EnsureFDFlags(m_terminal_fd, O_NONBLOCK);
    assert(status.Success());
  }

  Status status;
  m_sigchld_handle = mainloop.RegisterSignal(
      SIGCHLD, [this](MainLoopBase &) { SigchldHandler(); }, status);
  assert(m_sigchld_handle && status.Success());

  for (const auto &tid : tids) {
    NativeThreadLinux &thread = AddThread(tid);
    thread.SetStoppedBySignal(SIGSTOP);
    ThreadWasCreated(thread);
  }

  // Let our process instance know the thread has stopped.
  SetCurrentThreadID(tids[0]);
  SetState(StateType::eStateStopped, false);

  // Proccess any signals we received before installing our handler
  SigchldHandler();
}

llvm::Expected<std::vector<::pid_t>> NativeProcessLinux::Attach(::pid_t pid) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));

  Status status;
  // Use a map to keep track of the threads which we have attached/need to
  // attach.
  Host::TidMap tids_to_attach;
  while (Host::FindProcessThreads(pid, tids_to_attach)) {
    for (Host::TidMap::iterator it = tids_to_attach.begin();
         it != tids_to_attach.end();) {
      if (it->second == false) {
        lldb::tid_t tid = it->first;

        // Attach to the requested process.
        // An attach will cause the thread to stop with a SIGSTOP.
        if ((status = PtraceWrapper(PTRACE_ATTACH, tid)).Fail()) {
          // No such thread. The thread may have exited.
          // More error handling may be needed.
          if (status.GetError() == ESRCH) {
            it = tids_to_attach.erase(it);
            continue;
          }
          return status.ToError();
        }

        int wpid =
            llvm::sys::RetryAfterSignal(-1, ::waitpid, tid, nullptr, __WALL);
        // Need to use __WALL otherwise we receive an error with errno=ECHLD
        // At this point we should have a thread stopped if waitpid succeeds.
        if (wpid < 0) {
          // No such thread. The thread may have exited.
          // More error handling may be needed.
          if (errno == ESRCH) {
            it = tids_to_attach.erase(it);
            continue;
          }
          return llvm::errorCodeToError(
              std::error_code(errno, std::generic_category()));
        }

        if ((status = SetDefaultPtraceOpts(tid)).Fail())
          return status.ToError();

        LLDB_LOG(log, "adding tid = {0}", tid);
        it->second = true;
      }

      // move the loop forward
      ++it;
    }
  }

  size_t tid_count = tids_to_attach.size();
  if (tid_count == 0)
    return llvm::make_error<StringError>("No such process",
                                         llvm::inconvertibleErrorCode());

  std::vector<::pid_t> tids;
  tids.reserve(tid_count);
  for (const auto &p : tids_to_attach)
    tids.push_back(p.first);
  return std::move(tids);
}

Status NativeProcessLinux::SetDefaultPtraceOpts(lldb::pid_t pid) {
  long ptrace_opts = 0;

  // Have the child raise an event on exit.  This is used to keep the child in
  // limbo until it is destroyed.
  ptrace_opts |= PTRACE_O_TRACEEXIT;

  // Have the tracer trace threads which spawn in the inferior process.
  // TODO: if we want to support tracing the inferiors' child, add the
  // appropriate ptrace flags here (PTRACE_O_TRACEFORK, PTRACE_O_TRACEVFORK)
  ptrace_opts |= PTRACE_O_TRACECLONE;

  // Have the tracer notify us before execve returns
  // (needed to disable legacy SIGTRAP generation)
  ptrace_opts |= PTRACE_O_TRACEEXEC;

  return PtraceWrapper(PTRACE_SETOPTIONS, pid, nullptr, (void *)ptrace_opts);
}

// Handles all waitpid events from the inferior process.
void NativeProcessLinux::MonitorCallback(lldb::pid_t pid, bool exited,
                                         WaitStatus status) {
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS));

  // Certain activities differ based on whether the pid is the tid of the main
  // thread.
  const bool is_main_thread = (pid == GetID());

  // Handle when the thread exits.
  if (exited) {
    LLDB_LOG(log, "got exit signal({0}) , tid = {1} ({2} main thread)", signal,
             pid, is_main_thread ? "is" : "is not");

    // This is a thread that exited.  Ensure we're not tracking it anymore.
    const bool thread_found = StopTrackingThread(pid);

    if (is_main_thread) {
      // We only set the exit status and notify the delegate if we haven't
      // already set the process
      // state to an exited state.  We normally should have received a SIGTRAP |
      // (PTRACE_EVENT_EXIT << 8)
      // for the main thread.
      const bool already_notified = (GetState() == StateType::eStateExited) ||
                                    (GetState() == StateType::eStateCrashed);
      if (!already_notified) {
        LLDB_LOG(
            log,
            "tid = {0} handling main thread exit ({1}), expected exit state "
            "already set but state was {2} instead, setting exit state now",
            pid,
            thread_found ? "stopped tracking thread metadata"
                         : "thread metadata not found",
            GetState());
        // The main thread exited.  We're done monitoring.  Report to delegate.
        SetExitStatus(status, true);

        // Notify delegate that our process has exited.
        SetState(StateType::eStateExited, true);
      } else
        LLDB_LOG(log, "tid = {0} main thread now exited (%s)", pid,
                 thread_found ? "stopped tracking thread metadata"
                              : "thread metadata not found");
    } else {
      // Do we want to report to the delegate in this case?  I think not.  If
      // this was an orderly thread exit, we would already have received the
      // SIGTRAP | (PTRACE_EVENT_EXIT << 8) signal, and we would have done an
      // all-stop then.
      LLDB_LOG(log, "tid = {0} handling non-main thread exit (%s)", pid,
               thread_found ? "stopped tracking thread metadata"
                            : "thread metadata not found");
    }
    return;
  }

  siginfo_t info;
  const auto info_err = GetSignalInfo(pid, &info);
  auto thread_sp = GetThreadByID(pid);

  if (!thread_sp) {
    // Normally, the only situation when we cannot find the thread is if we have
    // just received a new thread notification. This is indicated by
    // GetSignalInfo() returning si_code == SI_USER and si_pid == 0
    LLDB_LOG(log, "received notification about an unknown tid {0}.", pid);

    if (info_err.Fail()) {
      LLDB_LOG(log,
               "(tid {0}) GetSignalInfo failed ({1}). "
               "Ingoring this notification.",
               pid, info_err);
      return;
    }

    LLDB_LOG(log, "tid {0}, si_code: {1}, si_pid: {2}", pid, info.si_code,
             info.si_pid);

    NativeThreadLinux &thread = AddThread(pid);

    // Resume the newly created thread.
    ResumeThread(thread, eStateRunning, LLDB_INVALID_SIGNAL_NUMBER);
    ThreadWasCreated(thread);
    return;
  }

  // Get details on the signal raised.
  if (info_err.Success()) {
    // We have retrieved the signal info.  Dispatch appropriately.
    if (info.si_signo == SIGTRAP)
      MonitorSIGTRAP(info, *thread_sp);
    else
      MonitorSignal(info, *thread_sp, exited);
  } else {
    if (info_err.GetError() == EINVAL) {
      // This is a group stop reception for this tid.
      // We can reach here if we reinject SIGSTOP, SIGSTP, SIGTTIN or SIGTTOU
      // into the tracee, triggering the group-stop mechanism. Normally
      // receiving these would stop the process, pending a SIGCONT. Simulating
      // this state in a debugger is hard and is generally not needed (one use
      // case is debugging background task being managed by a shell). For
      // general use, it is sufficient to stop the process in a signal-delivery
      // stop which happens before the group stop. This done by MonitorSignal
      // and works correctly for all signals.
      LLDB_LOG(log,
               "received a group stop for pid {0} tid {1}. Transparent "
               "handling of group stops not supported, resuming the "
               "thread.",
               GetID(), pid);
      ResumeThread(*thread_sp, thread_sp->GetState(),
                   LLDB_INVALID_SIGNAL_NUMBER);
    } else {
      // ptrace(GETSIGINFO) failed (but not due to group-stop).

      // A return value of ESRCH means the thread/process is no longer on the
      // system, so it was killed somehow outside of our control.  Either way,
      // we can't do anything with it anymore.

      // Stop tracking the metadata for the thread since it's entirely off the
      // system now.
      const bool thread_found = StopTrackingThread(pid);

      LLDB_LOG(log,
               "GetSignalInfo failed: {0}, tid = {1}, signal = {2}, "
               "status = {3}, main_thread = {4}, thread_found: {5}",
               info_err, pid, signal, status, is_main_thread, thread_found);

      if (is_main_thread) {
        // Notify the delegate - our process is not available but appears to
        // have been killed outside
        // our control.  Is eStateExited the right exit state in this case?
        SetExitStatus(status, true);
        SetState(StateType::eStateExited, true);
      } else {
        // This thread was pulled out from underneath us.  Anything to do here?
        // Do we want to do an all stop?
        LLDB_LOG(log,
                 "pid {0} tid {1} non-main thread exit occurred, didn't "
                 "tell delegate anything since thread disappeared out "
                 "from underneath us",
                 GetID(), pid);
      }
    }
  }
}

void NativeProcessLinux::WaitForNewThread(::pid_t tid) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));

  if (GetThreadByID(tid)) {
    // We are already tracking the thread - we got the event on the new thread
    // (see MonitorSignal) before this one. We are done.
    return;
  }

  // The thread is not tracked yet, let's wait for it to appear.
  int status = -1;
  LLDB_LOG(log,
           "received thread creation event for tid {0}. tid not tracked "
           "yet, waiting for thread to appear...",
           tid);
  ::pid_t wait_pid = llvm::sys::RetryAfterSignal(-1, ::waitpid, tid, &status, __WALL);
  // Since we are waiting on a specific tid, this must be the creation event.
  // But let's do some checks just in case.
  if (wait_pid != tid) {
    LLDB_LOG(log,
             "waiting for tid {0} failed. Assuming the thread has "
             "disappeared in the meantime",
             tid);
    // The only way I know of this could happen is if the whole process was
    // SIGKILLed in the mean time. In any case, we can't do anything about that
    // now.
    return;
  }
  if (WIFEXITED(status)) {
    LLDB_LOG(log,
             "waiting for tid {0} returned an 'exited' event. Not "
             "tracking the thread.",
             tid);
    // Also a very improbable event.
    return;
  }

  LLDB_LOG(log, "pid = {0}: tracking new thread tid {1}", GetID(), tid);
  NativeThreadLinux &new_thread = AddThread(tid);

  ResumeThread(new_thread, eStateRunning, LLDB_INVALID_SIGNAL_NUMBER);
  ThreadWasCreated(new_thread);
}

void NativeProcessLinux::MonitorSIGTRAP(const siginfo_t &info,
                                        NativeThreadLinux &thread) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));
  const bool is_main_thread = (thread.GetID() == GetID());

  assert(info.si_signo == SIGTRAP && "Unexpected child signal!");

  switch (info.si_code) {
  // TODO: these two cases are required if we want to support tracing of the
  // inferiors' children.  We'd need this to debug a monitor.
  // case (SIGTRAP | (PTRACE_EVENT_FORK << 8)):
  // case (SIGTRAP | (PTRACE_EVENT_VFORK << 8)):

  case (SIGTRAP | (PTRACE_EVENT_CLONE << 8)): {
    // This is the notification on the parent thread which informs us of new
    // thread
    // creation.
    // We don't want to do anything with the parent thread so we just resume it.
    // In case we
    // want to implement "break on thread creation" functionality, we would need
    // to stop
    // here.

    unsigned long event_message = 0;
    if (GetEventMessage(thread.GetID(), &event_message).Fail()) {
      LLDB_LOG(log,
               "pid {0} received thread creation event but "
               "GetEventMessage failed so we don't know the new tid",
               thread.GetID());
    } else
      WaitForNewThread(event_message);

    ResumeThread(thread, thread.GetState(), LLDB_INVALID_SIGNAL_NUMBER);
    break;
  }

  case (SIGTRAP | (PTRACE_EVENT_EXEC << 8)): {
    LLDB_LOG(log, "received exec event, code = {0}", info.si_code ^ SIGTRAP);

    // Exec clears any pending notifications.
    m_pending_notification_tid = LLDB_INVALID_THREAD_ID;

    // Remove all but the main thread here.  Linux fork creates a new process
    // which only copies the main thread.
    LLDB_LOG(log, "exec received, stop tracking all but main thread");

    for (auto i = m_threads.begin(); i != m_threads.end();) {
      if ((*i)->GetID() == GetID())
        i = m_threads.erase(i);
      else
        ++i;
    }
    assert(m_threads.size() == 1);
    auto *main_thread = static_cast<NativeThreadLinux *>(m_threads[0].get());

    SetCurrentThreadID(main_thread->GetID());
    main_thread->SetStoppedByExec();

    // Tell coordinator about about the "new" (since exec) stopped main thread.
    ThreadWasCreated(*main_thread);

    // Let our delegate know we have just exec'd.
    NotifyDidExec();

    // Let the process know we're stopped.
    StopRunningThreads(main_thread->GetID());

    break;
  }

  case (SIGTRAP | (PTRACE_EVENT_EXIT << 8)): {
    // The inferior process or one of its threads is about to exit.
    // We don't want to do anything with the thread so we just resume it. In
    // case we
    // want to implement "break on thread exit" functionality, we would need to
    // stop
    // here.

    unsigned long data = 0;
    if (GetEventMessage(thread.GetID(), &data).Fail())
      data = -1;

    LLDB_LOG(log,
             "received PTRACE_EVENT_EXIT, data = {0:x}, WIFEXITED={1}, "
             "WIFSIGNALED={2}, pid = {3}, main_thread = {4}",
             data, WIFEXITED(data), WIFSIGNALED(data), thread.GetID(),
             is_main_thread);

    if (is_main_thread)
      SetExitStatus(WaitStatus::Decode(data), true);

    StateType state = thread.GetState();
    if (!StateIsRunningState(state)) {
      // Due to a kernel bug, we may sometimes get this stop after the inferior
      // gets a
      // SIGKILL. This confuses our state tracking logic in ResumeThread(),
      // since normally,
      // we should not be receiving any ptrace events while the inferior is
      // stopped. This
      // makes sure that the inferior is resumed and exits normally.
      state = eStateRunning;
    }
    ResumeThread(thread, state, LLDB_INVALID_SIGNAL_NUMBER);

    break;
  }

  case 0:
  case TRAP_TRACE:  // We receive this on single stepping.
  case TRAP_HWBKPT: // We receive this on watchpoint hit
  {
    // If a watchpoint was hit, report it
    uint32_t wp_index;
    Status error = thread.GetRegisterContext()->GetWatchpointHitIndex(
        wp_index, (uintptr_t)info.si_addr);
    if (error.Fail())
      LLDB_LOG(log,
               "received error while checking for watchpoint hits, pid = "
               "{0}, error = {1}",
               thread.GetID(), error);
    if (wp_index != LLDB_INVALID_INDEX32) {
      MonitorWatchpoint(thread, wp_index);
      break;
    }

    // If a breakpoint was hit, report it
    uint32_t bp_index;
    error = thread.GetRegisterContext()->GetHardwareBreakHitIndex(
        bp_index, (uintptr_t)info.si_addr);
    if (error.Fail())
      LLDB_LOG(log, "received error while checking for hardware "
                    "breakpoint hits, pid = {0}, error = {1}",
               thread.GetID(), error);
    if (bp_index != LLDB_INVALID_INDEX32) {
      MonitorBreakpoint(thread);
      break;
    }

    // Otherwise, report step over
    MonitorTrace(thread);
    break;
  }

  case SI_KERNEL:
#if defined __mips__
    // For mips there is no special signal for watchpoint
    // So we check for watchpoint in kernel trap
    {
      // If a watchpoint was hit, report it
      uint32_t wp_index;
      Status error = thread.GetRegisterContext()->GetWatchpointHitIndex(
          wp_index, LLDB_INVALID_ADDRESS);
      if (error.Fail())
        LLDB_LOG(log,
                 "received error while checking for watchpoint hits, pid = "
                 "{0}, error = {1}",
                 thread.GetID(), error);
      if (wp_index != LLDB_INVALID_INDEX32) {
        MonitorWatchpoint(thread, wp_index);
        break;
      }
    }
// NO BREAK
#endif
  case TRAP_BRKPT:
    MonitorBreakpoint(thread);
    break;

  case SIGTRAP:
  case (SIGTRAP | 0x80):
    LLDB_LOG(
        log,
        "received unknown SIGTRAP stop event ({0}, pid {1} tid {2}, resuming",
        info.si_code, GetID(), thread.GetID());

    // Ignore these signals until we know more about them.
    ResumeThread(thread, thread.GetState(), LLDB_INVALID_SIGNAL_NUMBER);
    break;

  default:
    LLDB_LOG(log, "received unknown SIGTRAP stop event ({0}, pid {1} tid {2}",
             info.si_code, GetID(), thread.GetID());
    MonitorSignal(info, thread, false);
    break;
  }
}

void NativeProcessLinux::MonitorTrace(NativeThreadLinux &thread) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));
  LLDB_LOG(log, "received trace event, pid = {0}", thread.GetID());

  // This thread is currently stopped.
  thread.SetStoppedByTrace();

  StopRunningThreads(thread.GetID());
}

void NativeProcessLinux::MonitorBreakpoint(NativeThreadLinux &thread) {
  Log *log(
      GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_BREAKPOINTS));
  LLDB_LOG(log, "received breakpoint event, pid = {0}", thread.GetID());

  // Mark the thread as stopped at breakpoint.
  thread.SetStoppedByBreakpoint();
  Status error = FixupBreakpointPCAsNeeded(thread);
  if (error.Fail())
    LLDB_LOG(log, "pid = {0} fixup: {1}", thread.GetID(), error);

  if (m_threads_stepping_with_breakpoint.find(thread.GetID()) !=
      m_threads_stepping_with_breakpoint.end())
    thread.SetStoppedByTrace();

  StopRunningThreads(thread.GetID());
}

void NativeProcessLinux::MonitorWatchpoint(NativeThreadLinux &thread,
                                           uint32_t wp_index) {
  Log *log(
      GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_WATCHPOINTS));
  LLDB_LOG(log, "received watchpoint event, pid = {0}, wp_index = {1}",
           thread.GetID(), wp_index);

  // Mark the thread as stopped at watchpoint.
  // The address is at (lldb::addr_t)info->si_addr if we need it.
  thread.SetStoppedByWatchpoint(wp_index);

  // We need to tell all other running threads before we notify the delegate
  // about this stop.
  StopRunningThreads(thread.GetID());
}

void NativeProcessLinux::MonitorSignal(const siginfo_t &info,
                                       NativeThreadLinux &thread, bool exited) {
  const int signo = info.si_signo;
  const bool is_from_llgs = info.si_pid == getpid();

  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));

  // POSIX says that process behaviour is undefined after it ignores a SIGFPE,
  // SIGILL, SIGSEGV, or SIGBUS *unless* that signal was generated by a
  // kill(2) or raise(3).  Similarly for tgkill(2) on Linux.
  //
  // IOW, user generated signals never generate what we consider to be a
  // "crash".
  //
  // Similarly, ACK signals generated by this monitor.

  // Handle the signal.
  LLDB_LOG(log,
           "received signal {0} ({1}) with code {2}, (siginfo pid = {3}, "
           "waitpid pid = {4})",
           Host::GetSignalAsCString(signo), signo, info.si_code,
           thread.GetID());

  // Check for thread stop notification.
  if (is_from_llgs && (info.si_code == SI_TKILL) && (signo == SIGSTOP)) {
    // This is a tgkill()-based stop.
    LLDB_LOG(log, "pid {0} tid {1}, thread stopped", GetID(), thread.GetID());

    // Check that we're not already marked with a stop reason.
    // Note this thread really shouldn't already be marked as stopped - if we
    // were, that would imply that the kernel signaled us with the thread
    // stopping which we handled and marked as stopped, and that, without an
    // intervening resume, we received another stop.  It is more likely that we
    // are missing the marking of a run state somewhere if we find that the
    // thread was marked as stopped.
    const StateType thread_state = thread.GetState();
    if (!StateIsStoppedState(thread_state, false)) {
      // An inferior thread has stopped because of a SIGSTOP we have sent it.
      // Generally, these are not important stops and we don't want to report
      // them as they are just used to stop other threads when one thread (the
      // one with the *real* stop reason) hits a breakpoint (watchpoint,
      // etc...). However, in the case of an asynchronous Interrupt(), this *is*
      // the real stop reason, so we leave the signal intact if this is the
      // thread that was chosen as the triggering thread.
      if (m_pending_notification_tid != LLDB_INVALID_THREAD_ID) {
        if (m_pending_notification_tid == thread.GetID())
          thread.SetStoppedBySignal(SIGSTOP, &info);
        else
          thread.SetStoppedWithNoReason();

        SetCurrentThreadID(thread.GetID());
        SignalIfAllThreadsStopped();
      } else {
        // We can end up here if stop was initiated by LLGS but by this time a
        // thread stop has occurred - maybe initiated by another event.
        Status error = ResumeThread(thread, thread.GetState(), 0);
        if (error.Fail())
          LLDB_LOG(log, "failed to resume thread {0}: {1}", thread.GetID(),
                   error);
      }
    } else {
      LLDB_LOG(log,
               "pid {0} tid {1}, thread was already marked as a stopped "
               "state (state={2}), leaving stop signal as is",
               GetID(), thread.GetID(), thread_state);
      SignalIfAllThreadsStopped();
    }

    // Done handling.
    return;
  }

  // Check if debugger should stop at this signal or just ignore it
  // and resume the inferior.
  if (m_signals_to_ignore.find(signo) != m_signals_to_ignore.end()) {
     ResumeThread(thread, thread.GetState(), signo);
     return;
  }

  // This thread is stopped.
  LLDB_LOG(log, "received signal {0}", Host::GetSignalAsCString(signo));
  thread.SetStoppedBySignal(signo, &info);

  // Send a stop to the debugger after we get all other threads to stop.
  StopRunningThreads(thread.GetID());
}

namespace {

struct EmulatorBaton {
  NativeProcessLinux *m_process;
  NativeRegisterContext *m_reg_context;

  // eRegisterKindDWARF -> RegsiterValue
  std::unordered_map<uint32_t, RegisterValue> m_register_values;

  EmulatorBaton(NativeProcessLinux *process, NativeRegisterContext *reg_context)
      : m_process(process), m_reg_context(reg_context) {}
};

} // anonymous namespace

static size_t ReadMemoryCallback(EmulateInstruction *instruction, void *baton,
                                 const EmulateInstruction::Context &context,
                                 lldb::addr_t addr, void *dst, size_t length) {
  EmulatorBaton *emulator_baton = static_cast<EmulatorBaton *>(baton);

  size_t bytes_read;
  emulator_baton->m_process->ReadMemory(addr, dst, length, bytes_read);
  return bytes_read;
}

static bool ReadRegisterCallback(EmulateInstruction *instruction, void *baton,
                                 const RegisterInfo *reg_info,
                                 RegisterValue &reg_value) {
  EmulatorBaton *emulator_baton = static_cast<EmulatorBaton *>(baton);

  auto it = emulator_baton->m_register_values.find(
      reg_info->kinds[eRegisterKindDWARF]);
  if (it != emulator_baton->m_register_values.end()) {
    reg_value = it->second;
    return true;
  }

  // The emulator only fill in the dwarf regsiter numbers (and in some case
  // the generic register numbers). Get the full register info from the
  // register context based on the dwarf register numbers.
  const RegisterInfo *full_reg_info =
      emulator_baton->m_reg_context->GetRegisterInfo(
          eRegisterKindDWARF, reg_info->kinds[eRegisterKindDWARF]);

  Status error =
      emulator_baton->m_reg_context->ReadRegister(full_reg_info, reg_value);
  if (error.Success())
    return true;

  return false;
}

static bool WriteRegisterCallback(EmulateInstruction *instruction, void *baton,
                                  const EmulateInstruction::Context &context,
                                  const RegisterInfo *reg_info,
                                  const RegisterValue &reg_value) {
  EmulatorBaton *emulator_baton = static_cast<EmulatorBaton *>(baton);
  emulator_baton->m_register_values[reg_info->kinds[eRegisterKindDWARF]] =
      reg_value;
  return true;
}

static size_t WriteMemoryCallback(EmulateInstruction *instruction, void *baton,
                                  const EmulateInstruction::Context &context,
                                  lldb::addr_t addr, const void *dst,
                                  size_t length) {
  return length;
}

static lldb::addr_t ReadFlags(NativeRegisterContext *regsiter_context) {
  const RegisterInfo *flags_info = regsiter_context->GetRegisterInfo(
      eRegisterKindGeneric, LLDB_REGNUM_GENERIC_FLAGS);
  return regsiter_context->ReadRegisterAsUnsigned(flags_info,
                                                  LLDB_INVALID_ADDRESS);
}

Status
NativeProcessLinux::SetupSoftwareSingleStepping(NativeThreadLinux &thread) {
  Status error;
  NativeRegisterContextSP register_context_sp = thread.GetRegisterContext();

  std::unique_ptr<EmulateInstruction> emulator_ap(
      EmulateInstruction::FindPlugin(m_arch, eInstructionTypePCModifying,
                                     nullptr));

  if (emulator_ap == nullptr)
    return Status("Instruction emulator not found!");

  EmulatorBaton baton(this, register_context_sp.get());
  emulator_ap->SetBaton(&baton);
  emulator_ap->SetReadMemCallback(&ReadMemoryCallback);
  emulator_ap->SetReadRegCallback(&ReadRegisterCallback);
  emulator_ap->SetWriteMemCallback(&WriteMemoryCallback);
  emulator_ap->SetWriteRegCallback(&WriteRegisterCallback);

  if (!emulator_ap->ReadInstruction())
    return Status("Read instruction failed!");

  bool emulation_result =
      emulator_ap->EvaluateInstruction(eEmulateInstructionOptionAutoAdvancePC);

  const RegisterInfo *reg_info_pc = register_context_sp->GetRegisterInfo(
      eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
  const RegisterInfo *reg_info_flags = register_context_sp->GetRegisterInfo(
      eRegisterKindGeneric, LLDB_REGNUM_GENERIC_FLAGS);

  auto pc_it =
      baton.m_register_values.find(reg_info_pc->kinds[eRegisterKindDWARF]);
  auto flags_it =
      baton.m_register_values.find(reg_info_flags->kinds[eRegisterKindDWARF]);

  lldb::addr_t next_pc;
  lldb::addr_t next_flags;
  if (emulation_result) {
    assert(pc_it != baton.m_register_values.end() &&
           "Emulation was successfull but PC wasn't updated");
    next_pc = pc_it->second.GetAsUInt64();

    if (flags_it != baton.m_register_values.end())
      next_flags = flags_it->second.GetAsUInt64();
    else
      next_flags = ReadFlags(register_context_sp.get());
  } else if (pc_it == baton.m_register_values.end()) {
    // Emulate instruction failed and it haven't changed PC. Advance PC
    // with the size of the current opcode because the emulation of all
    // PC modifying instruction should be successful. The failure most
    // likely caused by a not supported instruction which don't modify PC.
    next_pc =
        register_context_sp->GetPC() + emulator_ap->GetOpcode().GetByteSize();
    next_flags = ReadFlags(register_context_sp.get());
  } else {
    // The instruction emulation failed after it modified the PC. It is an
    // unknown error where we can't continue because the next instruction is
    // modifying the PC but we don't  know how.
    return Status("Instruction emulation failed unexpectedly.");
  }

  if (m_arch.GetMachine() == llvm::Triple::arm) {
    if (next_flags & 0x20) {
      // Thumb mode
      error = SetSoftwareBreakpoint(next_pc, 2);
    } else {
      // Arm mode
      error = SetSoftwareBreakpoint(next_pc, 4);
    }
  } else if (m_arch.GetMachine() == llvm::Triple::mips64 ||
             m_arch.GetMachine() == llvm::Triple::mips64el ||
             m_arch.GetMachine() == llvm::Triple::mips ||
             m_arch.GetMachine() == llvm::Triple::mipsel ||
             m_arch.GetMachine() == llvm::Triple::ppc64le)
    error = SetSoftwareBreakpoint(next_pc, 4);
  else {
    // No size hint is given for the next breakpoint
    error = SetSoftwareBreakpoint(next_pc, 0);
  }

  // If setting the breakpoint fails because next_pc is out of
  // the address space, ignore it and let the debugee segfault.
  if (error.GetError() == EIO || error.GetError() == EFAULT) {
    return Status();
  } else if (error.Fail())
    return error;

  m_threads_stepping_with_breakpoint.insert({thread.GetID(), next_pc});

  return Status();
}

bool NativeProcessLinux::SupportHardwareSingleStepping() const {
  if (m_arch.GetMachine() == llvm::Triple::arm ||
      m_arch.GetMachine() == llvm::Triple::mips64 ||
      m_arch.GetMachine() == llvm::Triple::mips64el ||
      m_arch.GetMachine() == llvm::Triple::mips ||
      m_arch.GetMachine() == llvm::Triple::mipsel)
    return false;
  return true;
}

Status NativeProcessLinux::Resume(const ResumeActionList &resume_actions) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));
  LLDB_LOG(log, "pid {0}", GetID());

  bool software_single_step = !SupportHardwareSingleStepping();

  if (software_single_step) {
    for (const auto &thread : m_threads) {
      assert(thread && "thread list should not contain NULL threads");

      const ResumeAction *const action =
          resume_actions.GetActionForThread(thread->GetID(), true);
      if (action == nullptr)
        continue;

      if (action->state == eStateStepping) {
        Status error = SetupSoftwareSingleStepping(
            static_cast<NativeThreadLinux &>(*thread));
        if (error.Fail())
          return error;
      }
    }
  }

  for (const auto &thread : m_threads) {
    assert(thread && "thread list should not contain NULL threads");

    const ResumeAction *const action =
        resume_actions.GetActionForThread(thread->GetID(), true);

    if (action == nullptr) {
      LLDB_LOG(log, "no action specified for pid {0} tid {1}", GetID(),
               thread->GetID());
      continue;
    }

    LLDB_LOG(log, "processing resume action state {0} for pid {1} tid {2}",
             action->state, GetID(), thread->GetID());

    switch (action->state) {
    case eStateRunning:
    case eStateStepping: {
      // Run the thread, possibly feeding it the signal.
      const int signo = action->signal;
      ResumeThread(static_cast<NativeThreadLinux &>(*thread), action->state,
                   signo);
      break;
    }

    case eStateSuspended:
    case eStateStopped:
      llvm_unreachable("Unexpected state");

    default:
      return Status("NativeProcessLinux::%s (): unexpected state %s specified "
                    "for pid %" PRIu64 ", tid %" PRIu64,
                    __FUNCTION__, StateAsCString(action->state), GetID(),
                    thread->GetID());
    }
  }

  return Status();
}

Status NativeProcessLinux::Halt() {
  Status error;

  if (kill(GetID(), SIGSTOP) != 0)
    error.SetErrorToErrno();

  return error;
}

Status NativeProcessLinux::Detach() {
  Status error;

  // Stop monitoring the inferior.
  m_sigchld_handle.reset();

  // Tell ptrace to detach from the process.
  if (GetID() == LLDB_INVALID_PROCESS_ID)
    return error;

  for (const auto &thread : m_threads) {
    Status e = Detach(thread->GetID());
    if (e.Fail())
      error =
          e; // Save the error, but still attempt to detach from other threads.
  }

  m_processor_trace_monitor.clear();
  m_pt_proces_trace_id = LLDB_INVALID_UID;

  return error;
}

Status NativeProcessLinux::Signal(int signo) {
  Status error;

  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));
  LLDB_LOG(log, "sending signal {0} ({1}) to pid {1}", signo,
           Host::GetSignalAsCString(signo), GetID());

  if (kill(GetID(), signo))
    error.SetErrorToErrno();

  return error;
}

Status NativeProcessLinux::Interrupt() {
  // Pick a running thread (or if none, a not-dead stopped thread) as
  // the chosen thread that will be the stop-reason thread.
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));

  NativeThreadProtocol *running_thread = nullptr;
  NativeThreadProtocol *stopped_thread = nullptr;

  LLDB_LOG(log, "selecting running thread for interrupt target");
  for (const auto &thread : m_threads) {
    // If we have a running or stepping thread, we'll call that the
    // target of the interrupt.
    const auto thread_state = thread->GetState();
    if (thread_state == eStateRunning || thread_state == eStateStepping) {
      running_thread = thread.get();
      break;
    } else if (!stopped_thread && StateIsStoppedState(thread_state, true)) {
      // Remember the first non-dead stopped thread.  We'll use that as a backup
      // if there are no running threads.
      stopped_thread = thread.get();
    }
  }

  if (!running_thread && !stopped_thread) {
    Status error("found no running/stepping or live stopped threads as target "
                 "for interrupt");
    LLDB_LOG(log, "skipping due to error: {0}", error);

    return error;
  }

  NativeThreadProtocol *deferred_signal_thread =
      running_thread ? running_thread : stopped_thread;

  LLDB_LOG(log, "pid {0} {1} tid {2} chosen for interrupt target", GetID(),
           running_thread ? "running" : "stopped",
           deferred_signal_thread->GetID());

  StopRunningThreads(deferred_signal_thread->GetID());

  return Status();
}

Status NativeProcessLinux::Kill() {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));
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
             m_state);
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

  if (kill(GetID(), SIGKILL) != 0) {
    error.SetErrorToErrno();
    return error;
  }

  return error;
}

static Status
ParseMemoryRegionInfoFromProcMapsLine(llvm::StringRef &maps_line,
                                      MemoryRegionInfo &memory_region_info) {
  memory_region_info.Clear();

  StringExtractor line_extractor(maps_line);

  // Format: {address_start_hex}-{address_end_hex} perms offset  dev   inode
  // pathname
  // perms: rwxp   (letter is present if set, '-' if not, final character is
  // p=private, s=shared).

  // Parse out the starting address
  lldb::addr_t start_address = line_extractor.GetHexMaxU64(false, 0);

  // Parse out hyphen separating start and end address from range.
  if (!line_extractor.GetBytesLeft() || (line_extractor.GetChar() != '-'))
    return Status(
        "malformed /proc/{pid}/maps entry, missing dash between address range");

  // Parse out the ending address
  lldb::addr_t end_address = line_extractor.GetHexMaxU64(false, start_address);

  // Parse out the space after the address.
  if (!line_extractor.GetBytesLeft() || (line_extractor.GetChar() != ' '))
    return Status(
        "malformed /proc/{pid}/maps entry, missing space after range");

  // Save the range.
  memory_region_info.GetRange().SetRangeBase(start_address);
  memory_region_info.GetRange().SetRangeEnd(end_address);

  // Any memory region in /proc/{pid}/maps is by definition mapped into the
  // process.
  memory_region_info.SetMapped(MemoryRegionInfo::OptionalBool::eYes);

  // Parse out each permission entry.
  if (line_extractor.GetBytesLeft() < 4)
    return Status("malformed /proc/{pid}/maps entry, missing some portion of "
                  "permissions");

  // Handle read permission.
  const char read_perm_char = line_extractor.GetChar();
  if (read_perm_char == 'r')
    memory_region_info.SetReadable(MemoryRegionInfo::OptionalBool::eYes);
  else if (read_perm_char == '-')
    memory_region_info.SetReadable(MemoryRegionInfo::OptionalBool::eNo);
  else
    return Status("unexpected /proc/{pid}/maps read permission char");

  // Handle write permission.
  const char write_perm_char = line_extractor.GetChar();
  if (write_perm_char == 'w')
    memory_region_info.SetWritable(MemoryRegionInfo::OptionalBool::eYes);
  else if (write_perm_char == '-')
    memory_region_info.SetWritable(MemoryRegionInfo::OptionalBool::eNo);
  else
    return Status("unexpected /proc/{pid}/maps write permission char");

  // Handle execute permission.
  const char exec_perm_char = line_extractor.GetChar();
  if (exec_perm_char == 'x')
    memory_region_info.SetExecutable(MemoryRegionInfo::OptionalBool::eYes);
  else if (exec_perm_char == '-')
    memory_region_info.SetExecutable(MemoryRegionInfo::OptionalBool::eNo);
  else
    return Status("unexpected /proc/{pid}/maps exec permission char");

  line_extractor.GetChar();              // Read the private bit
  line_extractor.SkipSpaces();           // Skip the separator
  line_extractor.GetHexMaxU64(false, 0); // Read the offset
  line_extractor.GetHexMaxU64(false, 0); // Read the major device number
  line_extractor.GetChar();              // Read the device id separator
  line_extractor.GetHexMaxU64(false, 0); // Read the major device number
  line_extractor.SkipSpaces();           // Skip the separator
  line_extractor.GetU64(0, 10);          // Read the inode number

  line_extractor.SkipSpaces();
  const char *name = line_extractor.Peek();
  if (name)
    memory_region_info.SetName(name);

  return Status();
}

Status NativeProcessLinux::GetMemoryRegionInfo(lldb::addr_t load_addr,
                                               MemoryRegionInfo &range_info) {
  // FIXME review that the final memory region returned extends to the end of
  // the virtual address space,
  // with no perms if it is not mapped.

  // Use an approach that reads memory regions from /proc/{pid}/maps.
  // Assume proc maps entries are in ascending order.
  // FIXME assert if we find differently.

  if (m_supports_mem_region == LazyBool::eLazyBoolNo) {
    // We're done.
    return Status("unsupported");
  }

  Status error = PopulateMemoryRegionCache();
  if (error.Fail()) {
    return error;
  }

  lldb::addr_t prev_base_address = 0;

  // FIXME start by finding the last region that is <= target address using
  // binary search.  Data is sorted.
  // There can be a ton of regions on pthreads apps with lots of threads.
  for (auto it = m_mem_region_cache.begin(); it != m_mem_region_cache.end();
       ++it) {
    MemoryRegionInfo &proc_entry_info = it->first;

    // Sanity check assumption that /proc/{pid}/maps entries are ascending.
    assert((proc_entry_info.GetRange().GetRangeBase() >= prev_base_address) &&
           "descending /proc/pid/maps entries detected, unexpected");
    prev_base_address = proc_entry_info.GetRange().GetRangeBase();
    UNUSED_IF_ASSERT_DISABLED(prev_base_address);

    // If the target address comes before this entry, indicate distance to next
    // region.
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
      // The target address is within the memory region we're processing here.
      range_info = proc_entry_info;
      return error;
    }

    // The target memory address comes somewhere after the region we just
    // parsed.
  }

  // If we made it here, we didn't find an entry that contained the given
  // address. Return the
  // load_addr as start and the amount of bytes betwwen load address and the end
  // of the memory as
  // size.
  range_info.GetRange().SetRangeBase(load_addr);
  range_info.GetRange().SetRangeEnd(LLDB_INVALID_ADDRESS);
  range_info.SetReadable(MemoryRegionInfo::OptionalBool::eNo);
  range_info.SetWritable(MemoryRegionInfo::OptionalBool::eNo);
  range_info.SetExecutable(MemoryRegionInfo::OptionalBool::eNo);
  range_info.SetMapped(MemoryRegionInfo::OptionalBool::eNo);
  return error;
}

Status NativeProcessLinux::PopulateMemoryRegionCache() {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));

  // If our cache is empty, pull the latest.  There should always be at least
  // one memory region if memory region handling is supported.
  if (!m_mem_region_cache.empty()) {
    LLDB_LOG(log, "reusing {0} cached memory region entries",
             m_mem_region_cache.size());
    return Status();
  }

  auto BufferOrError = getProcFile(GetID(), "maps");
  if (!BufferOrError) {
    m_supports_mem_region = LazyBool::eLazyBoolNo;
    return BufferOrError.getError();
  }
  StringRef Rest = BufferOrError.get()->getBuffer();
  while (! Rest.empty()) {
    StringRef Line;
    std::tie(Line, Rest) = Rest.split('\n');
    MemoryRegionInfo info;
    const Status parse_error =
        ParseMemoryRegionInfoFromProcMapsLine(Line, info);
    if (parse_error.Fail()) {
      LLDB_LOG(log, "failed to parse proc maps line '{0}': {1}", Line,
               parse_error);
      m_supports_mem_region = LazyBool::eLazyBoolNo;
      return parse_error;
    }
    m_mem_region_cache.emplace_back(
        info, FileSpec(info.GetName().GetCString(), true));
  }

  if (m_mem_region_cache.empty()) {
    // No entries after attempting to read them.  This shouldn't happen if
    // /proc/{pid}/maps is supported. Assume we don't support map entries
    // via procfs.
    m_supports_mem_region = LazyBool::eLazyBoolNo;
    LLDB_LOG(log,
             "failed to find any procfs maps entries, assuming no support "
             "for memory region metadata retrieval");
    return Status("not supported");
  }

  LLDB_LOG(log, "read {0} memory region entries from /proc/{1}/maps",
           m_mem_region_cache.size(), GetID());

  // We support memory retrieval, remember that.
  m_supports_mem_region = LazyBool::eLazyBoolYes;
  return Status();
}

void NativeProcessLinux::DoStopIDBumped(uint32_t newBumpId) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));
  LLDB_LOG(log, "newBumpId={0}", newBumpId);
  LLDB_LOG(log, "clearing {0} entries from memory region cache",
           m_mem_region_cache.size());
  m_mem_region_cache.clear();
}

Status NativeProcessLinux::AllocateMemory(size_t size, uint32_t permissions,
                                          lldb::addr_t &addr) {
// FIXME implementing this requires the equivalent of
// InferiorCallPOSIX::InferiorCallMmap, which depends on
// functional ThreadPlans working with Native*Protocol.
#if 1
  return Status("not implemented yet");
#else
  addr = LLDB_INVALID_ADDRESS;

  unsigned prot = 0;
  if (permissions & lldb::ePermissionsReadable)
    prot |= eMmapProtRead;
  if (permissions & lldb::ePermissionsWritable)
    prot |= eMmapProtWrite;
  if (permissions & lldb::ePermissionsExecutable)
    prot |= eMmapProtExec;

  // TODO implement this directly in NativeProcessLinux
  // (and lift to NativeProcessPOSIX if/when that class is
  // refactored out).
  if (InferiorCallMmap(this, addr, 0, size, prot,
                       eMmapFlagsAnon | eMmapFlagsPrivate, -1, 0)) {
    m_addr_to_mmap_size[addr] = size;
    return Status();
  } else {
    addr = LLDB_INVALID_ADDRESS;
    return Status("unable to allocate %" PRIu64
                  " bytes of memory with permissions %s",
                  size, GetPermissionsAsCString(permissions));
  }
#endif
}

Status NativeProcessLinux::DeallocateMemory(lldb::addr_t addr) {
  // FIXME see comments in AllocateMemory - required lower-level
  // bits not in place yet (ThreadPlans)
  return Status("not implemented");
}

lldb::addr_t NativeProcessLinux::GetSharedLibraryInfoAddress() {
  // punt on this for now
  return LLDB_INVALID_ADDRESS;
}

size_t NativeProcessLinux::UpdateThreads() {
  // The NativeProcessLinux monitoring threads are always up to date
  // with respect to thread state and they keep the thread list
  // populated properly. All this method needs to do is return the
  // thread count.
  return m_threads.size();
}

bool NativeProcessLinux::GetArchitecture(ArchSpec &arch) const {
  arch = m_arch;
  return true;
}

Status NativeProcessLinux::GetSoftwareBreakpointPCOffset(
    uint32_t &actual_opcode_size) {
  // FIXME put this behind a breakpoint protocol class that can be
  // set per architecture.  Need ARM, MIPS support here.
  static const uint8_t g_i386_opcode[] = {0xCC};
  static const uint8_t g_s390x_opcode[] = {0x00, 0x01};
  static const uint8_t g_ppc64le_opcode[] = {0x08, 0x00, 0xe0, 0x7f}; // trap

  switch (m_arch.GetMachine()) {
  case llvm::Triple::x86:
  case llvm::Triple::x86_64:
    actual_opcode_size = static_cast<uint32_t>(sizeof(g_i386_opcode));
    return Status();

  case llvm::Triple::systemz:
    actual_opcode_size = static_cast<uint32_t>(sizeof(g_s390x_opcode));
    return Status();

  case llvm::Triple::ppc64le:
    actual_opcode_size = static_cast<uint32_t>(sizeof(g_ppc64le_opcode));
    return Status();

  case llvm::Triple::arm:
  case llvm::Triple::aarch64:
  case llvm::Triple::mips64:
  case llvm::Triple::mips64el:
  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
    // On these architectures the PC don't get updated for breakpoint hits
    actual_opcode_size = 0;
    return Status();

  default:
    assert(false && "CPU type not supported!");
    return Status("CPU type not supported");
  }
}

Status NativeProcessLinux::SetBreakpoint(lldb::addr_t addr, uint32_t size,
                                         bool hardware) {
  if (hardware)
    return SetHardwareBreakpoint(addr, size);
  else
    return SetSoftwareBreakpoint(addr, size);
}

Status NativeProcessLinux::RemoveBreakpoint(lldb::addr_t addr, bool hardware) {
  if (hardware)
    return RemoveHardwareBreakpoint(addr);
  else
    return NativeProcessProtocol::RemoveBreakpoint(addr);
}

Status NativeProcessLinux::GetSoftwareBreakpointTrapOpcode(
    size_t trap_opcode_size_hint, size_t &actual_opcode_size,
    const uint8_t *&trap_opcode_bytes) {
  // FIXME put this behind a breakpoint protocol class that can be set per
  // architecture.  Need MIPS support here.
  static const uint8_t g_aarch64_opcode[] = {0x00, 0x00, 0x20, 0xd4};
  // The ARM reference recommends the use of 0xe7fddefe and 0xdefe but the
  // linux kernel does otherwise.
  static const uint8_t g_arm_breakpoint_opcode[] = {0xf0, 0x01, 0xf0, 0xe7};
  static const uint8_t g_i386_opcode[] = {0xCC};
  static const uint8_t g_mips64_opcode[] = {0x00, 0x00, 0x00, 0x0d};
  static const uint8_t g_mips64el_opcode[] = {0x0d, 0x00, 0x00, 0x00};
  static const uint8_t g_s390x_opcode[] = {0x00, 0x01};
  static const uint8_t g_thumb_breakpoint_opcode[] = {0x01, 0xde};
  static const uint8_t g_ppc64le_opcode[] = {0x08, 0x00, 0xe0, 0x7f}; // trap

  switch (m_arch.GetMachine()) {
  case llvm::Triple::aarch64:
    trap_opcode_bytes = g_aarch64_opcode;
    actual_opcode_size = sizeof(g_aarch64_opcode);
    return Status();

  case llvm::Triple::arm:
    switch (trap_opcode_size_hint) {
    case 2:
      trap_opcode_bytes = g_thumb_breakpoint_opcode;
      actual_opcode_size = sizeof(g_thumb_breakpoint_opcode);
      return Status();
    case 4:
      trap_opcode_bytes = g_arm_breakpoint_opcode;
      actual_opcode_size = sizeof(g_arm_breakpoint_opcode);
      return Status();
    default:
      assert(false && "Unrecognised trap opcode size hint!");
      return Status("Unrecognised trap opcode size hint!");
    }

  case llvm::Triple::x86:
  case llvm::Triple::x86_64:
    trap_opcode_bytes = g_i386_opcode;
    actual_opcode_size = sizeof(g_i386_opcode);
    return Status();

  case llvm::Triple::mips:
  case llvm::Triple::mips64:
    trap_opcode_bytes = g_mips64_opcode;
    actual_opcode_size = sizeof(g_mips64_opcode);
    return Status();

  case llvm::Triple::mipsel:
  case llvm::Triple::mips64el:
    trap_opcode_bytes = g_mips64el_opcode;
    actual_opcode_size = sizeof(g_mips64el_opcode);
    return Status();

  case llvm::Triple::systemz:
    trap_opcode_bytes = g_s390x_opcode;
    actual_opcode_size = sizeof(g_s390x_opcode);
    return Status();

  case llvm::Triple::ppc64le:
    trap_opcode_bytes = g_ppc64le_opcode;
    actual_opcode_size = sizeof(g_ppc64le_opcode);
    return Status();

  default:
    assert(false && "CPU type not supported!");
    return Status("CPU type not supported");
  }
}

#if 0
ProcessMessage::CrashReason
NativeProcessLinux::GetCrashReasonForSIGSEGV(const siginfo_t *info)
{
    ProcessMessage::CrashReason reason;
    assert(info->si_signo == SIGSEGV);

    reason = ProcessMessage::eInvalidCrashReason;

    switch (info->si_code)
    {
    default:
        assert(false && "unexpected si_code for SIGSEGV");
        break;
    case SI_KERNEL:
        // Linux will occasionally send spurious SI_KERNEL codes.
        // (this is poorly documented in sigaction)
        // One way to get this is via unaligned SIMD loads.
        reason = ProcessMessage::eInvalidAddress; // for lack of anything better
        break;
    case SEGV_MAPERR:
        reason = ProcessMessage::eInvalidAddress;
        break;
    case SEGV_ACCERR:
        reason = ProcessMessage::ePrivilegedAddress;
        break;
    }

    return reason;
}
#endif

#if 0
ProcessMessage::CrashReason
NativeProcessLinux::GetCrashReasonForSIGILL(const siginfo_t *info)
{
    ProcessMessage::CrashReason reason;
    assert(info->si_signo == SIGILL);

    reason = ProcessMessage::eInvalidCrashReason;

    switch (info->si_code)
    {
    default:
        assert(false && "unexpected si_code for SIGILL");
        break;
    case ILL_ILLOPC:
        reason = ProcessMessage::eIllegalOpcode;
        break;
    case ILL_ILLOPN:
        reason = ProcessMessage::eIllegalOperand;
        break;
    case ILL_ILLADR:
        reason = ProcessMessage::eIllegalAddressingMode;
        break;
    case ILL_ILLTRP:
        reason = ProcessMessage::eIllegalTrap;
        break;
    case ILL_PRVOPC:
        reason = ProcessMessage::ePrivilegedOpcode;
        break;
    case ILL_PRVREG:
        reason = ProcessMessage::ePrivilegedRegister;
        break;
    case ILL_COPROC:
        reason = ProcessMessage::eCoprocessorError;
        break;
    case ILL_BADSTK:
        reason = ProcessMessage::eInternalStackError;
        break;
    }

    return reason;
}
#endif

#if 0
ProcessMessage::CrashReason
NativeProcessLinux::GetCrashReasonForSIGFPE(const siginfo_t *info)
{
    ProcessMessage::CrashReason reason;
    assert(info->si_signo == SIGFPE);

    reason = ProcessMessage::eInvalidCrashReason;

    switch (info->si_code)
    {
    default:
        assert(false && "unexpected si_code for SIGFPE");
        break;
    case FPE_INTDIV:
        reason = ProcessMessage::eIntegerDivideByZero;
        break;
    case FPE_INTOVF:
        reason = ProcessMessage::eIntegerOverflow;
        break;
    case FPE_FLTDIV:
        reason = ProcessMessage::eFloatDivideByZero;
        break;
    case FPE_FLTOVF:
        reason = ProcessMessage::eFloatOverflow;
        break;
    case FPE_FLTUND:
        reason = ProcessMessage::eFloatUnderflow;
        break;
    case FPE_FLTRES:
        reason = ProcessMessage::eFloatInexactResult;
        break;
    case FPE_FLTINV:
        reason = ProcessMessage::eFloatInvalidOperation;
        break;
    case FPE_FLTSUB:
        reason = ProcessMessage::eFloatSubscriptRange;
        break;
    }

    return reason;
}
#endif

#if 0
ProcessMessage::CrashReason
NativeProcessLinux::GetCrashReasonForSIGBUS(const siginfo_t *info)
{
    ProcessMessage::CrashReason reason;
    assert(info->si_signo == SIGBUS);

    reason = ProcessMessage::eInvalidCrashReason;

    switch (info->si_code)
    {
    default:
        assert(false && "unexpected si_code for SIGBUS");
        break;
    case BUS_ADRALN:
        reason = ProcessMessage::eIllegalAlignment;
        break;
    case BUS_ADRERR:
        reason = ProcessMessage::eIllegalAddress;
        break;
    case BUS_OBJERR:
        reason = ProcessMessage::eHardwareError;
        break;
    }

    return reason;
}
#endif

Status NativeProcessLinux::ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                                      size_t &bytes_read) {
  if (ProcessVmReadvSupported()) {
    // The process_vm_readv path is about 50 times faster than ptrace api. We
    // want to use
    // this syscall if it is supported.

    const ::pid_t pid = GetID();

    struct iovec local_iov, remote_iov;
    local_iov.iov_base = buf;
    local_iov.iov_len = size;
    remote_iov.iov_base = reinterpret_cast<void *>(addr);
    remote_iov.iov_len = size;

    bytes_read = process_vm_readv(pid, &local_iov, 1, &remote_iov, 1, 0);
    const bool success = bytes_read == size;

    Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));
    LLDB_LOG(log,
             "using process_vm_readv to read {0} bytes from inferior "
             "address {1:x}: {2}",
             size, addr, success ? "Success" : llvm::sys::StrError(errno));

    if (success)
      return Status();
    // else the call failed for some reason, let's retry the read using ptrace
    // api.
  }

  unsigned char *dst = static_cast<unsigned char *>(buf);
  size_t remainder;
  long data;

  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_MEMORY));
  LLDB_LOG(log, "addr = {0}, buf = {1}, size = {2}", addr, buf, size);

  for (bytes_read = 0; bytes_read < size; bytes_read += remainder) {
    Status error = NativeProcessLinux::PtraceWrapper(
        PTRACE_PEEKDATA, GetID(), (void *)addr, nullptr, 0, &data);
    if (error.Fail())
      return error;

    remainder = size - bytes_read;
    remainder = remainder > k_ptrace_word_size ? k_ptrace_word_size : remainder;

    // Copy the data into our buffer
    memcpy(dst, &data, remainder);

    LLDB_LOG(log, "[{0:x}]:{1:x}", addr, data);
    addr += k_ptrace_word_size;
    dst += k_ptrace_word_size;
  }
  return Status();
}

Status NativeProcessLinux::ReadMemoryWithoutTrap(lldb::addr_t addr, void *buf,
                                                 size_t size,
                                                 size_t &bytes_read) {
  Status error = ReadMemory(addr, buf, size, bytes_read);
  if (error.Fail())
    return error;
  return m_breakpoint_list.RemoveTrapsFromBuffer(addr, buf, size);
}

Status NativeProcessLinux::WriteMemory(lldb::addr_t addr, const void *buf,
                                       size_t size, size_t &bytes_written) {
  const unsigned char *src = static_cast<const unsigned char *>(buf);
  size_t remainder;
  Status error;

  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_MEMORY));
  LLDB_LOG(log, "addr = {0}, buf = {1}, size = {2}", addr, buf, size);

  for (bytes_written = 0; bytes_written < size; bytes_written += remainder) {
    remainder = size - bytes_written;
    remainder = remainder > k_ptrace_word_size ? k_ptrace_word_size : remainder;

    if (remainder == k_ptrace_word_size) {
      unsigned long data = 0;
      memcpy(&data, src, k_ptrace_word_size);

      LLDB_LOG(log, "[{0:x}]:{1:x}", addr, data);
      error = NativeProcessLinux::PtraceWrapper(PTRACE_POKEDATA, GetID(),
                                                (void *)addr, (void *)data);
      if (error.Fail())
        return error;
    } else {
      unsigned char buff[8];
      size_t bytes_read;
      error = ReadMemory(addr, buff, k_ptrace_word_size, bytes_read);
      if (error.Fail())
        return error;

      memcpy(buff, src, remainder);

      size_t bytes_written_rec;
      error = WriteMemory(addr, buff, k_ptrace_word_size, bytes_written_rec);
      if (error.Fail())
        return error;

      LLDB_LOG(log, "[{0:x}]:{1:x} ({2:x})", addr, *(const unsigned long *)src,
               *(unsigned long *)buff);
    }

    addr += k_ptrace_word_size;
    src += k_ptrace_word_size;
  }
  return error;
}

Status NativeProcessLinux::GetSignalInfo(lldb::tid_t tid, void *siginfo) {
  return PtraceWrapper(PTRACE_GETSIGINFO, tid, nullptr, siginfo);
}

Status NativeProcessLinux::GetEventMessage(lldb::tid_t tid,
                                           unsigned long *message) {
  return PtraceWrapper(PTRACE_GETEVENTMSG, tid, nullptr, message);
}

Status NativeProcessLinux::Detach(lldb::tid_t tid) {
  if (tid == LLDB_INVALID_THREAD_ID)
    return Status();

  return PtraceWrapper(PTRACE_DETACH, tid);
}

bool NativeProcessLinux::HasThreadNoLock(lldb::tid_t thread_id) {
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

bool NativeProcessLinux::StopTrackingThread(lldb::tid_t thread_id) {
  Log *const log = ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_THREAD);
  LLDB_LOG(log, "tid: {0})", thread_id);

  bool found = false;
  for (auto it = m_threads.begin(); it != m_threads.end(); ++it) {
    if (*it && ((*it)->GetID() == thread_id)) {
      m_threads.erase(it);
      found = true;
      break;
    }
  }

  if (found)
    StopTracingForThread(thread_id);
  SignalIfAllThreadsStopped();
  return found;
}

NativeThreadLinux &NativeProcessLinux::AddThread(lldb::tid_t thread_id) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_THREAD));
  LLDB_LOG(log, "pid {0} adding thread with tid {1}", GetID(), thread_id);

  assert(!HasThreadNoLock(thread_id) &&
         "attempted to add a thread by id that already exists");

  // If this is the first thread, save it as the current thread
  if (m_threads.empty())
    SetCurrentThreadID(thread_id);

  m_threads.push_back(llvm::make_unique<NativeThreadLinux>(*this, thread_id));

  if (m_pt_proces_trace_id != LLDB_INVALID_UID) {
    auto traceMonitor = ProcessorTraceMonitor::Create(
        GetID(), thread_id, m_pt_process_trace_config, true);
    if (traceMonitor) {
      m_pt_traced_thread_group.insert(thread_id);
      m_processor_trace_monitor.insert(
          std::make_pair(thread_id, std::move(*traceMonitor)));
    } else {
      LLDB_LOG(log, "failed to start trace on thread {0}", thread_id);
      Status error(traceMonitor.takeError());
      LLDB_LOG(log, "error {0}", error);
    }
  }

  return static_cast<NativeThreadLinux &>(*m_threads.back());
}

Status
NativeProcessLinux::FixupBreakpointPCAsNeeded(NativeThreadLinux &thread) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_BREAKPOINTS));

  Status error;

  // Find out the size of a breakpoint (might depend on where we are in the
  // code).
  NativeRegisterContextSP context_sp = thread.GetRegisterContext();
  if (!context_sp) {
    error.SetErrorString("cannot get a NativeRegisterContext for the thread");
    LLDB_LOG(log, "failed: {0}", error);
    return error;
  }

  uint32_t breakpoint_size = 0;
  error = GetSoftwareBreakpointPCOffset(breakpoint_size);
  if (error.Fail()) {
    LLDB_LOG(log, "GetBreakpointSize() failed: {0}", error);
    return error;
  } else
    LLDB_LOG(log, "breakpoint size: {0}", breakpoint_size);

  // First try probing for a breakpoint at a software breakpoint location: PC -
  // breakpoint size.
  const lldb::addr_t initial_pc_addr =
      context_sp->GetPCfromBreakpointLocation();
  lldb::addr_t breakpoint_addr = initial_pc_addr;
  if (breakpoint_size > 0) {
    // Do not allow breakpoint probe to wrap around.
    if (breakpoint_addr >= breakpoint_size)
      breakpoint_addr -= breakpoint_size;
  }

  // Check if we stopped because of a breakpoint.
  NativeBreakpointSP breakpoint_sp;
  error = m_breakpoint_list.GetBreakpoint(breakpoint_addr, breakpoint_sp);
  if (!error.Success() || !breakpoint_sp) {
    // We didn't find one at a software probe location.  Nothing to do.
    LLDB_LOG(log,
             "pid {0} no lldb breakpoint found at current pc with "
             "adjustment: {1}",
             GetID(), breakpoint_addr);
    return Status();
  }

  // If the breakpoint is not a software breakpoint, nothing to do.
  if (!breakpoint_sp->IsSoftwareBreakpoint()) {
    LLDB_LOG(
        log,
        "pid {0} breakpoint found at {1:x}, not software, nothing to adjust",
        GetID(), breakpoint_addr);
    return Status();
  }

  //
  // We have a software breakpoint and need to adjust the PC.
  //

  // Sanity check.
  if (breakpoint_size == 0) {
    // Nothing to do!  How did we get here?
    LLDB_LOG(log,
             "pid {0} breakpoint found at {1:x}, it is software, but the "
             "size is zero, nothing to do (unexpected)",
             GetID(), breakpoint_addr);
    return Status();
  }

  // Change the program counter.
  LLDB_LOG(log, "pid {0} tid {1}: changing PC from {2:x} to {3:x}", GetID(),
           thread.GetID(), initial_pc_addr, breakpoint_addr);

  error = context_sp->SetPC(breakpoint_addr);
  if (error.Fail()) {
    LLDB_LOG(log, "pid {0} tid {1}: failed to set PC: {2}", GetID(),
             thread.GetID(), error);
    return error;
  }

  return error;
}

Status NativeProcessLinux::GetLoadedModuleFileSpec(const char *module_path,
                                                   FileSpec &file_spec) {
  Status error = PopulateMemoryRegionCache();
  if (error.Fail())
    return error;

  FileSpec module_file_spec(module_path, true);

  file_spec.Clear();
  for (const auto &it : m_mem_region_cache) {
    if (it.second.GetFilename() == module_file_spec.GetFilename()) {
      file_spec = it.second;
      return Status();
    }
  }
  return Status("Module file (%s) not found in /proc/%" PRIu64 "/maps file!",
                module_file_spec.GetFilename().AsCString(), GetID());
}

Status NativeProcessLinux::GetFileLoadAddress(const llvm::StringRef &file_name,
                                              lldb::addr_t &load_addr) {
  load_addr = LLDB_INVALID_ADDRESS;
  Status error = PopulateMemoryRegionCache();
  if (error.Fail())
    return error;

  FileSpec file(file_name, false);
  for (const auto &it : m_mem_region_cache) {
    if (it.second == file) {
      load_addr = it.first.GetRange().GetRangeBase();
      return Status();
    }
  }
  return Status("No load address found for specified file.");
}

NativeThreadLinux *NativeProcessLinux::GetThreadByID(lldb::tid_t tid) {
  return static_cast<NativeThreadLinux *>(
      NativeProcessProtocol::GetThreadByID(tid));
}

Status NativeProcessLinux::ResumeThread(NativeThreadLinux &thread,
                                        lldb::StateType state, int signo) {
  Log *const log = ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_THREAD);
  LLDB_LOG(log, "tid: {0}", thread.GetID());

  // Before we do the resume below, first check if we have a pending
  // stop notification that is currently waiting for
  // all threads to stop.  This is potentially a buggy situation since
  // we're ostensibly waiting for threads to stop before we send out the
  // pending notification, and here we are resuming one before we send
  // out the pending stop notification.
  if (m_pending_notification_tid != LLDB_INVALID_THREAD_ID) {
    LLDB_LOG(log,
             "about to resume tid {0} per explicit request but we have a "
             "pending stop notification (tid {1}) that is actively "
             "waiting for this thread to stop. Valid sequence of events?",
             thread.GetID(), m_pending_notification_tid);
  }

  // Request a resume.  We expect this to be synchronous and the system
  // to reflect it is running after this completes.
  switch (state) {
  case eStateRunning: {
    const auto resume_result = thread.Resume(signo);
    if (resume_result.Success())
      SetState(eStateRunning, true);
    return resume_result;
  }
  case eStateStepping: {
    const auto step_result = thread.SingleStep(signo);
    if (step_result.Success())
      SetState(eStateRunning, true);
    return step_result;
  }
  default:
    LLDB_LOG(log, "Unhandled state {0}.", state);
    llvm_unreachable("Unhandled state for resume");
  }
}

//===----------------------------------------------------------------------===//

void NativeProcessLinux::StopRunningThreads(const lldb::tid_t triggering_tid) {
  Log *const log = ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_THREAD);
  LLDB_LOG(log, "about to process event: (triggering_tid: {0})",
           triggering_tid);

  m_pending_notification_tid = triggering_tid;

  // Request a stop for all the thread stops that need to be stopped
  // and are not already known to be stopped.
  for (const auto &thread : m_threads) {
    if (StateIsRunningState(thread->GetState()))
      static_cast<NativeThreadLinux *>(thread.get())->RequestStop();
  }

  SignalIfAllThreadsStopped();
  LLDB_LOG(log, "event processing done");
}

void NativeProcessLinux::SignalIfAllThreadsStopped() {
  if (m_pending_notification_tid == LLDB_INVALID_THREAD_ID)
    return; // No pending notification. Nothing to do.

  for (const auto &thread_sp : m_threads) {
    if (StateIsRunningState(thread_sp->GetState()))
      return; // Some threads are still running. Don't signal yet.
  }

  // We have a pending notification and all threads have stopped.
  Log *log(
      GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PROCESS | LIBLLDB_LOG_BREAKPOINTS));

  // Clear any temporary breakpoints we used to implement software single
  // stepping.
  for (const auto &thread_info : m_threads_stepping_with_breakpoint) {
    Status error = RemoveBreakpoint(thread_info.second);
    if (error.Fail())
      LLDB_LOG(log, "pid = {0} remove stepping breakpoint: {1}",
               thread_info.first, error);
  }
  m_threads_stepping_with_breakpoint.clear();

  // Notify the delegate about the stop
  SetCurrentThreadID(m_pending_notification_tid);
  SetState(StateType::eStateStopped, true);
  m_pending_notification_tid = LLDB_INVALID_THREAD_ID;
}

void NativeProcessLinux::ThreadWasCreated(NativeThreadLinux &thread) {
  Log *const log = ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_THREAD);
  LLDB_LOG(log, "tid: {0}", thread.GetID());

  if (m_pending_notification_tid != LLDB_INVALID_THREAD_ID &&
      StateIsRunningState(thread.GetState())) {
    // We will need to wait for this new thread to stop as well before firing
    // the
    // notification.
    thread.RequestStop();
  }
}

void NativeProcessLinux::SigchldHandler() {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));
  // Process all pending waitpid notifications.
  while (true) {
    int status = -1;
    ::pid_t wait_pid = llvm::sys::RetryAfterSignal(-1, ::waitpid, -1, &status,
                                          __WALL | __WNOTHREAD | WNOHANG);

    if (wait_pid == 0)
      break; // We are done.

    if (wait_pid == -1) {
      Status error(errno, eErrorTypePOSIX);
      LLDB_LOG(log, "waitpid (-1, &status, _) failed: {0}", error);
      break;
    }

    WaitStatus wait_status = WaitStatus::Decode(status);
    bool exited = wait_status.type == WaitStatus::Exit ||
                  (wait_status.type == WaitStatus::Signal &&
                   wait_pid == static_cast<::pid_t>(GetID()));

    LLDB_LOG(
        log,
        "waitpid (-1, &status, _) => pid = {0}, status = {1}, exited = {2}",
        wait_pid, wait_status, exited);

    MonitorCallback(wait_pid, exited, wait_status);
  }
}

// Wrapper for ptrace to catch errors and log calls.
// Note that ptrace sets errno on error because -1 can be a valid result (i.e.
// for PTRACE_PEEK*)
Status NativeProcessLinux::PtraceWrapper(int req, lldb::pid_t pid, void *addr,
                                         void *data, size_t data_size,
                                         long *result) {
  Status error;
  long int ret;

  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PTRACE));

  PtraceDisplayBytes(req, data, data_size);

  errno = 0;
  if (req == PTRACE_GETREGSET || req == PTRACE_SETREGSET)
    ret = ptrace(static_cast<__ptrace_request>(req), static_cast<::pid_t>(pid),
                 *(unsigned int *)addr, data);
  else
    ret = ptrace(static_cast<__ptrace_request>(req), static_cast<::pid_t>(pid),
                 addr, data);

  if (ret == -1)
    error.SetErrorToErrno();

  if (result)
    *result = ret;

  LLDB_LOG(log, "ptrace({0}, {1}, {2}, {3}, {4})={5:x}", req, pid, addr, data,
           data_size, ret);

  PtraceDisplayBytes(req, data, data_size);

  if (error.Fail())
    LLDB_LOG(log, "ptrace() failed: {0}", error);

  return error;
}

llvm::Expected<ProcessorTraceMonitor &>
NativeProcessLinux::LookupProcessorTraceInstance(lldb::user_id_t traceid,
                                                 lldb::tid_t thread) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PTRACE));
  if (thread == LLDB_INVALID_THREAD_ID && traceid == m_pt_proces_trace_id) {
    LLDB_LOG(log, "thread not specified: {0}", traceid);
    return Status("tracing not active thread not specified").ToError();
  }

  for (auto& iter : m_processor_trace_monitor) {
    if (traceid == iter.second->GetTraceID() &&
        (thread == iter.first || thread == LLDB_INVALID_THREAD_ID))
      return *(iter.second);
  }

  LLDB_LOG(log, "traceid not being traced: {0}", traceid);
  return Status("tracing not active for this thread").ToError();
}

Status NativeProcessLinux::GetMetaData(lldb::user_id_t traceid,
                                       lldb::tid_t thread,
                                       llvm::MutableArrayRef<uint8_t> &buffer,
                                       size_t offset) {
  TraceOptions trace_options;
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PTRACE));
  Status error;

  LLDB_LOG(log, "traceid {0}", traceid);

  auto perf_monitor = LookupProcessorTraceInstance(traceid, thread);
  if (!perf_monitor) {
    LLDB_LOG(log, "traceid not being traced: {0}", traceid);
    buffer = buffer.slice(buffer.size());
    error = perf_monitor.takeError();
    return error;
  }
  return (*perf_monitor).ReadPerfTraceData(buffer, offset);
}

Status NativeProcessLinux::GetData(lldb::user_id_t traceid, lldb::tid_t thread,
                                   llvm::MutableArrayRef<uint8_t> &buffer,
                                   size_t offset) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PTRACE));
  Status error;

  LLDB_LOG(log, "traceid {0}", traceid);

  auto perf_monitor = LookupProcessorTraceInstance(traceid, thread);
  if (!perf_monitor) {
    LLDB_LOG(log, "traceid not being traced: {0}", traceid);
    buffer = buffer.slice(buffer.size());
    error = perf_monitor.takeError();
    return error;
  }
  return (*perf_monitor).ReadPerfTraceAux(buffer, offset);
}

Status NativeProcessLinux::GetTraceConfig(lldb::user_id_t traceid,
                                          TraceOptions &config) {
  Status error;
  if (config.getThreadID() == LLDB_INVALID_THREAD_ID &&
      m_pt_proces_trace_id == traceid) {
    if (m_pt_proces_trace_id == LLDB_INVALID_UID) {
      error.SetErrorString("tracing not active for this process");
      return error;
    }
    config = m_pt_process_trace_config;
  } else {
    auto perf_monitor =
        LookupProcessorTraceInstance(traceid, config.getThreadID());
    if (!perf_monitor) {
      error = perf_monitor.takeError();
      return error;
    }
    error = (*perf_monitor).GetTraceConfig(config);
  }
  return error;
}

lldb::user_id_t
NativeProcessLinux::StartTraceGroup(const TraceOptions &config,
                                           Status &error) {

  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PTRACE));
  if (config.getType() != TraceType::eTraceTypeProcessorTrace)
    return LLDB_INVALID_UID;

  if (m_pt_proces_trace_id != LLDB_INVALID_UID) {
    error.SetErrorString("tracing already active on this process");
    return m_pt_proces_trace_id;
  }

  for (const auto &thread_sp : m_threads) {
    if (auto traceInstance = ProcessorTraceMonitor::Create(
            GetID(), thread_sp->GetID(), config, true)) {
      m_pt_traced_thread_group.insert(thread_sp->GetID());
      m_processor_trace_monitor.insert(
          std::make_pair(thread_sp->GetID(), std::move(*traceInstance)));
    }
  }

  m_pt_process_trace_config = config;
  error = ProcessorTraceMonitor::GetCPUType(m_pt_process_trace_config);

  // Trace on Complete process will have traceid of 0
  m_pt_proces_trace_id = 0;

  LLDB_LOG(log, "Process Trace ID {0}", m_pt_proces_trace_id);
  return m_pt_proces_trace_id;
}

lldb::user_id_t NativeProcessLinux::StartTrace(const TraceOptions &config,
                                               Status &error) {
  if (config.getType() != TraceType::eTraceTypeProcessorTrace)
    return NativeProcessProtocol::StartTrace(config, error);

  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PTRACE));

  lldb::tid_t threadid = config.getThreadID();

  if (threadid == LLDB_INVALID_THREAD_ID)
    return StartTraceGroup(config, error);

  auto thread_sp = GetThreadByID(threadid);
  if (!thread_sp) {
    // Thread not tracked by lldb so don't trace.
    error.SetErrorString("invalid thread id");
    return LLDB_INVALID_UID;
  }

  const auto &iter = m_processor_trace_monitor.find(threadid);
  if (iter != m_processor_trace_monitor.end()) {
    LLDB_LOG(log, "Thread already being traced");
    error.SetErrorString("tracing already active on this thread");
    return LLDB_INVALID_UID;
  }

  auto traceMonitor =
      ProcessorTraceMonitor::Create(GetID(), threadid, config, false);
  if (!traceMonitor) {
    error = traceMonitor.takeError();
    LLDB_LOG(log, "error {0}", error);
    return LLDB_INVALID_UID;
  }
  lldb::user_id_t ret_trace_id = (*traceMonitor)->GetTraceID();
  m_processor_trace_monitor.insert(
      std::make_pair(threadid, std::move(*traceMonitor)));
  return ret_trace_id;
}

Status NativeProcessLinux::StopTracingForThread(lldb::tid_t thread) {
  Status error;
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PTRACE));
  LLDB_LOG(log, "Thread {0}", thread);

  const auto& iter = m_processor_trace_monitor.find(thread);
  if (iter == m_processor_trace_monitor.end()) {
    error.SetErrorString("tracing not active for this thread");
    return error;
  }

  if (iter->second->GetTraceID() == m_pt_proces_trace_id) {
    // traceid maps to the whole process so we have to erase it from the
    // thread group.
    LLDB_LOG(log, "traceid maps to process");
    m_pt_traced_thread_group.erase(thread);
  }
  m_processor_trace_monitor.erase(iter);

  return error;
}

Status NativeProcessLinux::StopTrace(lldb::user_id_t traceid,
                                     lldb::tid_t thread) {
  Status error;

  TraceOptions trace_options;
  trace_options.setThreadID(thread);
  error = NativeProcessLinux::GetTraceConfig(traceid, trace_options);

  if (error.Fail())
    return error;

  switch (trace_options.getType()) {
  case lldb::TraceType::eTraceTypeProcessorTrace:
    if (traceid == m_pt_proces_trace_id &&
        thread == LLDB_INVALID_THREAD_ID)
      StopProcessorTracingOnProcess();
    else
      error = StopProcessorTracingOnThread(traceid, thread);
    break;
  default:
    error.SetErrorString("trace not supported");
    break;
  }

  return error;
}

void NativeProcessLinux::StopProcessorTracingOnProcess() {
  for (auto thread_id_iter : m_pt_traced_thread_group)
    m_processor_trace_monitor.erase(thread_id_iter);
  m_pt_traced_thread_group.clear();
  m_pt_proces_trace_id = LLDB_INVALID_UID;
}

Status NativeProcessLinux::StopProcessorTracingOnThread(lldb::user_id_t traceid,
                                                        lldb::tid_t thread) {
  Status error;
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PTRACE));

  if (thread == LLDB_INVALID_THREAD_ID) {
    for (auto& iter : m_processor_trace_monitor) {
      if (iter.second->GetTraceID() == traceid) {
        // Stopping a trace instance for an individual thread
        // hence there will only be one traceid that can match.
        m_processor_trace_monitor.erase(iter.first);
        return error;
      }
      LLDB_LOG(log, "Trace ID {0}", iter.second->GetTraceID());
    }

    LLDB_LOG(log, "Invalid TraceID");
    error.SetErrorString("invalid trace id");
    return error;
  }

  // thread is specified so we can use find function on the map.
  const auto& iter = m_processor_trace_monitor.find(thread);
  if (iter == m_processor_trace_monitor.end()) {
    // thread not found in our map.
    LLDB_LOG(log, "thread not being traced");
    error.SetErrorString("tracing not active for this thread");
    return error;
  }
  if (iter->second->GetTraceID() != traceid) {
    // traceid did not match so it has to be invalid.
    LLDB_LOG(log, "Invalid TraceID");
    error.SetErrorString("invalid trace id");
    return error;
  }

  LLDB_LOG(log, "UID - {0} , Thread -{1}", traceid, thread);

  if (traceid == m_pt_proces_trace_id) {
    // traceid maps to the whole process so we have to erase it from the
    // thread group.
    LLDB_LOG(log, "traceid maps to process");
    m_pt_traced_thread_group.erase(thread);
  }
  m_processor_trace_monitor.erase(iter);

  return error;
}
