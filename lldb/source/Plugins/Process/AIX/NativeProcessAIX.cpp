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
#define DECLARE_REGISTER_INFOS_PPC64_STRUCT
#include "Plugins/Process/Utility/RegisterInfos_ppc64.h"
#undef DECLARE_REGISTER_INFOS_PPC64_STRUCT

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
    return llvm::createStringError("could not sync with inferior process");
  }
  LLDB_LOG(log, "inferior started, now in stopped state");

  ProcessInstanceInfo Info;
  if (!Host::GetProcessInfo(pid, Info)) {
    return llvm::make_error<StringError>("Cannot get process architecture",
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
  while (true) {
    auto wait_result = WaitPid();
    if (!wait_result)
      return;
  }
}

void NativeProcessAIX::Manager::CollectThread(::pid_t tid) {}

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
  if (llvm::Error err = PtraceWrapper(PT_ATTACH, pid).takeError())
    return err;

  int wpid = llvm::sys::RetryAfterSignal(-1, ::waitpid, pid, nullptr, WNOHANG);
  if (wpid <= 0)
    return llvm::errorCodeToError(errnoAsErrorCode());
  LLDB_LOG(log, "adding pid = {0}", pid);

  return std::vector<::pid_t>{pid};
}

bool NativeProcessAIX::SupportHardwareSingleStepping() const { return false; }

Status NativeProcessAIX::Resume(const ResumeActionList &resume_actions) {
  return Status("unsupported");
}

Status NativeProcessAIX::Halt() { return Status("unsupported"); }

Status NativeProcessAIX::Detach() { return Status("unsupported"); }

Status NativeProcessAIX::Signal(int signo) { return Status("unsupported"); }

Status NativeProcessAIX::Interrupt() { return Status("unsupported"); }

Status NativeProcessAIX::Kill() {

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

  llvm::Expected<int> result =
      PtraceWrapper(PT_KILL, GetID(), nullptr, nullptr, 0);
  if (!result) {
    std::string error_string = std::string("Kill failed for process. error: ") +
                               llvm::toString(result.takeError());
    error.FromErrorString(error_string.c_str());
  }
  return error;
}

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

Status NativeProcessAIX::GetFileLoadAddress(const llvm::StringRef &file_name,
                                            lldb::addr_t &load_addr) {
  return Status("unsupported");
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
  return PtraceWrapper(PT_DETACH, tid).takeError();
}

template <typename GPR_T, typename PTSPRS_T>
int GetSPRs(int req, lldb::tid_t tid, GPR_T *gpr) {
  PTSPRS_T sprs;

  int ret = ptrace64(req, tid, reinterpret_cast<long long>(&sprs), 0, 0);

  if (ret != -1) {
    gpr->cr = sprs.pt_cr;
    gpr->msr = sprs.pt_msr;
    gpr->xer = sprs.pt_xer;
    gpr->lr = sprs.pt_lr;
    gpr->ctr = sprs.pt_ctr;
    gpr->pc = sprs.pt_iar;
  }
  return ret;
}

template <typename GPR_T, typename PTSPRS_T>
int SetSPRs(int req, lldb::tid_t tid, GPR_T *gpr) {
  PTSPRS_T sprs;

  sprs.pt_cr = gpr->cr;
  sprs.pt_msr = gpr->msr;
  sprs.pt_xer = gpr->xer;
  sprs.pt_lr = gpr->lr;
  sprs.pt_ctr = gpr->ctr;
  sprs.pt_iar = gpr->pc;

  return ptrace64(req, tid, reinterpret_cast<long long>(&sprs), 0, 0);
}

llvm::Expected<int> NativeProcessAIX::PtraceWrapper(int req, lldb::pid_t pid,
                                                    void *addr, void *data,
                                                    size_t data_size) {
  int ret = 0;
  Log *log = GetLog(POSIXLog::Ptrace);
  // PTT_* requests require a thread ID (TID).
  // Each entry under /proc/<pid>/lwp/ represents a thread,
  // and the directory name corresponds to its thread ID (TID).
  // A process may contain multiple threads.
  // TODO: With multi-threading support, iterate over all entries to enumerate
  // available TIDs and retrieve the target debugging thread ID.
  llvm::SmallString<128> proc_lwp_dir;
  llvm::sys::path::append(proc_lwp_dir, "/proc/", std::to_string(pid), "/lwp/");

  lldb::tid_t tid = 0;
  std::error_code ec;
  bool result;
  if (!llvm::sys::fs::is_directory(proc_lwp_dir, result) && result) {
    for (sys::fs::directory_iterator it(proc_lwp_dir, ec), end;
         it != end && !ec; it.increment(ec)) {
      llvm::StringRef name = llvm::sys::path::filename(it->path());

      if (name == "." || name == "..")
        continue;

      if (!name.getAsInteger(10, tid)) {
        break;
      }
    }
  }

  switch (req) {
  // On AIX, ptrace exposes differently. GPRs and SPRs are handled via separate
  // requests: PTT_READ_GPRS reads only GPRs & PTT_READ_SPRS is required to
  // fetch SPRs. Similarly, writes are also split across:
  // PTT_WRITE_GPRS & PTT_WRITE_SPRS.
  case PTT_READ_GPRS:
    if (data_size == sizeof(GPR_PPC)) // 32bit SPRs read
      ret = GetSPRs<GPR_PPC, ptsprs>(PTT_READ_SPRS, tid,
                                     static_cast<GPR_PPC *>(data));
    else if (data_size == sizeof(GPR_PPC64)) // 64bit SPRs read
      ret = GetSPRs<GPR_PPC64, ptxsprs>(PTT_READ_SPRS, tid,
                                        static_cast<GPR_PPC64 *>(data));

    if (ret != -1)
      ret = ptrace64(req, tid, reinterpret_cast<long long>(data), 0,
                     0); // read GPRs
    break;

  case PTT_WRITE_GPRS:
    if (data_size == sizeof(GPR_PPC)) // 32bit SPRs write
      ret = SetSPRs<GPR_PPC, ptsprs>(PTT_WRITE_SPRS, tid,
                                     static_cast<GPR_PPC *>(data));
    else if (data_size == sizeof(GPR_PPC64)) // 64bit SPRS write
      ret = SetSPRs<GPR_PPC64, ptxsprs>(PTT_WRITE_SPRS, tid,
                                        static_cast<GPR_PPC64 *>(data));

    if (ret != -1)
      ret = ptrace64(req, tid, reinterpret_cast<long long>(data), 0,
                     0); // write GPRs
    break;
  case PT_ATTACH:
  case PT_DETACH:
  case PT_KILL:
    ret = ptrace64(req, pid, 0, 0, nullptr);
    break;
  default:
    llvm_unreachable("PT_ request not supported yet.");
  }

  LLDB_LOG(log, "ptrace({0}, {1}, {2}, {3}, {4})={5:x}", req, pid, addr, data,
           data_size, ret);

  if (ret == -1) {
    LLDB_LOG(log, "ptrace() failed");
    return llvm::errorCodeToError(errnoAsErrorCode());
  }
  return ret;
}
