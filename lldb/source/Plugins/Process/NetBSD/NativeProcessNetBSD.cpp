//===-- NativeProcessNetBSD.cpp ------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NativeProcessNetBSD.h"

// C Includes

// C++ Includes

// Other libraries and framework includes
#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"
#include "lldb/Core/State.h"
#include "lldb/Host/HostProcess.h"
#include "lldb/Host/common/NativeBreakpoint.h"
#include "lldb/Host/common/NativeRegisterContext.h"
#include "lldb/Host/posix/ProcessLauncherPosixFork.h"
#include "lldb/Target/Process.h"

// System includes - They have to be included after framework includes because
// they define some
// macros which collide with variable names in other modules
// clang-format off
#include <sys/types.h>
#include <sys/ptrace.h>
#include <sys/sysctl.h>
#include <sys/wait.h>
#include <uvm/uvm_prot.h>
#include <elf.h>
#include <util.h>
// clang-format on

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_netbsd;
using namespace llvm;

static ExitType convert_pid_status_to_exit_type(int status) {
  if (WIFEXITED(status))
    return ExitType::eExitTypeExit;
  else if (WIFSIGNALED(status))
    return ExitType::eExitTypeSignal;
  else if (WIFSTOPPED(status))
    return ExitType::eExitTypeStop;
  else {
    // We don't know what this is.
    return ExitType::eExitTypeInvalid;
  }
}

static int convert_pid_status_to_return_code(int status) {
  if (WIFEXITED(status))
    return WEXITSTATUS(status);
  else if (WIFSIGNALED(status))
    return WTERMSIG(status);
  else if (WIFSTOPPED(status))
    return WSTOPSIG(status);
  else {
    // We don't know what this is.
    return ExitType::eExitTypeInvalid;
  }
}

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

Status NativeProcessProtocol::Launch(
    ProcessLaunchInfo &launch_info,
    NativeProcessProtocol::NativeDelegate &native_delegate, MainLoop &mainloop,
    NativeProcessProtocolSP &native_process_sp) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));

  Status error;

  // Verify the working directory is valid if one was specified.
  FileSpec working_dir{launch_info.GetWorkingDirectory()};
  if (working_dir && (!working_dir.ResolvePath() ||
                      !llvm::sys::fs::is_directory(working_dir.GetPath()))) {
    error.SetErrorStringWithFormat("No such file or directory: %s",
                                   working_dir.GetCString());
    return error;
  }

  // Create the NativeProcessNetBSD in launch mode.
  native_process_sp.reset(new NativeProcessNetBSD());

  if (!native_process_sp->RegisterNativeDelegate(native_delegate)) {
    native_process_sp.reset();
    error.SetErrorStringWithFormat("failed to register the native delegate");
    return error;
  }

  error = std::static_pointer_cast<NativeProcessNetBSD>(native_process_sp)
              ->LaunchInferior(mainloop, launch_info);

  if (error.Fail()) {
    native_process_sp.reset();
    LLDB_LOG(log, "failed to launch process: {0}", error);
    return error;
  }

  launch_info.SetProcessID(native_process_sp->GetID());

  return error;
}

Status NativeProcessProtocol::Attach(
    lldb::pid_t pid, NativeProcessProtocol::NativeDelegate &native_delegate,
    MainLoop &mainloop, NativeProcessProtocolSP &native_process_sp) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));
  LLDB_LOG(log, "pid = {0:x}", pid);

  // Retrieve the architecture for the running process.
  ArchSpec process_arch;
  Status error = ResolveProcessArchitecture(pid, process_arch);
  if (!error.Success())
    return error;

  std::shared_ptr<NativeProcessNetBSD> native_process_netbsd_sp(
      new NativeProcessNetBSD());

  if (!native_process_netbsd_sp->RegisterNativeDelegate(native_delegate)) {
    error.SetErrorStringWithFormat("failed to register the native delegate");
    return error;
  }

  native_process_netbsd_sp->AttachToInferior(mainloop, pid, error);
  if (!error.Success())
    return error;

  native_process_sp = native_process_netbsd_sp;
  return error;
}

// -----------------------------------------------------------------------------
// Public Instance Methods
// -----------------------------------------------------------------------------

NativeProcessNetBSD::NativeProcessNetBSD()
    : NativeProcessProtocol(LLDB_INVALID_PROCESS_ID), m_arch(),
      m_supports_mem_region(eLazyBoolCalculate), m_mem_region_cache() {}

// Handles all waitpid events from the inferior process.
void NativeProcessNetBSD::MonitorCallback(lldb::pid_t pid, int signal) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));

  switch (signal) {
  case SIGTRAP:
    return MonitorSIGTRAP(pid);
  case SIGSTOP:
    return MonitorSIGSTOP(pid);
  default:
    return MonitorSignal(pid, signal);
  }
}

void NativeProcessNetBSD::MonitorExited(lldb::pid_t pid, int signal,
                                        int status) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));

  LLDB_LOG(log, "got exit signal({0}) , pid = {1}", signal, pid);

  /* Stop Tracking All Threads attached to Process */
  m_threads.clear();

  SetExitStatus(convert_pid_status_to_exit_type(status),
                convert_pid_status_to_return_code(status), nullptr, true);

  // Notify delegate that our process has exited.
  SetState(StateType::eStateExited, true);
}

void NativeProcessNetBSD::MonitorSIGSTOP(lldb::pid_t pid) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));
  ptrace_siginfo_t info;

  const auto siginfo_err =
      PtraceWrapper(PT_GET_SIGINFO, pid, &info, sizeof(info));

  // Get details on the signal raised.
  if (siginfo_err.Success()) {
    // Handle SIGSTOP from LLGS (LLDB GDB Server)
    if (info.psi_siginfo.si_code == SI_USER &&
        info.psi_siginfo.si_pid == ::getpid()) {
      /* Stop Tracking All Threads attached to Process */
      for (const auto &thread_sp : m_threads) {
        static_pointer_cast<NativeThreadNetBSD>(thread_sp)->SetStoppedBySignal(
            SIGSTOP, &info.psi_siginfo);
      }
    }
  }
}

void NativeProcessNetBSD::MonitorSIGTRAP(lldb::pid_t pid) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));
  ptrace_siginfo_t info;

  const auto siginfo_err =
      PtraceWrapper(PT_GET_SIGINFO, pid, &info, sizeof(info));

  // Get details on the signal raised.
  if (siginfo_err.Fail()) {
    return;
  }

  switch (info.psi_siginfo.si_code) {
  case TRAP_BRKPT:
    for (const auto &thread_sp : m_threads) {
      static_pointer_cast<NativeThreadNetBSD>(thread_sp)
          ->SetStoppedByBreakpoint();
      FixupBreakpointPCAsNeeded(
          *static_pointer_cast<NativeThreadNetBSD>(thread_sp));
    }
    SetState(StateType::eStateStopped, true);
    break;
  case TRAP_TRACE:
    for (const auto &thread_sp : m_threads) {
      static_pointer_cast<NativeThreadNetBSD>(thread_sp)->SetStoppedByTrace();
    }
    SetState(StateType::eStateStopped, true);
    break;
  case TRAP_EXEC: {
    Status error = ReinitializeThreads();
    if (error.Fail()) {
      SetState(StateType::eStateInvalid);
      return;
    }

    // Let our delegate know we have just exec'd.
    NotifyDidExec();

    for (const auto &thread_sp : m_threads) {
      static_pointer_cast<NativeThreadNetBSD>(thread_sp)->SetStoppedByExec();
    }
    SetState(StateType::eStateStopped, true);
  } break;
  case TRAP_DBREG: {
    // If a watchpoint was hit, report it
    uint32_t wp_index;
    Status error =
        static_pointer_cast<NativeThreadNetBSD>(m_threads[info.psi_lwpid])
            ->GetRegisterContext()
            ->GetWatchpointHitIndex(wp_index,
                                    (uintptr_t)info.psi_siginfo.si_addr);
    if (error.Fail())
      LLDB_LOG(log,
               "received error while checking for watchpoint hits, pid = "
               "{0}, LWP = {1}, error = {2}",
               GetID(), info.psi_lwpid, error);
    if (wp_index != LLDB_INVALID_INDEX32) {
      for (const auto &thread_sp : m_threads) {
        static_pointer_cast<NativeThreadNetBSD>(thread_sp)
            ->SetStoppedByWatchpoint(wp_index);
      }
      SetState(StateType::eStateStopped, true);
      break;
    }

    // If a breakpoint was hit, report it
    uint32_t bp_index;
    error = static_pointer_cast<NativeThreadNetBSD>(m_threads[info.psi_lwpid])
                ->GetRegisterContext()
                ->GetHardwareBreakHitIndex(bp_index,
                                           (uintptr_t)info.psi_siginfo.si_addr);
    if (error.Fail())
      LLDB_LOG(log,
               "received error while checking for hardware "
               "breakpoint hits, pid = {0}, LWP = {1}, error = {2}",
               GetID(), info.psi_lwpid, error);
    if (bp_index != LLDB_INVALID_INDEX32) {
      for (const auto &thread_sp : m_threads) {
        static_pointer_cast<NativeThreadNetBSD>(thread_sp)
            ->SetStoppedByBreakpoint();
      }
      SetState(StateType::eStateStopped, true);
      break;
    }
  } break;
  }
}

void NativeProcessNetBSD::MonitorSignal(lldb::pid_t pid, int signal) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));

  ptrace_siginfo_t info;
  const auto siginfo_err =
      PtraceWrapper(PT_GET_SIGINFO, pid, &info, sizeof(info));

  for (const auto &thread_sp : m_threads) {
    static_pointer_cast<NativeThreadNetBSD>(thread_sp)->SetStoppedBySignal(
        info.psi_siginfo.si_signo, &info.psi_siginfo);
  }
  SetState(StateType::eStateStopped, true);
}

Status NativeProcessNetBSD::PtraceWrapper(int req, lldb::pid_t pid, void *addr,
                                          int data, int *result) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PTRACE));
  Status error;
  int ret;

  errno = 0;
  ret = ptrace(req, static_cast<::pid_t>(pid), addr, data);

  if (ret == -1)
    error.SetErrorToErrno();

  if (result)
    *result = ret;

  LLDB_LOG(log, "ptrace({0}, {1}, {2}, {3})={4:x}", req, pid, addr, data, ret);

  if (error.Fail())
    LLDB_LOG(log, "ptrace() failed: {0}", error);

  return error;
}

Status NativeProcessNetBSD::GetSoftwareBreakpointPCOffset(
    uint32_t &actual_opcode_size) {
  // FIXME put this behind a breakpoint protocol class that can be
  // set per architecture.  Need ARM, MIPS support here.
  static const uint8_t g_i386_opcode[] = {0xCC};
  switch (m_arch.GetMachine()) {
  case llvm::Triple::x86_64:
    actual_opcode_size = static_cast<uint32_t>(sizeof(g_i386_opcode));
    return Status();
  default:
    assert(false && "CPU type not supported!");
    return Status("CPU type not supported");
  }
}

Status
NativeProcessNetBSD::FixupBreakpointPCAsNeeded(NativeThreadNetBSD &thread) {
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
  // First try probing for a breakpoint at a software breakpoint location: PC
  // - breakpoint size.
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

Status NativeProcessNetBSD::Resume(const ResumeActionList &resume_actions) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));
  LLDB_LOG(log, "pid {0}", GetID());

  const auto &thread_sp = m_threads[0];
  const ResumeAction *const action =
      resume_actions.GetActionForThread(thread_sp->GetID(), true);

  if (action == nullptr) {
    LLDB_LOG(log, "no action specified for pid {0} tid {1}", GetID(),
             thread_sp->GetID());
    return Status();
  }

  Status error;

  switch (action->state) {
  case eStateRunning: {
    // Run the thread, possibly feeding it the signal.
    error = NativeProcessNetBSD::PtraceWrapper(PT_CONTINUE, GetID(), (void *)1,
                                               action->signal);
    if (!error.Success())
      return error;
    for (const auto &thread_sp : m_threads) {
      static_pointer_cast<NativeThreadNetBSD>(thread_sp)->SetRunning();
    }
    SetState(eStateRunning, true);
    break;
  }
  case eStateStepping:
    // Run the thread, possibly feeding it the signal.
    error = NativeProcessNetBSD::PtraceWrapper(PT_STEP, GetID(), (void *)1,
                                               action->signal);
    if (!error.Success())
      return error;
    for (const auto &thread_sp : m_threads) {
      static_pointer_cast<NativeThreadNetBSD>(thread_sp)->SetStepping();
    }
    SetState(eStateStepping, true);
    break;

  case eStateSuspended:
  case eStateStopped:
    llvm_unreachable("Unexpected state");

  default:
    return Status("NativeProcessNetBSD::%s (): unexpected state %s specified "
                  "for pid %" PRIu64 ", tid %" PRIu64,
                  __FUNCTION__, StateAsCString(action->state), GetID(),
                  thread_sp->GetID());
  }

  return Status();
}

Status NativeProcessNetBSD::Halt() {
  Status error;

  if (kill(GetID(), SIGSTOP) != 0)
    error.SetErrorToErrno();

  return error;
}

Status NativeProcessNetBSD::Detach() {
  Status error;

  // Stop monitoring the inferior.
  m_sigchld_handle.reset();

  // Tell ptrace to detach from the process.
  if (GetID() == LLDB_INVALID_PROCESS_ID)
    return error;

  return PtraceWrapper(PT_DETACH, GetID());
}

Status NativeProcessNetBSD::Signal(int signo) {
  Status error;

  if (kill(GetID(), signo))
    error.SetErrorToErrno();

  return error;
}

Status NativeProcessNetBSD::Kill() {
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

  if (kill(GetID(), SIGKILL) != 0) {
    error.SetErrorToErrno();
    return error;
  }

  return error;
}

Status NativeProcessNetBSD::GetMemoryRegionInfo(lldb::addr_t load_addr,
                                                MemoryRegionInfo &range_info) {

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
    // Sanity check assumption that memory map entries are ascending.
    assert((proc_entry_info.GetRange().GetRangeBase() >= prev_base_address) &&
           "descending memory map entries detected, unexpected");
    prev_base_address = proc_entry_info.GetRange().GetRangeBase();
    UNUSED_IF_ASSERT_DISABLED(prev_base_address);
    // If the target address comes before this entry, indicate distance to
    // next region.
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
  // load_addr as start and the amount of bytes betwwen load address and the
  // end of the memory as size.
  range_info.GetRange().SetRangeBase(load_addr);
  range_info.GetRange().SetRangeEnd(LLDB_INVALID_ADDRESS);
  range_info.SetReadable(MemoryRegionInfo::OptionalBool::eNo);
  range_info.SetWritable(MemoryRegionInfo::OptionalBool::eNo);
  range_info.SetExecutable(MemoryRegionInfo::OptionalBool::eNo);
  range_info.SetMapped(MemoryRegionInfo::OptionalBool::eNo);
  return error;
}

Status NativeProcessNetBSD::PopulateMemoryRegionCache() {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));
  // If our cache is empty, pull the latest.  There should always be at least
  // one memory region if memory region handling is supported.
  if (!m_mem_region_cache.empty()) {
    LLDB_LOG(log, "reusing {0} cached memory region entries",
             m_mem_region_cache.size());
    return Status();
  }

  struct kinfo_vmentry *vm;
  size_t count, i;
  vm = kinfo_getvmmap(GetID(), &count);
  if (vm == NULL) {
    m_supports_mem_region = LazyBool::eLazyBoolNo;
    Status error;
    error.SetErrorString("not supported");
    return error;
  }
  for (i = 0; i < count; i++) {
    MemoryRegionInfo info;
    info.Clear();
    info.GetRange().SetRangeBase(vm[i].kve_start);
    info.GetRange().SetRangeEnd(vm[i].kve_end);
    info.SetMapped(MemoryRegionInfo::OptionalBool::eYes);

    if (vm[i].kve_protection & VM_PROT_READ)
      info.SetReadable(MemoryRegionInfo::OptionalBool::eYes);
    else
      info.SetReadable(MemoryRegionInfo::OptionalBool::eNo);

    if (vm[i].kve_protection & VM_PROT_WRITE)
      info.SetWritable(MemoryRegionInfo::OptionalBool::eYes);
    else
      info.SetWritable(MemoryRegionInfo::OptionalBool::eNo);

    if (vm[i].kve_protection & VM_PROT_EXECUTE)
      info.SetExecutable(MemoryRegionInfo::OptionalBool::eYes);
    else
      info.SetExecutable(MemoryRegionInfo::OptionalBool::eNo);

    if (vm[i].kve_path[0])
      info.SetName(vm[i].kve_path);

    m_mem_region_cache.emplace_back(
        info, FileSpec(info.GetName().GetCString(), true));
  }
  free(vm);

  if (m_mem_region_cache.empty()) {
    // No entries after attempting to read them.  This shouldn't happen.
    // Assume we don't support map entries.
    LLDB_LOG(log, "failed to find any vmmap entries, assuming no support "
                  "for memory region metadata retrieval");
    m_supports_mem_region = LazyBool::eLazyBoolNo;
    Status error;
    error.SetErrorString("not supported");
    return error;
  }
  LLDB_LOG(log, "read {0} memory region entries from process {1}",
           m_mem_region_cache.size(), GetID());
  // We support memory retrieval, remember that.
  m_supports_mem_region = LazyBool::eLazyBoolYes;
  return Status();
}

Status NativeProcessNetBSD::AllocateMemory(size_t size, uint32_t permissions,
                                           lldb::addr_t &addr) {
  return Status("Unimplemented");
}

Status NativeProcessNetBSD::DeallocateMemory(lldb::addr_t addr) {
  return Status("Unimplemented");
}

lldb::addr_t NativeProcessNetBSD::GetSharedLibraryInfoAddress() {
  // punt on this for now
  return LLDB_INVALID_ADDRESS;
}

size_t NativeProcessNetBSD::UpdateThreads() { return m_threads.size(); }

bool NativeProcessNetBSD::GetArchitecture(ArchSpec &arch) const {
  arch = m_arch;
  return true;
}

Status NativeProcessNetBSD::SetBreakpoint(lldb::addr_t addr, uint32_t size,
                                          bool hardware) {
  if (hardware)
    return Status("NativeProcessNetBSD does not support hardware breakpoints");
  else
    return SetSoftwareBreakpoint(addr, size);
}

Status NativeProcessNetBSD::GetSoftwareBreakpointTrapOpcode(
    size_t trap_opcode_size_hint, size_t &actual_opcode_size,
    const uint8_t *&trap_opcode_bytes) {
  static const uint8_t g_i386_opcode[] = {0xCC};

  switch (m_arch.GetMachine()) {
  case llvm::Triple::x86:
  case llvm::Triple::x86_64:
    trap_opcode_bytes = g_i386_opcode;
    actual_opcode_size = sizeof(g_i386_opcode);
    return Status();
  default:
    assert(false && "CPU type not supported!");
    return Status("CPU type not supported");
  }
}

Status NativeProcessNetBSD::GetLoadedModuleFileSpec(const char *module_path,
                                                    FileSpec &file_spec) {
  return Status("Unimplemented");
}

Status NativeProcessNetBSD::GetFileLoadAddress(const llvm::StringRef &file_name,
                                               lldb::addr_t &load_addr) {
  load_addr = LLDB_INVALID_ADDRESS;
  return Status();
}

Status NativeProcessNetBSD::LaunchInferior(MainLoop &mainloop,
                                           ProcessLaunchInfo &launch_info) {
  Status error;
  m_sigchld_handle = mainloop.RegisterSignal(
      SIGCHLD, [this](MainLoopBase &) { SigchldHandler(); }, error);
  if (!m_sigchld_handle)
    return error;

  SetState(eStateLaunching);

  ::pid_t pid = ProcessLauncherPosixFork()
                    .LaunchProcess(launch_info, error)
                    .GetProcessId();
  if (error.Fail())
    return error;

  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));

  // Wait for the child process to trap on its call to execve.
  ::pid_t wpid;
  int status;
  if ((wpid = waitpid(pid, &status, 0)) < 0) {
    error.SetErrorToErrno();
    LLDB_LOG(log, "waitpid for inferior failed with %s", error);

    // Mark the inferior as invalid.
    // FIXME this could really use a new state - eStateLaunchFailure.  For
    // now, using eStateInvalid.
    SetState(StateType::eStateInvalid);

    return error;
  }
  assert(WIFSTOPPED(status) && (wpid == static_cast<::pid_t>(pid)) &&
         "Could not sync with inferior process.");

  LLDB_LOG(log, "inferior started, now in stopped state");

  // Release the master terminal descriptor and pass it off to the
  // NativeProcessNetBSD instance.  Similarly stash the inferior pid.
  m_terminal_fd = launch_info.GetPTY().ReleaseMasterFileDescriptor();
  m_pid = pid;
  launch_info.SetProcessID(pid);

  if (m_terminal_fd != -1) {
    error = EnsureFDFlags(m_terminal_fd, O_NONBLOCK);
    if (error.Fail()) {
      LLDB_LOG(log,
               "inferior EnsureFDFlags failed for ensuring terminal "
               "O_NONBLOCK setting: {0}",
               error);

      // Mark the inferior as invalid.
      // FIXME this could really use a new state - eStateLaunchFailure.  For
      // now, using eStateInvalid.
      SetState(StateType::eStateInvalid);

      return error;
    }
  }

  LLDB_LOG(log, "adding pid = {0}", pid);

  ResolveProcessArchitecture(m_pid, m_arch);

  error = ReinitializeThreads();
  if (error.Fail()) {
    SetState(StateType::eStateInvalid);
    return error;
  }

  for (const auto &thread_sp : m_threads) {
    static_pointer_cast<NativeThreadNetBSD>(thread_sp)->SetStoppedBySignal(
        SIGSTOP);
  }

  /* Set process stopped */
  SetState(StateType::eStateStopped);

  if (error.Fail())
    LLDB_LOG(log, "inferior launching failed {0}", error);
  return error;
}

void NativeProcessNetBSD::AttachToInferior(MainLoop &mainloop, lldb::pid_t pid,
                                           Status &error) {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));
  LLDB_LOG(log, "pid = {0:x}", pid);

  m_sigchld_handle = mainloop.RegisterSignal(
      SIGCHLD, [this](MainLoopBase &) { SigchldHandler(); }, error);
  if (!m_sigchld_handle)
    return;

  error = ResolveProcessArchitecture(pid, m_arch);
  if (!error.Success())
    return;

  // Set the architecture to the exe architecture.
  LLDB_LOG(log, "pid = {0:x}, detected architecture {1}", pid,
           m_arch.GetArchitectureName());

  m_pid = pid;
  SetState(eStateAttaching);

  Attach(pid, error);
}

void NativeProcessNetBSD::SigchldHandler() {
  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_PROCESS));
  // Process all pending waitpid notifications.
  int status;
  ::pid_t wait_pid = waitpid(GetID(), &status, WALLSIG | WNOHANG);

  if (wait_pid == 0)
    return; // We are done.

  if (wait_pid == -1) {
    if (errno == EINTR)
      return;

    Status error(errno, eErrorTypePOSIX);
    LLDB_LOG(log, "waitpid ({0}, &status, _) failed: {1}", GetID(), error);
  }

  bool exited = false;
  int signal = 0;
  int exit_status = 0;
  const char *status_cstr = nullptr;
  if (WIFSTOPPED(status)) {
    signal = WSTOPSIG(status);
    status_cstr = "STOPPED";
  } else if (WIFEXITED(status)) {
    exit_status = WEXITSTATUS(status);
    status_cstr = "EXITED";
    exited = true;
  } else if (WIFSIGNALED(status)) {
    signal = WTERMSIG(status);
    status_cstr = "SIGNALED";
    if (wait_pid == static_cast<::pid_t>(GetID())) {
      exited = true;
      exit_status = -1;
    }
  } else
    status_cstr = "(\?\?\?)";

  LLDB_LOG(log,
           "waitpid ({0}, &status, _) => pid = {1}, status = {2:x} "
           "({3}), signal = {4}, exit_state = {5}",
           GetID(), wait_pid, status, status_cstr, signal, exit_status);

  if (exited)
    MonitorExited(wait_pid, signal, exit_status);
  else
    MonitorCallback(wait_pid, signal);
}

NativeThreadNetBSDSP NativeProcessNetBSD::AddThread(lldb::tid_t thread_id) {

  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_THREAD));
  LLDB_LOG(log, "pid {0} adding thread with tid {1}", GetID(), thread_id);

  assert(!HasThreadNoLock(thread_id) &&
         "attempted to add a thread by id that already exists");

  // If this is the first thread, save it as the current thread
  if (m_threads.empty())
    SetCurrentThreadID(thread_id);

  auto thread_sp = std::make_shared<NativeThreadNetBSD>(this, thread_id);
  m_threads.push_back(thread_sp);
  return thread_sp;
}

::pid_t NativeProcessNetBSD::Attach(lldb::pid_t pid, Status &error) {
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));

  if (pid <= 1) {
    error.SetErrorToGenericError();
    error.SetErrorString("Attaching to process 1 is not allowed.");
    return -1;
  }

  // Attach to the requested process.
  // An attach will cause the thread to stop with a SIGSTOP.
  error = PtraceWrapper(PT_ATTACH, pid);
  if (error.Fail())
    return -1;

  int status;
  // Need to use WALLSIG otherwise we receive an error with errno=ECHLD
  // At this point we should have a thread stopped if waitpid succeeds.
  if ((status = waitpid(pid, NULL, WALLSIG)) < 0)
    return -1;

  m_pid = pid;

  /* Initialize threads */
  error = ReinitializeThreads();
  if (error.Fail()) {
    SetState(StateType::eStateInvalid);
    return -1;
  }

  for (const auto &thread_sp : m_threads) {
    static_pointer_cast<NativeThreadNetBSD>(thread_sp)->SetStoppedBySignal(
        SIGSTOP);
  }

  // Let our process instance know the thread has stopped.
  SetState(StateType::eStateStopped);

  return pid;
}

Status NativeProcessNetBSD::ReadMemory(lldb::addr_t addr, void *buf,
                                       size_t size, size_t &bytes_read) {
  unsigned char *dst = static_cast<unsigned char *>(buf);
  struct ptrace_io_desc io;

  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_MEMORY));
  LLDB_LOG(log, "addr = {0}, buf = {1}, size = {2}", addr, buf, size);

  bytes_read = 0;
  io.piod_op = PIOD_READ_D;
  io.piod_len = size;

  do {
    io.piod_offs = (void *)(addr + bytes_read);
    io.piod_addr = dst + bytes_read;

    Status error = NativeProcessNetBSD::PtraceWrapper(PT_IO, GetID(), &io);
    if (error.Fail())
      return error;

    bytes_read = io.piod_len;
    io.piod_len = size - bytes_read;
  } while (bytes_read < size);

  return Status();
}

Status NativeProcessNetBSD::ReadMemoryWithoutTrap(lldb::addr_t addr, void *buf,
                                                  size_t size,
                                                  size_t &bytes_read) {
  Status error = ReadMemory(addr, buf, size, bytes_read);
  if (error.Fail())
    return error;
  return m_breakpoint_list.RemoveTrapsFromBuffer(addr, buf, size);
}

Status NativeProcessNetBSD::WriteMemory(lldb::addr_t addr, const void *buf,
                                        size_t size, size_t &bytes_written) {
  const unsigned char *src = static_cast<const unsigned char *>(buf);
  Status error;
  struct ptrace_io_desc io;

  Log *log(ProcessPOSIXLog::GetLogIfAllCategoriesSet(POSIX_LOG_MEMORY));
  LLDB_LOG(log, "addr = {0}, buf = {1}, size = {2}", addr, buf, size);

  bytes_written = 0;
  io.piod_op = PIOD_WRITE_D;
  io.piod_len = size;

  do {
    io.piod_addr = (void *)(src + bytes_written);
    io.piod_offs = (void *)(addr + bytes_written);

    Status error = NativeProcessNetBSD::PtraceWrapper(PT_IO, GetID(), &io);
    if (error.Fail())
      return error;

    bytes_written = io.piod_len;
    io.piod_len = size - bytes_written;
  } while (bytes_written < size);

  return error;
}

llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
NativeProcessNetBSD::GetAuxvData() const {
  /*
   * ELF_AUX_ENTRIES is currently restricted to kernel
   * (<sys/exec_elf.h> r. 1.155 specifies 15)
   *
   * ptrace(2) returns the whole AUXV including extra fiels after AT_NULL this
   * information isn't needed.
   */
  size_t auxv_size = 100 * sizeof(AuxInfo);

  ErrorOr<std::unique_ptr<MemoryBuffer>> buf =
      llvm::MemoryBuffer::getNewMemBuffer(auxv_size);

  struct ptrace_io_desc io = {.piod_op = PIOD_READ_AUXV,
                              .piod_offs = 0,
                              .piod_addr = (void *)buf.get()->getBufferStart(),
                              .piod_len = auxv_size};

  Status error = NativeProcessNetBSD::PtraceWrapper(PT_IO, GetID(), &io);

  if (error.Fail())
    return std::error_code(error.GetError(), std::generic_category());

  if (io.piod_len < 1)
    return std::error_code(ECANCELED, std::generic_category());

  return buf;
}

Status NativeProcessNetBSD::ReinitializeThreads() {
  // Clear old threads
  m_threads.clear();

  // Initialize new thread
  struct ptrace_lwpinfo info = {};
  Status error = PtraceWrapper(PT_LWPINFO, GetID(), &info, sizeof(info));
  if (error.Fail()) {
    return error;
  }
  // Reinitialize from scratch threads and register them in process
  while (info.pl_lwpid != 0) {
    NativeThreadNetBSDSP thread_sp = AddThread(info.pl_lwpid);
    error = PtraceWrapper(PT_LWPINFO, GetID(), &info, sizeof(info));
    if (error.Fail()) {
      return error;
    }
  }

  return error;
}
