//===-- NativeProcessWindows.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/windows/windows.h"
#include <psapi.h>

#include "NativeProcessWindows.h"
#include "NativeThreadWindows.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostNativeProcessBase.h"
#include "lldb/Host/HostProcess.h"
#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Host/PseudoTerminal.h"
#include "lldb/Host/windows/AutoHandle.h"
#include "lldb/Host/windows/ConnectionConPTYWindows.h"
#include "lldb/Host/windows/HostThreadWindows.h"
#include "lldb/Host/windows/ProcessLauncherWindows.h"
#include "lldb/Host/windows/PseudoConsole.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/State.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

#include "DebuggerThread.h"
#include "ExceptionRecord.h"
#include "ProcessWindowsLog.h"

#include <tlhelp32.h>

#pragma warning(disable : 4005)
#include "winternl.h"
#include <ntstatus.h>

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

namespace lldb_private {

NativeProcessWindows::NativeProcessWindows(ProcessLaunchInfo &launch_info,
                                           NativeDelegate &delegate,
                                           llvm::Error &E)
    : NativeProcessProtocol(
          LLDB_INVALID_PROCESS_ID,
          PseudoTerminal::invalid_fd, // NativeProcessWindows owns the ConPTY.
          delegate),
      ProcessDebugger(), m_arch(launch_info.GetArchitecture()),
      m_stdio_communication("lldb.NativeProcessWindows.stdio") {
  ErrorAsOutParameter EOut(&E);
  DebugDelegateSP delegate_sp(new NativeDebugDelegate(*this));
  E = LaunchProcess(launch_info, delegate_sp).ToError();
  if (E)
    return;

  SetID(GetDebuggedProcessId());

  m_pty = launch_info.TakePTY();
  StartStdioForwarding();
}

NativeProcessWindows::NativeProcessWindows(lldb::pid_t pid, int terminal_fd,
                                           NativeDelegate &delegate,
                                           llvm::Error &E)
    : NativeProcessProtocol(pid, terminal_fd, delegate), ProcessDebugger(),
      m_stdio_communication("lldb.NativeProcessWindows.stdio") {
  ErrorAsOutParameter EOut(&E);
  DebugDelegateSP delegate_sp(new NativeDebugDelegate(*this));
  ProcessAttachInfo attach_info;
  attach_info.SetProcessID(pid);
  E = AttachProcess(pid, attach_info, delegate_sp).ToError();
  if (E)
    return;

  SetID(GetDebuggedProcessId());

  ProcessInstanceInfo info;
  if (!Host::GetProcessInfo(pid, info)) {
    E = createStringError(inconvertibleErrorCode(),
                          "Cannot get process information");
    return;
  }
  m_arch = info.GetArchitecture();
}

Status NativeProcessWindows::Resume(const ResumeActionList &resume_actions) {
  Log *log = GetLog(WindowsLog::Process);
  Status error;
  llvm::sys::ScopedLock lock(m_mutex);

  StateType state = GetState();
  if (state == eStateStopped || state == eStateCrashed) {
    LLDB_LOG(log, "process {0} is in state {1}.  Resuming...",
             GetDebuggedProcessId(), state);
    LLDB_LOG(log, "resuming {0} threads.", m_threads.size());

    m_pending_library_events = false;

    bool failed = false;
    for (uint32_t i = 0; i < m_threads.size(); ++i) {
      auto thread = static_cast<NativeThreadWindows *>(m_threads[i].get());
      const ResumeAction *const action =
          resume_actions.GetActionForThread(thread->GetID(), true);
      if (action == nullptr)
        continue;

      switch (action->state) {
      case eStateRunning:
      case eStateStepping: {
        Status result = thread->DoResume(action->state);
        if (result.Fail()) {
          failed = true;
          LLDB_LOG(log,
                   "Trying to resume thread at index {0}, but failed with "
                   "error {1}.",
                   i, result);
        }
        break;
      }
      case eStateSuspended:
      case eStateStopped:
        break;

      default:
        return Status::FromErrorStringWithFormat(
            "NativeProcessWindows::%s (): unexpected state %s specified "
            "for pid %" PRIu64 ", tid %" PRIu64,
            __FUNCTION__, StateAsCString(action->state), GetID(),
            thread->GetID());
      }
    }

    if (failed) {
      error = Status::FromErrorString("NativeProcessWindows::DoResume failed");
    } else {
      SetState(eStateRunning);
    }

    // Resume the debug loop.
    ExceptionRecordSP active_exception =
        m_session_data->m_debugger->GetActiveException().lock();
    if (active_exception) {
      // Resume the process and continue processing debug events.  Mask the
      // exception so that from the process's view, there is no indication that
      // anything happened.
      m_session_data->m_debugger->ContinueAsyncException(
          ExceptionResult::MaskException);
    }
  } else {
    LLDB_LOG(log, "error: process {0} is in state {1}.  Returning...",
             GetDebuggedProcessId(), GetState());
  }

  return error;
}

NativeThreadWindows *
NativeProcessWindows::GetThreadByID(lldb::tid_t thread_id) {
  return static_cast<NativeThreadWindows *>(
      NativeProcessProtocol::GetThreadByID(thread_id));
}

Status NativeProcessWindows::Halt() {
  bool caused_stop = false;
  StateType state = GetState();
  if (state != eStateStopped) {
    m_pending_halt = true;
    Status err = HaltProcess(caused_stop);
    if (err.Fail() || !caused_stop)
      m_pending_halt = false;
    return err;
  }
  return Status();
}

Status NativeProcessWindows::Detach() {
  Status error;
  Log *log = GetLog(WindowsLog::Process);
  StateType state = GetState();
  if (state != eStateExited && state != eStateDetached) {
    error = DetachProcess();
    if (error.Success())
      SetState(eStateDetached);
    else
      LLDB_LOG(log, "Detaching process error: {0}", error);
  } else {
    error = Status::FromErrorStringWithFormatv(
        "error: process {0} in state = {1}, but "
        "cannot detach it in this state.",
        GetID(), state);
    LLDB_LOG(log, "error: {0}", error);
  }
  return error;
}

Status NativeProcessWindows::Signal(int signo) {
  Status error;
  error = Status::FromErrorString(
      "Windows does not support sending signals to processes");
  return error;
}

Status NativeProcessWindows::Interrupt() { return Halt(); }

Status NativeProcessWindows::Kill() {
  StateType state = GetState();
  return DestroyProcess(state);
}

Status NativeProcessWindows::IgnoreSignals(llvm::ArrayRef<int> signals) {
  return Status();
}

Status NativeProcessWindows::GetMemoryRegionInfo(lldb::addr_t load_addr,
                                                 MemoryRegionInfo &range_info) {
  return ProcessDebugger::GetMemoryRegionInfo(load_addr, range_info);
}

Status NativeProcessWindows::ReadMemory(lldb::addr_t addr, void *buf,
                                        size_t size, size_t &bytes_read) {
  return ProcessDebugger::ReadMemory(addr, buf, size, bytes_read);
}

Status NativeProcessWindows::WriteMemory(lldb::addr_t addr, const void *buf,
                                         size_t size, size_t &bytes_written) {
  return ProcessDebugger::WriteMemory(addr, buf, size, bytes_written);
}

llvm::Expected<lldb::addr_t>
NativeProcessWindows::AllocateMemory(size_t size, uint32_t permissions) {
  lldb::addr_t addr;
  Status ST = ProcessDebugger::AllocateMemory(size, permissions, addr);
  if (ST.Success())
    return addr;
  return ST.ToError();
}

llvm::Error NativeProcessWindows::DeallocateMemory(lldb::addr_t addr) {
  return ProcessDebugger::DeallocateMemory(addr).ToError();
}

lldb::addr_t NativeProcessWindows::GetSharedLibraryInfoAddress() { return 0; }

bool NativeProcessWindows::IsAlive() const {
  StateType state = GetState();
  switch (state) {
  case eStateCrashed:
  case eStateDetached:
  case eStateExited:
  case eStateInvalid:
  case eStateUnloaded:
    return false;
  default:
    return true;
  }
}

void NativeProcessWindows::SetStopReasonForThread(NativeThreadWindows &thread,
                                                  lldb::StopReason reason,
                                                  std::string description) {
  SetCurrentThreadID(thread.GetID());

  ThreadStopInfo stop_info;
  stop_info.reason = reason;
  // No signal support on Windows but required to provide a 'valid' signum.
  stop_info.signo = SIGTRAP;

  if (reason == StopReason::eStopReasonException) {
    stop_info.details.exception.type = 0;
    stop_info.details.exception.data_count = 0;
  }

  thread.SetStopReason(stop_info, description);
}

void NativeProcessWindows::StopThread(lldb::tid_t thread_id,
                                      lldb::StopReason reason,
                                      std::string description) {
  NativeThreadWindows *thread = GetThreadByID(thread_id);
  if (!thread)
    return;

  Log *log = GetLog(WindowsLog::Thread);
  for (uint32_t i = 0; i < m_threads.size(); ++i) {
    auto t = static_cast<NativeThreadWindows *>(m_threads[i].get());
    if (Status error = t->DoStop(); error.Fail())
      LLDB_LOG(log, "failed to stop thread {0}: {1}", t->GetID(), error);
  }
  SetStopReasonForThread(*thread, reason, description);
}

size_t NativeProcessWindows::UpdateThreads() { return m_threads.size(); }

llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
NativeProcessWindows::GetAuxvData() const {
  // Not available on this target.
  return llvm::errc::not_supported;
}

llvm::Expected<llvm::ArrayRef<uint8_t>>
NativeProcessWindows::GetSoftwareBreakpointTrapOpcode(size_t size_hint) {
  static const uint8_t g_aarch64_opcode[] = {0x00, 0x00, 0x3e,
                                             0xd4};     // brk #0xf000
  static const uint8_t g_thumb_opcode[] = {0xfe, 0xde}; // udf #0xfe

  switch (GetArchitecture().GetMachine()) {
  case llvm::Triple::aarch64:
    return llvm::ArrayRef(g_aarch64_opcode);

  case llvm::Triple::arm:
  case llvm::Triple::thumb:
    return llvm::ArrayRef(g_thumb_opcode);

  default:
    return NativeProcessProtocol::GetSoftwareBreakpointTrapOpcode(size_hint);
  }
}

size_t NativeProcessWindows::GetSoftwareBreakpointPCOffset() {
  // Windows always reports an incremented PC after a breakpoint is hit,
  // even on ARM.
  return cantFail(GetSoftwareBreakpointTrapOpcode(0)).size();
}

bool NativeProcessWindows::FindSoftwareBreakpoint(lldb::addr_t addr) {
  return m_software_breakpoints.find(addr) != m_software_breakpoints.end();
}

Status NativeProcessWindows::SetBreakpoint(lldb::addr_t addr, uint32_t size,
                                           bool hardware) {
  if (hardware)
    return SetHardwareBreakpoint(addr, size);
  return SetSoftwareBreakpoint(addr, size);
}

Status NativeProcessWindows::RemoveBreakpoint(lldb::addr_t addr,
                                              bool hardware) {
  if (hardware)
    return RemoveHardwareBreakpoint(addr);
  return RemoveSoftwareBreakpoint(addr);
}

Status NativeProcessWindows::CacheLoadedModules() {
  Status error;
  if (!m_loaded_modules.empty())
    return Status();

  // Retrieve loaded modules by a Target/Module free implemenation.
  AutoHandle snapshot(CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, GetID()));
  if (snapshot.IsValid()) {
    MODULEENTRY32W me;
    me.dwSize = sizeof(MODULEENTRY32W);
    if (Module32FirstW(snapshot.get(), &me)) {
      do {
        std::string path;
        if (!llvm::convertWideToUTF8(me.szExePath, path))
          continue;

        FileSpec file_spec(path);
        FileSystem::Instance().Resolve(file_spec);
        m_loaded_modules[file_spec] = (addr_t)me.modBaseAddr;
      } while (Module32Next(snapshot.get(), &me));
    }

    if (!m_loaded_modules.empty())
      return Status();
  }

  error = Status(::GetLastError(), lldb::ErrorType::eErrorTypeWin32);
  return error;
}

Status NativeProcessWindows::GetLoadedModuleFileSpec(const char *module_path,
                                                     FileSpec &file_spec) {
  Status error = CacheLoadedModules();
  if (error.Fail())
    return error;

  FileSpec module_file_spec(module_path);
  FileSystem::Instance().Resolve(module_file_spec);
  for (auto &it : m_loaded_modules) {
    if (it.first == module_file_spec) {
      file_spec = it.first;
      return Status();
    }
  }
  return Status::FromErrorStringWithFormat(
      "Module (%s) not found in process %" PRIu64 "!",
      module_file_spec.GetPath().c_str(), GetID());
}

Status
NativeProcessWindows::GetFileLoadAddress(const llvm::StringRef &file_name,
                                         lldb::addr_t &load_addr) {
  Status error = CacheLoadedModules();
  if (error.Fail())
    return error;

  load_addr = LLDB_INVALID_ADDRESS;
  FileSpec file_spec(file_name);
  FileSystem::Instance().Resolve(file_spec);
  for (auto &it : m_loaded_modules) {
    if (it.first == file_spec) {
      load_addr = it.second;
      return Status();
    }
  }
  return Status::FromErrorStringWithFormat(
      "Can't get loaded address of file (%s) in process %" PRIu64 "!",
      file_spec.GetPath().c_str(), GetID());
}

llvm::Expected<std::vector<LoadedLibraryInfo>>
NativeProcessWindows::GetLoadedLibraries() {
  if (Status error = CacheLoadedModules(); error.Fail())
    return error.ToError();

  std::vector<LoadedLibraryInfo> libs;
  libs.reserve(m_loaded_modules.size());
  for (const auto &[file_spec, base] : m_loaded_modules) {
    LoadedLibraryInfo info;
    info.name = file_spec.GetPath();
    info.base_addr = base;
    libs.push_back(std::move(info));
  }
  return libs;
}

bool NativeProcessWindows::HasPendingLibraryEvents() {
  return m_pending_library_events;
}

void NativeProcessWindows::OnExitProcess(uint32_t exit_code) {
  Log *log = GetLog(WindowsLog::Process);
  LLDB_LOG(log, "Process {0} exited with code {1}", GetID(), exit_code);

  // Closing the ConPTY signals EOF on the parent-side STDOUT pipe so the
  // read thread can exit. Tear it down before the debuggee is destroyed.
  StopStdioForwarding();

  ProcessDebugger::OnExitProcess(exit_code);

  // No signal involved.  It is just an exit event.
  WaitStatus wait_status(WaitStatus::Exit, exit_code);
  SetExitStatus(wait_status, true);

  // Notify the native delegate.
  SetState(eStateExited, true);
}

void NativeProcessWindows::OnDebuggerConnected(lldb::addr_t image_base) {
  Log *log = GetLog(WindowsLog::Process);
  LLDB_LOG(log, "Debugger connected to process {0}. Image base = {1:x}",
           GetDebuggedProcessId(), image_base);

  // This is the earliest chance we can resolve the process ID and
  // architecture if we don't know them yet.
  if (GetID() == LLDB_INVALID_PROCESS_ID)
    SetID(GetDebuggedProcessId());

  if (GetArchitecture().GetMachine() == llvm::Triple::UnknownArch) {
    ProcessInstanceInfo process_info;
    if (!Host::GetProcessInfo(GetDebuggedProcessId(), process_info)) {
      LLDB_LOG(log, "Cannot get process information during debugger connecting "
                    "to process");
      return;
    }
    SetArchitecture(process_info.GetArchitecture());
  }

  // The very first one shall always be the main thread.
  assert(m_threads.empty());
  m_threads.push_back(std::make_unique<NativeThreadWindows>(
      *this, m_session_data->m_debugger->GetMainThread()));
}

ExceptionResult
NativeProcessWindows::HandleSingleStepException(const ExceptionRecord &record) {
  Log *log = GetLog(WindowsLog::Exception);
  uint32_t wp_id = LLDB_INVALID_INDEX32;
#ifndef __aarch64__
  if (NativeThreadWindows *thread = GetThreadByID(record.GetThreadID())) {
    NativeRegisterContextWindows &reg_ctx = thread->GetRegisterContext();
    Status error =
        reg_ctx.GetWatchpointHitIndex(wp_id, record.GetExceptionAddress());
    if (error.Fail())
      LLDB_LOG(log,
               "received error while checking for watchpoint hits, pid = "
               "{0}, error = {1}",
               thread->GetID(), error);
    if (wp_id != LLDB_INVALID_INDEX32) {
      addr_t wp_addr = reg_ctx.GetWatchpointAddress(wp_id);
      addr_t wp_hit_addr = reg_ctx.GetWatchpointHitAddress(wp_id);
      std::string desc =
          formatv("{0} {1} {2}", wp_addr, wp_id, wp_hit_addr).str();
      StopThread(record.GetThreadID(), StopReason::eStopReasonWatchpoint, desc);
    }
  }
#endif
  if (wp_id == LLDB_INVALID_INDEX32)
    StopThread(record.GetThreadID(), StopReason::eStopReasonTrace);

  SetState(eStateStopped, true);
  return ExceptionResult::MaskException;
}

ExceptionResult
NativeProcessWindows::HandleBreakpointException(const ExceptionRecord &record) {
  Log *log = GetLog(WindowsLog::Exception);
  const auto exception_addr = record.GetExceptionAddress();
  const auto thread_id = record.GetThreadID();

  if (NativeThreadWindows *stop_thread = GetThreadByID(thread_id)) {
    auto &reg_ctx = stop_thread->GetRegisterContext();

    if (FindSoftwareBreakpoint(exception_addr)) {
      LLDB_LOG(log, "Hit non-loader breakpoint at address {0:x}.",
               exception_addr);
      StopThread(thread_id, StopReason::eStopReasonBreakpoint);
      // The current PC is AFTER the BP opcode, on all architectures.
      reg_ctx.SetPC(reg_ctx.GetPC() - GetSoftwareBreakpointPCOffset());
      SetState(eStateStopped, true);
      return ExceptionResult::MaskException;
    }

    // This block of code will only be entered in case of a hardware
    // watchpoint or breakpoint hit on AArch64. However, we only handle
    // hardware watchpoints below as breakpoints are not yet supported.
    const std::vector<ULONG_PTR> &args = record.GetExceptionArguments();
    // Check that the ExceptionInformation array of EXCEPTION_RECORD
    // contains at least two elements: the first is a read-write flag
    // indicating the type of data access operation (read or write) while
    // the second contains the virtual address of the accessed data.
    if (args.size() >= 2) {
      uint32_t hw_id = LLDB_INVALID_INDEX32;
      Status error = reg_ctx.GetWatchpointHitIndex(hw_id, args[1]);
      if (error.Fail())
        LLDB_LOG(log,
                 "received error while checking for watchpoint hits, pid = "
                 "{0}, error = {1}",
                 thread_id, error);

      if (hw_id != LLDB_INVALID_INDEX32) {
        std::string desc =
            formatv("{0} {1} {2}", reg_ctx.GetWatchpointAddress(hw_id), hw_id,
                    exception_addr)
                .str();
        StopThread(thread_id, StopReason::eStopReasonWatchpoint, desc);
        SetState(eStateStopped, true);
        return ExceptionResult::MaskException;
      }
    }
  }

  if (!m_initial_stop_seen) {
    m_initial_stop_seen = true;
    LLDB_LOG(log,
             "Hit loader breakpoint at address {0:x}, setting initial stop "
             "event.",
             exception_addr);

    // We are required to report the reason for the first stop after
    // launching or being attached.
    if (NativeThreadWindows *thread = GetThreadByID(thread_id))
      SetStopReasonForThread(*thread, StopReason::eStopReasonBreakpoint);

    // Do not notify the native delegate (e.g. llgs) since at this moment
    // the program hasn't returned from Manager::Launch() and the delegate
    // might not have an valid native process to operate on.
    SetState(eStateStopped, false);

    // Hit the initial stop. Continue the application.
    return ExceptionResult::BreakInDebugger;
  }

  // Any remaining STATUS_BREAKPOINT is a breakpoint instruction in the
  // program's own code (e.g. `__debugbreak()` or `__builtin_debugtrap()`).
  // Stop the debugger and let the user decide what to do.
  if (m_pending_halt) {
    LLDB_LOG(log,
             "DebugBreakProcess injection treated as Halt SIGSTOP for tid "
             "{0:x}",
             thread_id);
    m_pending_halt = false;
    ThreadStopInfo signal_info;
    signal_info.reason = StopReason::eStopReasonSignal;
    signal_info.signo = 19; // SIGSTOP on POSIX

    // Halt all threads at the kernel level.
    for (uint32_t i = 0; i < m_threads.size(); ++i) {
      auto t = static_cast<NativeThreadWindows *>(m_threads[i].get());
      if (Status err = t->DoStop(); err.Fail()) {
        LLDB_LOG(log, "Failed to stop thread {1:x}: {0}", t->GetID(),
                 err.GetError());
        exit(1);
      }
    }
    if (!m_threads.empty()) {
      auto first = static_cast<NativeThreadWindows *>(m_threads[0].get());
      first->SetStopReason(signal_info, "interrupt");
    }
    SetCurrentThreadID(thread_id);
    if (NativeThreadWindows *injected = GetThreadByID(thread_id))
      injected->SetStopReason(signal_info, "interrupt");
    SetState(eStateStopped, true);
    return ExceptionResult::BreakInDebugger;
  }

  std::string desc = formatv("Exception {0:x8} encountered at address {1:x8}",
                             record.GetExceptionCode(), exception_addr)
                         .str();
  StopThread(thread_id, StopReason::eStopReasonException, std::move(desc));
  SetState(eStateStopped, true);
  return ExceptionResult::MaskException;
}

ExceptionResult
NativeProcessWindows::HandleGenericException(bool first_chance,
                                             const ExceptionRecord &record) {
  Log *log = GetLog(WindowsLog::Exception);
  LLDB_LOG(log,
           "Debugger thread reported exception {0:x} at address {1:x} "
           "(first_chance={2})",
           record.GetExceptionCode(), record.GetExceptionAddress(),
           first_chance);

  if (first_chance)
    return ExceptionResult::SendToApplication;

  std::string desc;
  llvm::raw_string_ostream desc_stream(desc);
  desc_stream << "Exception " << llvm::format_hex(record.GetExceptionCode(), 8)
              << " encountered at address "
              << llvm::format_hex(record.GetExceptionAddress(), 8);
  StopThread(record.GetThreadID(), StopReason::eStopReasonException,
             desc.c_str());

  SetState(eStateStopped, true);
  return ExceptionResult::BreakInDebugger;
}

ExceptionResult
NativeProcessWindows::OnDebugException(bool first_chance,
                                       const ExceptionRecord &record) {
  llvm::sys::ScopedLock lock(m_mutex);

  // Let the debugger establish the internal status.
  ProcessDebugger::OnDebugException(first_chance, record);

  switch (record.GetExceptionCode()) {
  case DWORD(STATUS_SINGLE_STEP):
  case STATUS_WX86_SINGLE_STEP:
    return HandleSingleStepException(record);
  case DWORD(STATUS_BREAKPOINT):
  case STATUS_WX86_BREAKPOINT:
    return HandleBreakpointException(record);
  default:
    return HandleGenericException(first_chance, record);
  }
}

void NativeProcessWindows::OnCreateThread(const HostThread &new_thread) {
  llvm::sys::ScopedLock lock(m_mutex);

  auto thread = std::make_unique<NativeThreadWindows>(*this, new_thread);
  thread->GetRegisterContext().ClearAllHardwareWatchpoints();
  for (const auto &pair : GetWatchpointMap()) {
    const NativeWatchpoint &wp = pair.second;
    thread->SetWatchpoint(wp.m_addr, wp.m_size, wp.m_watch_flags,
                          wp.m_hardware);
  }

  if (StateType state = GetState();
      state == eStateStopped || state == eStateCrashed) {
    if (Status error = thread->DoStop(); error.Fail()) {
      Log *log = GetLog(WindowsLog::Thread);
      LLDB_LOG(log, "failed to suspend newly-created thread {0}: {1}",
               thread->GetID(), error);
    }
    ThreadStopInfo stop_info;
    stop_info.reason = lldb::eStopReasonNone;
    thread->SetStopReason(stop_info, "");
  }

  m_threads.push_back(std::move(thread));
}

void NativeProcessWindows::OnExitThread(lldb::tid_t thread_id,
                                        uint32_t exit_code) {
  llvm::sys::ScopedLock lock(m_mutex);
  NativeThreadWindows *thread = GetThreadByID(thread_id);
  if (!thread)
    return;

  for (auto t = m_threads.begin(); t != m_threads.end();) {
    if ((*t)->GetID() == thread_id) {
      t = m_threads.erase(t);
    } else {
      ++t;
    }
  }
}

void NativeProcessWindows::OnLoadDll(const ModuleSpec &module_spec,
                                     lldb::addr_t module_addr) {
  m_loaded_modules.clear();
  m_pending_library_events = true;
}

void NativeProcessWindows::OnUnloadDll(lldb::addr_t module_addr) {
  m_loaded_modules.clear();
  m_pending_library_events = true;
}

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
NativeProcessWindows::Manager::Launch(
    ProcessLaunchInfo &launch_info,
    NativeProcessProtocol::NativeDelegate &native_delegate) {
  Error E = Error::success();
  auto process_up = std::unique_ptr<NativeProcessWindows>(
      new NativeProcessWindows(launch_info, native_delegate, E));
  if (E)
    return std::move(E);
  return std::move(process_up);
}

llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
NativeProcessWindows::Manager::Attach(
    lldb::pid_t pid, NativeProcessProtocol::NativeDelegate &native_delegate) {
  Error E = Error::success();
  // Set pty primary fd invalid since it is not available.
  auto process_up = std::unique_ptr<NativeProcessWindows>(
      new NativeProcessWindows(pid, -1, native_delegate, E));
  if (E)
    return std::move(E);
  return std::move(process_up);
}

NativeProcessWindows::~NativeProcessWindows() { StopStdioForwarding(); }

void NativeProcessWindows::StartStdioForwarding() {
  if (!m_pty || !m_pty->IsConnected())
    return;

  m_stdio_communication.SetConnection(
      std::make_unique<ConnectionConPTY>(m_pty));
  if (!m_stdio_communication.IsConnected())
    return;
  m_stdio_communication.SetReadThreadBytesReceivedCallback(
      &NativeProcessWindows::STDIOReadThreadBytesReceived, this);
  m_stdio_communication.StartReadThread();
}

void NativeProcessWindows::StopStdioForwarding() {
  if (!m_stdio_communication.HasConnection())
    return;

  if (m_pty)
    m_pty->Close();

  if (m_stdio_communication.ReadThreadIsRunning())
    m_stdio_communication.JoinReadThread();

  if (m_stdio_communication.HasConnection())
    m_stdio_communication.Disconnect();
}

void NativeProcessWindows::STDIOReadThreadBytesReceived(void *baton,
                                                        const void *src,
                                                        size_t src_len) {
  auto *self = static_cast<NativeProcessWindows *>(baton);
  if (src_len == 0)
    return;
  self->m_delegate.NewProcessOutput(
      self, llvm::StringRef(static_cast<const char *>(src), src_len));
}
} // namespace lldb_private
