//===-- ProcessWindows.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProcessWindows.h"

// Windows includes
#include "lldb/Host/windows/windows.h"
#include <psapi.h>

#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostNativeProcessBase.h"
#include "lldb/Host/HostProcess.h"
#include "lldb/Host/windows/HostThreadWindows.h"
#include "lldb/Host/windows/windows.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/State.h"

#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

#include "DebuggerThread.h"
#include "ExceptionRecord.h"
#include "ForwardDecl.h"
#include "LocalDebugDelegate.h"
#include "ProcessWindowsLog.h"
#include "TargetThreadWindows.h"

using namespace lldb;
using namespace lldb_private;

namespace {
std::string GetProcessExecutableName(HANDLE process_handle) {
  std::vector<wchar_t> file_name;
  DWORD file_name_size = MAX_PATH; // first guess, not an absolute limit
  DWORD copied = 0;
  do {
    file_name_size *= 2;
    file_name.resize(file_name_size);
    copied = ::GetModuleFileNameExW(process_handle, NULL, file_name.data(),
                                    file_name_size);
  } while (copied >= file_name_size);
  file_name.resize(copied);
  std::string result;
  llvm::convertWideToUTF8(file_name.data(), result);
  return result;
}

std::string GetProcessExecutableName(DWORD pid) {
  std::string file_name;
  HANDLE process_handle =
      ::OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, pid);
  if (process_handle != NULL) {
    file_name = GetProcessExecutableName(process_handle);
    ::CloseHandle(process_handle);
  }
  return file_name;
}

DWORD ConvertLldbToWinApiProtect(uint32_t protect) {
  // We also can process a read / write permissions here, but if the debugger
  // will make later a write into the allocated memory, it will fail. To get
  // around it is possible inside DoWriteMemory to remember memory permissions,
  // allow write, write and restore permissions, but for now we process only
  // the executable permission.
  //
  // TODO: Process permissions other than executable
  if (protect & ePermissionsExecutable)
    return PAGE_EXECUTE_READWRITE;

  return PAGE_READWRITE;
}

} // anonymous namespace

namespace lldb_private {

// We store a pointer to this class in the ProcessWindows, so that we don't
// expose Windows-specific types and implementation details from a public
// header file.
class ProcessWindowsData {
public:
  ProcessWindowsData(bool stop_at_entry) : m_stop_at_entry(stop_at_entry) {
    m_initial_stop_event = ::CreateEvent(nullptr, TRUE, FALSE, nullptr);
  }

  ~ProcessWindowsData() { ::CloseHandle(m_initial_stop_event); }

  Status m_launch_error;
  DebuggerThreadSP m_debugger;
  StopInfoSP m_pending_stop_info;
  HANDLE m_initial_stop_event = nullptr;
  bool m_initial_stop_received = false;
  bool m_stop_at_entry;
  std::map<lldb::tid_t, HostThread> m_new_threads;
  std::set<lldb::tid_t> m_exited_threads;
};

ProcessSP ProcessWindows::CreateInstance(lldb::TargetSP target_sp,
                                         lldb::ListenerSP listener_sp,
                                         const FileSpec *) {
  return ProcessSP(new ProcessWindows(target_sp, listener_sp));
}

void ProcessWindows::Initialize() {
  static std::once_flag g_once_flag;

  llvm::call_once(g_once_flag, []() {
    PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                  GetPluginDescriptionStatic(), CreateInstance);
  });
}

void ProcessWindows::Terminate() {}

lldb_private::ConstString ProcessWindows::GetPluginNameStatic() {
  static ConstString g_name("windows");
  return g_name;
}

const char *ProcessWindows::GetPluginDescriptionStatic() {
  return "Process plugin for Windows";
}

//------------------------------------------------------------------------------
// Constructors and destructors.

ProcessWindows::ProcessWindows(lldb::TargetSP target_sp,
                               lldb::ListenerSP listener_sp)
    : lldb_private::Process(target_sp, listener_sp) {}

ProcessWindows::~ProcessWindows() {}

size_t ProcessWindows::GetSTDOUT(char *buf, size_t buf_size, Status &error) {
  error.SetErrorString("GetSTDOUT unsupported on Windows");
  return 0;
}

size_t ProcessWindows::GetSTDERR(char *buf, size_t buf_size, Status &error) {
  error.SetErrorString("GetSTDERR unsupported on Windows");
  return 0;
}

size_t ProcessWindows::PutSTDIN(const char *buf, size_t buf_size,
                                Status &error) {
  error.SetErrorString("PutSTDIN unsupported on Windows");
  return 0;
}

//------------------------------------------------------------------------------
// ProcessInterface protocol.

lldb_private::ConstString ProcessWindows::GetPluginName() {
  return GetPluginNameStatic();
}

uint32_t ProcessWindows::GetPluginVersion() { return 1; }

Status ProcessWindows::EnableBreakpointSite(BreakpointSite *bp_site) {
  Log *log = ProcessWindowsLog::GetLogIfAny(WINDOWS_LOG_BREAKPOINTS);
  LLDB_LOG(log, "bp_site = {0:x}, id={1}, addr={2:x}", bp_site,
           bp_site->GetID(), bp_site->GetLoadAddress());

  Status error = EnableSoftwareBreakpoint(bp_site);
  if (!error.Success())
    LLDB_LOG(log, "error: {0}", error);
  return error;
}

Status ProcessWindows::DisableBreakpointSite(BreakpointSite *bp_site) {
  Log *log = ProcessWindowsLog::GetLogIfAny(WINDOWS_LOG_BREAKPOINTS);
  LLDB_LOG(log, "bp_site = {0:x}, id={1}, addr={2:x}", bp_site,
           bp_site->GetID(), bp_site->GetLoadAddress());

  Status error = DisableSoftwareBreakpoint(bp_site);

  if (!error.Success())
    LLDB_LOG(log, "error: {0}", error);
  return error;
}

Status ProcessWindows::DoDetach(bool keep_stopped) {
  Log *log = ProcessWindowsLog::GetLogIfAny(WINDOWS_LOG_PROCESS);
  DebuggerThreadSP debugger_thread;
  StateType private_state;
  {
    // Acquire the lock only long enough to get the DebuggerThread.
    // StopDebugging() will trigger a call back into ProcessWindows which will
    // also acquire the lock.  Thus we have to release the lock before calling
    // StopDebugging().
    llvm::sys::ScopedLock lock(m_mutex);

    private_state = GetPrivateState();

    if (!m_session_data) {
      LLDB_LOG(log, "state = {0}, but there is no active session.",
               private_state);
      return Status();
    }

    debugger_thread = m_session_data->m_debugger;
  }

  Status error;
  if (private_state != eStateExited && private_state != eStateDetached) {
    LLDB_LOG(log, "detaching from process {0} while state = {1}.",
             debugger_thread->GetProcess().GetNativeProcess().GetSystemHandle(),
             private_state);
    error = debugger_thread->StopDebugging(false);
    if (error.Success()) {
      SetPrivateState(eStateDetached);
    }

    // By the time StopDebugging returns, there is no more debugger thread, so
    // we can be assured that no other thread will race for the session data.
    m_session_data.reset();
  } else {
    LLDB_LOG(
        log,
        "error: process {0} in state = {1}, but cannot destroy in this state.",
        debugger_thread->GetProcess().GetNativeProcess().GetSystemHandle(),
        private_state);
  }

  return error;
}

Status ProcessWindows::DoLaunch(Module *exe_module,
                                ProcessLaunchInfo &launch_info) {
  // Even though m_session_data is accessed here, it is before a debugger
  // thread has been kicked off.  So there's no race conditions, and it
  // shouldn't be necessary to acquire the mutex.

  Log *log = ProcessWindowsLog::GetLogIfAny(WINDOWS_LOG_PROCESS);
  Status result;

  FileSpec working_dir = launch_info.GetWorkingDirectory();
  namespace fs = llvm::sys::fs;
  if (working_dir) {
    FileSystem::Instance().Resolve(working_dir);
    if (!FileSystem::Instance().IsDirectory(working_dir)) {
      result.SetErrorStringWithFormat("No such file or directory: %s",
                                      working_dir.GetCString());
      return result;
    }
  }

  if (!launch_info.GetFlags().Test(eLaunchFlagDebug)) {
    StreamString stream;
    stream.Printf("ProcessWindows unable to launch '%s'.  ProcessWindows can "
                  "only be used for debug launches.",
                  launch_info.GetExecutableFile().GetPath().c_str());
    std::string message = stream.GetString();
    result.SetErrorString(message.c_str());

    LLDB_LOG(log, "error: {0}", message);
    return result;
  }

  bool stop_at_entry = launch_info.GetFlags().Test(eLaunchFlagStopAtEntry);
  m_session_data.reset(new ProcessWindowsData(stop_at_entry));

  DebugDelegateSP delegate(new LocalDebugDelegate(shared_from_this()));
  m_session_data->m_debugger.reset(new DebuggerThread(delegate));
  DebuggerThreadSP debugger = m_session_data->m_debugger;

  // Kick off the DebugLaunch asynchronously and wait for it to complete.
  result = debugger->DebugLaunch(launch_info);
  if (result.Fail()) {
    LLDB_LOG(log, "failed launching '{0}'. {1}",
             launch_info.GetExecutableFile().GetPath(), result);
    return result;
  }

  HostProcess process;
  Status error = WaitForDebuggerConnection(debugger, process);
  if (error.Fail()) {
    LLDB_LOG(log, "failed launching '{0}'. {1}",
             launch_info.GetExecutableFile().GetPath(), error);
    return error;
  }

  LLDB_LOG(log, "successfully launched '{0}'",
           launch_info.GetExecutableFile().GetPath());

  // We've hit the initial stop.  If eLaunchFlagsStopAtEntry was specified, the
  // private state should already be set to eStateStopped as a result of
  // hitting the initial breakpoint.  If it was not set, the breakpoint should
  // have already been resumed from and the private state should already be
  // eStateRunning.
  launch_info.SetProcessID(process.GetProcessId());
  SetID(process.GetProcessId());

  return result;
}

Status
ProcessWindows::DoAttachToProcessWithID(lldb::pid_t pid,
                                        const ProcessAttachInfo &attach_info) {
  Log *log = ProcessWindowsLog::GetLogIfAny(WINDOWS_LOG_PROCESS);
  m_session_data.reset(
      new ProcessWindowsData(!attach_info.GetContinueOnceAttached()));

  DebugDelegateSP delegate(new LocalDebugDelegate(shared_from_this()));
  DebuggerThreadSP debugger(new DebuggerThread(delegate));

  m_session_data->m_debugger = debugger;

  DWORD process_id = static_cast<DWORD>(pid);
  Status error = debugger->DebugAttach(process_id, attach_info);
  if (error.Fail()) {
    LLDB_LOG(
        log,
        "encountered an error occurred initiating the asynchronous attach. {0}",
        error);
    return error;
  }

  HostProcess process;
  error = WaitForDebuggerConnection(debugger, process);
  if (error.Fail()) {
    LLDB_LOG(log,
             "encountered an error waiting for the debugger to connect. {0}",
             error);
    return error;
  }

  LLDB_LOG(log, "successfully attached to process with pid={0}", process_id);

  // We've hit the initial stop.  If eLaunchFlagsStopAtEntry was specified, the
  // private state should already be set to eStateStopped as a result of
  // hitting the initial breakpoint.  If it was not set, the breakpoint should
  // have already been resumed from and the private state should already be
  // eStateRunning.
  SetID(process.GetProcessId());
  return error;
}

Status ProcessWindows::DoResume() {
  Log *log = ProcessWindowsLog::GetLogIfAny(WINDOWS_LOG_PROCESS);
  llvm::sys::ScopedLock lock(m_mutex);
  Status error;

  StateType private_state = GetPrivateState();
  if (private_state == eStateStopped || private_state == eStateCrashed) {
    LLDB_LOG(log, "process {0} is in state {1}.  Resuming...",
             m_session_data->m_debugger->GetProcess().GetProcessId(),
             GetPrivateState());

    ExceptionRecordSP active_exception =
        m_session_data->m_debugger->GetActiveException().lock();
    if (active_exception) {
      // Resume the process and continue processing debug events.  Mask the
      // exception so that from the process's view, there is no indication that
      // anything happened.
      m_session_data->m_debugger->ContinueAsyncException(
          ExceptionResult::MaskException);
    }

    LLDB_LOG(log, "resuming {0} threads.", m_thread_list.GetSize());

    bool failed = false;
    for (uint32_t i = 0; i < m_thread_list.GetSize(); ++i) {
      auto thread = std::static_pointer_cast<TargetThreadWindows>(
          m_thread_list.GetThreadAtIndex(i));
      Status result = thread->DoResume();
      if (result.Fail()) {
        failed = true;
        LLDB_LOG(
            log,
            "Trying to resume thread at index {0}, but failed with error {1}.",
            i, result);
      }
    }

    if (failed) {
      error.SetErrorString("ProcessWindows::DoResume failed");
      return error;
    } else {
      SetPrivateState(eStateRunning);
    }
  } else {
    LLDB_LOG(log, "error: process {0} is in state {1}.  Returning...",
             m_session_data->m_debugger->GetProcess().GetProcessId(),
             GetPrivateState());
  }
  return error;
}

Status ProcessWindows::DoDestroy() {
  Log *log = ProcessWindowsLog::GetLogIfAny(WINDOWS_LOG_PROCESS);
  DebuggerThreadSP debugger_thread;
  StateType private_state;
  {
    // Acquire this lock inside an inner scope, only long enough to get the
    // DebuggerThread. StopDebugging() will trigger a call back into
    // ProcessWindows which will acquire the lock again, so we need to not
    // deadlock.
    llvm::sys::ScopedLock lock(m_mutex);

    private_state = GetPrivateState();

    if (!m_session_data) {
      LLDB_LOG(log, "warning: state = {0}, but there is no active session.",
               private_state);
      return Status();
    }

    debugger_thread = m_session_data->m_debugger;
  }

  Status error;
  if (private_state != eStateExited && private_state != eStateDetached) {
    LLDB_LOG(log, "Shutting down process {0} while state = {1}.",
             debugger_thread->GetProcess().GetNativeProcess().GetSystemHandle(),
             private_state);
    error = debugger_thread->StopDebugging(true);

    // By the time StopDebugging returns, there is no more debugger thread, so
    // we can be assured that no other thread will race for the session data.
    m_session_data.reset();
  } else {
    LLDB_LOG(log, "cannot destroy process {0} while state = {1}",
             debugger_thread->GetProcess().GetNativeProcess().GetSystemHandle(),
             private_state);
  }

  return error;
}

Status ProcessWindows::DoHalt(bool &caused_stop) {
  Log *log = ProcessWindowsLog::GetLogIfAny(WINDOWS_LOG_PROCESS);
  Status error;
  StateType state = GetPrivateState();
  if (state == eStateStopped)
    caused_stop = false;
  else {
    llvm::sys::ScopedLock lock(m_mutex);
    caused_stop = ::DebugBreakProcess(m_session_data->m_debugger->GetProcess()
                                          .GetNativeProcess()
                                          .GetSystemHandle());
    if (!caused_stop) {
      error.SetError(::GetLastError(), eErrorTypeWin32);
      LLDB_LOG(log, "DebugBreakProcess failed with error {0}", error);
    }
  }
  return error;
}

void ProcessWindows::DidLaunch() {
  ArchSpec arch_spec;
  DidAttach(arch_spec);
}

void ProcessWindows::DidAttach(ArchSpec &arch_spec) {
  llvm::sys::ScopedLock lock(m_mutex);

  // The initial stop won't broadcast the state change event, so account for
  // that here.
  if (m_session_data && GetPrivateState() == eStateStopped &&
      m_session_data->m_stop_at_entry)
    RefreshStateAfterStop();
}

void ProcessWindows::RefreshStateAfterStop() {
  Log *log = ProcessWindowsLog::GetLogIfAny(WINDOWS_LOG_EXCEPTION);
  llvm::sys::ScopedLock lock(m_mutex);

  if (!m_session_data) {
    LLDB_LOG(log, "no active session.  Returning...");
    return;
  }

  m_thread_list.RefreshStateAfterStop();

  std::weak_ptr<ExceptionRecord> exception_record =
      m_session_data->m_debugger->GetActiveException();
  ExceptionRecordSP active_exception = exception_record.lock();
  if (!active_exception) {
    LLDB_LOG(log,
             "there is no active exception in process {0}.  Why is the "
             "process stopped?",
             m_session_data->m_debugger->GetProcess().GetProcessId());
    return;
  }

  StopInfoSP stop_info;
  m_thread_list.SetSelectedThreadByID(active_exception->GetThreadID());
  ThreadSP stop_thread = m_thread_list.GetSelectedThread();
  if (!stop_thread)
    return;

  switch (active_exception->GetExceptionCode()) {
  case EXCEPTION_SINGLE_STEP: {
    RegisterContextSP register_context = stop_thread->GetRegisterContext();
    const uint64_t pc = register_context->GetPC();
    BreakpointSiteSP site(GetBreakpointSiteList().FindByAddress(pc));
    if (site && site->ValidForThisThread(stop_thread.get())) {
      LLDB_LOG(log,
               "Single-stepped onto a breakpoint in process {0} at "
               "address {1:x} with breakpoint site {2}",
               m_session_data->m_debugger->GetProcess().GetProcessId(), pc,
               site->GetID());
      stop_info = StopInfo::CreateStopReasonWithBreakpointSiteID(*stop_thread,
                                                                 site->GetID());
      stop_thread->SetStopInfo(stop_info);
    } else {
      LLDB_LOG(log, "single stepping thread {0}", stop_thread->GetID());
      stop_info = StopInfo::CreateStopReasonToTrace(*stop_thread);
      stop_thread->SetStopInfo(stop_info);
    }
    return;
  }

  case EXCEPTION_BREAKPOINT: {
    RegisterContextSP register_context = stop_thread->GetRegisterContext();

    // The current EIP is AFTER the BP opcode, which is one byte.
    uint64_t pc = register_context->GetPC() - 1;

    BreakpointSiteSP site(GetBreakpointSiteList().FindByAddress(pc));
    if (site) {
      LLDB_LOG(log,
               "detected breakpoint in process {0} at address {1:x} with "
               "breakpoint site {2}",
               m_session_data->m_debugger->GetProcess().GetProcessId(), pc,
               site->GetID());

      if (site->ValidForThisThread(stop_thread.get())) {
        LLDB_LOG(log,
                 "Breakpoint site {0} is valid for this thread ({1:x}), "
                 "creating stop info.",
                 site->GetID(), stop_thread->GetID());

        stop_info = StopInfo::CreateStopReasonWithBreakpointSiteID(
            *stop_thread, site->GetID());
        register_context->SetPC(pc);
      } else {
        LLDB_LOG(log,
                 "Breakpoint site {0} is not valid for this thread, "
                 "creating empty stop info.",
                 site->GetID());
      }
      stop_thread->SetStopInfo(stop_info);
      return;
    } else {
      // The thread hit a hard-coded breakpoint like an `int 3` or
      // `__debugbreak()`.
      LLDB_LOG(log,
               "No breakpoint site matches for this thread. __debugbreak()?  "
               "Creating stop info with the exception.");
      // FALLTHROUGH:  We'll treat this as a generic exception record in the
      // default case.
    }
  }

  default: {
    std::string desc;
    llvm::raw_string_ostream desc_stream(desc);
    desc_stream << "Exception "
                << llvm::format_hex(active_exception->GetExceptionCode(), 8)
                << " encountered at address "
                << llvm::format_hex(active_exception->GetExceptionAddress(), 8);
    stop_info = StopInfo::CreateStopReasonWithException(
        *stop_thread, desc_stream.str().c_str());
    stop_thread->SetStopInfo(stop_info);
    LLDB_LOG(log, "{0}", desc_stream.str());
    return;
  }
  }
}

bool ProcessWindows::CanDebug(lldb::TargetSP target_sp,
                              bool plugin_specified_by_name) {
  if (plugin_specified_by_name)
    return true;

  // For now we are just making sure the file exists for a given module
  ModuleSP exe_module_sp(target_sp->GetExecutableModule());
  if (exe_module_sp.get())
    return FileSystem::Instance().Exists(exe_module_sp->GetFileSpec());
  // However, if there is no executable module, we return true since we might
  // be preparing to attach.
  return true;
}

bool ProcessWindows::UpdateThreadList(ThreadList &old_thread_list,
                                      ThreadList &new_thread_list) {
  Log *log = ProcessWindowsLog::GetLogIfAny(WINDOWS_LOG_THREAD);
  // Add all the threads that were previously running and for which we did not
  // detect a thread exited event.
  int new_size = 0;
  int continued_threads = 0;
  int exited_threads = 0;
  int new_threads = 0;

  for (ThreadSP old_thread : old_thread_list.Threads()) {
    lldb::tid_t old_thread_id = old_thread->GetID();
    auto exited_thread_iter =
        m_session_data->m_exited_threads.find(old_thread_id);
    if (exited_thread_iter == m_session_data->m_exited_threads.end()) {
      new_thread_list.AddThread(old_thread);
      ++new_size;
      ++continued_threads;
      LLDB_LOGV(log, "Thread {0} was running and is still running.",
                old_thread_id);
    } else {
      LLDB_LOGV(log, "Thread {0} was running and has exited.", old_thread_id);
      ++exited_threads;
    }
  }

  // Also add all the threads that are new since the last time we broke into
  // the debugger.
  for (const auto &thread_info : m_session_data->m_new_threads) {
    ThreadSP thread(new TargetThreadWindows(*this, thread_info.second));
    thread->SetID(thread_info.first);
    new_thread_list.AddThread(thread);
    ++new_size;
    ++new_threads;
    LLDB_LOGV(log, "Thread {0} is new since last update.", thread_info.first);
  }

  LLDB_LOG(log, "{0} new threads, {1} old threads, {2} exited threads.",
           new_threads, continued_threads, exited_threads);

  m_session_data->m_new_threads.clear();
  m_session_data->m_exited_threads.clear();

  return new_size > 0;
}

bool ProcessWindows::IsAlive() {
  StateType state = GetPrivateState();
  switch (state) {
  case eStateCrashed:
  case eStateDetached:
  case eStateUnloaded:
  case eStateExited:
  case eStateInvalid:
    return false;
  default:
    return true;
  }
}

size_t ProcessWindows::DoReadMemory(lldb::addr_t vm_addr, void *buf,
                                    size_t size, Status &error) {
  Log *log = ProcessWindowsLog::GetLogIfAny(WINDOWS_LOG_MEMORY);
  llvm::sys::ScopedLock lock(m_mutex);

  if (!m_session_data)
    return 0;

  LLDB_LOG(log, "attempting to read {0} bytes from address {1:x}", size,
           vm_addr);

  HostProcess process = m_session_data->m_debugger->GetProcess();
  void *addr = reinterpret_cast<void *>(vm_addr);
  SIZE_T bytes_read = 0;
  if (!ReadProcessMemory(process.GetNativeProcess().GetSystemHandle(), addr,
                         buf, size, &bytes_read)) {
    // Reading from the process can fail for a number of reasons - set the
    // error code and make sure that the number of bytes read is set back to 0
    // because in some scenarios the value of bytes_read returned from the API
    // is garbage.
    error.SetError(GetLastError(), eErrorTypeWin32);
    LLDB_LOG(log, "reading failed with error: {0}", error);
    bytes_read = 0;
  }
  return bytes_read;
}

size_t ProcessWindows::DoWriteMemory(lldb::addr_t vm_addr, const void *buf,
                                     size_t size, Status &error) {
  Log *log = ProcessWindowsLog::GetLogIfAny(WINDOWS_LOG_MEMORY);
  llvm::sys::ScopedLock lock(m_mutex);
  LLDB_LOG(log, "attempting to write {0} bytes into address {1:x}", size,
           vm_addr);

  if (!m_session_data) {
    LLDB_LOG(log, "cannot write, there is no active debugger connection.");
    return 0;
  }

  HostProcess process = m_session_data->m_debugger->GetProcess();
  void *addr = reinterpret_cast<void *>(vm_addr);
  SIZE_T bytes_written = 0;
  lldb::process_t handle = process.GetNativeProcess().GetSystemHandle();
  if (WriteProcessMemory(handle, addr, buf, size, &bytes_written))
    FlushInstructionCache(handle, addr, bytes_written);
  else {
    error.SetError(GetLastError(), eErrorTypeWin32);
    LLDB_LOG(log, "writing failed with error: {0}", error);
  }
  return bytes_written;
}

lldb::addr_t ProcessWindows::DoAllocateMemory(size_t size, uint32_t permissions,
                                              Status &error) {
  Log *log = ProcessWindowsLog::GetLogIfAny(WINDOWS_LOG_MEMORY);
  llvm::sys::ScopedLock lock(m_mutex);
  LLDB_LOG(log, "attempting to allocate {0} bytes with permissions {1}", size,
           permissions);

  if (!m_session_data) {
    LLDB_LOG(log, "cannot allocate, there is no active debugger connection.");
    error.SetErrorString(
        "cannot allocate, there is no active debugger connection");
    return LLDB_INVALID_ADDRESS;
  }

  HostProcess process = m_session_data->m_debugger->GetProcess();
  lldb::process_t handle = process.GetNativeProcess().GetSystemHandle();
  auto protect = ConvertLldbToWinApiProtect(permissions);
  auto result = VirtualAllocEx(handle, nullptr, size, MEM_COMMIT, protect);
  if (!result) {
    error.SetError(GetLastError(), eErrorTypeWin32);
    LLDB_LOG(log, "allocating failed with error: {0}", error);
    return LLDB_INVALID_ADDRESS;
  }

  return reinterpret_cast<addr_t>(result);
}

Status ProcessWindows::DoDeallocateMemory(lldb::addr_t ptr) {
  Status result;

  Log *log = ProcessWindowsLog::GetLogIfAny(WINDOWS_LOG_MEMORY);
  llvm::sys::ScopedLock lock(m_mutex);
  LLDB_LOG(log, "attempting to deallocate bytes at address {0}", ptr);

  if (!m_session_data) {
    LLDB_LOG(log, "cannot deallocate, there is no active debugger connection.");
    result.SetErrorString(
        "cannot deallocate, there is no active debugger connection");
    return result;
  }

  HostProcess process = m_session_data->m_debugger->GetProcess();
  lldb::process_t handle = process.GetNativeProcess().GetSystemHandle();
  if (!VirtualFreeEx(handle, reinterpret_cast<LPVOID>(ptr), 0, MEM_RELEASE)) {
    result.SetError(GetLastError(), eErrorTypeWin32);
    LLDB_LOG(log, "deallocating failed with error: {0}", result);
    return result;
  }

  return result;
}

Status ProcessWindows::GetMemoryRegionInfo(lldb::addr_t vm_addr,
                                           MemoryRegionInfo &info) {
  Log *log = ProcessWindowsLog::GetLogIfAny(WINDOWS_LOG_MEMORY);
  Status error;
  llvm::sys::ScopedLock lock(m_mutex);
  info.Clear();

  if (!m_session_data) {
    error.SetErrorString(
        "GetMemoryRegionInfo called with no debugging session.");
    LLDB_LOG(log, "error: {0}", error);
    return error;
  }
  HostProcess process = m_session_data->m_debugger->GetProcess();
  lldb::process_t handle = process.GetNativeProcess().GetSystemHandle();
  if (handle == nullptr || handle == LLDB_INVALID_PROCESS) {
    error.SetErrorString(
        "GetMemoryRegionInfo called with an invalid target process.");
    LLDB_LOG(log, "error: {0}", error);
    return error;
  }

  LLDB_LOG(log, "getting info for address {0:x}", vm_addr);

  void *addr = reinterpret_cast<void *>(vm_addr);
  MEMORY_BASIC_INFORMATION mem_info = {};
  SIZE_T result = ::VirtualQueryEx(handle, addr, &mem_info, sizeof(mem_info));
  if (result == 0) {
    if (::GetLastError() == ERROR_INVALID_PARAMETER) {
      // ERROR_INVALID_PARAMETER is returned if VirtualQueryEx is called with
      // an address past the highest accessible address. We should return a
      // range from the vm_addr to LLDB_INVALID_ADDRESS
      info.GetRange().SetRangeBase(vm_addr);
      info.GetRange().SetRangeEnd(LLDB_INVALID_ADDRESS);
      info.SetReadable(MemoryRegionInfo::eNo);
      info.SetExecutable(MemoryRegionInfo::eNo);
      info.SetWritable(MemoryRegionInfo::eNo);
      info.SetMapped(MemoryRegionInfo::eNo);
      return error;
    } else {
      error.SetError(::GetLastError(), eErrorTypeWin32);
      LLDB_LOG(log,
               "VirtualQueryEx returned error {0} while getting memory "
               "region info for address {1:x}",
               error, vm_addr);
      return error;
    }
  }

  // Protect bits are only valid for MEM_COMMIT regions.
  if (mem_info.State == MEM_COMMIT) {
    const bool readable = IsPageReadable(mem_info.Protect);
    const bool executable = IsPageExecutable(mem_info.Protect);
    const bool writable = IsPageWritable(mem_info.Protect);
    info.SetReadable(readable ? MemoryRegionInfo::eYes : MemoryRegionInfo::eNo);
    info.SetExecutable(executable ? MemoryRegionInfo::eYes
                                  : MemoryRegionInfo::eNo);
    info.SetWritable(writable ? MemoryRegionInfo::eYes : MemoryRegionInfo::eNo);
  } else {
    info.SetReadable(MemoryRegionInfo::eNo);
    info.SetExecutable(MemoryRegionInfo::eNo);
    info.SetWritable(MemoryRegionInfo::eNo);
  }

  // AllocationBase is defined for MEM_COMMIT and MEM_RESERVE but not MEM_FREE.
  if (mem_info.State != MEM_FREE) {
    info.GetRange().SetRangeBase(
        reinterpret_cast<addr_t>(mem_info.AllocationBase));
    info.GetRange().SetRangeEnd(reinterpret_cast<addr_t>(mem_info.BaseAddress) +
                                mem_info.RegionSize);
    info.SetMapped(MemoryRegionInfo::eYes);
  } else {
    // In the unmapped case we need to return the distance to the next block of
    // memory. VirtualQueryEx nearly does that except that it gives the
    // distance from the start of the page containing vm_addr.
    SYSTEM_INFO data;
    GetSystemInfo(&data);
    DWORD page_offset = vm_addr % data.dwPageSize;
    info.GetRange().SetRangeBase(vm_addr);
    info.GetRange().SetByteSize(mem_info.RegionSize - page_offset);
    info.SetMapped(MemoryRegionInfo::eNo);
  }

  error.SetError(::GetLastError(), eErrorTypeWin32);
  LLDB_LOGV(log,
            "Memory region info for address {0}: readable={1}, "
            "executable={2}, writable={3}",
            vm_addr, info.GetReadable(), info.GetExecutable(),
            info.GetWritable());
  return error;
}

lldb::addr_t ProcessWindows::GetImageInfoAddress() {
  Target &target = GetTarget();
  ObjectFile *obj_file = target.GetExecutableModule()->GetObjectFile();
  Address addr = obj_file->GetImageInfoAddress(&target);
  if (addr.IsValid())
    return addr.GetLoadAddress(&target);
  else
    return LLDB_INVALID_ADDRESS;
}

DynamicLoaderWindowsDYLD *ProcessWindows::GetDynamicLoader() {
  if (m_dyld_up.get() == NULL)
    m_dyld_up.reset(DynamicLoader::FindPlugin(
        this, DynamicLoaderWindowsDYLD::GetPluginNameStatic().GetCString()));
  return static_cast<DynamicLoaderWindowsDYLD *>(m_dyld_up.get());
}

void ProcessWindows::OnExitProcess(uint32_t exit_code) {
  // No need to acquire the lock since m_session_data isn't accessed.
  Log *log = ProcessWindowsLog::GetLogIfAny(WINDOWS_LOG_PROCESS);
  LLDB_LOG(log, "Process {0} exited with code {1}", GetID(), exit_code);

  TargetSP target = CalculateTarget();
  if (target) {
    ModuleSP executable_module = target->GetExecutableModule();
    ModuleList unloaded_modules;
    unloaded_modules.Append(executable_module);
    target->ModulesDidUnload(unloaded_modules, true);
  }

  SetProcessExitStatus(GetID(), true, 0, exit_code);
  SetPrivateState(eStateExited);

  // If the process exits before any initial stop then notify the debugger 
  // of the error otherwise WaitForDebuggerConnection() will be blocked.
  // An example of this issue is when a process fails to load a dependent DLL. 
  if (m_session_data && !m_session_data->m_initial_stop_received) {
    Status error(exit_code, eErrorTypeWin32);
    OnDebuggerError(error, 0);
  }
}

void ProcessWindows::OnDebuggerConnected(lldb::addr_t image_base) {
  DebuggerThreadSP debugger = m_session_data->m_debugger;
  Log *log = ProcessWindowsLog::GetLogIfAny(WINDOWS_LOG_PROCESS);
  LLDB_LOG(log, "Debugger connected to process {0}.  Image base = {1:x}",
           debugger->GetProcess().GetProcessId(), image_base);

  ModuleSP module = GetTarget().GetExecutableModule();
  if (!module) {
    // During attach, we won't have the executable module, so find it now.
    const DWORD pid = debugger->GetProcess().GetProcessId();
    const std::string file_name = GetProcessExecutableName(pid);
    if (file_name.empty()) {
      return;
    }

    FileSpec executable_file(file_name);
    FileSystem::Instance().Resolve(executable_file);
    ModuleSpec module_spec(executable_file);
    Status error;
    module = GetTarget().GetSharedModule(module_spec, &error);
    if (!module) {
      return;
    }

    GetTarget().SetExecutableModule(module, eLoadDependentsNo);
  }

  if (auto dyld = GetDynamicLoader())
    dyld->OnLoadModule(module, ModuleSpec(), image_base);

  // Add the main executable module to the list of pending module loads.  We
  // can't call GetTarget().ModulesDidLoad() here because we still haven't
  // returned from DoLaunch() / DoAttach() yet so the target may not have set
  // the process instance to `this` yet.
  llvm::sys::ScopedLock lock(m_mutex);
  const HostThreadWindows &wmain_thread =
      debugger->GetMainThread().GetNativeThread();
  m_session_data->m_new_threads[wmain_thread.GetThreadId()] =
      debugger->GetMainThread();
}

ExceptionResult
ProcessWindows::OnDebugException(bool first_chance,
                                 const ExceptionRecord &record) {
  Log *log = ProcessWindowsLog::GetLogIfAny(WINDOWS_LOG_EXCEPTION);
  llvm::sys::ScopedLock lock(m_mutex);

  // FIXME: Without this check, occasionally when running the test suite there
  // is
  // an issue where m_session_data can be null.  It's not clear how this could
  // happen but it only surfaces while running the test suite.  In order to
  // properly diagnose this, we probably need to first figure allow the test
  // suite to print out full lldb logs, and then add logging to the process
  // plugin.
  if (!m_session_data) {
    LLDB_LOG(log,
             "Debugger thread reported exception {0:x} at address {1:x}, "
             "but there is no session.",
             record.GetExceptionCode(), record.GetExceptionAddress());
    return ExceptionResult::SendToApplication;
  }

  if (!first_chance) {
    // Not any second chance exception is an application crash by definition.
    // It may be an expression evaluation crash.
    SetPrivateState(eStateStopped);
  }

  ExceptionResult result = ExceptionResult::SendToApplication;
  switch (record.GetExceptionCode()) {
  case EXCEPTION_BREAKPOINT:
    // Handle breakpoints at the first chance.
    result = ExceptionResult::BreakInDebugger;

    if (!m_session_data->m_initial_stop_received) {
      LLDB_LOG(
          log,
          "Hit loader breakpoint at address {0:x}, setting initial stop event.",
          record.GetExceptionAddress());
      m_session_data->m_initial_stop_received = true;
      ::SetEvent(m_session_data->m_initial_stop_event);
    } else {
      LLDB_LOG(log, "Hit non-loader breakpoint at address {0:x}.",
               record.GetExceptionAddress());
    }
    SetPrivateState(eStateStopped);
    break;
  case EXCEPTION_SINGLE_STEP:
    result = ExceptionResult::BreakInDebugger;
    SetPrivateState(eStateStopped);
    break;
  default:
    LLDB_LOG(log,
             "Debugger thread reported exception {0:x} at address {1:x} "
             "(first_chance={2})",
             record.GetExceptionCode(), record.GetExceptionAddress(),
             first_chance);
    // For non-breakpoints, give the application a chance to handle the
    // exception first.
    if (first_chance)
      result = ExceptionResult::SendToApplication;
    else
      result = ExceptionResult::BreakInDebugger;
  }

  return result;
}

void ProcessWindows::OnCreateThread(const HostThread &new_thread) {
  llvm::sys::ScopedLock lock(m_mutex);
  const HostThreadWindows &wnew_thread = new_thread.GetNativeThread();
  m_session_data->m_new_threads[wnew_thread.GetThreadId()] = new_thread;
}

void ProcessWindows::OnExitThread(lldb::tid_t thread_id, uint32_t exit_code) {
  llvm::sys::ScopedLock lock(m_mutex);

  // On a forced termination, we may get exit thread events after the session
  // data has been cleaned up.
  if (!m_session_data)
    return;

  // A thread may have started and exited before the debugger stopped allowing a
  // refresh.
  // Just remove it from the new threads list in that case.
  auto iter = m_session_data->m_new_threads.find(thread_id);
  if (iter != m_session_data->m_new_threads.end())
    m_session_data->m_new_threads.erase(iter);
  else
    m_session_data->m_exited_threads.insert(thread_id);
}

void ProcessWindows::OnLoadDll(const ModuleSpec &module_spec,
                               lldb::addr_t module_addr) {
  if (auto dyld = GetDynamicLoader())
    dyld->OnLoadModule(nullptr, module_spec, module_addr);
}

void ProcessWindows::OnUnloadDll(lldb::addr_t module_addr) {
  if (auto dyld = GetDynamicLoader())
    dyld->OnUnloadModule(module_addr);
}

void ProcessWindows::OnDebugString(const std::string &string) {}

void ProcessWindows::OnDebuggerError(const Status &error, uint32_t type) {
  llvm::sys::ScopedLock lock(m_mutex);
  Log *log = ProcessWindowsLog::GetLogIfAny(WINDOWS_LOG_PROCESS);

  if (m_session_data->m_initial_stop_received) {
    // This happened while debugging.  Do we shutdown the debugging session,
    // try to continue, or do something else?
    LLDB_LOG(log,
             "Error {0} occurred during debugging.  Unexpected behavior "
             "may result.  {1}",
             error.GetError(), error);
  } else {
    // If we haven't actually launched the process yet, this was an error
    // launching the process.  Set the internal error and signal the initial
    // stop event so that the DoLaunch method wakes up and returns a failure.
    m_session_data->m_launch_error = error;
    ::SetEvent(m_session_data->m_initial_stop_event);
    LLDB_LOG(
        log,
        "Error {0} occurred launching the process before the initial stop. {1}",
        error.GetError(), error);
    return;
  }
}

Status ProcessWindows::WaitForDebuggerConnection(DebuggerThreadSP debugger,
                                                 HostProcess &process) {
  Status result;
  Log *log = ProcessWindowsLog::GetLogIfAny(WINDOWS_LOG_PROCESS |
                                            WINDOWS_LOG_BREAKPOINTS);
  LLDB_LOG(log, "Waiting for loader breakpoint.");

  // Block this function until we receive the initial stop from the process.
  if (::WaitForSingleObject(m_session_data->m_initial_stop_event, INFINITE) ==
      WAIT_OBJECT_0) {
    LLDB_LOG(log, "hit loader breakpoint, returning.");

    process = debugger->GetProcess();
    return m_session_data->m_launch_error;
  } else
    return Status(::GetLastError(), eErrorTypeWin32);
}

// The Windows page protection bits are NOT independent masks that can be
// bitwise-ORed together.  For example, PAGE_EXECUTE_READ is not (PAGE_EXECUTE
// | PAGE_READ).  To test for an access type, it's necessary to test for any of
// the bits that provide that access type.
bool ProcessWindows::IsPageReadable(uint32_t protect) {
  return (protect & PAGE_NOACCESS) == 0;
}

bool ProcessWindows::IsPageWritable(uint32_t protect) {
  return (protect & (PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY |
                     PAGE_READWRITE | PAGE_WRITECOPY)) != 0;
}

bool ProcessWindows::IsPageExecutable(uint32_t protect) {
  return (protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
                     PAGE_EXECUTE_WRITECOPY)) != 0;
}

} // namespace lldb_private
