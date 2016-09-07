//===-- ProcessWindowsLive.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Windows includes
#include "lldb/Host/windows/windows.h"
#include <psapi.h>

// C++ Includes
#include <list>
#include <mutex>
#include <set>
#include <vector>

// Other libraries and framework includes
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/State.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostNativeProcessBase.h"
#include "lldb/Host/HostNativeThreadBase.h"
#include "lldb/Host/HostProcess.h"
#include "lldb/Host/MonitoringProcessLauncher.h"
#include "lldb/Host/ThreadLauncher.h"
#include "lldb/Host/windows/HostThreadWindows.h"
#include "lldb/Host/windows/ProcessLauncherWindows.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/FileAction.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"

#include "Plugins/Process/Windows/Common/ProcessWindowsLog.h"

#include "DebuggerThread.h"
#include "ExceptionRecord.h"
#include "LocalDebugDelegate.h"
#include "ProcessWindowsLive.h"
#include "TargetThreadWindowsLive.h"

#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace lldb;
using namespace lldb_private;

#define BOOL_STR(b) ((b) ? "true" : "false")

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

} // anonymous namespace

namespace lldb_private {

// We store a pointer to this class in the ProcessWindows, so that we don't
// expose Windows
// OS specific types and implementation details from a public header file.
class ProcessWindowsData {
public:
  ProcessWindowsData(bool stop_at_entry)
      : m_stop_at_entry(stop_at_entry), m_initial_stop_event(nullptr),
        m_initial_stop_received(false) {
    m_initial_stop_event = ::CreateEvent(nullptr, TRUE, FALSE, nullptr);
  }

  ~ProcessWindowsData() { ::CloseHandle(m_initial_stop_event); }

  lldb_private::Error m_launch_error;
  lldb_private::DebuggerThreadSP m_debugger;
  StopInfoSP m_pending_stop_info;
  HANDLE m_initial_stop_event;
  bool m_stop_at_entry;
  bool m_initial_stop_received;
  std::map<lldb::tid_t, HostThread> m_new_threads;
  std::set<lldb::tid_t> m_exited_threads;
};
}
//------------------------------------------------------------------------------
// Static functions.

ProcessSP ProcessWindowsLive::CreateInstance(lldb::TargetSP target_sp,
                                             lldb::ListenerSP listener_sp,
                                             const FileSpec *) {
  return ProcessSP(new ProcessWindowsLive(target_sp, listener_sp));
}

void ProcessWindowsLive::Initialize() {
  static std::once_flag g_once_flag;

  std::call_once(g_once_flag, []() {
    PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                  GetPluginDescriptionStatic(), CreateInstance);
  });
}

//------------------------------------------------------------------------------
// Constructors and destructors.

ProcessWindowsLive::ProcessWindowsLive(lldb::TargetSP target_sp,
                                       lldb::ListenerSP listener_sp)
    : lldb_private::ProcessWindows(target_sp, listener_sp) {}

ProcessWindowsLive::~ProcessWindowsLive() {}

void ProcessWindowsLive::Terminate() {}

lldb_private::ConstString ProcessWindowsLive::GetPluginNameStatic() {
  static ConstString g_name("windows");
  return g_name;
}

const char *ProcessWindowsLive::GetPluginDescriptionStatic() {
  return "Process plugin for Windows";
}

Error ProcessWindowsLive::EnableBreakpointSite(BreakpointSite *bp_site) {
  WINLOG_IFALL(WINDOWS_LOG_BREAKPOINTS,
               "EnableBreakpointSite called with bp_site 0x%p "
               "(id=%d, addr=0x%x)",
               bp_site->GetID(), bp_site->GetLoadAddress());

  Error error = EnableSoftwareBreakpoint(bp_site);
  if (!error.Success()) {
    WINERR_IFALL(WINDOWS_LOG_BREAKPOINTS, "EnableBreakpointSite failed.  %s",
                 error.AsCString());
  }
  return error;
}

Error ProcessWindowsLive::DisableBreakpointSite(BreakpointSite *bp_site) {
  WINLOG_IFALL(WINDOWS_LOG_BREAKPOINTS,
               "DisableBreakpointSite called with bp_site 0x%p "
               "(id=%d, addr=0x%x)",
               bp_site, bp_site->GetID(), bp_site->GetLoadAddress());

  Error error = DisableSoftwareBreakpoint(bp_site);

  if (!error.Success()) {
    WINERR_IFALL(WINDOWS_LOG_BREAKPOINTS, "DisableBreakpointSite failed.  %s",
                 error.AsCString());
  }
  return error;
}

bool ProcessWindowsLive::UpdateThreadList(ThreadList &old_thread_list,
                                          ThreadList &new_thread_list) {
  // Add all the threads that were previously running and for which we did not
  // detect a thread
  // exited event.
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
      WINLOGV_IFALL(
          WINDOWS_LOG_THREAD,
          "UpdateThreadList - Thread %u was running and is still running.",
          old_thread_id);
    } else {
      WINLOGV_IFALL(WINDOWS_LOG_THREAD,
                    "UpdateThreadList - Thread %u was running and has exited.",
                    old_thread_id);
      ++exited_threads;
    }
  }

  // Also add all the threads that are new since the last time we broke into the
  // debugger.
  for (const auto &thread_info : m_session_data->m_new_threads) {
    ThreadSP thread(new TargetThreadWindowsLive(*this, thread_info.second));
    thread->SetID(thread_info.first);
    new_thread_list.AddThread(thread);
    ++new_size;
    ++new_threads;
    WINLOGV_IFALL(WINDOWS_LOG_THREAD,
                  "UpdateThreadList - Thread %u is new since last update.",
                  thread_info.first);
  }

  WINLOG_IFALL(
      WINDOWS_LOG_THREAD,
      "UpdateThreadList - %d new threads, %d old threads, %d exited threads.",
      new_threads, continued_threads, exited_threads);

  m_session_data->m_new_threads.clear();
  m_session_data->m_exited_threads.clear();

  return new_size > 0;
}

Error ProcessWindowsLive::DoLaunch(Module *exe_module,
                                   ProcessLaunchInfo &launch_info) {
  // Even though m_session_data is accessed here, it is before a debugger thread
  // has been
  // kicked off.  So there's no race conditions, and it shouldn't be necessary
  // to acquire
  // the mutex.

  Error result;
  if (!launch_info.GetFlags().Test(eLaunchFlagDebug)) {
    StreamString stream;
    stream.Printf("ProcessWindows unable to launch '%s'.  ProcessWindows can "
                  "only be used for debug launches.",
                  launch_info.GetExecutableFile().GetPath().c_str());
    std::string message = stream.GetString();
    result.SetErrorString(message.c_str());

    WINERR_IFALL(WINDOWS_LOG_PROCESS, message.c_str());
    return result;
  }

  bool stop_at_entry = launch_info.GetFlags().Test(eLaunchFlagStopAtEntry);
  m_session_data.reset(new ProcessWindowsData(stop_at_entry));

  SetPrivateState(eStateLaunching);
  DebugDelegateSP delegate(new LocalDebugDelegate(shared_from_this()));
  m_session_data->m_debugger.reset(new DebuggerThread(delegate));
  DebuggerThreadSP debugger = m_session_data->m_debugger;

  // Kick off the DebugLaunch asynchronously and wait for it to complete.
  result = debugger->DebugLaunch(launch_info);
  if (result.Fail()) {
    WINERR_IFALL(WINDOWS_LOG_PROCESS, "DoLaunch failed launching '%s'.  %s",
                 launch_info.GetExecutableFile().GetPath().c_str(),
                 result.AsCString());
    return result;
  }

  HostProcess process;
  Error error = WaitForDebuggerConnection(debugger, process);
  if (error.Fail()) {
    WINERR_IFALL(WINDOWS_LOG_PROCESS, "DoLaunch failed launching '%s'.  %s",
                 launch_info.GetExecutableFile().GetPath().c_str(),
                 error.AsCString());
    return error;
  }

  WINLOG_IFALL(WINDOWS_LOG_PROCESS, "DoLaunch successfully launched '%s'",
               launch_info.GetExecutableFile().GetPath().c_str());

  // We've hit the initial stop.  If eLaunchFlagsStopAtEntry was specified, the
  // private state
  // should already be set to eStateStopped as a result of hitting the initial
  // breakpoint.  If
  // it was not set, the breakpoint should have already been resumed from and
  // the private state
  // should already be eStateRunning.
  launch_info.SetProcessID(process.GetProcessId());
  SetID(process.GetProcessId());

  return result;
}

Error ProcessWindowsLive::DoAttachToProcessWithID(
    lldb::pid_t pid, const ProcessAttachInfo &attach_info) {
  m_session_data.reset(
      new ProcessWindowsData(!attach_info.GetContinueOnceAttached()));

  DebugDelegateSP delegate(new LocalDebugDelegate(shared_from_this()));
  DebuggerThreadSP debugger(new DebuggerThread(delegate));

  m_session_data->m_debugger = debugger;

  DWORD process_id = static_cast<DWORD>(pid);
  Error error = debugger->DebugAttach(process_id, attach_info);
  if (error.Fail()) {
    WINLOG_IFALL(WINDOWS_LOG_PROCESS, "DoAttachToProcessWithID encountered an "
                                      "error occurred initiating the "
                                      "asynchronous attach.  %s",
                 error.AsCString());
    return error;
  }

  HostProcess process;
  error = WaitForDebuggerConnection(debugger, process);
  if (error.Fail()) {
    WINLOG_IFALL(WINDOWS_LOG_PROCESS, "DoAttachToProcessWithID encountered an "
                                      "error waiting for the debugger to "
                                      "connect.  %s",
                 error.AsCString());
    return error;
  }

  WINLOG_IFALL(
      WINDOWS_LOG_PROCESS,
      "DoAttachToProcessWithID successfully attached to process with pid=%u",
      process_id);

  // We've hit the initial stop.  If eLaunchFlagsStopAtEntry was specified, the
  // private state
  // should already be set to eStateStopped as a result of hitting the initial
  // breakpoint.  If
  // it was not set, the breakpoint should have already been resumed from and
  // the private state
  // should already be eStateRunning.
  SetID(process.GetProcessId());
  return error;
}

Error ProcessWindowsLive::WaitForDebuggerConnection(DebuggerThreadSP debugger,
                                                    HostProcess &process) {
  Error result;
  WINLOG_IFANY(WINDOWS_LOG_PROCESS | WINDOWS_LOG_BREAKPOINTS,
               "WaitForDebuggerConnection Waiting for loader breakpoint.");

  // Block this function until we receive the initial stop from the process.
  if (::WaitForSingleObject(m_session_data->m_initial_stop_event, INFINITE) ==
      WAIT_OBJECT_0) {
    WINLOG_IFANY(WINDOWS_LOG_PROCESS | WINDOWS_LOG_BREAKPOINTS,
                 "WaitForDebuggerConnection hit loader breakpoint, returning.");

    process = debugger->GetProcess();
    return m_session_data->m_launch_error;
  } else
    return Error(::GetLastError(), eErrorTypeWin32);
}

Error ProcessWindowsLive::DoResume() {
  llvm::sys::ScopedLock lock(m_mutex);
  Error error;

  StateType private_state = GetPrivateState();
  if (private_state == eStateStopped || private_state == eStateCrashed) {
    WINLOG_IFALL(
        WINDOWS_LOG_PROCESS,
        "DoResume called for process %I64u while state is %u.  Resuming...",
        m_session_data->m_debugger->GetProcess().GetProcessId(),
        GetPrivateState());

    ExceptionRecordSP active_exception =
        m_session_data->m_debugger->GetActiveException().lock();
    if (active_exception) {
      // Resume the process and continue processing debug events.  Mask
      // the exception so that from the process's view, there is no
      // indication that anything happened.
      m_session_data->m_debugger->ContinueAsyncException(
          ExceptionResult::MaskException);
    }

    WINLOG_IFANY(WINDOWS_LOG_PROCESS | WINDOWS_LOG_THREAD,
                 "DoResume resuming %u threads.", m_thread_list.GetSize());

    for (int i = 0; i < m_thread_list.GetSize(); ++i) {
      auto thread = std::static_pointer_cast<TargetThreadWindowsLive>(
          m_thread_list.GetThreadAtIndex(i));
      thread->DoResume();
    }

    SetPrivateState(eStateRunning);
  } else {
    WINERR_IFALL(
        WINDOWS_LOG_PROCESS,
        "DoResume called for process %I64u but state is %u.  Returning...",
        m_session_data->m_debugger->GetProcess().GetProcessId(),
        GetPrivateState());
  }
  return error;
}

//------------------------------------------------------------------------------
// ProcessInterface protocol.

lldb_private::ConstString ProcessWindowsLive::GetPluginName() {
  return GetPluginNameStatic();
}

uint32_t ProcessWindowsLive::GetPluginVersion() { return 1; }

Error ProcessWindowsLive::DoDetach(bool keep_stopped) {
  DebuggerThreadSP debugger_thread;
  StateType private_state;
  {
    // Acquire the lock only long enough to get the DebuggerThread.
    // StopDebugging() will trigger a call back into ProcessWindows which
    // will also acquire the lock.  Thus we have to release the lock before
    // calling StopDebugging().
    llvm::sys::ScopedLock lock(m_mutex);

    private_state = GetPrivateState();

    if (!m_session_data) {
      WINWARN_IFALL(
          WINDOWS_LOG_PROCESS,
          "DoDetach called while state = %u, but there is no active session.",
          private_state);
      return Error();
    }

    debugger_thread = m_session_data->m_debugger;
  }

  Error error;
  if (private_state != eStateExited && private_state != eStateDetached) {
    WINLOG_IFALL(
        WINDOWS_LOG_PROCESS,
        "DoDetach called for process %I64u while state = %u.  Detaching...",
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
    WINERR_IFALL(
        WINDOWS_LOG_PROCESS, "DoDetach called for process %I64u while state = "
                             "%u, but cannot destroy in this state.",
        debugger_thread->GetProcess().GetNativeProcess().GetSystemHandle(),
        private_state);
  }

  return error;
}

Error ProcessWindowsLive::DoDestroy() {
  DebuggerThreadSP debugger_thread;
  StateType private_state;
  {
    // Acquire this lock inside an inner scope, only long enough to get the
    // DebuggerThread.
    // StopDebugging() will trigger a call back into ProcessWindows which will
    // acquire the lock
    // again, so we need to not deadlock.
    llvm::sys::ScopedLock lock(m_mutex);

    private_state = GetPrivateState();

    if (!m_session_data) {
      WINWARN_IFALL(
          WINDOWS_LOG_PROCESS,
          "DoDestroy called while state = %u, but there is no active session.",
          private_state);
      return Error();
    }

    debugger_thread = m_session_data->m_debugger;
  }

  Error error;
  if (private_state != eStateExited && private_state != eStateDetached) {
    WINLOG_IFALL(
        WINDOWS_LOG_PROCESS, "DoDestroy called for process %I64u while state = "
                             "%u.  Shutting down...",
        debugger_thread->GetProcess().GetNativeProcess().GetSystemHandle(),
        private_state);
    error = debugger_thread->StopDebugging(true);

    // By the time StopDebugging returns, there is no more debugger thread, so
    // we can be assured that no other thread will race for the session data.
    m_session_data.reset();
  } else {
    WINERR_IFALL(
        WINDOWS_LOG_PROCESS, "DoDestroy called for process %I64u while state = "
                             "%u, but cannot destroy in this state.",
        debugger_thread->GetProcess().GetNativeProcess().GetSystemHandle(),
        private_state);
  }

  return error;
}

void ProcessWindowsLive::RefreshStateAfterStop() {
  llvm::sys::ScopedLock lock(m_mutex);

  if (!m_session_data) {
    WINWARN_IFALL(
        WINDOWS_LOG_PROCESS,
        "RefreshStateAfterStop called with no active session.  Returning...");
    return;
  }

  m_thread_list.RefreshStateAfterStop();

  std::weak_ptr<ExceptionRecord> exception_record =
      m_session_data->m_debugger->GetActiveException();
  ExceptionRecordSP active_exception = exception_record.lock();
  if (!active_exception) {
    WINERR_IFALL(
        WINDOWS_LOG_PROCESS,
        "RefreshStateAfterStop called for process %I64u but there is no "
        "active exception.  Why is the process stopped?",
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
      WINLOG_IFANY(WINDOWS_LOG_BREAKPOINTS | WINDOWS_LOG_EXCEPTION |
                       WINDOWS_LOG_STEP,
                   "Single-stepped onto a breakpoint in process %I64u at "
                   "address 0x%I64x with breakpoint site %d",
                   m_session_data->m_debugger->GetProcess().GetProcessId(), pc,
                   site->GetID());
      stop_info = StopInfo::CreateStopReasonWithBreakpointSiteID(*stop_thread,
                                                                 site->GetID());
      stop_thread->SetStopInfo(stop_info);
    } else {
      WINLOG_IFANY(WINDOWS_LOG_EXCEPTION | WINDOWS_LOG_STEP,
                   "RefreshStateAfterStop single stepping thread %u",
                   stop_thread->GetID());
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
      WINLOG_IFANY(
          WINDOWS_LOG_BREAKPOINTS | WINDOWS_LOG_EXCEPTION,
          "RefreshStateAfterStop detected breakpoint in process %I64u at "
          "address 0x%I64x with breakpoint site %d",
          m_session_data->m_debugger->GetProcess().GetProcessId(), pc,
          site->GetID());

      if (site->ValidForThisThread(stop_thread.get())) {
        WINLOG_IFALL(WINDOWS_LOG_BREAKPOINTS | WINDOWS_LOG_EXCEPTION,
                     "Breakpoint site %d is valid for this thread (0x%I64x), "
                     "creating stop info.",
                     site->GetID(), stop_thread->GetID());

        stop_info = StopInfo::CreateStopReasonWithBreakpointSiteID(
            *stop_thread, site->GetID());
        register_context->SetPC(pc);
      } else {
        WINLOG_IFALL(WINDOWS_LOG_BREAKPOINTS | WINDOWS_LOG_EXCEPTION,
                     "Breakpoint site %d is not valid for this thread, "
                     "creating empty stop info.",
                     site->GetID());
      }
      stop_thread->SetStopInfo(stop_info);
      return;
    } else {
      // The thread hit a hard-coded breakpoint like an `int 3` or
      // `__debugbreak()`.
      WINLOG_IFALL(
          WINDOWS_LOG_BREAKPOINTS | WINDOWS_LOG_EXCEPTION,
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
    WINLOG_IFALL(WINDOWS_LOG_EXCEPTION, desc_stream.str().c_str());
    return;
  }
  }
}

bool ProcessWindowsLive::IsAlive() {
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

Error ProcessWindowsLive::DoHalt(bool &caused_stop) {
  Error error;
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
      WINERR_IFALL(
          WINDOWS_LOG_PROCESS,
          "DoHalt called DebugBreakProcess, but it failed with error %u",
          error.GetError());
    }
  }
  return error;
}

void ProcessWindowsLive::DidLaunch() {
  ArchSpec arch_spec;
  DidAttach(arch_spec);
}

void ProcessWindowsLive::DidAttach(ArchSpec &arch_spec) {
  llvm::sys::ScopedLock lock(m_mutex);

  // The initial stop won't broadcast the state change event, so account for
  // that here.
  if (m_session_data && GetPrivateState() == eStateStopped &&
      m_session_data->m_stop_at_entry)
    RefreshStateAfterStop();
}

size_t ProcessWindowsLive::DoReadMemory(lldb::addr_t vm_addr, void *buf,
                                        size_t size, Error &error) {
  llvm::sys::ScopedLock lock(m_mutex);

  if (!m_session_data)
    return 0;

  WINLOG_IFALL(WINDOWS_LOG_MEMORY,
               "DoReadMemory attempting to read %u bytes from address 0x%I64x",
               size, vm_addr);

  HostProcess process = m_session_data->m_debugger->GetProcess();
  void *addr = reinterpret_cast<void *>(vm_addr);
  SIZE_T bytes_read = 0;
  if (!ReadProcessMemory(process.GetNativeProcess().GetSystemHandle(), addr,
                         buf, size, &bytes_read)) {
    error.SetError(GetLastError(), eErrorTypeWin32);
    WINERR_IFALL(WINDOWS_LOG_MEMORY, "DoReadMemory failed with error code %u",
                 error.GetError());
  }
  return bytes_read;
}

size_t ProcessWindowsLive::DoWriteMemory(lldb::addr_t vm_addr, const void *buf,
                                         size_t size, Error &error) {
  llvm::sys::ScopedLock lock(m_mutex);
  WINLOG_IFALL(
      WINDOWS_LOG_MEMORY,
      "DoWriteMemory attempting to write %u bytes into address 0x%I64x", size,
      vm_addr);

  if (!m_session_data) {
    WINERR_IFANY(
        WINDOWS_LOG_MEMORY,
        "DoWriteMemory cannot write, there is no active debugger connection.");
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
    WINLOG_IFALL(WINDOWS_LOG_MEMORY, "DoWriteMemory failed with error code %u",
                 error.GetError());
  }
  return bytes_written;
}

Error ProcessWindowsLive::GetMemoryRegionInfo(lldb::addr_t vm_addr,
                                              MemoryRegionInfo &info) {
  Error error;
  llvm::sys::ScopedLock lock(m_mutex);
  info.Clear();

  if (!m_session_data) {
    error.SetErrorString(
        "GetMemoryRegionInfo called with no debugging session.");
    WINERR_IFALL(WINDOWS_LOG_MEMORY, error.AsCString());
    return error;
  }
  HostProcess process = m_session_data->m_debugger->GetProcess();
  lldb::process_t handle = process.GetNativeProcess().GetSystemHandle();
  if (handle == nullptr || handle == LLDB_INVALID_PROCESS) {
    error.SetErrorString(
        "GetMemoryRegionInfo called with an invalid target process.");
    WINERR_IFALL(WINDOWS_LOG_MEMORY, error.AsCString());
    return error;
  }

  WINLOG_IFALL(WINDOWS_LOG_MEMORY,
               "GetMemoryRegionInfo getting info for address 0x%I64x", vm_addr);

  void *addr = reinterpret_cast<void *>(vm_addr);
  MEMORY_BASIC_INFORMATION mem_info = {0};
  SIZE_T result = ::VirtualQueryEx(handle, addr, &mem_info, sizeof(mem_info));
  if (result == 0) {
    if (::GetLastError() == ERROR_INVALID_PARAMETER) {
      // ERROR_INVALID_PARAMETER is returned if VirtualQueryEx is called with an
      // address
      // past the highest accessible address. We should return a range from the
      // vm_addr
      // to LLDB_INVALID_ADDRESS
      info.GetRange().SetRangeBase(vm_addr);
      info.GetRange().SetRangeEnd(LLDB_INVALID_ADDRESS);
      info.SetReadable(MemoryRegionInfo::eNo);
      info.SetExecutable(MemoryRegionInfo::eNo);
      info.SetWritable(MemoryRegionInfo::eNo);
      info.SetMapped(MemoryRegionInfo::eNo);
      return error;
    } else {
      error.SetError(::GetLastError(), eErrorTypeWin32);
      WINERR_IFALL(WINDOWS_LOG_MEMORY, "VirtualQueryEx returned error %u while "
                                       "getting memory region info for address "
                                       "0x%I64x",
                   error.GetError(), vm_addr);
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
    // memory.
    // VirtualQueryEx nearly does that except that it gives the distance from
    // the start
    // of the page containing vm_addr.
    SYSTEM_INFO data;
    GetSystemInfo(&data);
    DWORD page_offset = vm_addr % data.dwPageSize;
    info.GetRange().SetRangeBase(vm_addr);
    info.GetRange().SetByteSize(mem_info.RegionSize - page_offset);
    info.SetMapped(MemoryRegionInfo::eNo);
  }

  error.SetError(::GetLastError(), eErrorTypeWin32);
  WINLOGV_IFALL(WINDOWS_LOG_MEMORY, "Memory region info for address 0x%I64u: "
                                    "readable=%s, executable=%s, writable=%s",
                BOOL_STR(info.GetReadable()), BOOL_STR(info.GetExecutable()),
                BOOL_STR(info.GetWritable()));
  return error;
}

bool ProcessWindowsLive::CanDebug(lldb::TargetSP target_sp,
                                  bool plugin_specified_by_name) {
  if (plugin_specified_by_name)
    return true;

  // For now we are just making sure the file exists for a given module
  ModuleSP exe_module_sp(target_sp->GetExecutableModule());
  if (exe_module_sp.get())
    return exe_module_sp->GetFileSpec().Exists();
  // However, if there is no executable module, we return true since we might be
  // preparing to attach.
  return true;
}

void ProcessWindowsLive::OnExitProcess(uint32_t exit_code) {
  // No need to acquire the lock since m_session_data isn't accessed.
  WINLOG_IFALL(WINDOWS_LOG_PROCESS, "Process %u exited with code %u", GetID(),
               exit_code);

  TargetSP target = m_target_sp.lock();
  if (target) {
    ModuleSP executable_module = target->GetExecutableModule();
    ModuleList unloaded_modules;
    unloaded_modules.Append(executable_module);
    target->ModulesDidUnload(unloaded_modules, true);
  }

  SetProcessExitStatus(GetID(), true, 0, exit_code);
  SetPrivateState(eStateExited);
}

void ProcessWindowsLive::OnDebuggerConnected(lldb::addr_t image_base) {
  DebuggerThreadSP debugger = m_session_data->m_debugger;

  WINLOG_IFALL(WINDOWS_LOG_PROCESS,
               "Debugger connected to process %I64u.  Image base = 0x%I64x",
               debugger->GetProcess().GetProcessId(), image_base);

  ModuleSP module = GetTarget().GetExecutableModule();
  if (!module) {
    // During attach, we won't have the executable module, so find it now.
    const DWORD pid = debugger->GetProcess().GetProcessId();
    const std::string file_name = GetProcessExecutableName(pid);
    if (file_name.empty()) {
      return;
    }

    FileSpec executable_file(file_name, true);
    ModuleSpec module_spec(executable_file);
    Error error;
    module = GetTarget().GetSharedModule(module_spec, &error);
    if (!module) {
      return;
    }

    GetTarget().SetExecutableModule(module, false);
  }

  bool load_addr_changed;
  module->SetLoadAddress(GetTarget(), image_base, false, load_addr_changed);

  ModuleList loaded_modules;
  loaded_modules.Append(module);
  GetTarget().ModulesDidLoad(loaded_modules);

  // Add the main executable module to the list of pending module loads.  We
  // can't call
  // GetTarget().ModulesDidLoad() here because we still haven't returned from
  // DoLaunch() / DoAttach() yet
  // so the target may not have set the process instance to `this` yet.
  llvm::sys::ScopedLock lock(m_mutex);
  const HostThreadWindows &wmain_thread =
      debugger->GetMainThread().GetNativeThread();
  m_session_data->m_new_threads[wmain_thread.GetThreadId()] =
      debugger->GetMainThread();
}

ExceptionResult
ProcessWindowsLive::OnDebugException(bool first_chance,
                                     const ExceptionRecord &record) {
  llvm::sys::ScopedLock lock(m_mutex);

  // FIXME: Without this check, occasionally when running the test suite there
  // is
  // an issue where m_session_data can be null.  It's not clear how this could
  // happen
  // but it only surfaces while running the test suite.  In order to properly
  // diagnose
  // this, we probably need to first figure allow the test suite to print out
  // full
  // lldb logs, and then add logging to the process plugin.
  if (!m_session_data) {
    WINERR_IFANY(WINDOWS_LOG_EXCEPTION, "Debugger thread reported exception "
                                        "0x%x at address 0x%I64x, but there is "
                                        "no session.",
                 record.GetExceptionCode(), record.GetExceptionAddress());
    return ExceptionResult::SendToApplication;
  }

  if (!first_chance) {
    // Any second chance exception is an application crash by definition.
    SetPrivateState(eStateCrashed);
  }

  ExceptionResult result = ExceptionResult::SendToApplication;
  switch (record.GetExceptionCode()) {
  case EXCEPTION_BREAKPOINT:
    // Handle breakpoints at the first chance.
    result = ExceptionResult::BreakInDebugger;

    if (!m_session_data->m_initial_stop_received) {
      WINLOG_IFANY(WINDOWS_LOG_BREAKPOINTS, "Hit loader breakpoint at address "
                                            "0x%I64x, setting initial stop "
                                            "event.",
                   record.GetExceptionAddress());
      m_session_data->m_initial_stop_received = true;
      ::SetEvent(m_session_data->m_initial_stop_event);
    } else {
      WINLOG_IFANY(WINDOWS_LOG_BREAKPOINTS,
                   "Hit non-loader breakpoint at address 0x%I64x.",
                   record.GetExceptionAddress());
    }
    SetPrivateState(eStateStopped);
    break;
  case EXCEPTION_SINGLE_STEP:
    result = ExceptionResult::BreakInDebugger;
    SetPrivateState(eStateStopped);
    break;
  default:
    WINLOG_IFANY(WINDOWS_LOG_EXCEPTION, "Debugger thread reported exception "
                                        "0x%x at address 0x%I64x "
                                        "(first_chance=%s)",
                 record.GetExceptionCode(), record.GetExceptionAddress(),
                 BOOL_STR(first_chance));
    // For non-breakpoints, give the application a chance to handle the
    // exception first.
    if (first_chance)
      result = ExceptionResult::SendToApplication;
    else
      result = ExceptionResult::BreakInDebugger;
  }

  return result;
}

void ProcessWindowsLive::OnCreateThread(const HostThread &new_thread) {
  llvm::sys::ScopedLock lock(m_mutex);
  const HostThreadWindows &wnew_thread = new_thread.GetNativeThread();
  m_session_data->m_new_threads[wnew_thread.GetThreadId()] = new_thread;
}

void ProcessWindowsLive::OnExitThread(lldb::tid_t thread_id,
                                      uint32_t exit_code) {
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

void ProcessWindowsLive::OnLoadDll(const ModuleSpec &module_spec,
                                   lldb::addr_t module_addr) {
  // Confusingly, there is no Target::AddSharedModule.  Instead, calling
  // GetSharedModule() with
  // a new module will add it to the module list and return a corresponding
  // ModuleSP.
  Error error;
  ModuleSP module = GetTarget().GetSharedModule(module_spec, &error);
  bool load_addr_changed = false;
  module->SetLoadAddress(GetTarget(), module_addr, false, load_addr_changed);

  ModuleList loaded_modules;
  loaded_modules.Append(module);
  GetTarget().ModulesDidLoad(loaded_modules);
}

void ProcessWindowsLive::OnUnloadDll(lldb::addr_t module_addr) {
  Address resolved_addr;
  if (GetTarget().ResolveLoadAddress(module_addr, resolved_addr)) {
    ModuleSP module = resolved_addr.GetModule();
    if (module) {
      ModuleList unloaded_modules;
      unloaded_modules.Append(module);
      GetTarget().ModulesDidUnload(unloaded_modules, false);
    }
  }
}

void ProcessWindowsLive::OnDebugString(const std::string &string) {}

void ProcessWindowsLive::OnDebuggerError(const Error &error, uint32_t type) {
  llvm::sys::ScopedLock lock(m_mutex);

  if (m_session_data->m_initial_stop_received) {
    // This happened while debugging.  Do we shutdown the debugging session, try
    // to continue,
    // or do something else?
    WINERR_IFALL(WINDOWS_LOG_PROCESS, "Error %u occurred during debugging.  "
                                      "Unexpected behavior may result.  %s",
                 error.GetError(), error.AsCString());
  } else {
    // If we haven't actually launched the process yet, this was an error
    // launching the
    // process.  Set the internal error and signal the initial stop event so
    // that the DoLaunch
    // method wakes up and returns a failure.
    m_session_data->m_launch_error = error;
    ::SetEvent(m_session_data->m_initial_stop_event);
    WINERR_IFALL(
        WINDOWS_LOG_PROCESS,
        "Error %u occurred launching the process before the initial stop.  %s",
        error.GetError(), error.AsCString());
    return;
  }
}
