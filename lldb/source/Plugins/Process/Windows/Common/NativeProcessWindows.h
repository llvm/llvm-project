//===-- NativeProcessWindows.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_NativeProcessWindows_h_
#define liblldb_NativeProcessWindows_h_

#include "lldb/Core/ThreadedCommunication.h"
#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/lldb-forward.h"

#include "IDebugDelegate.h"
#include "ProcessDebugger.h"

namespace lldb_private {

class HostProcess;
class NativeProcessWindows;
class NativeThreadWindows;
class NativeDebugDelegate;
class PseudoConsole;

using NativeDebugDelegateSP = std::shared_ptr<NativeDebugDelegate>;

//------------------------------------------------------------------
// NativeProcessWindows
//------------------------------------------------------------------
class NativeProcessWindows : public NativeProcessProtocol,
                             public ProcessDebugger {

public:
  class Manager : public NativeProcessProtocol::Manager {
  public:
    using NativeProcessProtocol::Manager::Manager;

    llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
    Launch(ProcessLaunchInfo &launch_info,
           NativeDelegate &native_delegate) override;

    llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
    Attach(lldb::pid_t pid, NativeDelegate &native_delegate) override;

    Extension GetSupportedExtensions() const override {
      return Extension::libraries;
    }
  };

  ~NativeProcessWindows() override;

  Status Resume(const ResumeActionList &resume_actions) override;

  Status Halt() override;

  Status Detach() override;

  Status Signal(int signo) override;

  Status Interrupt() override;

  Status Kill() override;

  Status IgnoreSignals(llvm::ArrayRef<int> signals) override;

  Status GetMemoryRegionInfo(lldb::addr_t load_addr,
                             MemoryRegionInfo &range_info) override;

  Status ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                    size_t &bytes_read) override;

  Status WriteMemory(lldb::addr_t addr, const void *buf, size_t size,
                     size_t &bytes_written) override;

  llvm::Expected<lldb::addr_t> AllocateMemory(size_t size,
                                              uint32_t permissions) override;

  llvm::Error DeallocateMemory(lldb::addr_t addr) override;

  lldb::addr_t GetSharedLibraryInfoAddress() override;

  bool IsAlive() const override;

  size_t UpdateThreads() override;

  const ArchSpec &GetArchitecture() const override { return m_arch; }

  void SetArchitecture(const ArchSpec &arch_spec) { m_arch = arch_spec; }

  Status SetBreakpoint(lldb::addr_t addr, uint32_t size,
                       bool hardware) override;

  Status RemoveBreakpoint(lldb::addr_t addr, bool hardware = false) override;

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  GetAuxvData() const override;

  Status GetLoadedModuleFileSpec(const char *module_path,
                                 FileSpec &file_spec) override;

  Status GetFileLoadAddress(const llvm::StringRef &file_name,
                            lldb::addr_t &load_addr) override;

  llvm::Expected<std::vector<LoadedLibraryInfo>> GetLoadedLibraries() override;

  bool HasPendingLibraryEvents() override;

  void SetClientSupportsLibrariesRead(bool v) {
    m_client_supports_libraries_read = v;
  }

  /// Forwards to NativeProcessProtocol's bookkeeping.
  void SetEnabledExtensions(Extension flags) override {
    NativeProcessProtocol::SetEnabledExtensions(flags);
    SetClientSupportsLibrariesRead(
        bool(flags & (Extension::libraries | Extension::libraries_svr4)));
  }

  /// Forward bytes from the gdb-remote `I` packet into the inferior's
  /// ConPTY-backed stdin via `m_stdio_communication.Write` →
  /// `ConnectionConPTY::Write` → `WriteFile` on the parent-side STDIN
  /// HANDLE. Returns the number of bytes written (0 if the PTY is
  /// disconnected or write fails).
  size_t WriteStdin(const void *buf, size_t len, Status &error) override;

  // ProcessDebugger Overrides
  void OnExitProcess(uint32_t exit_code) override;
  void OnDebuggerConnected(lldb::addr_t image_base) override;
  ExceptionResult OnDebugException(bool first_chance,
                                   const ExceptionRecord &record) override;
  void OnCreateThread(const HostThread &thread) override;
  void OnExitThread(lldb::tid_t thread_id, uint32_t exit_code) override;
  DllEventAction OnLoadDll(const ModuleSpec &module_spec,
                           lldb::addr_t module_addr,
                           lldb::tid_t thread_id) override;
  DllEventAction OnUnloadDll(lldb::addr_t module_addr,
                             lldb::tid_t thread_id) override;
  void OnDebugString(lldb::addr_t debug_string_addr, bool is_unicode,
                     uint16_t length_lower_word) override;

protected:
  NativeThreadWindows *GetThreadByID(lldb::tid_t thread_id);

  llvm::Expected<llvm::ArrayRef<uint8_t>>
  GetSoftwareBreakpointTrapOpcode(size_t size_hint) override;

  size_t GetSoftwareBreakpointPCOffset() override;

  bool FindSoftwareBreakpoint(lldb::addr_t addr);

  void StopThread(lldb::tid_t thread_id, lldb::StopReason reason,
                  std::string description = "");

  void SetStopReasonForThread(NativeThreadWindows &thread,
                              lldb::StopReason reason,
                              std::string description = "");

private:
  ArchSpec m_arch;

  NativeProcessWindows(ProcessLaunchInfo &launch_info, NativeDelegate &delegate,
                       llvm::Error &E);

  NativeProcessWindows(lldb::pid_t pid, int terminal_fd,
                       NativeDelegate &delegate, llvm::Error &E);

  ExceptionResult HandleSingleStepException(const ExceptionRecord &record);
  ExceptionResult HandleBreakpointException(const ExceptionRecord &record);
  ExceptionResult HandleGenericException(bool first_chance,
                                         const ExceptionRecord &record);

  Status CacheLoadedModules();
  std::map<lldb_private::FileSpec, lldb::addr_t> m_loaded_modules;

  /// Set whenever an OS DLL load/unload event has been seen since the last stop
  /// reply.
  bool m_pending_library_events = false;

  /// Whether we've seen the loader breakpoint that fires once per process at
  /// launch / attach.
  bool m_initial_stop_seen = false;

  bool m_expecting_loader_int3 = false;

  /// Set when Halt() / Interrupt() schedules a DebugBreakProcess injection.
  bool m_pending_halt = false;

  bool m_client_supports_libraries_read = false;

  /// PseudoConsole for the lldb-server stdio-forwarding path.
  std::shared_ptr<PseudoConsole> m_pty;

  /// Wraps a ConnectionConPTY around the PTY's parent-side STDOUT HANDLE.
  ThreadedCommunication m_stdio_communication;

  /// Bridge between m_stdio_communication's read thread and
  /// NativeDelegate::NewProcessOutput.
  static void STDIOReadThreadBytesReceived(void *baton, const void *src,
                                           size_t src_len);

  /// Wire up m_stdio_communication on m_pty's STDOUT HANDLE.
  void StartStdioForwarding();

  /// Tear down the read thread and disconnect m_stdio_communication.
  void StopStdioForwarding();
};

//------------------------------------------------------------------
// NativeDebugDelegate
//------------------------------------------------------------------
class NativeDebugDelegate : public IDebugDelegate {
public:
  NativeDebugDelegate(NativeProcessWindows &process) : m_process(process) {}

  void OnExitProcess(uint32_t exit_code) override {
    m_process.OnExitProcess(exit_code);
  }

  void OnDebuggerConnected(lldb::addr_t image_base) override {
    m_process.OnDebuggerConnected(image_base);
  }

  ExceptionResult OnDebugException(bool first_chance,
                                   const ExceptionRecord &record) override {
    return m_process.OnDebugException(first_chance, record);
  }

  void OnCreateThread(const HostThread &thread) override {
    m_process.OnCreateThread(thread);
  }

  void OnExitThread(lldb::tid_t thread_id, uint32_t exit_code) override {
    m_process.OnExitThread(thread_id, exit_code);
  }

  DllEventAction OnLoadDll(const lldb_private::ModuleSpec &module_spec,
                           lldb::addr_t module_addr,
                           lldb::tid_t thread_id) override {
    return m_process.OnLoadDll(module_spec, module_addr, thread_id);
  }

  DllEventAction OnUnloadDll(lldb::addr_t module_addr,
                             lldb::tid_t thread_id) override {
    return m_process.OnUnloadDll(module_addr, thread_id);
  }

  void OnDebugString(lldb::addr_t debug_string_addr, bool is_unicode,
                     uint16_t length_lower_word) override {
    m_process.OnDebugString(debug_string_addr, is_unicode, length_lower_word);
  }

  void OnDebuggerError(const Status &error, uint32_t type) override {
    return m_process.OnDebuggerError(error, type);
  }

private:
  NativeProcessWindows &m_process;
};

} // namespace lldb_private

#endif // #ifndef liblldb_NativeProcessWindows_h_
