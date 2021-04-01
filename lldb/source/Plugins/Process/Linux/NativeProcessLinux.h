//===-- NativeProcessLinux.h ---------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_NativeProcessLinux_H_
#define liblldb_NativeProcessLinux_H_

#include <csignal>
#include <unordered_set>

#include "lldb/Host/Debug.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Host/linux/Support.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/lldb-types.h"

#include "IntelPTManager.h"
#include "NativeThreadLinux.h"
#include "Plugins/Process/POSIX/NativeProcessELF.h"
#include "Plugins/Process/Utility/NativeProcessSoftwareSingleStep.h"

namespace lldb_private {
class Status;
class Scalar;

namespace process_linux {
/// \class NativeProcessLinux
/// Manages communication with the inferior (debugee) process.
///
/// Upon construction, this class prepares and launches an inferior process
/// for debugging.
///
/// Changes in the inferior process state are broadcasted.
class NativeProcessLinux : public NativeProcessELF,
                           private NativeProcessSoftwareSingleStep {
public:
  class Factory : public NativeProcessProtocol::Factory {
  public:
    llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
    Launch(ProcessLaunchInfo &launch_info, NativeDelegate &native_delegate,
           MainLoop &mainloop) const override;

    llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
    Attach(lldb::pid_t pid, NativeDelegate &native_delegate,
           MainLoop &mainloop) const override;
  };

  // NativeProcessProtocol Interface
  Status Resume(const ResumeActionList &resume_actions) override;

  Status Halt() override;

  Status Detach() override;

  Status Signal(int signo) override;

  Status Interrupt() override;

  Status Kill() override;

  Status GetMemoryRegionInfo(lldb::addr_t load_addr,
                             MemoryRegionInfo &range_info) override;

  Status ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                    size_t &bytes_read) override;

  Status WriteMemory(lldb::addr_t addr, const void *buf, size_t size,
                     size_t &bytes_written) override;

  llvm::Expected<lldb::addr_t> AllocateMemory(size_t size,
                                              uint32_t permissions) override;

  llvm::Error DeallocateMemory(lldb::addr_t addr) override;

  size_t UpdateThreads() override;

  const ArchSpec &GetArchitecture() const override { return m_arch; }

  Status SetBreakpoint(lldb::addr_t addr, uint32_t size,
                       bool hardware) override;

  Status RemoveBreakpoint(lldb::addr_t addr, bool hardware = false) override;

  void DoStopIDBumped(uint32_t newBumpId) override;

  Status GetLoadedModuleFileSpec(const char *module_path,
                                 FileSpec &file_spec) override;

  Status GetFileLoadAddress(const llvm::StringRef &file_name,
                            lldb::addr_t &load_addr) override;

  NativeThreadLinux *GetThreadByID(lldb::tid_t id);
  NativeThreadLinux *GetCurrentThread();

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  GetAuxvData() const override {
    return getProcFile(GetID(), "auxv");
  }

  /// Tracing
  /// These methods implement the jLLDBTrace packets
  /// \{
  llvm::Error TraceStart(llvm::StringRef json_request,
                         llvm::StringRef type) override;

  llvm::Error TraceStop(const TraceStopRequest &request) override;

  llvm::Expected<llvm::json::Value>
  TraceGetState(llvm::StringRef type) override;

  llvm::Expected<std::vector<uint8_t>>
  TraceGetBinaryData(const TraceGetBinaryDataRequest &request) override;

  llvm::Expected<TraceSupportedResponse> TraceSupported() override;
  /// }

  // Interface used by NativeRegisterContext-derived classes.
  static Status PtraceWrapper(int req, lldb::pid_t pid, void *addr = nullptr,
                              void *data = nullptr, size_t data_size = 0,
                              long *result = nullptr);

  bool SupportHardwareSingleStepping() const;

protected:
  llvm::Expected<llvm::ArrayRef<uint8_t>>
  GetSoftwareBreakpointTrapOpcode(size_t size_hint) override;

  llvm::Expected<uint64_t> Syscall(llvm::ArrayRef<uint64_t> args);

private:
  MainLoop::SignalHandleUP m_sigchld_handle;
  ArchSpec m_arch;

  LazyBool m_supports_mem_region = eLazyBoolCalculate;
  std::vector<std::pair<MemoryRegionInfo, FileSpec>> m_mem_region_cache;

  lldb::tid_t m_pending_notification_tid = LLDB_INVALID_THREAD_ID;

  /// Inferior memory (allocated by us) and its size.
  llvm::DenseMap<lldb::addr_t, lldb::addr_t> m_allocated_memory;

  // Private Instance Methods
  NativeProcessLinux(::pid_t pid, int terminal_fd, NativeDelegate &delegate,
                     const ArchSpec &arch, MainLoop &mainloop,
                     llvm::ArrayRef<::pid_t> tids);

  // Returns a list of process threads that we have attached to.
  static llvm::Expected<std::vector<::pid_t>> Attach(::pid_t pid);

  static Status SetDefaultPtraceOpts(const lldb::pid_t);

  void MonitorCallback(lldb::pid_t pid, bool exited, WaitStatus status);

  void WaitForNewThread(::pid_t tid);

  void MonitorSIGTRAP(const siginfo_t &info, NativeThreadLinux &thread);

  void MonitorTrace(NativeThreadLinux &thread);

  void MonitorBreakpoint(NativeThreadLinux &thread);

  void MonitorWatchpoint(NativeThreadLinux &thread, uint32_t wp_index);

  void MonitorSignal(const siginfo_t &info, NativeThreadLinux &thread,
                     bool exited);

  bool HasThreadNoLock(lldb::tid_t thread_id);

  bool StopTrackingThread(lldb::tid_t thread_id);

  /// Create a new thread.
  ///
  /// If process tracing is enabled and the thread can't be traced, then the
  /// thread is left stopped with a \a eStopReasonProcessorTrace status, and
  /// then the process is stopped.
  ///
  /// \param[in] resume
  ///     If a tracing error didn't happen, then resume the thread after
  ///     creation if \b true, or leave it stopped with SIGSTOP if \b false.
  NativeThreadLinux &AddThread(lldb::tid_t thread_id, bool resume);

  /// Start tracing a new thread if process tracing is enabled.
  ///
  /// Trace mechanisms should modify this method to provide automatic tracing
  /// for new threads.
  Status NotifyTracersOfNewThread(lldb::tid_t tid);

  /// Stop tracing threads upon a destroy event.
  ///
  /// Trace mechanisms should modify this method to provide automatic trace
  /// stopping for threads being destroyed.
  Status NotifyTracersOfThreadDestroyed(lldb::tid_t tid);

  /// Writes a siginfo_t structure corresponding to the given thread ID to the
  /// memory region pointed to by \p siginfo.
  Status GetSignalInfo(lldb::tid_t tid, void *siginfo);

  /// Writes the raw event message code (vis-a-vis PTRACE_GETEVENTMSG)
  /// corresponding to the given thread ID to the memory pointed to by @p
  /// message.
  Status GetEventMessage(lldb::tid_t tid, unsigned long *message);

  void NotifyThreadDeath(lldb::tid_t tid);

  Status Detach(lldb::tid_t tid);

  // This method is requests a stop on all threads which are still running. It
  // sets up a
  // deferred delegate notification, which will fire once threads report as
  // stopped. The
  // triggerring_tid will be set as the current thread (main stop reason).
  void StopRunningThreads(lldb::tid_t triggering_tid);

  // Notify the delegate if all threads have stopped.
  void SignalIfAllThreadsStopped();

  // Resume the given thread, optionally passing it the given signal. The type
  // of resume
  // operation (continue, single-step) depends on the state parameter.
  Status ResumeThread(NativeThreadLinux &thread, lldb::StateType state,
                      int signo);

  void ThreadWasCreated(NativeThreadLinux &thread);

  void SigchldHandler();

  Status PopulateMemoryRegionCache();

  /// Manages Intel PT process and thread traces.
  IntelPTManager m_intel_pt_manager;
};

} // namespace process_linux
} // namespace lldb_private

#endif // #ifndef liblldb_NativeProcessLinux_H_
