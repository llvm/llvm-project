//===-- NativeProcessLinux.h ---------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

#include "NativeThreadLinux.h"
#include "ProcessorTrace.h"
#include "lldb/Host/common/NativeProcessProtocol.h"

namespace lldb_private {
class Status;
class Scalar;

namespace process_linux {
/// @class NativeProcessLinux
/// @brief Manages communication with the inferior (debugee) process.
///
/// Upon construction, this class prepares and launches an inferior process for
/// debugging.
///
/// Changes in the inferior process state are broadcasted.
class NativeProcessLinux : public NativeProcessProtocol {
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

  // ---------------------------------------------------------------------
  // NativeProcessProtocol Interface
  // ---------------------------------------------------------------------
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

  Status ReadMemoryWithoutTrap(lldb::addr_t addr, void *buf, size_t size,
                               size_t &bytes_read) override;

  Status WriteMemory(lldb::addr_t addr, const void *buf, size_t size,
                     size_t &bytes_written) override;

  Status AllocateMemory(size_t size, uint32_t permissions,
                        lldb::addr_t &addr) override;

  Status DeallocateMemory(lldb::addr_t addr) override;

  lldb::addr_t GetSharedLibraryInfoAddress() override;

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

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  GetAuxvData() const override {
    return getProcFile(GetID(), "auxv");
  }

  lldb::user_id_t StartTrace(const TraceOptions &config,
                             Status &error) override;

  Status StopTrace(lldb::user_id_t traceid,
                   lldb::tid_t thread) override;

  Status GetData(lldb::user_id_t traceid, lldb::tid_t thread,
                 llvm::MutableArrayRef<uint8_t> &buffer,
                 size_t offset = 0) override;

  Status GetMetaData(lldb::user_id_t traceid, lldb::tid_t thread,
                     llvm::MutableArrayRef<uint8_t> &buffer,
                     size_t offset = 0) override;

  Status GetTraceConfig(lldb::user_id_t traceid, TraceOptions &config) override;

  // ---------------------------------------------------------------------
  // Interface used by NativeRegisterContext-derived classes.
  // ---------------------------------------------------------------------
  static Status PtraceWrapper(int req, lldb::pid_t pid, void *addr = nullptr,
                              void *data = nullptr, size_t data_size = 0,
                              long *result = nullptr);

  bool SupportHardwareSingleStepping() const;

protected:
  // ---------------------------------------------------------------------
  // NativeProcessProtocol protected interface
  // ---------------------------------------------------------------------
  Status
  GetSoftwareBreakpointTrapOpcode(size_t trap_opcode_size_hint,
                                  size_t &actual_opcode_size,
                                  const uint8_t *&trap_opcode_bytes) override;

private:
  MainLoop::SignalHandleUP m_sigchld_handle;
  ArchSpec m_arch;

  LazyBool m_supports_mem_region = eLazyBoolCalculate;
  std::vector<std::pair<MemoryRegionInfo, FileSpec>> m_mem_region_cache;

  lldb::tid_t m_pending_notification_tid = LLDB_INVALID_THREAD_ID;

  // List of thread ids stepping with a breakpoint with the address of
  // the relevan breakpoint
  std::map<lldb::tid_t, lldb::addr_t> m_threads_stepping_with_breakpoint;

  // ---------------------------------------------------------------------
  // Private Instance Methods
  // ---------------------------------------------------------------------
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

  Status SetupSoftwareSingleStepping(NativeThreadLinux &thread);

#if 0
        static ::ProcessMessage::CrashReason
        GetCrashReasonForSIGSEGV(const siginfo_t *info);

        static ::ProcessMessage::CrashReason
        GetCrashReasonForSIGILL(const siginfo_t *info);

        static ::ProcessMessage::CrashReason
        GetCrashReasonForSIGFPE(const siginfo_t *info);

        static ::ProcessMessage::CrashReason
        GetCrashReasonForSIGBUS(const siginfo_t *info);
#endif

  bool HasThreadNoLock(lldb::tid_t thread_id);

  bool StopTrackingThread(lldb::tid_t thread_id);

  NativeThreadLinux &AddThread(lldb::tid_t thread_id);

  Status GetSoftwareBreakpointPCOffset(uint32_t &actual_opcode_size);

  Status FixupBreakpointPCAsNeeded(NativeThreadLinux &thread);

  /// Writes a siginfo_t structure corresponding to the given thread ID to the
  /// memory region pointed to by @p siginfo.
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

  lldb::user_id_t StartTraceGroup(const TraceOptions &config,
                                         Status &error);

  // This function is intended to be used to stop tracing
  // on a thread that exited.
  Status StopTracingForThread(lldb::tid_t thread);

  // The below function as the name suggests, looks up a ProcessorTrace
  // instance from the m_processor_trace_monitor map. In the case of
  // process tracing where the traceid passed would map to the complete
  // process, it is mandatory to provide a threadid to obtain a trace
  // instance (since ProcessorTrace is tied to a thread). In the other
  // scenario that an individual thread is being traced, just the traceid
  // is sufficient to obtain the actual ProcessorTrace instance.
  llvm::Expected<ProcessorTraceMonitor &>
  LookupProcessorTraceInstance(lldb::user_id_t traceid, lldb::tid_t thread);

  // Stops tracing on individual threads being traced. Not intended
  // to be used to stop tracing on complete process.
  Status StopProcessorTracingOnThread(lldb::user_id_t traceid,
                                      lldb::tid_t thread);

  // Intended to stop tracing on complete process.
  // Should not be used for stopping trace on
  // individual threads.
  void StopProcessorTracingOnProcess();

  llvm::DenseMap<lldb::tid_t, ProcessorTraceMonitorUP>
      m_processor_trace_monitor;

  // Set for tracking threads being traced under
  // same process user id.
  llvm::DenseSet<lldb::tid_t> m_pt_traced_thread_group;

  lldb::user_id_t m_pt_proces_trace_id = LLDB_INVALID_UID;
  TraceOptions m_pt_process_trace_config;
};

} // namespace process_linux
} // namespace lldb_private

#endif // #ifndef liblldb_NativeProcessLinux_H_
