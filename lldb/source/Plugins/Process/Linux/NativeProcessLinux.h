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

// C++ Includes
#include <unordered_set>

// Other libraries and framework includes
#include "lldb/Core/ArchSpec.h"
#include "lldb/Host/Debug.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Host/linux/Support.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/lldb-types.h"

#include "NativeThreadLinux.h"
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
  friend Status NativeProcessProtocol::Launch(
      ProcessLaunchInfo &launch_info, NativeDelegate &native_delegate,
      MainLoop &mainloop, NativeProcessProtocolSP &process_sp);

  friend Status NativeProcessProtocol::Attach(
      lldb::pid_t pid, NativeProcessProtocol::NativeDelegate &native_delegate,
      MainLoop &mainloop, NativeProcessProtocolSP &process_sp);

public:
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

  bool GetArchitecture(ArchSpec &arch) const override;

  Status SetBreakpoint(lldb::addr_t addr, uint32_t size,
                       bool hardware) override;

  Status RemoveBreakpoint(lldb::addr_t addr, bool hardware = false) override;

  void DoStopIDBumped(uint32_t newBumpId) override;

  Status GetLoadedModuleFileSpec(const char *module_path,
                                 FileSpec &file_spec) override;

  Status GetFileLoadAddress(const llvm::StringRef &file_name,
                            lldb::addr_t &load_addr) override;

  NativeThreadLinuxSP GetThreadByID(lldb::tid_t id);

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  GetAuxvData() const override {
    return getProcFile(GetID(), "auxv");
  }

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

  LazyBool m_supports_mem_region;
  std::vector<std::pair<MemoryRegionInfo, FileSpec>> m_mem_region_cache;

  lldb::tid_t m_pending_notification_tid;

  // List of thread ids stepping with a breakpoint with the address of
  // the relevan breakpoint
  std::map<lldb::tid_t, lldb::addr_t> m_threads_stepping_with_breakpoint;

  // ---------------------------------------------------------------------
  // Private Instance Methods
  // ---------------------------------------------------------------------
  NativeProcessLinux();

  Status LaunchInferior(MainLoop &mainloop, ProcessLaunchInfo &launch_info);

  /// Attaches to an existing process.  Forms the
  /// implementation of Process::DoAttach
  void AttachToInferior(MainLoop &mainloop, lldb::pid_t pid, Status &error);

  ::pid_t Attach(lldb::pid_t pid, Status &error);

  static Status SetDefaultPtraceOpts(const lldb::pid_t);

  static void *MonitorThread(void *baton);

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

  NativeThreadLinuxSP AddThread(lldb::tid_t thread_id);

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
};

} // namespace process_linux
} // namespace lldb_private

#endif // #ifndef liblldb_NativeProcessLinux_H_
