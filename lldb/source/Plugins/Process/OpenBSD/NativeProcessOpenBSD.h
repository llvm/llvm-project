//===-- NativeProcessOpenBSD.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_OPENBSD_NATIVEPROCESSOPENBSD_H
#define LLDB_SOURCE_PLUGINS_PROCESS_OPENBSD_NATIVEPROCESSOPENBSD_H

#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/FileSpec.h"

#include "NativeThreadOpenBSD.h"
#include "Plugins/Process/POSIX/NativeProcessELF.h"

namespace lldb_private {
namespace process_openbsd {
/// @class NativeProcessOpenBSD
/// Manages communication with the inferior (debugee) process.
///
/// Upon construction, this class prepares and launches an inferior process
/// for debugging.
///
/// Changes in the inferior process state are broadcasted.
class NativeProcessOpenBSD : public NativeProcessELF {
public:
  class Manager : public NativeProcessProtocol::Manager {
  public:
    using NativeProcessProtocol::Manager::Manager;

    llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
    Launch(ProcessLaunchInfo &launch_info,
           NativeDelegate &native_delegate) override;

    llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
    Attach(lldb::pid_t pid, NativeDelegate &native_delegate) override;

    Extension GetSupportedExtensions() const override;
  };

  // NativeProcessProtocol Interface
  Status Resume(const ResumeActionList &resume_actions) override;

  Status Halt() override;

  Status Detach() override;

  Status Signal(int signo) override;

  Status Kill() override;

  Status GetMemoryRegionInfo(lldb::addr_t load_addr,
                             MemoryRegionInfo &range_info) override;

  Status ReadMemory(const ProcessAddress &addr, void *buf, size_t size,
                    size_t &bytes_read) override;

  Status DoWriteMemory(lldb::addr_t addr, const void *buf, size_t size,
                       size_t &bytes_written) override;

  llvm::Expected<lldb::addr_t> AllocateMemory(size_t size,
                                              uint32_t permissions) override;

  llvm::Error DeallocateMemory(lldb::addr_t addr) override;

  size_t UpdateThreads() override;

  const ArchSpec &GetArchitecture() const override { return m_arch; }

  Status SetBreakpoint(lldb::addr_t addr, uint32_t size,
                       bool hardware) override;

  Status GetLoadedModuleFileSpec(const char *module_path,
                                 FileSpec &file_spec) override;

  Status GetFileLoadAddress(const llvm::StringRef &file_name,
                            lldb::addr_t &load_addr) override;

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  GetAuxvData() const override;

  void SetEnabledExtensions(Extension flags) override;

  // Interface used by NativeRegisterContext-derived classes.
  static Status PtraceWrapper(int req, lldb::pid_t pid, void *addr = nullptr,
                              int data = 0, int *result = nullptr);

private:
  MainLoop::SignalHandleUP m_sigchld_handle;
  ArchSpec m_arch;
  MainLoop &m_main_loop;
  LazyBool m_supports_mem_region = eLazyBoolCalculate;
  // Thread that was single-stepped by the last Resume(), if any.
  lldb::tid_t m_step_tid = LLDB_INVALID_THREAD_ID;
  // Thread that reported the last stop.
  lldb::tid_t m_trapped_tid = LLDB_INVALID_THREAD_ID;
  // The other process of the last fork seen. The kernel keeps reporting the
  // last fork in the ptrace state until the next one.
  ::pid_t m_last_fork_pid = 0;

  // Private Instance Methods
  NativeProcessOpenBSD(::pid_t pid, int terminal_fd, NativeDelegate &delegate,
                       const ArchSpec &arch, MainLoop &mainloop);

  bool HasThreadNoLock(lldb::tid_t thread_id);

  NativeThreadOpenBSD &AddThread(lldb::tid_t thread_id);

  void MonitorCallback(lldb::pid_t pid, int signal);
  void MonitorExited(lldb::pid_t pid, WaitStatus status);
  void MonitorSIGTRAP(lldb::pid_t pid);
  void MonitorSignal(lldb::pid_t pid, int signal);
  void MonitorFork(::pid_t child_pid, NativeThreadOpenBSD &parent_thread);

  Status PopulateMemoryRegionCache();
  void SigchldHandler();

  Status Attach();
  Status RefreshThreads();
  lldb::tid_t GetTrappedThreadID();
  ::pid_t GetForkedChildPID();
  Status SetForkReporting(bool enable);
};

} // namespace process_openbsd
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PROCESS_OPENBSD_NATIVEPROCESSOPENBSD_H
