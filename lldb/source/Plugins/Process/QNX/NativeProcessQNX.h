//===-- NativeProcessQNX.h ------------------------------------ -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_NativeProcessQNX_H_
#define liblldb_NativeProcessQNX_H_

#include <sys/procfs.h>

#include "Plugins/Process/POSIX/NativeProcessELF.h"
#include "Plugins/Process/Utility/NativeProcessSoftwareSingleStep.h"

#include "lldb/lldb-forward.h"

#include "lldb/Target/MemoryRegionInfo.h"
#include "llvm/ADT/SmallPtrSet.h"

#include "NativeThreadQNX.h"

namespace lldb_private {
namespace process_qnx {
/// \class NativeProcessQNX
/// Manages communication with the inferior (debuggee) process.
///
/// Upon construction, this class prepares and launches an inferior process
/// for debugging.
///
/// Changes in the inferior process state are broadcasted.
class NativeProcessQNX : public NativeProcessELF {
public:
  class Manager : public NativeProcessProtocol::Manager {
  public:
    Manager(MainLoop &mainloop);

    llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
    Launch(ProcessLaunchInfo &launch_info,
           NativeDelegate &native_delegate) override;

    llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
    Attach(lldb::pid_t pid, NativeDelegate &native_delegate) override;

    Extension GetSupportedExtensions() const override;

    void AddProcess(NativeProcessQNX &process) { m_processes.insert(&process); }

    void RemoveProcess(NativeProcessQNX &process) {
      m_processes.erase(&process);
    }

  private:
    MainLoop::SignalHandleUP m_sigchld_handle;

    llvm::SmallPtrSet<NativeProcessQNX *, 2> m_processes;

    void SigchldHandler();
  };

  // NativeProcessProtocol interface.

  ~NativeProcessQNX() override { m_manager.RemoveProcess(*this); }

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

  size_t UpdateThreads() override;

  const ArchSpec &GetArchitecture() const override { return m_arch; }

  Status SetBreakpoint(lldb::addr_t addr, uint32_t size,
                       bool hardware) override;

  Status RemoveBreakpoint(lldb::addr_t addr, bool hardware = false) override;

  Status GetLoadedModuleFileSpec(const char *module_path,
                                 FileSpec &file_spec) override;
  Status GetFileLoadAddress(const llvm::StringRef &file_name,
                            lldb::addr_t &load_addr) override;

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  GetAuxvData() const override;

  // Interface used by NativeRegisterContext-derived classes.

  static Status DevctlWrapper(int fd, int dcmd, void *dev_data_ptr = nullptr,
                              size_t n_bytes = 0, int *dev_info_ptr = nullptr);

  int GetFileDescriptor() const { return m_fd; }

private:
  Manager &m_manager;
  ArchSpec m_arch;
  LazyBool m_supports_mem_region = eLazyBoolCalculate;
  std::vector<std::pair<MemoryRegionInfo, FileSpec>> m_mem_region_cache;
  lldb::FileUP m_file_up;
  int m_fd;

  // Private instance methods.

  NativeProcessQNX(::pid_t pid, int terminal_fd, NativeDelegate &delegate,
                   Manager &manager, const ArchSpec &arch, Status &status);

  NativeThreadQNX &AddThread(lldb::tid_t thread_id);

  void RemoveThread(lldb::tid_t thread_id);

  void MonitorInterrupt();

  void MonitorCallback(procfs_status &proc_status);

  void MonitorSIGTRAP(procfs_status &proc_status);

  void MonitorSIGSTOP();

  void MonitorSignal(procfs_status &proc_status);

  void MonitorExited(procfs_status &proc_status);

  void MonitorThread(procfs_status &proc_status);

  Status PopulateMemoryRegionCache();

  Status Attach();

  Status SetupTrace();

  Status ReinitializeThreads();

  Status ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                    size_t &bytes_read) const;
};

} // namespace process_qnx
} // namespace lldb_private

#endif // #ifndef liblldb_NativeProcessQNX_H_
