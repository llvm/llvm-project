//===-- NativeProcessAIX.h ---------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBLLDB_NATIVEPROCESSAIX_H_
#define LIBLLDB_NATIVEPROCESSAIX_H_

#include "Plugins/Process/Utility/NativeProcessSoftwareSingleStep.h"
#include "lldb/Host/Debug.h"
#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/lldb-types.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <csignal>
#include <unordered_set>

namespace lldb_private {

namespace process_aix {
/// \class NativeProcessAIX
/// Manages communication with the inferior (debugee) process.
///
/// Upon construction, this class prepares and launches an inferior process
/// for debugging.
///
/// Changes in the inferior process state are broadcasted.
class NativeProcessAIX : public NativeProcessProtocol {
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

    void AddProcess(NativeProcessAIX &process) { m_processes.insert(&process); }

    void RemoveProcess(NativeProcessAIX &process) {
      m_processes.erase(&process);
    }

    // Collect an event for the given tid, waiting for it if necessary.
    void CollectThread(::pid_t tid);

  private:
    MainLoop::SignalHandleUP m_sigchld_handle;

    llvm::SmallPtrSet<NativeProcessAIX *, 2> m_processes;

    void SigchldHandler();
  };

  // NativeProcessProtocol Interface

  ~NativeProcessAIX() override { m_manager.RemoveProcess(*this); }

  Status Resume(const ResumeActionList &resume_actions) override;

  Status Halt() override;

  Status Detach() override;

  Status Signal(int signo) override;

  Status Interrupt() override;

  Status Kill() override;

  const ArchSpec &GetArchitecture() const override { return m_arch; }

  Status SetBreakpoint(lldb::addr_t addr, uint32_t size,
                       bool hardware) override;

  Status RemoveBreakpoint(lldb::addr_t addr, bool hardware = false) override;

  Status GetFileLoadAddress(const llvm::StringRef &file_name,
                            lldb::addr_t &load_addr) override;

  static Status PtraceWrapper(int req, lldb::pid_t pid, void *addr = nullptr,
                              void *data = nullptr, size_t data_size = 0,
                              long *result = nullptr);

private:
  Manager &m_manager;
  ArchSpec m_arch;

  // Private Instance Methods
  NativeProcessAIX(::pid_t pid, int terminal_fd, NativeDelegate &delegate,
                   const ArchSpec &arch, Manager &manager,
                   llvm::ArrayRef<::pid_t> tids);

  // Returns a list of process threads that we have attached to.
  static llvm::Expected<std::vector<::pid_t>> Attach(::pid_t pid);

  void MonitorSIGTRAP(const WaitStatus status, NativeThreadAIX &thread);

  void MonitorBreakpoint(NativeThreadAIX &thread);

  Status Detach(lldb::tid_t tid);

  void SigchldHandler();
};

} // namespace process_aix
} // namespace lldb_private

#endif // #ifndef LIBLLDB_NATIVEPROCESSAIX_H_
