//===-- ProcessDpu.h ---------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessDpu_H_
#define liblldb_ProcessDpu_H_

#include <csignal>
#include <unordered_set>

#include "lldb/Host/Debug.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Host/linux/Support.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/lldb-forward.h" // for IOObjectSP
#include "lldb/lldb-types.h"

#include "ThreadDpu.h"
#include "lldb/Host/common/NativeProcessProtocol.h"

namespace lldb_private {
class Status;
class Scalar;

namespace dpu {
class Dpu;
}

namespace process_dpu {
/// @class ProcessDpu
/// Manages communication with the inferior (debugee) process.
///
/// Changes in the inferior process state are broadcasted.
class ProcessDpu : public NativeProcessProtocol {
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

  Status ReadMemory(lldb::addr_t addr, void *buf, size_t size,
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

  Status GetLoadedModuleFileSpec(const char *module_path,
                                 FileSpec &file_spec) override;

  Status GetFileLoadAddress(const llvm::StringRef &file_name,
                            lldb::addr_t &load_addr) override;

  ThreadDpu *GetThreadByID(lldb::tid_t id);

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  GetAuxvData() const override {
    return getProcFile(GetID(), "auxv");
  }

  // ---------------------------------------------------------------------
  // Interface used by RegisterContext-derived classes.
  // ---------------------------------------------------------------------
  bool SupportHardwareSingleStepping() const;

  // ---------------------------------------------------------------------
  // Other methods
  // ---------------------------------------------------------------------
  void GetThreadContext(int thread_index, uint32_t *&regs, uint16_t *&pc,
                        bool *&zf, bool *&cf);

  lldb::StateType GetThreadState(int thread_index, std::string &description,
                                 lldb::StopReason &stop_reason);

private:
  ProcessDpu(::pid_t pid, int terminal_fd, NativeDelegate &delegate,
             const ArchSpec &arch, MainLoop &mainloop, dpu::Dpu *dpu);

  void InterfaceTimerCallback();

  ArchSpec m_arch;
  lldb::IOObjectSP m_timer_fd;
  MainLoop::ReadHandleUP m_timer_handle;
  dpu::Dpu *m_dpu;
};

} // namespace process_dpu
} // namespace lldb_private

#endif // #ifndef liblldb_ProcessDpu_H_
