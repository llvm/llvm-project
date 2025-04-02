//===-- ProcessMockGPU.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PROCESSMOCKGPU_H
#define LLDB_TOOLS_LLDB_SERVER_PROCESSMOCKGPU_H

#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Utility/ProcessInfo.h"

namespace lldb_private {
namespace lldb_server {

/// \class ProcessMockGPU
/// Abstract class that extends \a NativeProcessProtocol for a mock GPU. This
/// class is used to unit testing the GPU plugins in lldb-server.
class ProcessMockGPU : public NativeProcessProtocol {
  // TODO: change NativeProcessProtocol::GetArchitecture() to return by value
  mutable ArchSpec m_arch;
  ProcessInstanceInfo m_process_info;

public:
  ProcessMockGPU(lldb::pid_t pid, NativeDelegate &delegate);

  Status Resume(const ResumeActionList &resume_actions) override;

  Status Halt() override;

  Status Detach() override;

  /// Sends a process a UNIX signal \a signal.
  ///
  /// \return
  ///     Returns an error object.
  Status Signal(int signo) override;

  /// Tells a process to interrupt all operations as if by a Ctrl-C.
  ///
  /// The default implementation will send a local host's equivalent of
  /// a SIGSTOP to the process via the NativeProcessProtocol::Signal()
  /// operation.
  ///
  /// \return
  ///     Returns an error object.
  Status Interrupt() override;

  Status Kill() override;

  Status ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                    size_t &bytes_read) override;

  Status WriteMemory(lldb::addr_t addr, const void *buf, size_t size,
                     size_t &bytes_written) override;

  lldb::addr_t GetSharedLibraryInfoAddress() override;

  size_t UpdateThreads() override;

  const ArchSpec &GetArchitecture() const override;

  // Breakpoint functions
  Status SetBreakpoint(lldb::addr_t addr, uint32_t size,
                       bool hardware) override;

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  GetAuxvData() const override;

  Status GetLoadedModuleFileSpec(const char *module_path,
                                 FileSpec &file_spec) override;

  Status GetFileLoadAddress(const llvm::StringRef &file_name,
                            lldb::addr_t &load_addr) override;

  bool GetProcessInfo(ProcessInstanceInfo &info) override;

  // Custom accessors
  void SetLaunchInfo(ProcessLaunchInfo &launch_info);
};

class ProcessManagerMockGPU : public NativeProcessProtocol::Manager {
public:
  ProcessManagerMockGPU(MainLoop &mainloop)
      : NativeProcessProtocol::Manager(mainloop) {}

  llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
  Launch(ProcessLaunchInfo &launch_info,
         NativeProcessProtocol::NativeDelegate &native_delegate) override;

  llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
  Attach(lldb::pid_t pid,
         NativeProcessProtocol::NativeDelegate &native_delegate) override;
};

} // namespace lldb_server
} // namespace lldb_private

#endif
