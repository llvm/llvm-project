//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_SERVER_PLUGINS_ACCELERATOR_MOCK_PROCESSMOCKACCELERATOR_H
#define LLDB_TOOLS_LLDB_SERVER_PLUGINS_ACCELERATOR_MOCK_PROCESSMOCKACCELERATOR_H

#include "lldb/Host/common/NativeProcessProtocol.h"
#include "lldb/Utility/ArchSpec.h"

#include <string>
#include <utility>

namespace lldb_private {
namespace lldb_server {

/// A minimal, always-stopped fake process that serves a GDB remote connection
/// for the mock accelerator plugin, letting the client connect a second
/// (accelerator) target without any real hardware.
class ProcessMockAccelerator : public NativeProcessProtocol {
public:
  class Manager : public NativeProcessProtocol::Manager {
  public:
    Manager(MainLoop &mainloop, ArchSpec arch,
            std::string dynamic_loader_library_path)
        : NativeProcessProtocol::Manager(mainloop), m_arch(std::move(arch)),
          m_dynamic_loader_library_path(
              std::move(dynamic_loader_library_path)) {}

    llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
    Launch(ProcessLaunchInfo &launch_info,
           NativeDelegate &native_delegate) override;

    llvm::Expected<std::unique_ptr<NativeProcessProtocol>>
    Attach(lldb::pid_t pid, NativeDelegate &native_delegate) override;

    Extension GetSupportedExtensions() const override {
      return Extension::address_spaces;
    }

  private:
    ArchSpec m_arch;
    std::string m_dynamic_loader_library_path;
  };

  ProcessMockAccelerator(lldb::pid_t pid, ArchSpec arch,
                         std::string dynamic_loader_library_path,
                         NativeDelegate &delegate);

  Status Resume(const ResumeActionList &resume_actions) override;
  Status Halt() override;
  Status Detach() override;
  Status Signal(int signo) override;
  Status Kill() override;

  Status ReadMemory(const ProcessAddress &addr, void *buf, size_t size,
                    size_t &bytes_read) override;
  Status DoWriteMemory(lldb::addr_t addr, const void *buf, size_t size,
                       size_t &bytes_written) override;

  lldb::addr_t GetSharedLibraryInfoAddress() override;
  size_t UpdateThreads() override;
  const ArchSpec &GetArchitecture() const override;

  Status SetBreakpoint(lldb::addr_t addr, uint32_t size,
                       bool hardware) override;

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  GetAuxvData() const override;

  Status GetLoadedModuleFileSpec(const char *module_path,
                                 FileSpec &file_spec) override;
  Status GetFileLoadAddress(const llvm::StringRef &file_name,
                            lldb::addr_t &load_addr) override;

  std::vector<AddressSpaceInfo> GetAddressSpaces() override;

  std::optional<AcceleratorDynamicLoaderResponse>
  GetAcceleratorDynamicLoaderLibraryInfos(
      const AcceleratorDynamicLoaderArgs &args) override;

private:
  ArchSpec m_arch;
  std::string m_dynamic_loader_library_path;
};

} // namespace lldb_server
} // namespace lldb_private

#endif // LLDB_TOOLS_LLDB_SERVER_PLUGINS_ACCELERATOR_MOCK_PROCESSMOCKACCELERATOR_H
