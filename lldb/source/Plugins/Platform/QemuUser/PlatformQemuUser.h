//===-- PlatformQemuUser.h ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Platform.h"

namespace lldb_private {

class PlatformQemuUser : public Platform {
public:
  static void Initialize();
  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "qemu-user"; }
  static llvm::StringRef GetPluginDescriptionStatic();

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }
  llvm::StringRef GetDescription() override {
    return GetPluginDescriptionStatic();
  }

  UserIDResolver &GetUserIDResolver() override {
    return HostInfo::GetUserIDResolver();
  }

  std::vector<ArchSpec> GetSupportedArchitectures() override;

  lldb::ProcessSP DebugProcess(ProcessLaunchInfo &launch_info,
                               Debugger &debugger, Target &target,
                               Status &error) override;

  lldb::ProcessSP Attach(ProcessAttachInfo &attach_info, Debugger &debugger,
                         Target *target, Status &status) override {
    status.SetErrorString("Not supported");
    return nullptr;
  }

  bool IsConnected() const override { return true; }

  void CalculateTrapHandlerSymbolNames() override {}

  Environment GetEnvironment() override;

  MmapArgList GetMmapArgumentList(const ArchSpec &arch, lldb::addr_t addr,
                                  lldb::addr_t length, unsigned prot,
                                  unsigned flags, lldb::addr_t fd,
                                  lldb::addr_t offset) override {
    return Platform::GetHostPlatform()->GetMmapArgumentList(
        arch, addr, length, prot, flags, fd, offset);
  }

private:
  static lldb::PlatformSP CreateInstance(bool force, const ArchSpec *arch);
  static void DebuggerInitialize(Debugger &debugger);

  PlatformQemuUser() : Platform(/*is_host=*/false) {}
};

} // namespace lldb_private
