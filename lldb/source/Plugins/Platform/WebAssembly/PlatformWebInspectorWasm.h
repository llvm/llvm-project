//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PLATFORM_WEBASSEMBLY_PLATFORMWEBINSPECTORWASM_H
#define LLDB_SOURCE_PLUGINS_PLATFORM_WEBASSEMBLY_PLATFORMWEBINSPECTORWASM_H

#include "PlatformWasm.h"

namespace lldb_private {

class PlatformWebInspectorWasm : public PlatformWasm {
public:
  static void Initialize();
  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "webinspector-wasm"; }
  static llvm::StringRef GetPluginDescriptionStatic();

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }
  llvm::StringRef GetDescription() override {
    return GetPluginDescriptionStatic();
  }

  ~PlatformWebInspectorWasm() override;

  Status ConnectRemote(Args &args) override;

  lldb::ProcessSP Attach(ProcessAttachInfo &attach_info, Debugger &debugger,
                         Target *target, Status &status) override;

  lldb::ProcessSP DebugProcess(ProcessLaunchInfo &launch_info,
                               Debugger &debugger, Target &target,
                               Status &error) override;

  uint32_t FindProcesses(const ProcessInstanceInfoMatch &match_info,
                         ProcessInstanceInfoList &proc_infos) override;

  bool GetProcessInfo(lldb::pid_t pid, ProcessInstanceInfo &proc_info) override;

private:
  static lldb::PlatformSP CreateInstance(bool force, const ArchSpec *arch);

  PlatformWebInspectorWasm() = default;

  llvm::Error LaunchPlatformServer();
  llvm::Error EnsureConnected();

  lldb::pid_t m_server_pid = LLDB_INVALID_PROCESS_ID;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PLATFORM_WEBASSEMBLY_PLATFORMWEBINSPECTORWASM_H
