//===-- ScriptedPlatform.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SCRIPTED_PLATFORM_H
#define LLDB_SOURCE_PLUGINS_SCRIPTED_PLATFORM_H

#include "lldb/Target/Platform.h"
#include "lldb/Utility/ScriptedMetadata.h"

namespace lldb_private {

class ScriptedPlatform : public Platform {
public:
  ScriptedPlatform();

  ~ScriptedPlatform() override;

  llvm::Error SetupScriptedObject();

  static lldb::PlatformSP CreateInstance(bool force, const ArchSpec *arch);

  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "scripted-platform"; }

  static llvm::StringRef GetDescriptionStatic() {
    return "Scripted Platform plug-in.";
  }

  llvm::StringRef GetDescription() override { return GetDescriptionStatic(); }

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  std::vector<ArchSpec>
  GetSupportedArchitectures(const ArchSpec &process_host_arch) override;

  bool IsConnected() const override { return true; }

  lldb::ProcessSP Attach(lldb_private::ProcessAttachInfo &attach_info,
                         lldb_private::Debugger &debugger,
                         lldb_private::Target *target,
                         lldb_private::Status &error) override;

  uint32_t FindProcesses(const ProcessInstanceInfoMatch &match_info,
                         ProcessInstanceInfoList &proc_infos) override;

  bool GetProcessInfo(lldb::pid_t pid, ProcessInstanceInfo &proc_info) override;

  Status LaunchProcess(ProcessLaunchInfo &launch_info) override;

  Status KillProcess(const lldb::pid_t pid) override;

  void CalculateTrapHandlerSymbolNames() override {}

  llvm::Error ReloadMetadata() override;

private:
  void CheckInterpreterAndScriptObject() const {
    assert(m_interface_up && "Invalid Scripted Platform Interface.");
  }

  ScriptedPlatform(const ScriptedPlatform &) = delete;
  const ScriptedPlatform &operator=(const ScriptedPlatform &) = delete;

  ScriptedPlatformInterface &GetInterface() const;

  llvm::Expected<ProcessInstanceInfo>
  ParseProcessInfo(StructuredData::Dictionary &dict, lldb::pid_t pid) const;

  static bool IsScriptLanguageSupported(lldb::ScriptLanguage language);

  lldb::ScriptedPlatformInterfaceUP m_interface_up;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SCRIPTED_PLATFORM_H
