//===-- ScriptedProcessPythonInterface.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDPROCESSPYTHONINTERFACE_H
#define LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDPROCESSPYTHONINTERFACE_H

#include "lldb/Host/Config.h"
#include "lldb/Interpreter/Interfaces/ScriptedProcessInterface.h"

#if LLDB_ENABLE_PYTHON

#include "ScriptedPythonInterface.h"

#include <optional>

namespace lldb_private {
class ScriptedProcessPythonInterface : public ScriptedProcessInterface,
                                       public ScriptedPythonInterface,
                                       public PluginInterface {
public:
  ScriptedProcessPythonInterface(ScriptInterpreterPythonImpl &interpreter);

  llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(const llvm::StringRef class_name,
                     ExecutionContext &exe_ctx,
                     StructuredData::DictionarySP args_sp,
                     StructuredData::Generic *script_obj = nullptr) override;

  llvm::SmallVector<AbstractMethodRequirement>
  GetAbstractMethodRequirements() const override {
    return llvm::SmallVector<AbstractMethodRequirement>(
        {{"read_memory_at_address", 4},
         {"is_alive"},
         {"get_scripted_thread_plugin"}});
  }

  StructuredData::DictionarySP GetCapabilities() override;

  Status Attach(const ProcessAttachInfo &attach_info) override;

  Status Launch() override;

  Status Resume() override;

  std::optional<MemoryRegionInfo>
  GetMemoryRegionContainingAddress(lldb::addr_t address,
                                   Status &error) override;

  StructuredData::DictionarySP GetThreadsInfo() override;

  bool CreateBreakpoint(lldb::addr_t addr, Status &error) override;

  lldb::DataExtractorSP ReadMemoryAtAddress(lldb::addr_t address, size_t size,
                                            Status &error) override;

  lldb::offset_t WriteMemoryAtAddress(lldb::addr_t addr,
                                      lldb::DataExtractorSP data_sp,
                                      Status &error) override;

  StructuredData::ArraySP GetLoadedImages() override;

  lldb::pid_t GetProcessID() override;

  bool IsAlive() override;

  std::optional<std::string> GetScriptedThreadPluginName() override;

  StructuredData::DictionarySP GetMetadata() override;

  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() {
    return "ScriptedProcessPythonInterface";
  }

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

private:
  lldb::ScriptedThreadInterfaceSP CreateScriptedThreadInterface() override;
};
} // namespace lldb_private

#endif // LLDB_ENABLE_PYTHON
#endif // LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDPROCESSPYTHONINTERFACE_H
