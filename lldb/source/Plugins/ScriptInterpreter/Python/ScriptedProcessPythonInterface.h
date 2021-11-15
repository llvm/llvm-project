//===-- ScriptedProcessPythonInterface.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_SCRIPTEDPROCESSPYTHONINTERFACE_H
#define LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_SCRIPTEDPROCESSPYTHONINTERFACE_H

#include "lldb/Host/Config.h"

#if LLDB_ENABLE_PYTHON

#include "ScriptedPythonInterface.h"
#include "lldb/Interpreter/ScriptedProcessInterface.h"

namespace lldb_private {
class ScriptedProcessPythonInterface : public ScriptedProcessInterface,
                                       public ScriptedPythonInterface {
public:
  ScriptedProcessPythonInterface(ScriptInterpreterPythonImpl &interpreter);

  StructuredData::GenericSP
  CreatePluginObject(const llvm::StringRef class_name,
                     ExecutionContext &exe_ctx,
                     StructuredData::DictionarySP args_sp) override;

  Status Launch() override;

  Status Resume() override;

  bool ShouldStop() override;

  Status Stop() override;

  llvm::Optional<MemoryRegionInfo>
  GetMemoryRegionContainingAddress(lldb::addr_t address,
                                   Status &error) override;

  StructuredData::DictionarySP GetThreadWithID(lldb::tid_t tid) override;

  StructuredData::DictionarySP GetRegistersForThread(lldb::tid_t tid) override;

  lldb::DataExtractorSP ReadMemoryAtAddress(lldb::addr_t address, size_t size,
                                            Status &error) override;

  StructuredData::DictionarySP GetLoadedImages() override;

  lldb::pid_t GetProcessID() override;

  bool IsAlive() override;

  llvm::Optional<std::string> GetScriptedThreadPluginName() override;

private:
  lldb::ScriptedThreadInterfaceSP GetScriptedThreadInterface() override;
};
} // namespace lldb_private

#endif // LLDB_ENABLE_PYTHON
#endif // LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_SCRIPTEDPROCESSPYTHONINTERFACE_H
