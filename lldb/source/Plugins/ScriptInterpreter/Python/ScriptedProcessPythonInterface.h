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

#include "lldb/Interpreter/ScriptedProcessInterface.h"

namespace lldb_private {
class ScriptInterpreterPythonImpl;
class ScriptedProcessPythonInterface : public ScriptedProcessInterface {
public:
  ScriptedProcessPythonInterface(ScriptInterpreterPythonImpl &interpreter)
      : ScriptedProcessInterface(), m_interpreter(interpreter) {}

  StructuredData::GenericSP
  CreatePluginObject(const llvm::StringRef class_name, lldb::TargetSP target_sp,
                     StructuredData::DictionarySP args_sp) override;

  Status Launch() override;

  Status Resume() override;

  bool ShouldStop() override;

  Status Stop() override;

  lldb::MemoryRegionInfoSP
  GetMemoryRegionContainingAddress(lldb::addr_t address) override;

  StructuredData::DictionarySP GetThreadWithID(lldb::tid_t tid) override;

  StructuredData::DictionarySP GetRegistersForThread(lldb::tid_t tid) override;

  lldb::DataExtractorSP ReadMemoryAtAddress(lldb::addr_t address, size_t size,
                                            Status &error) override;

  StructuredData::DictionarySP GetLoadedImages() override;

  lldb::pid_t GetProcessID() override;

  bool IsAlive() override;

protected:
  llvm::Optional<unsigned long long>
  GetGenericInteger(llvm::StringRef method_name);
  Status GetStatusFromMethod(llvm::StringRef method_name);

private:
  // The lifetime is managed by the ScriptInterpreter
  ScriptInterpreterPythonImpl &m_interpreter;
  StructuredData::GenericSP m_object_instance_sp;
};
} // namespace lldb_private

#endif // LLDB_ENABLE_PYTHON
#endif // LLDB_PLUGINS_SCRIPTINTERPRETER_PYTHON_SCRIPTEDPROCESSPYTHONINTERFACE_H
