//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDSTACKFRAMERECOGNIZERPYTHONINTERFACE_H
#define LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDSTACKFRAMERECOGNIZERPYTHONINTERFACE_H

#include "lldb/Interpreter/Interfaces/ScriptedStackFrameRecognizerInterface.h"

#include "ScriptedPythonInterface.h"
namespace lldb_private {

class ScriptedStackFrameRecognizerPythonInterface
    : public ScriptedStackFrameRecognizerInterface,
      public ScriptedPythonInterface,
      public PluginInterface {
public:
  ScriptedStackFrameRecognizerPythonInterface(
      ScriptInterpreterPythonImpl &interpreter);

  llvm::Expected<StructuredData::GenericSP>
  CreatePluginObject(const ScriptedMetadata &scripted_metadata) override;

  llvm::SmallVector<AbstractMethodRequirement>
  GetAbstractMethodRequirements() const override {
    return {};
  }

  lldb::ValueObjectListSP
  GetRecognizedArguments(lldb::StackFrameSP frame_sp) override;

  bool ShouldHide(lldb::StackFrameSP frame_sp) override;

  lldb::StackFrameSP
  SelectMostRelevantFrame(lldb::StackFrameSP frame_sp) override;

  lldb::ValueObjectSP GetException(lldb::StackFrameSP frame_sp) override;

  std::string GetStopDescription(lldb::StackFrameSP frame_sp) override;

  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() {
    return "ScriptedStackFrameRecognizerPythonInterface";
  }

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }
};
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SCRIPTINTERPRETER_PYTHON_INTERFACES_SCRIPTEDSTACKFRAMERECOGNIZERPYTHONINTERFACE_H
