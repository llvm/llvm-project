//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lldb-python.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/lldb-enumerations.h"

#include "../SWIGPythonBridge.h"
#include "../ScriptInterpreterPythonImpl.h"
#include "ScriptedStackFrameRecognizerPythonInterface.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::python;

ScriptedStackFrameRecognizerPythonInterface::
    ScriptedStackFrameRecognizerPythonInterface(
        ScriptInterpreterPythonImpl &interpreter)
    : ScriptedStackFrameRecognizerInterface(),
      ScriptedPythonInterface(interpreter) {}

llvm::Expected<StructuredData::GenericSP>
ScriptedStackFrameRecognizerPythonInterface::CreatePluginObject(
    const ScriptedMetadata &scripted_metadata) {
  return ScriptedPythonInterface::CreatePluginObject(scripted_metadata,
                                                     nullptr);
}

lldb::ValueObjectListSP
ScriptedStackFrameRecognizerPythonInterface::GetRecognizedArguments(
    lldb::StackFrameSP frame_sp) {
  Status error;
  return Dispatch<lldb::ValueObjectListSP>("get_recognized_arguments", error,
                                           frame_sp);
}

bool ScriptedStackFrameRecognizerPythonInterface::ShouldHide(
    lldb::StackFrameSP frame_sp) {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("should_hide", error, frame_sp);

  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error))
    return false;

  return obj->GetBooleanValue();
}

lldb::StackFrameSP
ScriptedStackFrameRecognizerPythonInterface::SelectMostRelevantFrame(
    lldb::StackFrameSP frame_sp) {
  Status error;
  return Dispatch<lldb::StackFrameSP>("select_most_relevant_frame", error,
                                      frame_sp);
}

lldb::ValueObjectSP ScriptedStackFrameRecognizerPythonInterface::GetException(
    lldb::StackFrameSP frame_sp) {
  Status error;
  return Dispatch<lldb::ValueObjectSP>("get_exception", error, frame_sp);
}

std::string ScriptedStackFrameRecognizerPythonInterface::GetStopDescription(
    lldb::StackFrameSP frame_sp) {
  Status error;
  StructuredData::ObjectSP obj =
      Dispatch("get_stop_description", error, frame_sp);
  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error))
    return "";
  return obj->GetStringValue().str();
}

void ScriptedStackFrameRecognizerPythonInterface::Initialize() {
  const std::vector<llvm::StringRef> ci_usages = {
      "frame recognizer add -l <script-name> [-s <shlib> ...] "
      "[-n <symbol> ... | -x <symbol-regex>] [-f false] "
      "[--mangled-name-preference <mode>]"};
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(),
      "Recognize a stack frame and provide extra information about it "
      "(recognized arguments, exception object, stop description, "
      "hidden/most-relevant frame).",
      CreateInstance, eScriptedExtensionScriptedStackFrameRecognizer,
      eScriptLanguagePython, {ci_usages, {}});
}

void ScriptedStackFrameRecognizerPythonInterface::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}
