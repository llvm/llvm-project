//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lldb-python.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Target/ScriptedThreadPlan.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Utility/StructuredData.h"
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

lldb::ThreadPlanSP
ScriptedStackFrameRecognizerPythonInterface::GetStepThroughPlan(
    lldb::ThreadSP thread_sp) {
  Status error;
  StructuredData::DictionarySP dict_sp = Dispatch<StructuredData::DictionarySP>(
      "get_step_through_plan", error, thread_sp);
  if (error.Fail())
    return {};

  // The return value is an StructuredData::Dictionary with the class name and
  // the extra args for the call:
  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION,
                                                    dict_sp, error))
    return {};

  StructuredData::ObjectSP obj = dict_sp->GetValueForKey("class_name");
  if (!obj)
    return {};

  llvm::StringRef class_string = obj->GetStringValue();
  if (class_string.empty())
    return {};

  // Look for extra args, this is optional:
  StructuredData::Dictionary *extra_args_ptr = nullptr;
  StructuredData::DictionarySP extra_args_sp;
  if (dict_sp->GetValueForKeyAsDictionary("extra_args", extra_args_ptr))
    extra_args_sp = std::static_pointer_cast<StructuredData::Dictionary>(
        extra_args_ptr->shared_from_this());

  // Now make a new thread plan for stepping using the provided class name and
  // extra args.
  ScriptedMetadata plan_metadata(class_string, extra_args_sp);
  ThreadPlanSP step_through_plan_sp(
      new ScriptedThreadPlan(*thread_sp.get(), plan_metadata));
  step_through_plan_sp->SetStopOthers(true);

  return step_through_plan_sp;
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
