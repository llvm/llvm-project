//===-- ScriptedThreadPlanPythonInterface.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lldb-python.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/lldb-enumerations.h"

#include "../SWIGPythonBridge.h"
#include "../ScriptInterpreterPythonImpl.h"
#include "ScriptedThreadPlanPythonInterface.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::python;

ScriptedThreadPlanPythonInterface::ScriptedThreadPlanPythonInterface(
    ScriptInterpreterPythonImpl &interpreter)
    : ScriptedThreadPlanInterface(), ScriptedPythonInterface(interpreter) {}

llvm::Expected<StructuredData::GenericSP>
ScriptedThreadPlanPythonInterface::CreatePluginObject(
    const ScriptedMetadata &scripted_metadata,
    lldb::ThreadPlanSP thread_plan_sp) {
  StructuredDataImpl args_sp(scripted_metadata.GetArgsSP());
  return ScriptedPythonInterface::CreatePluginObject(scripted_metadata, nullptr,
                                                     thread_plan_sp, args_sp);
}

llvm::Expected<bool>
ScriptedThreadPlanPythonInterface::ExplainsStop(Event *event) {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("explains_stop", error, event);

  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error)) {
    if (!obj)
      return false;
    return error.ToError();
  }

  return obj->GetBooleanValue();
}

llvm::Expected<bool>
ScriptedThreadPlanPythonInterface::ShouldStop(Event *event) {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("should_stop", error, event);

  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error)) {
    if (!obj)
      return false;
    return error.ToError();
  }

  return obj->GetBooleanValue();
}

llvm::Expected<bool> ScriptedThreadPlanPythonInterface::IsStale() {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("is_stale", error);

  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error)) {
    if (!obj)
      return false;
    return error.ToError();
  }

  return obj->GetBooleanValue();
}

lldb::StateType ScriptedThreadPlanPythonInterface::GetRunState() {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("should_step", error);

  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error))
    return lldb::eStateStepping;

  // A thread plan's run state can formally be eStateSuspended, but that state
  // is decided by the thread plan negotiation, not by the plan itself.  So a
  // scripted plan's contract is only running or stepping: a bool.
  if (StructuredData::Boolean *should_step = obj->GetAsBoolean())
    return should_step->GetValue() ? lldb::eStateStepping : lldb::eStateRunning;

  if (Log *log = GetLog(LLDBLog::Script)) {
    StreamString reply;
    obj->Dump(reply, /*pretty_print=*/false);
    LLDB_LOG(log, "should_step returned {0}, not a bool; stepping.",
             reply.GetData());
  }
  return lldb::eStateStepping;
}

llvm::Error
ScriptedThreadPlanPythonInterface::GetStopDescription(lldb::StreamSP &stream) {
  Status error;
  Dispatch("stop_description", error, stream);

  if (error.Fail())
    return error.ToError();

  return llvm::Error::success();
}

void ScriptedThreadPlanPythonInterface::Initialize() {
  const std::vector<llvm::StringRef> ci_usages = {
      "thread step-scripted -C <script-name> [-k key -v value ...]"};
  const std::vector<llvm::StringRef> api_usages = {
      "SBThread.StepUsingScriptedThreadPlan"};
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(),
      llvm::StringRef("Alter thread stepping logic and stop reason"),
      CreateInstance, eScriptedExtensionScriptedThreadPlan,
      eScriptLanguagePython, {ci_usages, api_usages});
}

void ScriptedThreadPlanPythonInterface::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}
