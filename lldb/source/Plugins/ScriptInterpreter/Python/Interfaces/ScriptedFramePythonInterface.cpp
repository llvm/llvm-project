//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lldb-python.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Host/Config.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/ValueObject/ValueObject.h"
#include "lldb/lldb-enumerations.h"

#include "../SWIGPythonBridge.h"
#include "../ScriptInterpreterPythonImpl.h"
#include "ScriptedFramePythonInterface.h"
#include <optional>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::python;
using Locker = ScriptInterpreterPythonImpl::Locker;

ScriptedFramePythonInterface::ScriptedFramePythonInterface(
    ScriptInterpreterPythonImpl &interpreter)
    : ScriptedFrameInterface(), ScriptedPythonInterface(interpreter) {}

llvm::Expected<StructuredData::GenericSP>
ScriptedFramePythonInterface::CreatePluginObject(
    const llvm::StringRef class_name, ExecutionContext &exe_ctx,
    StructuredData::DictionarySP args_sp, StructuredData::Generic *script_obj) {
  ExecutionContextRefSP exe_ctx_ref_sp =
      std::make_shared<ExecutionContextRef>(exe_ctx);
  ScriptedMetadata scripted_metadata(class_name, args_sp);
  return ScriptedPythonInterface::CreatePluginObject(
      scripted_metadata, script_obj, exe_ctx_ref_sp, args_sp);
}

lldb::user_id_t ScriptedFramePythonInterface::GetID() {
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("get_id"), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return LLDB_INVALID_FRAME_ID;

  return obj->GetUnsignedIntegerValue(LLDB_INVALID_FRAME_ID);
}

lldb::addr_t ScriptedFramePythonInterface::GetPC() {
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("get_pc"), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return LLDB_INVALID_ADDRESS;

  return obj->GetUnsignedIntegerValue(LLDB_INVALID_ADDRESS);
}

lldb::addr_t ScriptedFramePythonInterface::GetCFA() {
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("get_cfa"), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return LLDB_INVALID_ADDRESS;

  return obj->GetUnsignedIntegerValue(LLDB_INVALID_ADDRESS);
}

std::optional<SymbolContext> ScriptedFramePythonInterface::GetSymbolContext() {
  llvm::Expected<SymbolContext> sym_ctx_or_err =
      Dispatch<SymbolContext>("get_symbol_context");
  if (!sym_ctx_or_err) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Script), sym_ctx_or_err.takeError(),
                   "get_symbol_context failed: {0}");
    return {};
  }

  return *sym_ctx_or_err;
}

std::optional<std::string> ScriptedFramePythonInterface::GetFunctionName() {
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("get_function_name"), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return {};

  return obj->GetStringValue().str();
}

std::optional<std::string>
ScriptedFramePythonInterface::GetDisplayFunctionName() {
  StructuredData::ObjectSP obj = LogAndDefault(
      Dispatch("get_display_function_name"), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return {};

  return obj->GetStringValue().str();
}

bool ScriptedFramePythonInterface::IsInlined() {
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("is_inlined"), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return false;

  return obj->GetBooleanValue();
}

bool ScriptedFramePythonInterface::IsArtificial() {
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("is_artificial"), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return false;

  return obj->GetBooleanValue();
}

bool ScriptedFramePythonInterface::IsHidden() {
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("is_hidden"), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return false;

  return obj->GetBooleanValue();
}

StructuredData::DictionarySP ScriptedFramePythonInterface::GetRegisterInfo() {
  StructuredData::DictionarySP dict =
      LogAndDefault(Dispatch<StructuredData::DictionarySP>("get_register_info"),
                    LLVM_PRETTY_FUNCTION);
  if (!dict)
    return {};

  return dict;
}

std::optional<std::string> ScriptedFramePythonInterface::GetRegisterContext() {
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("get_register_context"), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return {};

  return obj->GetStringValue().str();
}

lldb::ValueObjectListSP ScriptedFramePythonInterface::GetVariables() {
  return LogAndDefault(Dispatch<lldb::ValueObjectListSP>("get_variables"),
                       LLVM_PRETTY_FUNCTION);
}

std::optional<lldb::ValueType>
ScriptedFramePythonInterface::GetValueTypeForVariable(
    lldb::ValueObjectSP value) {
  return LogAndDefault(Dispatch<std::optional<lldb::ValueType>>(
                           "get_value_type_for_variable", std::move(value)),
                       LLVM_PRETTY_FUNCTION);
}

lldb::ValueObjectSP
ScriptedFramePythonInterface::GetValueObjectForVariableExpression(
    llvm::StringRef expr, uint32_t options, Status &status) {
  llvm::Expected<lldb::ValueObjectSP> val_or_err =
      Dispatch<lldb::ValueObjectSP>("get_value_for_variable_expression",
                                    expr.data(), options, status);
  if (!val_or_err) {
    status = Status::FromError(val_or_err.takeError());
    return {};
  }

  return *val_or_err;
}

llvm::Expected<ScriptedMetadata>
ScriptedFramePythonInterface::GetThreadPlanMetadataForStepType(
    lldb::StepType step_type) {
  ScriptedMetadata no_plan_return("", StructuredData::DictionarySP());

  // A frame that doesn't implement `get_plan_spec_for_step_type` simply has no
  // plan to offer, which Dispatch reports as an UnimplementedError. Any other
  // failure - notably an exception raised inside the method - propagates.
  llvm::Expected<std::optional<StructuredData::DictionarySP>> dict_or_err =
      DispatchToOptional<StructuredData::DictionarySP>(
          "get_plan_spec_for_step_type", step_type);
  if (!dict_or_err)
    return llvm::joinErrors(
        llvm::createStringError(
            "error dispatching get_plan_spec_for_step_type"),
        dict_or_err.takeError());

  // The return value is an StructuredData::Dictionary with the class name and
  // the extra args for the call:
  StructuredData::DictionarySP dict_sp = dict_or_err->value_or(nullptr);
  if (!dict_sp || !dict_sp->IsValid())
    return no_plan_return;

  StructuredData::ObjectSP obj = dict_sp->GetValueForKey("class_name");
  if (!obj)
    return llvm::createStringError("Required 'class_name' field not provided.");

  std::string class_string = obj->GetStringValue().str();
  // Passing out an empty class name is they way to say the frame provider
  // doesn't know how to step from here, and the regular method should be tried
  // instead. So we only need to make sure the class exists if we were given a
  // string:
  if (!class_string.empty()) {
    const char *class_str = class_string.c_str();
    if (!m_interpreter.CheckObjectExists(class_str))
      return llvm::createStringError(
          "class_name specified a class: '%s' that does not exist.", class_str);
  }

  // Look for extra args, this is optional:
  StructuredData::Dictionary *extra_args_ptr = nullptr;
  StructuredData::DictionarySP extra_args_sp;
  if (dict_sp->GetValueForKeyAsDictionary("extra_args", extra_args_ptr))
    extra_args_sp = std::static_pointer_cast<StructuredData::Dictionary>(
        extra_args_ptr->shared_from_this());

  // Now make a new thread plan for stepping using the provided class name and
  // extra args.
  ScriptedMetadata plan_metadata(class_string, extra_args_sp);
  return plan_metadata;
}

void ScriptedFramePythonInterface::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(),
      "Provide frame state for scripted threads and frame providers.",
      CreateInstance, eScriptedExtensionScriptedFrame, eScriptLanguagePython,
      {});
}

void ScriptedFramePythonInterface::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}
