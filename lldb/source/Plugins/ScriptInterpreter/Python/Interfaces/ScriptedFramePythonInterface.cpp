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
  Status error;
  StructuredData::ObjectSP obj = Dispatch("get_id", error);

  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error))
    return LLDB_INVALID_FRAME_ID;

  return obj->GetUnsignedIntegerValue(LLDB_INVALID_FRAME_ID);
}

lldb::addr_t ScriptedFramePythonInterface::GetPC() {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("get_pc", error);

  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error))
    return LLDB_INVALID_ADDRESS;

  return obj->GetUnsignedIntegerValue(LLDB_INVALID_ADDRESS);
}

lldb::addr_t ScriptedFramePythonInterface::GetCFA() {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("get_cfa", error);

  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error))
    return LLDB_INVALID_ADDRESS;

  return obj->GetUnsignedIntegerValue(LLDB_INVALID_ADDRESS);
}

std::optional<SymbolContext> ScriptedFramePythonInterface::GetSymbolContext() {
  Status error;
  auto sym_ctx = Dispatch<SymbolContext>("get_symbol_context", error);

  if (error.Fail()) {
    return ErrorWithMessage<SymbolContext>(LLVM_PRETTY_FUNCTION,
                                           error.AsCString(), error);
  }

  return sym_ctx;
}

std::optional<std::string> ScriptedFramePythonInterface::GetFunctionName() {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("get_function_name", error);

  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error))
    return {};

  return obj->GetStringValue().str();
}

std::optional<std::string>
ScriptedFramePythonInterface::GetDisplayFunctionName() {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("get_display_function_name", error);

  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error))
    return {};

  return obj->GetStringValue().str();
}

bool ScriptedFramePythonInterface::IsInlined() {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("is_inlined", error);

  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error))
    return false;

  return obj->GetBooleanValue();
}

bool ScriptedFramePythonInterface::IsArtificial() {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("is_artificial", error);

  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error))
    return false;

  return obj->GetBooleanValue();
}

bool ScriptedFramePythonInterface::IsHidden() {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("is_hidden", error);

  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error))
    return false;

  return obj->GetBooleanValue();
}

StructuredData::DictionarySP ScriptedFramePythonInterface::GetRegisterInfo() {
  Status error;
  StructuredData::DictionarySP dict =
      Dispatch<StructuredData::DictionarySP>("get_register_info", error);

  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, dict,
                                                    error))
    return {};

  return dict;
}

std::optional<std::string> ScriptedFramePythonInterface::GetRegisterContext() {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("get_register_context", error);

  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error))
    return {};

  return obj->GetAsString()->GetValue().str();
}

lldb::ValueObjectListSP ScriptedFramePythonInterface::GetVariables() {
  Status error;
  auto vals = Dispatch<lldb::ValueObjectListSP>("get_variables", error);

  if (error.Fail()) {
    return ErrorWithMessage<lldb::ValueObjectListSP>(LLVM_PRETTY_FUNCTION,
                                                     error.AsCString(), error);
  }

  return vals;
}

std::optional<lldb::ValueType>
ScriptedFramePythonInterface::GetValueTypeForVariable(
    lldb::ValueObjectSP value) {
  Status error;
  auto val = Dispatch<std::optional<lldb::ValueType>>(
      "get_value_type_for_variable", error, std::move(value));

  if (error.Fail()) {
    return ErrorWithMessage<std::optional<lldb::ValueType>>(
        LLVM_PRETTY_FUNCTION, error.AsCString(), error);
  }

  return val;
}

lldb::ValueObjectSP
ScriptedFramePythonInterface::GetValueObjectForVariableExpression(
    llvm::StringRef expr, uint32_t options, Status &status) {
  Status dispatch_error;
  auto val = Dispatch<lldb::ValueObjectSP>("get_value_for_variable_expression",
                                           dispatch_error, expr.data(), options,
                                           status);

  if (dispatch_error.Fail()) {
    return ErrorWithMessage<lldb::ValueObjectSP>(
        LLVM_PRETTY_FUNCTION, dispatch_error.AsCString(), dispatch_error);
  }

  return val;
}

llvm::Expected<ScriptedMetadata>
ScriptedFramePythonInterface::GetThreadPlanMetadataForStepType(
    lldb::StepType step_type) {
  Status error;

  ScriptedMetadata no_plan_return("", StructuredData::DictionarySP());
  StructuredData::DictionarySP dict_sp = Dispatch<StructuredData::DictionarySP>(
      "get_plan_spec_for_step_type", error, step_type);
  if (error.Fail()) {
    // There are two cases here.  The `get_plan_spec_for_step_type` didn't
    // exist, in which case we should return no_plan_return
    // FIXME - Dispatch should distinguish between these two cases in a way
    // that's more definitive than this.
    llvm::StringRef err_str(error.AsCString());
    if (err_str.contains(
            "object has no attribute 'get_plan_spec_for_step_type'"))
      return no_plan_return;
    else
      return llvm::createStringError(
          "error dispatching get_plan_spec_for_step_type: %s",
          error.AsCString());
  }

  // The return value is an StructuredData::Dictionary with the class name and
  // the extra args for the call:
  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION,
                                                    dict_sp, error))
    return llvm::createStringError(
        "return from get_plan_spec_for_step_type not a valid object: %s",
        error.AsCString());

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
