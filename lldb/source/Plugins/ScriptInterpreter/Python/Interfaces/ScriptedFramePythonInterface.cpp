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
