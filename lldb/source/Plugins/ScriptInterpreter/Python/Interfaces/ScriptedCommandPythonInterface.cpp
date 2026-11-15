//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lldb-python.h"

#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Utility/Args.h"
#include "lldb/Utility/ScriptedMetadata.h"
#include "lldb/lldb-enumerations.h"

#include "../SWIGPythonBridge.h"
#include "../ScriptInterpreterPythonImpl.h"
#include "ScriptedCommandPythonInterface.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::python;
using Locker = ScriptInterpreterPythonImpl::Locker;

ScriptedCommandPythonInterface::ScriptedCommandPythonInterface(
    ScriptInterpreterPythonImpl &interpreter)
    : ScriptedCommandInterface(), ScriptedPythonInterface(interpreter) {}

llvm::Expected<StructuredData::GenericSP>
ScriptedCommandPythonInterface::CreatePluginObject(
    llvm::StringRef class_name, lldb::DebuggerSP debugger_sp) {
  if (class_name.empty())
    return llvm::createStringError("empty class name");

  if (!debugger_sp)
    return llvm::createStringError("invalid Debugger pointer");

  m_debugger_sp = debugger_sp;
  ScriptedMetadata scripted_metadata(class_name,
                                     StructuredData::DictionarySP());
  return ScriptedPythonInterface::CreatePluginObject(
      scripted_metadata, /*script_obj=*/nullptr, debugger_sp);
}

bool ScriptedCommandPythonInterface::RunRawCommand(
    llvm::StringRef args, ScriptedCommandSynchronicity synchronicity,
    CommandReturnObject &cmd_retobj, Status &error,
    const ExecutionContext &exe_ctx) {
  lldb::DebuggerSP debugger_sp = m_debugger_sp;
  if (!debugger_sp) {
    error = Status::FromErrorString("invalid Debugger pointer");
    return false;
  }
  lldb::ExecutionContextRefSP exe_ctx_ref_sp(new ExecutionContextRef(exe_ctx));

  // Outer Locker sets up the session (with conditional NoSTDIN for
  // interactive commands and TearDownSession on exit) that Dispatch's inner
  // Locker doesn't provide. Nesting is only safe because Dispatch's own
  // Locker never requests InitSession itself.
  Locker py_lock(&m_interpreter,
                 Locker::AcquireLock | Locker::InitSession |
                     (cmd_retobj.GetInteractive() ? 0 : Locker::NoSTDIN),
                 Locker::FreeLock | Locker::TearDownSession);
  ScriptInterpreterPythonImpl::SynchronicityHandler synch_handler(
      debugger_sp, synchronicity);

  std::string args_str = args.str();
  llvm::Expected<StructuredData::ObjectSP> obj_or_err = Dispatch(
      "__call__", debugger_sp, args_str.c_str(), exe_ctx_ref_sp, &cmd_retobj);
  if (!obj_or_err) {
    error = Status::FromError(obj_or_err.takeError());
    return false;
  }

  return cmd_retobj.GetStatus() != eReturnStatusFailed;
}

bool ScriptedCommandPythonInterface::RunParsedCommand(
    Args &args, ScriptedCommandSynchronicity synchronicity,
    CommandReturnObject &cmd_retobj, Status &error,
    const ExecutionContext &exe_ctx) {
  lldb::DebuggerSP debugger_sp = m_debugger_sp;
  if (!debugger_sp) {
    error = Status::FromErrorString("invalid Debugger pointer");
    return false;
  }
  lldb::ExecutionContextRefSP exe_ctx_ref_sp(new ExecutionContextRef(exe_ctx));

  Locker py_lock(&m_interpreter,
                 Locker::AcquireLock | Locker::InitSession |
                     (cmd_retobj.GetInteractive() ? 0 : Locker::NoSTDIN),
                 Locker::FreeLock | Locker::TearDownSession);
  ScriptInterpreterPythonImpl::SynchronicityHandler synch_handler(
      debugger_sp, synchronicity);

  StructuredData::ArraySP args_arr_sp(new StructuredData::Array());
  for (const Args::ArgEntry &entry : args)
    args_arr_sp->AddStringItem(entry.ref());
  StructuredDataImpl args_impl(args_arr_sp);

  llvm::Expected<StructuredData::ObjectSP> obj_or_err =
      Dispatch("__call__", debugger_sp, args_impl, exe_ctx_ref_sp, &cmd_retobj);
  if (!obj_or_err) {
    error = Status::FromError(obj_or_err.takeError());
    return false;
  }

  return cmd_retobj.GetStatus() != eReturnStatusFailed;
}

std::optional<std::string>
ScriptedCommandPythonInterface::GetRepeatCommand(Args &args) {
  std::string command;
  args.GetQuotedCommandString(command);
  StructuredData::ObjectSP obj = LogAndDefault(
      Dispatch("get_repeat_command", command.c_str()), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return {};

  return obj->GetStringValue().str();
}

StructuredData::DictionarySP
ScriptedCommandPythonInterface::HandleArgumentCompletion(
    std::vector<std::string> &args, size_t args_pos, size_t char_in_arg) {
  StructuredData::ObjectSP obj = LogAndDefault(
      Dispatch("handle_argument_completion", args, args_pos, char_in_arg),
      LLVM_PRETTY_FUNCTION);
  if (!obj)
    return {};
  StructuredData::DictionarySP dict_sp(new StructuredData::Dictionary(obj));
  if (dict_sp->GetType() == lldb::eStructuredDataTypeInvalid)
    return {};
  return dict_sp;
}

StructuredData::DictionarySP
ScriptedCommandPythonInterface::HandleOptionArgumentCompletion(
    llvm::StringRef &long_option, size_t char_in_arg) {
  std::string long_option_str = long_option.str();
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("handle_option_argument_completion",
                             long_option_str.c_str(), char_in_arg),
                    LLVM_PRETTY_FUNCTION);
  if (!obj)
    return {};

  // A boolean return means: True means completion handled but no
  // completions; False means completion not handled (fall back to default).
  if (obj->GetType() == lldb::eStructuredDataTypeBoolean) {
    if (!obj->GetBooleanValue())
      return {};
    StructuredData::DictionarySP dict_sp(new StructuredData::Dictionary());
    dict_sp->AddBooleanItem("no-completion", true);
    return dict_sp;
  }

  StructuredData::DictionarySP dict_sp(new StructuredData::Dictionary(obj));
  if (dict_sp->GetType() == lldb::eStructuredDataTypeInvalid)
    return {};
  return dict_sp;
}

bool ScriptedCommandPythonInterface::GetShortHelp(std::string &dest) {
  dest.clear();
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("get_short_help"), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return false;
  dest = obj->GetStringValue().str();
  return !dest.empty();
}

bool ScriptedCommandPythonInterface::GetLongHelp(std::string &dest) {
  dest.clear();
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("get_long_help"), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return false;
  dest = obj->GetStringValue().str();
  return !dest.empty();
}

uint32_t ScriptedCommandPythonInterface::GetFlags() {
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("get_flags"), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return 0;

  return static_cast<uint32_t>(obj->GetUnsignedIntegerValue());
}

StructuredData::ObjectSP
ScriptedCommandPythonInterface::GetOptionsDefinition() {
  return LogAndDefault(Dispatch("get_options_definition"),
                       LLVM_PRETTY_FUNCTION);
}

StructuredData::ObjectSP
ScriptedCommandPythonInterface::GetArgumentsDefinition() {
  return LogAndDefault(Dispatch("get_args_definition"), LLVM_PRETTY_FUNCTION);
}

void ScriptedCommandPythonInterface::OptionParsingStarted() {
  LogAndDefault(Dispatch("option_parsing_started"), LLVM_PRETTY_FUNCTION);
}

bool ScriptedCommandPythonInterface::SetOptionValue(ExecutionContext *exe_ctx,
                                                    llvm::StringRef long_option,
                                                    llvm::StringRef value) {
  lldb::ExecutionContextRefSP exe_ctx_ref_sp;
  if (exe_ctx)
    exe_ctx_ref_sp = std::make_shared<ExecutionContextRef>(exe_ctx);
  std::string long_option_str = long_option.str();
  std::string value_str = value.str();
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("set_option_value", exe_ctx_ref_sp,
                             long_option_str.c_str(), value_str.c_str()),
                    LLVM_PRETTY_FUNCTION);
  if (!obj)
    return false;
  return obj->GetBooleanValue();
}

void ScriptedCommandPythonInterface::Initialize() {
  const std::vector<llvm::StringRef> ci_usages = {
      "command script add -c <ClassName> <cmd>",
      "command script add -p <ClassName> <cmd>"};
  const std::vector<llvm::StringRef> api_usages = {};
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(),
      "Implement a raw or parsed custom command backed by a Python class.",
      CreateInstance, eScriptedExtensionScriptedCommand, eScriptLanguagePython,
      ScriptedInterfaceUsages(ci_usages, api_usages));
}

void ScriptedCommandPythonInterface::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}
