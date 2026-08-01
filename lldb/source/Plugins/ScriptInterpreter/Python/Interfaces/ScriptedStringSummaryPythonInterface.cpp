//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lldb-python.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/ValueObject/ValueObject.h"
#include "lldb/lldb-enumerations.h"

#include "../ScriptInterpreterPythonImpl.h"
#include "ScriptedStringSummaryPythonInterface.h"

using namespace lldb;
using namespace lldb_private;

ScriptedStringSummaryPythonInterface::ScriptedStringSummaryPythonInterface(
    ScriptInterpreterPythonImpl &interpreter)
    : ScriptedStringSummaryInterface(), ScriptedPythonInterface(interpreter) {}

llvm::Expected<StructuredData::GenericSP>
ScriptedStringSummaryPythonInterface::CreatePluginObject(
    llvm::StringRef class_name) {
  if (class_name.empty())
    return llvm::createStringError("empty class name");

  return ScriptedPythonInterface::CreatePluginObject(
      ScriptedMetadata(class_name, nullptr), nullptr);
}

llvm::Expected<std::string> ScriptedStringSummaryPythonInterface::GetSummary(
    ValueObject &valobj, const TypeSummaryOptions &options) {
  Status error;
  StructuredData::ObjectSP obj =
      Dispatch("get_summary", error, valobj.GetSP(), options);
  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error))
    return error.ToError();
  return obj->GetStringValue().str();
}

void ScriptedStringSummaryPythonInterface::Initialize() {
  const std::vector<llvm::StringRef> ci_usages = {
      "type summary add -L <ClassName> [-K <key> -V <value> ...] <TypeName>"};
  const std::vector<llvm::StringRef> api_usages = {
      "SBTypeSummary.CreateWithClassName"};
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(),
      "Provide a summary string for a type, used by 'type summary add -l'",
      CreateInstance, eScriptedExtensionScriptedStringSummary,
      eScriptLanguagePython, {ci_usages, api_usages});
}

void ScriptedStringSummaryPythonInterface::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}
