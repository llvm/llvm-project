//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lldb-python.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Utility/ScriptedMetadata.h"
#include "lldb/ValueObject/ValueObject.h"
#include "lldb/lldb-enumerations.h"

#include "../SWIGPythonBridge.h"
#include "../ScriptInterpreterPythonImpl.h"
#include "ScriptedSyntheticChildrenPythonInterface.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::python;
using Locker = ScriptInterpreterPythonImpl::Locker;

ScriptedSyntheticChildrenPythonInterface::
    ScriptedSyntheticChildrenPythonInterface(
        ScriptInterpreterPythonImpl &interpreter)
    : ScriptedSyntheticChildrenInterface(),
      ScriptedPythonInterface(interpreter) {}

llvm::Expected<StructuredData::GenericSP>
ScriptedSyntheticChildrenPythonInterface::CreatePluginObject(
    llvm::StringRef class_name, ValueObject &backend) {
  if (class_name.empty())
    return llvm::createStringError("empty class name");

  ValueObjectSP valobj_sp = backend.GetSP();
  if (!valobj_sp)
    return llvm::createStringError("invalid backing value");

  Locker py_lock(&m_interpreter,
                 Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN,
                 Locker::FreeLock | Locker::TearDownSession);

  // Hand the provider's __init__ a fresh SBValue view of the backing value
  // with synthetic children disabled, so introspecting it doesn't recursively
  // re-enter this provider. `SetPreferSyntheticValue` lives on the SBValue's
  // ValueImpl, so this override doesn't affect the caller's original view.
  PythonObject val_arg =
      SWIGBridge::ToSWIGWrapper(valobj_sp, /*use_synthetic=*/false);

  ScriptedMetadata scripted_metadata(class_name,
                                     StructuredData::DictionarySP());
  return ScriptedPythonInterface::CreatePluginObject(
      scripted_metadata, /*script_obj=*/nullptr, std::move(val_arg));
}

llvm::Expected<uint32_t>
ScriptedSyntheticChildrenPythonInterface::CalculateNumChildren(uint32_t max) {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("num_children", error, max);
  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error))
    return 0;
  // Cap at max in case the provider ignores the argument (e.g. defines
  // `num_children(self)`) and returns an unbounded count.
  return std::min<uint32_t>(obj->GetUnsignedIntegerValue(), max);
}

lldb::ValueObjectSP
ScriptedSyntheticChildrenPythonInterface::GetChildAtIndex(uint32_t idx) {
  Status error;
  return Dispatch<lldb::ValueObjectSP>("get_child_at_index", error, idx);
}

llvm::Expected<uint32_t>
ScriptedSyntheticChildrenPythonInterface::GetIndexOfChildWithName(
    ConstString name) {
  Status error;
  StructuredData::ObjectSP obj =
      Dispatch("get_child_index", error, name.GetCString());
  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error))
    return llvm::createStringErrorV("type has no child named '{0}'", name);

  // `CreateStructuredObject` only produces a `SignedInteger` for values that
  // don't fit as unsigned, i.e. negative ones; a non-negative index comes
  // back as `UnsignedInteger` instead, so check the sign this way rather
  // than via `GetSignedIntegerValue`, which would misread every valid index.
  if (obj->GetAsSignedInteger())
    return llvm::createStringErrorV("type has no child named '{0}'", name);
  return static_cast<uint32_t>(obj->GetUnsignedIntegerValue());
}

lldb::ChildCacheState ScriptedSyntheticChildrenPythonInterface::Update() {
  Status error;
  // update() is optional; a missing method means "always refetch".
  StructuredData::ObjectSP obj = Dispatch("update", error);
  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error))
    return lldb::eRefetch;
  return obj->GetBooleanValue() ? lldb::eReuse : lldb::eRefetch;
}

bool ScriptedSyntheticChildrenPythonInterface::MightHaveChildren() {
  Status error;
  // has_children() is optional and defaults to True when missing.
  StructuredData::ObjectSP obj = Dispatch("has_children", error);
  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error))
    return true;
  return obj->GetBooleanValue();
}

lldb::ValueObjectSP
ScriptedSyntheticChildrenPythonInterface::GetSyntheticValue() {
  Status error;
  return Dispatch<lldb::ValueObjectSP>("get_value", error);
}

ConstString ScriptedSyntheticChildrenPythonInterface::GetSyntheticTypeName() {
  Status error;
  StructuredData::ObjectSP obj = Dispatch("get_type_name", error);
  if (!ScriptedInterface::CheckStructuredDataObject(LLVM_PRETTY_FUNCTION, obj,
                                                    error))
    return {};
  return ConstString(obj->GetStringValue());
}

void ScriptedSyntheticChildrenPythonInterface::Initialize() {
  const std::vector<llvm::StringRef> ci_usages = {
      "type synthetic add -l <ClassName> <TypeName>"};
  const std::vector<llvm::StringRef> api_usages = {
      "SBTypeSynthetic.CreateWithClassName"};
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(),
      "Provide synthetic children for a type, used by 'type synthetic add -l'",
      CreateInstance, eScriptedExtensionScriptedSyntheticChildren,
      eScriptLanguagePython, {ci_usages, api_usages});
}

void ScriptedSyntheticChildrenPythonInterface::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}
