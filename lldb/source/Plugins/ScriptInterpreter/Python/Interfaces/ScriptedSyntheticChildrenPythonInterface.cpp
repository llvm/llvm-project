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
  // This interface requires no abstract methods, so a provider that doesn't
  // implement `num_children` simply has no children rather than being broken.
  llvm::Expected<std::optional<StructuredData::ObjectSP>> obj_or_err =
      DispatchToOptional("num_children", max);
  if (!obj_or_err)
    return obj_or_err.takeError();

  StructuredData::ObjectSP obj = obj_or_err->value_or(nullptr);
  if (!obj || !obj->IsValid())
    return 0;
  // Cap at max in case the provider ignores the argument (e.g. defines
  // `num_children(self)`) and returns an unbounded count.
  return std::min<uint32_t>(obj->GetUnsignedIntegerValue(), max);
}

lldb::ValueObjectSP
ScriptedSyntheticChildrenPythonInterface::GetChildAtIndex(uint32_t idx) {
  return LogAndDefault(Dispatch<lldb::ValueObjectSP>("get_child_at_index", idx),
                       LLVM_PRETTY_FUNCTION);
}

llvm::Expected<uint32_t>
ScriptedSyntheticChildrenPythonInterface::GetIndexOfChildWithName(
    ConstString name) {
  // A provider without `get_child_index` has no child of that name, which is
  // a friendlier answer than "the method is missing".
  llvm::Expected<std::optional<StructuredData::ObjectSP>> obj_or_err =
      DispatchToOptional("get_child_index", name.GetCString());
  if (!obj_or_err)
    return obj_or_err.takeError();

  StructuredData::ObjectSP obj = obj_or_err->value_or(nullptr);
  if (!obj || !obj->IsValid())
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
  // update() is optional; a missing method means "always refetch".
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("update"), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return lldb::eRefetch;
  return obj->GetBooleanValue() ? lldb::eReuse : lldb::eRefetch;
}

bool ScriptedSyntheticChildrenPythonInterface::MightHaveChildren() {
  // has_children() is optional and defaults to True when missing.
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("has_children"), LLVM_PRETTY_FUNCTION);
  if (!obj)
    return true;
  return obj->GetBooleanValue();
}

lldb::ValueObjectSP
ScriptedSyntheticChildrenPythonInterface::GetSyntheticValue() {
  return LogAndDefault(Dispatch<lldb::ValueObjectSP>("get_value"),
                       LLVM_PRETTY_FUNCTION);
}

ConstString ScriptedSyntheticChildrenPythonInterface::GetSyntheticTypeName() {
  StructuredData::ObjectSP obj =
      LogAndDefault(Dispatch("get_type_name"), LLVM_PRETTY_FUNCTION);
  if (!obj)
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
