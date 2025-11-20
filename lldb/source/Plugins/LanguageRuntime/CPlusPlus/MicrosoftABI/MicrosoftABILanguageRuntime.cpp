//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MicrosoftABILanguageRuntime.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Target/Process.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE_ADV(MicrosoftABILanguageRuntime, CXXMicrosoftABI)

char MicrosoftABILanguageRuntime::ID = 0;

void MicrosoftABILanguageRuntime::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                "Microsoft ABI for the C++ language",
                                CreateInstance);
}

void MicrosoftABILanguageRuntime::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

LanguageRuntime *
MicrosoftABILanguageRuntime::CreateInstance(Process *process,
                                            lldb::LanguageType language) {
  if (!ShouldUseMicrosoftABI(process))
    return nullptr;

  if (!(language == eLanguageTypeC_plus_plus ||
        language == eLanguageTypeC_plus_plus_03 ||
        language == eLanguageTypeC_plus_plus_11 ||
        language == eLanguageTypeC_plus_plus_14))
    return nullptr;

  return new MicrosoftABILanguageRuntime(process);
}

llvm::Expected<LanguageRuntime::VTableInfo>
MicrosoftABILanguageRuntime::GetVTableInfo(ValueObject &in_value,
                                           bool check_type) {
  return llvm::createStringError("Not implemented");
}

bool MicrosoftABILanguageRuntime::GetDynamicTypeAndAddress(
    ValueObject &in_value, lldb::DynamicValueType use_dynamic,
    TypeAndOrName &class_type_or_name, Address &address,
    Value::ValueType &value_type, llvm::ArrayRef<uint8_t> &local_buffer) {
  return false;
}

TypeAndOrName MicrosoftABILanguageRuntime::FixUpDynamicType(
    const TypeAndOrName &type_and_or_name, ValueObject &static_value) {
  return type_and_or_name;
}

bool MicrosoftABILanguageRuntime::CouldHaveDynamicValue(ValueObject &in_value) {
  return false;
}

lldb::BreakpointResolverSP MicrosoftABILanguageRuntime::CreateExceptionResolver(
    const lldb::BreakpointSP &bkpt, bool catch_bp, bool throw_bp) {
  return nullptr;
}
