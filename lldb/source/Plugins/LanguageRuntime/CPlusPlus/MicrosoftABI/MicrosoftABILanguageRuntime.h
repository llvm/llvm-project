//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_CPLUSPLUS_MICROSOFTABI_MICROSOFTABILANGUAGERUNTIME_H
#define LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_CPLUSPLUS_MICROSOFTABI_MICROSOFTABILANGUAGERUNTIME_H

#include "lldb/Core/Value.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Target/LanguageRuntime.h"

#include "Plugins/LanguageRuntime/CPlusPlus/CPPLanguageRuntime.h"

namespace lldb_private {

class MicrosoftABILanguageRuntime : public CPPLanguageRuntime {
public:
  static void Initialize();

  static void Terminate();

  static lldb_private::LanguageRuntime *
  CreateInstance(Process *process, lldb::LanguageType language);

  static llvm::StringRef GetPluginNameStatic() { return "microsoft-abi"; }

  static char ID;

  bool isA(const void *ClassID) const override {
    return ClassID == &ID || CPPLanguageRuntime::isA(ClassID);
  }

  static bool classof(const LanguageRuntime *runtime) {
    return runtime->isA(&ID);
  }

  llvm::Expected<LanguageRuntime::VTableInfo>
  GetVTableInfo(ValueObject &in_value, bool check_type) override;

  bool GetDynamicTypeAndAddress(ValueObject &in_value,
                                lldb::DynamicValueType use_dynamic,
                                TypeAndOrName &class_type_or_name,
                                Address &address, Value::ValueType &value_type,
                                llvm::ArrayRef<uint8_t> &local_buffer) override;

  TypeAndOrName FixUpDynamicType(const TypeAndOrName &type_and_or_name,
                                 ValueObject &static_value) override;

  bool CouldHaveDynamicValue(ValueObject &in_value) override;

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  lldb::BreakpointResolverSP
  CreateExceptionResolver(const lldb::BreakpointSP &bkpt, bool catch_bp,
                          bool throw_bp) override;

private:
  MicrosoftABILanguageRuntime(Process *process) : CPPLanguageRuntime(process) {}
};

} // namespace lldb_private

#endif
