//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_CPLUSPLUS_ITANIUMABIRUNTIME_H
#define LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_CPLUSPLUS_ITANIUMABIRUNTIME_H

#include "CommonABIRuntime.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/ValueObject/ValueObject.h"

#include <vector>

namespace lldb_private {

class ItaniumABIRuntime : public CommonABIRuntime {
public:
  ItaniumABIRuntime(Process *process);

  llvm::StringRef GetName() const override { return "Itanium ABI runtime"; }

  bool IsVTableSymbol(Mangled &manged) const override;

  llvm::Expected<LanguageRuntime::VTableInfo>
  GetVTableInfo(ValueObject &in_value, bool check_type);

  bool GetDynamicTypeAndAddress(ValueObject &in_value,
                                lldb::DynamicValueType use_dynamic,
                                const LanguageRuntime::VTableInfo &vtable_info,
                                TypeAndOrName &class_type_or_name,
                                Address &dynamic_address) override;

  void AppendExceptionBreakpointFunctions(std::vector<const char *> &names,
                                          bool catch_bp, bool throw_bp,
                                          bool for_expressions) override;

  void AppendExceptionBreakpointFilterModules(FileSpecList &list,
                                              const Target &target) override;

  lldb::ValueObjectSP
  GetExceptionObjectForThread(lldb::ThreadSP thread_sp) override;

private:
  TypeAndOrName GetTypeInfo(ValueObject &in_value,
                            const LanguageRuntime::VTableInfo &vtable_info);

  using VTableInfoCache = std::map<Address, LanguageRuntime::VTableInfo>;

  VTableInfoCache m_vtable_info_map;
};

} // namespace lldb_private

#endif
