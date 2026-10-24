//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_CPLUSPLUS_MICROSOFTABIRUNTIME_H
#define LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_CPLUSPLUS_MICROSOFTABIRUNTIME_H

#include "CommonABIRuntime.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/ValueObject/ValueObject.h"

namespace lldb_private {

class MicrosoftABIRuntime : public CommonABIRuntime {
public:
  MicrosoftABIRuntime(Process *process);

  llvm::StringRef GetName() const override { return "Microsoft ABI runtime"; }

  bool GetDynamicTypeAndAddress(ValueObject &in_value,
                                lldb::DynamicValueType use_dynamic,
                                const LanguageRuntime::VTableInfo &vtable_info,
                                TypeAndOrName &class_type_or_name,
                                Address &dynamic_address) override;

  bool IsVTableSymbol(Mangled &mangled) const override;

private:
  TypeAndOrName
  GetTypeInfo(ValueObject &in_value,
              const LanguageRuntime::VTableInfo &vtable_info,
              llvm::SmallVectorImpl<llvm::StringRef> &vtable_components);

  llvm::Error TypeHasVTable(CompilerType type);

  std::optional<int64_t>
  GetOffsetToTop(ValueObject &in_value, TypeAndOrName &class_type_or_name,
                 const LanguageRuntime::VTableInfo &vtable_info);

  std::optional<int64_t>
  GetOffsetToTopFromRTTI(const LanguageRuntime::VTableInfo &vtable_info,
                         const CompilerType &dynamic_type);

  std::optional<int64_t>
  GetOffsetToTopFromAST(const CompilerType &in_type, ValueObject &in_value,
                        const CompilerType &dyn_type,
                        const LanguageRuntime::VTableInfo &vtable_info,
                        llvm::ArrayRef<llvm::StringRef> vtable_components);

  using OffsetToTopCache = std::map<Address, std::optional<int64_t>>;

  OffsetToTopCache m_offset_to_top_map;
};

} // namespace lldb_private

#endif
