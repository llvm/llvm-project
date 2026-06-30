//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_CPLUSPLUS_COMMONABIRUNTIME_H
#define LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_CPLUSPLUS_COMMONABIRUNTIME_H

#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/ValueObject/ValueObject.h"

#include <map>
#include <mutex>

namespace lldb_private {

class CommonABIRuntime {
public:
  virtual ~CommonABIRuntime() = default;

  virtual llvm::StringRef GetName() const = 0;

  virtual bool IsVTableSymbol(Mangled &mangled) const;

  virtual bool GetDynamicTypeAndAddress(
      ValueObject &in_value, lldb::DynamicValueType use_dynamic,
      const LanguageRuntime::VTableInfo &vtable_info,
      TypeAndOrName &class_type_or_name, Address &dynamic_address);

  virtual void
  AppendExceptionBreakpointFunctions(std::vector<const char *> &names,
                                     bool catch_bp, bool throw_bp,
                                     bool for_expressions);

  virtual void AppendExceptionBreakpointFilterModules(FileSpecList &list,
                                                      const Target &target);

  virtual lldb::ValueObjectSP
  GetExceptionObjectForThread(lldb::ThreadSP thread_sp);

protected:
  CommonABIRuntime(Process *process);

  lldb::TypeSP LookupTypeByName(llvm::StringRef type_name,
                                lldb::ModuleSP preferred_module) const;

  TypeAndOrName GetDynamicTypeInfo(const lldb_private::Address &vtable_addr);

  void SetDynamicTypeInfo(const lldb_private::Address &vtable_addr,
                          const TypeAndOrName &type_info);

protected:
  Process *m_process;
  std::mutex m_mutex;

private:
  using DynamicTypeCache = std::map<Address, TypeAndOrName>;

  DynamicTypeCache m_dynamic_type_map;
};

} // namespace lldb_private

#endif
