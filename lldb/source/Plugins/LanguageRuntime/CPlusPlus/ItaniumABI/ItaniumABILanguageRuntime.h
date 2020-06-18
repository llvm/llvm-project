//===-- ItaniumABILanguageRuntime.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_CPLUSPLUS_ITANIUMABI_ITANIUMABILANGUAGERUNTIME_H
#define LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_CPLUSPLUS_ITANIUMABI_ITANIUMABILANGUAGERUNTIME_H

#include <map>
#include <mutex>
#include <vector>

#include "lldb/Breakpoint/BreakpointResolver.h"
#include "lldb/Core/Value.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/lldb-private.h"

#include "Plugins/LanguageRuntime/CPlusPlus/CPPLanguageRuntime.h"

namespace lldb_private {

class ItaniumABILanguageRuntime : public lldb_private::CPPLanguageRuntime {
public:
  ~ItaniumABILanguageRuntime() override = default;

  // Static Functions
  static void Initialize();

  static void Terminate();

  static lldb_private::LanguageRuntime *
  CreateInstance(Process *process, lldb::LanguageType language);

  static lldb_private::ConstString GetPluginNameStatic();

  static char ID;

  bool isA(const void *ClassID) const override {
    return ClassID == &ID || CPPLanguageRuntime::isA(ClassID);
  }

  static bool classof(const LanguageRuntime *runtime) {
    return runtime->isA(&ID);
  }

  bool GetDynamicTypeAndAddress(ValueObject &in_value,
                                lldb::DynamicValueType use_dynamic,
                                TypeAndOrName &class_type_or_name,
                                Address &address,
                                Value::ValueType &value_type) override;

  TypeAndOrName FixUpDynamicType(const TypeAndOrName &type_and_or_name,
                                 ValueObject &static_value) override;

  bool CouldHaveDynamicValue(ValueObject &in_value) override;

  void SetExceptionBreakpoints() override;

  void ClearExceptionBreakpoints() override;

  bool ExceptionBreakpointsAreSet() override;

  bool ExceptionBreakpointsExplainStop(lldb::StopInfoSP stop_reason) override;

  lldb::BreakpointResolverSP
  CreateExceptionResolver(const lldb::BreakpointSP &bkpt,
                          bool catch_bp, bool throw_bp) override;

  lldb::SearchFilterSP CreateExceptionSearchFilter() override;
  
  lldb::ValueObjectSP GetExceptionObjectForThread(
      lldb::ThreadSP thread_sp) override;

  // PluginInterface protocol
  lldb_private::ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override;

protected:
  lldb::BreakpointResolverSP
  CreateExceptionResolver(const lldb::BreakpointSP &bkpt,
                          bool catch_bp, bool throw_bp, bool for_expressions);

  lldb::BreakpointSP CreateExceptionBreakpoint(bool catch_bp, bool throw_bp,
                                               bool for_expressions,
                                               bool is_internal);

private:
  typedef std::map<lldb_private::Address, TypeAndOrName> DynamicTypeCache;

  ItaniumABILanguageRuntime(Process *process)
      : // Call CreateInstance instead.
        lldb_private::CPPLanguageRuntime(process), m_cxx_exception_bp_sp(),
        m_dynamic_type_map(), m_dynamic_type_map_mutex() {}

  lldb::BreakpointSP m_cxx_exception_bp_sp;
  DynamicTypeCache m_dynamic_type_map;
  std::mutex m_dynamic_type_map_mutex;

  TypeAndOrName GetTypeInfoFromVTableAddress(ValueObject &in_value,
                                             lldb::addr_t original_ptr,
                                             lldb::addr_t vtable_addr);

  TypeAndOrName GetDynamicTypeInfo(const lldb_private::Address &vtable_addr);

  void SetDynamicTypeInfo(const lldb_private::Address &vtable_addr,
                          const TypeAndOrName &type_info);
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_CPLUSPLUS_ITANIUMABI_ITANIUMABILANGUAGERUNTIME_H
