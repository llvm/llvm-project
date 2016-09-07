//===-- GoLanguageRuntime.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_GoLanguageRuntime_h_
#define liblldb_GoLanguageRuntime_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/BreakpointResolver.h"
#include "lldb/Core/Value.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class GoLanguageRuntime : public lldb_private::LanguageRuntime {
public:
  ~GoLanguageRuntime() override = default;

  //------------------------------------------------------------------
  // Static Functions
  //------------------------------------------------------------------
  static void Initialize();

  static void Terminate();

  static lldb_private::LanguageRuntime *
  CreateInstance(Process *process, lldb::LanguageType language);

  static lldb_private::ConstString GetPluginNameStatic();

  lldb::LanguageType GetLanguageType() const override {
    return lldb::eLanguageTypeGo;
  }

  bool GetObjectDescription(Stream &str, ValueObject &object) override {
    // TODO(ribrdb): Maybe call String() method?
    return false;
  }

  bool GetObjectDescription(Stream &str, Value &value,
                            ExecutionContextScope *exe_scope) override {
    return false;
  }

  bool GetDynamicTypeAndAddress(ValueObject &in_value,
                                lldb::DynamicValueType use_dynamic,
                                TypeAndOrName &class_type_or_name,
                                Address &address,
                                Value::ValueType &value_type) override;

  bool CouldHaveDynamicValue(ValueObject &in_value) override;

  lldb::BreakpointResolverSP CreateExceptionResolver(Breakpoint *bkpt,
                                                     bool catch_bp,
                                                     bool throw_bp) override {
    return lldb::BreakpointResolverSP();
  }

  TypeAndOrName FixUpDynamicType(const TypeAndOrName &type_and_or_name,
                                 ValueObject &static_value) override;

  //------------------------------------------------------------------
  // PluginInterface protocol
  //------------------------------------------------------------------
  lldb_private::ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override;

private:
  GoLanguageRuntime(Process *process)
      : lldb_private::LanguageRuntime(process) {
  } // Call CreateInstance instead.
};

} // namespace lldb_private

#endif // liblldb_GoLanguageRuntime_h_
