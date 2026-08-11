//===-- GNUstepObjCRuntime.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_OBJC_GNUSTEPOBJCRUNTIME_GNUSTEPOBJCRUNTIME_H
#define LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_OBJC_GNUSTEPOBJCRUNTIME_GNUSTEPOBJCRUNTIME_H

#include "lldb/Target/LanguageRuntime.h"
#include "lldb/lldb-private.h"

#include "Plugins/LanguageRuntime/ObjC/ObjCLanguageRuntime.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <optional>

namespace lldb_private {

class GNUstepTaggedPointerVendor;

class GNUstepObjCRuntime : public lldb_private::ObjCLanguageRuntime {
public:
  ~GNUstepObjCRuntime() override;

  //
  //  PluginManager, PluginInterface and LLVM RTTI implementation
  //

  static char ID;

  static void Initialize();

  static void Terminate();

  static lldb_private::LanguageRuntime *
  CreateInstance(Process *process, lldb::LanguageType language);

  static llvm::StringRef GetPluginNameStatic() {
    return "gnustep-objc-libobjc2";
  }

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  void ModulesDidLoad(const ModuleList &module_list) override;

  bool isA(const void *ClassID) const override {
    return ClassID == &ID || ObjCLanguageRuntime::isA(ClassID);
  }

  static bool classof(const LanguageRuntime *runtime) {
    return runtime->isA(&ID);
  }

  //
  // LanguageRuntime implementation
  //
  llvm::Error GetObjectDescription(Stream &str, Value &value,
                                   ExecutionContextScope *exe_scope) override;

  llvm::Error GetObjectDescription(Stream &str, ValueObject &object) override;

  bool CouldHaveDynamicValue(ValueObject &in_value) override;

  bool GetDynamicTypeAndAddress(ValueObject &in_value,
                                lldb::DynamicValueType use_dynamic,
                                TypeAndOrName &class_type_or_name,
                                Address &address, Value::ValueType &value_type,
                                llvm::ArrayRef<uint8_t> &local_buffer) override;

  TypeAndOrName FixUpDynamicType(const TypeAndOrName &type_and_or_name,
                                 ValueObject &static_value) override;

  lldb::BreakpointResolverSP
  CreateExceptionResolver(const lldb::BreakpointSP &bkpt, bool catch_bp,
                          bool throw_bp) override;

  lldb::ThreadPlanSP GetStepThroughTrampolinePlan(Thread &thread,
                                                  bool stop_others) override;

  //
  // ObjCLanguageRuntime implementation
  //

  bool IsModuleObjCLibrary(const lldb::ModuleSP &module_sp) override;

  bool ReadObjCLibrary(const lldb::ModuleSP &module_sp) override;

  bool HasReadObjCLibrary() override { return m_objc_module_sp != nullptr; }

  llvm::Expected<std::unique_ptr<UtilityFunction>>
  CreateObjectChecker(std::string name, ExecutionContext &exe_ctx) override;

  /// Reported by `statistics dump` and the SB API, as AppleObjCRuntimeV2
  /// does. libobjc2 implements the GNUstep Objective-C ABI version 2.
  StructuredData::ObjectSP GetLanguageSpecificData(SymbolContext sc) override;

  ObjCRuntimeVersions GetRuntimeVersion() const override {
    return ObjCRuntimeVersions::eGNUstep_libobjc2;
  }

  void UpdateISAToDescriptorMapIfNeeded() override;

  /// Provides an IR pass that registers the expression module's Objective-C
  /// selectors with the runtime. JIT'd expression modules never run
  /// __objc_load, so their selector structures would otherwise reach
  /// objc_msgSend unregistered.
  bool GetIRPasses(LLVMUserExpression::IRPasses &custom_passes) override;

  /// gnustep-base implements the container-literal and boxed-expression
  /// protocol methods, so @[...], @{...} and @(...) are available.
  bool CalculateHasNewLiteralsAndIndexing() override { return true; }

  TaggedPointerVendor *GetTaggedPointerVendor() override;

  ClassDescriptorSP GetClassDescriptor(ValueObject &in_value) override;

  ClassDescriptorSP GetClassDescriptorFromISA(ObjCISA isa) override;

protected:
  // Call CreateInstance instead.
  GNUstepObjCRuntime(Process *process);

  /// Finds a complete Objective-C interface type named \p class_name in the
  /// target's debug info. Used to attach a real type to a dynamic value when
  /// the base class's symbol-name-keyed cache misses (the gnustep-2.x class
  /// symbol is not named after the class).
  lldb::TypeSP LookupClassTypeInDebugInfo(ConstString class_name);

  /// Address of gnustep-base's `const char *_NSPrintForDebugger(id)`, the
  /// same debugger hook AppleObjCRuntime uses. Resolved lazily; nullptr when
  /// gnustep-base is not loaded in the inferior.
  Address *GetPrintForDebuggerAddr();

public:
  /// Lazily-built FunctionCaller for libobjc2's
  /// `IMP objc_msg_lookup(id receiver, SEL selector)`, used by the
  /// step-through-trampoline plan. Returns nullptr if the symbol cannot be
  /// resolved.
  FunctionCaller *GetMsgLookupFunctionCaller();

protected:

  lldb::ModuleSP m_objc_module_sp;

  std::unique_ptr<Address> m_print_for_debugger_addr_up;

  std::unique_ptr<FunctionCaller> m_print_object_caller_up;

  std::unique_ptr<FunctionCaller> m_msg_lookup_caller_up;

  std::unique_ptr<GNUstepTaggedPointerVendor> m_tagged_pointer_vendor_up;

  /// Set when new modules arrive; cleared once the ISA-to-descriptor map has
  /// been refreshed, so the symbol sweep only reruns after module changes.
  bool m_isa_map_dirty = true;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_OBJC_GNUSTEPOBJCRUNTIME_GNUSTEPOBJCRUNTIME_H
