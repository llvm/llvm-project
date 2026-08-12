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

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <vector>

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

  bool CalculateHasNewLiteralsAndIndexing() override;

  TaggedPointerVendor *GetTaggedPointerVendor() override;

  ClassDescriptorSP GetClassDescriptor(ValueObject &in_value) override;

  ClassDescriptorSP GetClassDescriptorFromISA(ObjCISA isa) override;

  /// Lazily-built FunctionCaller for a utility function that resolves a
  /// method implementation via libobjc2's
  /// `IMP objc_msg_lookup(id receiver, SEL selector)`, used by the
  /// step-through-trampoline plan. Returns nullptr on failure. The caller is
  /// owned by the utility function and stays valid for the runtime's life.
  FunctionCaller *GetMsgLookupFunctionCaller(Thread &thread);

  /// Returns true if \p addr belongs to the module implementing the ObjC
  /// runtime. Method lookups that resolve there reached either the forwarding
  /// machinery or one of the runtime's own methods, neither of which has user
  /// source to step into.
  ///
  /// Always false when the runtime is linked into the executable, where its
  /// module is the program's own and the distinction cannot be drawn.
  bool IsRuntimeInternalAddress(lldb::addr_t addr);

protected:
  // Call CreateInstance instead.
  GNUstepObjCRuntime(Process *process);

  /// A libobjc2 message dispatch entry point, identified by the load address
  /// of its first instruction (resolved by symbol name, so it is robust to
  /// local labels sharing the address).
  struct DispatchEntryPoint {
    lldb::addr_t address;
    bool is_stret;
    bool is_sender;
  };

  /// Returns the dispatch entry point whose first instruction is at \p pc, if
  /// any. The entry-point address table is resolved on first use and dropped
  /// when modules are loaded.
  std::optional<DispatchEntryPoint> FindDispatchEntryPoint(lldb::addr_t pc);

  /// The prefix clang gives the runtime's public class symbols, which differs
  /// between object formats.
  llvm::StringRef GetClassSymbolPrefix();

  /// Adds every class statically defined by \p module_sp to the
  /// ISA-to-descriptor map.
  void AddClassesFromModule(const lldb::ModuleSP &module_sp);

  /// Finds a complete Objective-C interface type named \p class_name in the
  /// target's debug info. Used to attach a real type to a dynamic value when
  /// the base class's symbol-name-keyed cache misses (the gnustep-2.x class
  /// symbol is not named after the class).
  lldb::TypeSP LookupClassTypeInDebugInfo(ConstString class_name);

  /// Address of gnustep-base's `const char *_NSPrintForDebugger(id)`, the
  /// same debugger hook AppleObjCRuntime uses. Resolved lazily; nullptr when
  /// gnustep-base is not loaded in the inferior.
  Address *GetPrintForDebuggerAddr();

  lldb::ModuleSP m_objc_module_sp;

  std::unique_ptr<Address> m_print_for_debugger_addr_up;

  std::unique_ptr<FunctionCaller> m_print_object_caller_up;

  /// Utility function wrapping objc_msg_lookup; owns m_msg_lookup_caller.
  /// Guarded by m_msg_lookup_mutex, which also latches a failed build so it
  /// is not retried on every step.
  std::mutex m_msg_lookup_mutex;
  std::unique_ptr<UtilityFunction> m_msg_lookup_utility_up;
  FunctionCaller *m_msg_lookup_caller = nullptr;
  bool m_msg_lookup_failed = false;

  llvm::SmallVector<DispatchEntryPoint, 6> m_dispatch_entry_points;
  bool m_dispatch_entry_points_resolved = false;

  std::unique_ptr<GNUstepTaggedPointerVendor> m_tagged_pointer_vendor_up;

  /// Classes named here have no debug info, so the search is not repeated.
  std::set<ConstString> m_negative_type_cache;

  /// Modules seen since the last ISA-to-descriptor map update, so only new
  /// symbol tables have to be scanned.
  std::vector<lldb::ModuleSP> m_pending_modules;
  bool m_swept_all_modules = false;

  /// Set when new modules arrive; cleared once the ISA-to-descriptor map has
  /// been refreshed, so the symbol sweep only reruns after module changes.
  bool m_isa_map_dirty = true;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_OBJC_GNUSTEPOBJCRUNTIME_GNUSTEPOBJCRUNTIME_H
