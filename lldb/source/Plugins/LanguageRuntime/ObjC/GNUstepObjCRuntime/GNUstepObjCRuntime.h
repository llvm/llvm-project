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
class GNUstepObjCDeclVendor;

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

  /// CreateExceptionSearchFilter is deliberately not overridden. Apple's
  /// runtime narrows the filter to the module holding its throw entry point,
  /// which it can because that is always libobjc.A.dylib. Ours is not always
  /// in the runtime library: where clang routes @catch through the C++ ABI
  /// the entry point is libstdc++'s __cxa_begin_catch, so the search has to
  /// stay unrestricted.
  lldb::BreakpointResolverSP
  CreateExceptionResolver(const lldb::BreakpointSP &bkpt, bool catch_bp,
                          bool throw_bp) override;

  void SetExceptionBreakpoints() override;

  void ClearExceptionBreakpoints() override;

  bool ExceptionBreakpointsAreSet() override;

  bool ExceptionBreakpointsExplainStop(lldb::StopInfoSP stop_reason) override;

  /// The object of the Objective-C exception being thrown on \p thread, or an
  /// empty ValueObjectSP if the thread is not throwing one.
  lldb::ValueObjectSP
  GetExceptionObjectForThread(lldb::ThreadSP thread_sp) override;

  /// Thread whose frames are the call stack recorded inside \p exception_sp,
  /// for `thread exception`. gnustep-base captures it in -[NSException raise];
  /// an exception raised by a bare `@throw` has none.
  lldb::ThreadSP
  GetBacktraceThreadFromException(lldb::ValueObjectSP exception_sp) override;

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

  /// Realizes libobjc2's type encodings, which is what makes the ivar and
  /// method metadata usable as types. Built on first use and kept, because
  /// the types it returns are only valid in the AST it created them in.
  EncodingToTypeSP GetEncodingToType() override;

  /// Synthesizes Objective-C interface declarations from runtime metadata,
  /// so a class the debug info does not describe is still usable. Built on
  /// first use; the decls it owns are copied into the expression parser's
  /// AST as they are needed.
  DeclVendor *GetDeclVendor() override;

  /// Resolves \p base_type to the most complete description available.
  ///
  /// The inherited implementation consults debug info through
  /// LookupInCompleteClassCache, which keys on an eSymbolTypeObjCClass
  /// symbol that only Mach-O produces - so for gnustep-2.x it always misses
  /// and the runtime-synthesized interface wins by default. That is a
  /// downgrade wherever debug info exists, because the runtime's metadata
  /// cannot describe members inside a struct-typed ivar. Prefer debug info
  /// explicitly, and fall back to the inherited behaviour otherwise.
  std::optional<CompilerType> GetRuntimeType(CompilerType base_type) override;

  /// Byte offset of ivar \p ivar_name within \p parent_qual_type, or
  /// LLDB_INVALID_IVAR_OFFSET. Without this the offset comes from laying the
  /// class out as a plain struct, which libobjc2's does not match.
  size_t GetByteOffsetForIvar(CompilerType &parent_qual_type,
                              const char *ivar_name) override;

  /// Size of an Objective-C class, in bits.
  ///
  /// The inherited implementation derives this from the ivar list, as the end
  /// of the last ivar. That is wrong for libobjc2 twice over: it ignores
  /// trailing padding, and `objc_class::ivars` holds only the class's *own*
  /// ivars, so a class that declares none would report nothing at all while
  /// its sibling reported a size. The runtime already knows the answer -
  /// `instance_size` is the true body size once the class is resolved - so
  /// use that, and decline rather than guess when it is not yet available.
  std::optional<uint64_t>
  GetTypeBitSize(const CompilerType &compiler_type) override;

  /// The name of the selector at \p sel_addr, recovered from the symbol
  /// clang emits for it. Empty when there is no such symbol.
  ///
  /// This cannot be read from memory: __objc_load overwrites a selector's
  /// name field with a numeric dispatch index (selector_table.cc), so the
  /// string is gone by the time a debugger sees it. No code is run in the
  /// inferior to find it.
  ConstString GetSelectorName(lldb::addr_t sel_addr);

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

  /// Address of the runtime variable a JIT'd expression's symbol refers to,
  /// for the ivar-offset and class symbols the gnustep-2.x ABI emits. Only
  /// consulted after the inferior's own symbol table has been tried.
  lldb::addr_t LookupRuntimeSymbol(ConstString name) override;

  /// Splits `__objc_ivar_offset_<Class>.<ivar>.<mangled type encoding>` into
  /// its class and ivar names, false if \p symbol is not one. Static so it can
  /// be tested without a process.
  static bool ParseIvarOffsetSymbol(llvm::StringRef symbol,
                                    llvm::StringRef &class_name,
                                    llvm::StringRef &ivar_name);

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

  /// Builds an interface type for \p class_name from runtime metadata, for a
  /// class the debug info does not describe. The base class keeps its
  /// equivalent private, and reaches it only from GetRuntimeType - which takes
  /// a type rather than a name, so the dynamic-value path cannot use it.
  CompilerType LookupClassTypeInRuntime(ConstString class_name);

  /// Address of ivar \p ivar_name within the object at \p object_addr, found
  /// through the runtime's metadata so that it works for a class whose ivars
  /// the debug info does not describe - which is every gnustep-base class
  /// behind GS_EXPOSE.
  std::optional<lldb::addr_t> GetIvarAddress(lldb::addr_t object_addr,
                                             llvm::StringRef ivar_name);

  /// Lazily-built FunctionCaller for a utility function that reproduces
  /// gnustep-base's `_NSPrintForDebugger` (NSDebug.m) using nothing but
  /// libobjc2's exported API, so `po` works whether or not Foundation is
  /// loaded and regardless of whether gnustep-base exports that hook - it
  /// does on ELF but not from the MSVC DLL. Returns nullptr on failure; the
  /// caller is owned by the utility function and stays valid for the
  /// runtime's life.
  FunctionCaller *GetObjectDescriptionCaller(ExecutionContext &exe_ctx);

  /// Re-runs every exception breakpoint's resolver, so modules loaded before
  /// this runtime existed are considered.
  void ResolveExceptionBreakpoints();

  bool m_swept_exception_breakpoints = false;

  lldb::ModuleSP m_objc_module_sp;

  /// Utility function wrapping the -description/-UTF8String pair; owns
  /// m_description_caller. Guarded by m_description_mutex, which also
  /// latches a failed build so it is not retried on every `po`.
  std::mutex m_description_mutex;
  std::unique_ptr<UtilityFunction> m_description_utility_up;
  FunctionCaller *m_description_caller = nullptr;
  bool m_description_failed = false;

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

  /// Guards against a sweep that resolves a class re-entering the sweep.
  bool m_updating_isa_map = false;

  /// The internal breakpoint on the runtime's throw entry point, used by
  /// `process handle`/`thread exception` to stop where an exception is
  /// raised rather than where it is caught.
  lldb::BreakpointSP m_objc_exception_bp_sp;

  /// Torn down in an explicit order by the destructor, which see: the parser
  /// caches types belonging to the vendor's AST.
  EncodingToTypeSP m_encoding_to_type_sp;

  std::unique_ptr<GNUstepObjCDeclVendor> m_decl_vendor_up;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_OBJC_GNUSTEPOBJCRUNTIME_GNUSTEPOBJCRUNTIME_H
