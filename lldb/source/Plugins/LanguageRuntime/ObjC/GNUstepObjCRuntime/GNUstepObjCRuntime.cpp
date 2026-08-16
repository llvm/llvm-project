//===-- GNUstepObjCRuntime.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GNUstepObjCRuntime.h"
#include "GNUstepObjCClassDescriptor.h"
#include "GNUstepThreadPlanStepThroughObjCTrampoline.h"

#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"

#include "lldb/Core/Address.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Value.h"
#include "lldb/Expression/DiagnosticManager.h"
#include "lldb/Expression/FunctionCaller.h"
#include "lldb/Expression/UtilityFunction.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlanRunToAddress.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegularExpression.h"
#include "lldb/Utility/StructuredData.h"
#include "lldb/ValueObject/ValueObject.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Regex.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(GNUstepObjCRuntime)

namespace {
/// Registers the Objective-C selectors of a JIT'd expression module with the
/// libobjc2 runtime.
///
/// clang emits each selector as a `.objc_selector_<name>_<types>` global in
/// the `__objc_selectors` section: a {name, types} string pair that the
/// runtime's __objc_load rewrites into a registered selector when a module
/// is loaded. Expression modules are never loaded that way, so passing the
/// raw structure to objc_msgSend dispatches an unregistered selector (which
/// gnustep-base reports as e.g. "-[NSSmallInt ]"). Replace every use with
/// the result of sel_registerTypedName_np()/sel_registerName(), which
/// resolve against libobjc2 at expression link time.
class GNUstepObjCSelectorRegistrationPass : public llvm::ModulePass {
public:
  static char ID;

  GNUstepObjCSelectorRegistrationPass() : llvm::ModulePass(ID) {}

  llvm::StringRef getPassName() const override {
    return "GNUstep ObjC selector registration";
  }

  bool runOnModule(llvm::Module &module) override {
    // Section names differ by object format: "__objc_selectors" everywhere
    // except COFF, which sorts the runtime metadata into ".objcrt$SEL"
    // subsections (CGObjCGNU.cpp).
    llvm::SmallVector<llvm::GlobalVariable *, 8> sel_globals;
    for (llvm::GlobalVariable &gv : module.globals()) {
      if (!gv.hasSection())
        continue;
      llvm::StringRef section(gv.getSection());
      if (section.starts_with("__objc_selectors") ||
          section.starts_with(".objcrt$SEL"))
        sel_globals.push_back(&gv);
    }
    if (sel_globals.empty())
      return false;

    llvm::LLVMContext &ctx = module.getContext();
    llvm::PointerType *ptr_ty = llvm::PointerType::get(ctx, 0);
    llvm::FunctionCallee typed_reg;
    llvm::FunctionCallee untyped_reg;

    bool changed = false;
    for (llvm::GlobalVariable *gv : sel_globals) {
      if (!gv->hasInitializer())
        continue;
      auto *init = llvm::dyn_cast<llvm::ConstantStruct>(gv->getInitializer());
      if (!init || init->getNumOperands() < 1)
        continue;
      llvm::Constant *name_ptr = init->getOperand(0);
      llvm::Constant *types_ptr =
          init->getNumOperands() > 1 ? init->getOperand(1) : nullptr;
      const bool has_types = types_ptr && !types_ptr->isNullValue();

      // One registration call per function; the entry block dominates all
      // uses, including PHI incoming edges.
      llvm::SmallDenseMap<llvm::Function *, llvm::Value *, 4> call_per_fn;
      llvm::SmallVector<llvm::Use *, 8> uses;
      for (llvm::Use &use : gv->uses())
        uses.push_back(&use);
      for (llvm::Use *use : uses) {
        auto *inst = llvm::dyn_cast<llvm::Instruction>(use->getUser());
        if (!inst) {
          // A constant expression has no instruction to anchor the call to;
          // such a selector stays unregistered.
          LLDB_LOG(
              GetLog(LLDBLog::Expressions),
              "not registering selector used by a constant expression: {0}",
              gv->getName());
          continue;
        }
        llvm::Function *func = inst->getFunction();
        llvm::Value *&reg_call = call_per_fn[func];
        if (!reg_call) {
          llvm::IRBuilder<> builder(
              &*func->getEntryBlock().getFirstInsertionPt());
          if (has_types) {
            if (!typed_reg)
              typed_reg = module.getOrInsertFunction(
                  "sel_registerTypedName_np",
                  llvm::FunctionType::get(ptr_ty, {ptr_ty, ptr_ty},
                                          /*isVarArg=*/false));
            reg_call = builder.CreateCall(typed_reg, {name_ptr, types_ptr},
                                          "lldb.objc.sel");
          } else {
            if (!untyped_reg)
              untyped_reg = module.getOrInsertFunction(
                  "sel_registerName",
                  llvm::FunctionType::get(ptr_ty, {ptr_ty},
                                          /*isVarArg=*/false));
            reg_call =
                builder.CreateCall(untyped_reg, {name_ptr}, "lldb.objc.sel");
          }
        }
        use->set(reg_call);
        changed = true;
      }
    }
    return changed;
  }
};

char GNUstepObjCSelectorRegistrationPass::ID = 0;
} // namespace

char GNUstepObjCRuntime::ID = 0;

void GNUstepObjCRuntime::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), "GNUstep Objective-C Language Runtime - libobjc2",
      CreateInstance);
}

void GNUstepObjCRuntime::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

/// Returns true if \p module_sp defines (rather than merely references) a
/// function named \p name.
static bool ModuleDefinesFunction(const ModuleSP &module_sp,
                                  llvm::StringRef name) {
  if (!module_sp)
    return false;
  SymbolContextList sc_list;
  module_sp->FindSymbolsWithNameAndType(ConstString(name), eSymbolTypeCode,
                                        sc_list);
  bool defines_function = false;
  for (const SymbolContext &sc : sc_list) {
    // Every module compiled against libobjc2 carries an undefined reference
    // to __objc_load from its .objc_init constructor, so only a definition
    // identifies the runtime itself.
    if (sc.symbol && sc.symbol->GetAddress().IsValid()) {
      defines_function = true;
      break;
    }
  }
  if (!defines_function)
    return false;
  // On PE/COFF an importing module contains an import thunk that carries the
  // imported function's plain name and a valid code address, which the check
  // above cannot tell apart from a definition. Only the importer also has the
  // IAT pointer symbol `__imp_<name>`; the implementing module does not.
  SymbolContextList imp_list;
  module_sp->FindSymbolsWithNameAndType(ConstString(("__imp_" + name).str()),
                                        eSymbolTypeAny, imp_list);
  for (const SymbolContext &sc : imp_list)
    if (sc.symbol && sc.symbol->GetAddress().IsValid())
      return false;
  return true;
}

/// Finds the module implementing the libobjc2 runtime, identified by its
/// loader entry point. __objc_load is exported by every libobjc2 build on
/// every platform and does not exist in GCC's Objective-C runtime, so this
/// both avoids activating for an unrelated runtime and recognizes builds the
/// file name does not identify: a renamed library (LIBOBJC_NAME) or a static
/// libobjc2, whose symbols land in the executable itself.
static ModuleSP FindGNUstepObjCRuntimeModule(const ModuleList &modules) {
  std::lock_guard<std::recursive_mutex> guard(modules.GetMutex());
  const size_t num_modules = modules.GetSize();
  for (size_t i = 0; i < num_modules; i++) {
    ModuleSP module_sp = modules.GetModuleAtIndex(i);
    if (ModuleDefinesFunction(module_sp, "__objc_load"))
      return module_sp;
  }
  return ModuleSP();
}

LanguageRuntime *GNUstepObjCRuntime::CreateInstance(Process *process,
                                                    LanguageType language) {
  if (language != eLanguageTypeObjC)
    return nullptr;
  if (!process)
    return nullptr;

  Target &target = process->GetTarget();
  const llvm::Triple &TT = target.GetArchitecture().GetTriple();
  if (TT.getVendor() == llvm::Triple::VendorType::Apple)
    return nullptr;

  if (!FindGNUstepObjCRuntimeModule(target.GetImages()))
    return nullptr;

  return new GNUstepObjCRuntime(process);
}

GNUstepObjCRuntime::~GNUstepObjCRuntime() = default;

GNUstepObjCRuntime::GNUstepObjCRuntime(Process *process)
    : ObjCLanguageRuntime(process), m_objc_module_sp(nullptr),
      m_tagged_pointer_vendor_up(
          std::make_unique<GNUstepTaggedPointerVendor>(*process)) {
  ReadObjCLibraryIfNeeded(process->GetTarget().GetImages());
}

Address *GNUstepObjCRuntime::GetPrintForDebuggerAddr() {
  if (!m_print_for_debugger_addr_up) {
    SymbolContextList sc_list;
    GetTargetRef().GetImages().FindSymbolsWithNameAndType(
        ConstString("_NSPrintForDebugger"), eSymbolTypeCode, sc_list);
    for (const SymbolContext &sc : sc_list) {
      if (!sc.symbol)
        continue;
      m_print_for_debugger_addr_up =
          std::make_unique<Address>(sc.symbol->GetAddress());
      break;
    }
  }
  return m_print_for_debugger_addr_up.get();
}

llvm::Error GNUstepObjCRuntime::GetObjectDescription(Stream &str,
                                                     ValueObject &valobj) {
  CompilerType compiler_type(valobj.GetCompilerType());
  bool is_signed;
  // ObjC objects can only be pointers (or numbers that actually represent
  // pointers but haven't been typecast).
  if (!compiler_type.IsIntegerType(is_signed) && !compiler_type.IsPointerType())
    return llvm::createStringError("not a pointer type");

  Value val;
  if (!valobj.ResolveValue(val.GetScalar()))
    return llvm::createStringError("pointer value could not be resolved");

  // Value objects may not have a process in their ExecutionContextRef. But
  // we need one in the context we pass down to eventually call description.
  ExecutionContext exe_ctx;
  if (valobj.GetProcessSP()) {
    exe_ctx = ExecutionContext(valobj.GetExecutionContextRef());
  } else {
    exe_ctx.SetContext(valobj.GetTargetSP(), true);
    if (!exe_ctx.HasProcessScope())
      return llvm::createStringError("no process");
  }
  return GetObjectDescription(str, val, exe_ctx.GetBestExecutionContextScope());
}

llvm::Error
GNUstepObjCRuntime::GetObjectDescription(Stream &strm, Value &value,
                                         ExecutionContextScope *exe_scope) {
  // The libobjc2 runtime alone cannot describe objects; the hook lives in
  // gnustep-base (Foundation), just like on Darwin.
  Address *function_address = GetPrintForDebuggerAddr();
  if (!function_address)
    return llvm::createStringError(
        "gnustep-base is not loaded: _NSPrintForDebugger not found");

  ExecutionContext exe_ctx;
  exe_scope->CalculateExecutionContext(exe_ctx);
  Process *process = exe_ctx.GetProcessPtr();
  if (!process)
    return llvm::createStringError("no process");

  Target *target = exe_ctx.GetTargetPtr();
  TypeSystemClangSP scratch_ts_sp =
      ScratchTypeSystemClang::GetForTarget(*target);
  if (!scratch_ts_sp)
    return llvm::createStringError("no scratch type system");

  // The call thunk is compiled as plain C (no ObjC machinery needed in the
  // expression parser), so pass the object as `void *` and read back a
  // `const char *`.
  CompilerType void_ptr_type =
      scratch_ts_sp->GetBasicType(eBasicTypeVoid).GetPointerType();
  value.SetCompilerType(void_ptr_type);

  ValueList arg_value_list;
  arg_value_list.PushValue(value);

  CompilerType return_compiler_type = scratch_ts_sp->GetCStringType(true);
  Value ret;
  ret.SetCompilerType(return_compiler_type);

  if (!exe_ctx.GetFramePtr()) {
    Thread *thread = exe_ctx.GetThreadPtr();
    if (thread == nullptr) {
      exe_ctx.SetThreadSP(process->GetThreadList().GetSelectedThread());
      thread = exe_ctx.GetThreadPtr();
    }
    if (thread)
      exe_ctx.SetFrameSP(thread->GetSelectedFrame(DoNoSelectMostRelevantFrame));
  }

  DiagnosticManager diagnostics;
  lldb::addr_t wrapper_struct_addr = LLDB_INVALID_ADDRESS;

  if (!m_print_object_caller_up) {
    Status error;
    m_print_object_caller_up.reset(
        exe_scope->CalculateTarget()->GetFunctionCallerForLanguage(
            eLanguageTypeC, return_compiler_type, *function_address,
            arg_value_list, "gnustep-object-description", error));
    if (error.Fail()) {
      m_print_object_caller_up.reset();
      return llvm::createStringError(
          llvm::Twine("could not get function runner to call "
                      "_NSPrintForDebugger: ") +
          error.AsCString());
    }
    m_print_object_caller_up->InsertFunction(exe_ctx, wrapper_struct_addr,
                                             diagnostics);
  } else {
    m_print_object_caller_up->WriteFunctionArguments(
        exe_ctx, wrapper_struct_addr, arg_value_list, diagnostics);
  }

  EvaluateExpressionOptions options;
  options.SetUnwindOnError(true);
  options.SetTryAllThreads(true);
  options.SetStopOthers(true);
  options.SetIgnoreBreakpoints(true);
  options.SetTimeout(process->GetUtilityExpressionTimeout());
  options.SetIsForUtilityExpr(true);

  ExpressionResults results = m_print_object_caller_up->ExecuteFunction(
      exe_ctx, &wrapper_struct_addr, options, diagnostics, ret);
  if (results != eExpressionCompleted)
    return llvm::createStringError(
        "could not evaluate _NSPrintForDebugger in the inferior");

  addr_t result_ptr = ret.GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
  if (result_ptr == 0 || result_ptr == LLDB_INVALID_ADDRESS)
    return llvm::createStringError("object returned no description");

  char buf[512];
  size_t cstr_len = 0;
  size_t full_buffer_len = sizeof(buf) - 1;
  size_t curr_len = full_buffer_len;
  while (curr_len == full_buffer_len) {
    Status error;
    curr_len = process->ReadCStringFromMemory(result_ptr + cstr_len, buf,
                                              sizeof(buf), error);
    strm.Write(buf, curr_len);
    cstr_len += curr_len;
  }
  if (cstr_len > 0)
    return llvm::Error::success();
  return llvm::createStringError("empty object description");
}

bool GNUstepObjCRuntime::CouldHaveDynamicValue(ValueObject &in_value) {
  static constexpr bool check_cxx = false;
  static constexpr bool check_objc = true;
  return in_value.GetCompilerType().IsPossibleDynamicType(nullptr, check_cxx,
                                                          check_objc);
}

bool GNUstepObjCRuntime::GetDynamicTypeAndAddress(
    ValueObject &in_value, DynamicValueType use_dynamic,
    TypeAndOrName &class_type_or_name, Address &address,
    Value::ValueType &value_type, llvm::ArrayRef<uint8_t> &local_buffer) {
  class_type_or_name.Clear();
  value_type = Value::ValueType::Scalar;

  if (!CouldHaveDynamicValue(in_value))
    return false;

  ClassDescriptorSP objc_class_sp(GetNonKVOClassDescriptor(in_value));
  if (!objc_class_sp)
    return false;

  // The descriptor was built from the first word of the pointed-to memory.
  // For an instance that word is its class; for a class object it is the
  // metaclass, which libobjc2 gives the same name as the class. Reporting
  // that name here would present the class object as an instance of itself
  // (and, since a root class may declare `id isa`, recurse through it), so
  // values that turn out to be Class have no dynamic type. Every descriptor
  // this runtime creates derives from GNUstepObjCClassDescriptor, so the
  // cast is safe.
  if (static_cast<GNUstepObjCClassDescriptor *>(objc_class_sp.get())
          ->IsMetaclass())
    return false;

  ConstString class_name(objc_class_sp->GetClassName());
  if (!class_name)
    return false;

  const addr_t object_ptr = in_value.GetPointerValue().address;
  address.SetRawAddress(object_ptr);
  class_type_or_name.SetName(class_name);

  // Upgrade the bare class name to a real type when the inferior's debug
  // info defines the class. LookupInCompleteClassCache keys on an
  // eSymbolTypeObjCClass symbol named exactly after the class, which the
  // Apple ABI emits but the gnustep-2.x ABI does not (its class symbol is
  // "._OBJC_CLASS_<name>"), so on a cache miss query the debug info directly.
  TypeSP type_sp(objc_class_sp->GetType());
  if (!type_sp) {
    type_sp = LookupInCompleteClassCache(class_name);
    if (!type_sp)
      type_sp = LookupClassTypeInDebugInfo(class_name);
    if (type_sp)
      objc_class_sp->SetType(type_sp);
  }
  if (type_sp)
    class_type_or_name.SetTypeSP(type_sp);

  return !class_type_or_name.IsEmpty();
}

lldb::TypeSP
GNUstepObjCRuntime::LookupClassTypeInDebugInfo(ConstString class_name) {
  // Searching every module's debug info is expensive and happens for each
  // value on each stop, so remember the classes that have no debug info. The
  // cache is dropped whenever new modules arrive.
  if (m_negative_type_cache.count(class_name))
    return TypeSP();

  TypeQuery query(class_name.GetStringRef(), TypeQueryOptions::e_exact_match);
  TypeResults results;
  GetTargetRef().GetImages().FindTypes(nullptr, query, results);
  for (const TypeSP &type_sp : results.GetTypeMap().Types()) {
    if (type_sp && TypeSystemClang::IsObjCObjectOrInterfaceType(
                       type_sp->GetForwardCompilerType()))
      return type_sp;
  }
  m_negative_type_cache.insert(class_name);
  return TypeSP();
}

bool GNUstepObjCRuntime::CalculateHasNewLiteralsAndIndexing() {
  // The literal and subscripting syntax lowers to calls on Foundation
  // classes, which live in gnustep-base rather than in the runtime itself.
  // Claiming support without them makes such expressions compile and then
  // fail inside the inferior, so require the classes to be present.
  static constexpr llvm::StringLiteral g_required_classes[] = {
      "NSArray", "NSDictionary", "NSNumber", "NSString"};

  const llvm::StringRef prefix = GetClassSymbolPrefix();
  const ModuleList &images = GetTargetRef().GetImages();
  for (llvm::StringRef class_name : g_required_classes) {
    SymbolContextList sc_list;
    images.FindSymbolsWithNameAndType(
        ConstString(prefix.str() + class_name.str()), eSymbolTypeAny, sc_list);
    if (sc_list.GetSize() == 0)
      return false;
  }
  return true;
}

TypeAndOrName
GNUstepObjCRuntime::FixUpDynamicType(const TypeAndOrName &type_and_or_name,
                                     ValueObject &static_value) {
  CompilerType static_type(static_value.GetCompilerType());
  Flags static_type_flags(static_type.GetTypeInfo());

  TypeAndOrName ret(type_and_or_name);
  if (type_and_or_name.HasType()) {
    // The type will always be the type of the dynamic object.  If our parent's
    // type was a pointer, then our type should be a pointer to the type of the
    // dynamic object.  If a reference, then the original type should be
    // okay...
    CompilerType orig_type = type_and_or_name.GetCompilerType();
    CompilerType corrected_type = orig_type;
    if (static_type_flags.AllSet(eTypeIsPointer))
      corrected_type = orig_type.GetPointerType();
    ret.SetCompilerType(corrected_type);
  } else {
    // If we are here we need to adjust our dynamic type name to include the
    // correct & or * symbol
    std::string corrected_name(type_and_or_name.GetName().GetCString());
    if (static_type_flags.AllSet(eTypeIsPointer))
      corrected_name.append(" *");
    // the parent type should be a correctly pointer'ed or referenc'ed type
    ret.SetCompilerType(static_type);
    ret.SetName(corrected_name.c_str());
  }
  return ret;
}

BreakpointResolverSP
GNUstepObjCRuntime::CreateExceptionResolver(const BreakpointSP &bkpt,
                                            bool catch_bp, bool throw_bp) {
  BreakpointResolverSP resolver_sp;

  if (throw_bp)
    resolver_sp = std::make_shared<BreakpointResolverName>(
        bkpt, "objc_exception_throw", eFunctionNameTypeBase,
        eLanguageTypeUnknown, Breakpoint::Exact, 0,
        /*offset_is_insn_count = */ false, eLazyBoolNo);

  return resolver_sp;
}

llvm::Expected<std::unique_ptr<UtilityFunction>>
GNUstepObjCRuntime::CreateObjectChecker(std::string name,
                                        ExecutionContext &exe_ctx) {
  // TODO: This function is supposed to check whether an ObjC selector is
  // present for an object. Might be implemented similar as in the Apple V2
  // runtime.
  const char *function_template = R"(
    extern "C" void
    %s(void *$__lldb_arg_obj, void *$__lldb_arg_selector) {}
  )";

  char empty_function_code[2048];
  int len = ::snprintf(empty_function_code, sizeof(empty_function_code),
                       function_template, name.c_str());

  assert(len < (int)sizeof(empty_function_code));
  UNUSED_IF_ASSERT_DISABLED(len);

  return GetTargetRef().CreateUtilityFunction(empty_function_code, name,
                                              eLanguageTypeC, exe_ctx);
}

ThreadPlanSP
GNUstepObjCRuntime::GetStepThroughTrampolinePlan(Thread &thread,
                                                 bool stop_others) {
  // Only act when stopped at the first instruction of a known libobjc2
  // dispatch entry point (where the argument registers still hold the
  // receiver and selector). Match by resolving each entry point's address by
  // name rather than by the symbol at the PC: libobjc2's hand-written
  // assembly places local labels (e.g. __objc_block_trampoline_end_sret) at
  // the same address as objc_msgSend, so the symbol found at an address is
  // not reliably the dispatch symbol.
  Process *process = thread.GetProcess().get();
  if (!process)
    return {};
  const addr_t pc = thread.GetRegisterContext()->GetPC();
  Target &target = GetTargetRef();

  std::optional<DispatchEntryPoint> entry = FindDispatchEntryPoint(pc);
  if (!entry)
    return {};
  const bool is_stret = entry->is_stret;
  const bool is_sender = entry->is_sender;

  ABISP abi_sp = process->GetABI();
  if (!abi_sp)
    return {};
  TypeSystemClangSP scratch_ts_sp =
      ScratchTypeSystemClang::GetForTarget(target);
  if (!scratch_ts_sp)
    return {};
  CompilerType void_ptr_type =
      scratch_ts_sp->GetBasicType(eBasicTypeVoid).GetPointerType();
  Value void_ptr_value;
  void_ptr_value.SetValueType(Value::ValueType::Scalar);
  void_ptr_value.SetCompilerType(void_ptr_type);

  ValueList argument_values;
  argument_values.PushValue(void_ptr_value);
  argument_values.PushValue(void_ptr_value);
  argument_values.PushValue(void_ptr_value);
  if (!abi_sp->GetArgumentValues(thread, argument_values))
    return {};

  // With struct return the sret pointer occupies the first argument slot.
  const uint32_t receiver_idx = is_stret ? 1 : 0;
  const uint32_t sel_idx = is_stret ? 2 : 1;
  addr_t receiver =
      argument_values.GetValueAtIndex(receiver_idx)->GetScalar().ULongLong();
  const addr_t selector =
      argument_values.GetValueAtIndex(sel_idx)->GetScalar().ULongLong();

  if (is_sender) {
    // objc_msg_lookup_sender takes `id *receiver`.
    Status error;
    receiver = process->ReadPointerFromMemory(receiver, error);
    if (error.Fail())
      return {};
  }

  // A message to nil does not dispatch anywhere.
  if (receiver == 0 || receiver == LLDB_INVALID_ADDRESS)
    return {};

  // Consult the method cache before running anything in the inferior.
  // Tagged pointers skip the cache: their ISA is not the object's first word.
  addr_t isa = LLDB_INVALID_ADDRESS;
  if (!(m_tagged_pointer_vendor_up &&
        m_tagged_pointer_vendor_up->IsPossibleTaggedPointer(receiver))) {
    Status error;
    const addr_t isa_candidate =
        process->ReadPointerFromMemory(receiver, error);
    if (error.Success())
      isa = isa_candidate;
  }
  if (isa != LLDB_INVALID_ADDRESS) {
    const addr_t cached_imp = LookupInMethodCache(isa, selector);
    if (cached_imp != LLDB_INVALID_ADDRESS) {
      Address imp_addr;
      imp_addr.SetOpcodeLoadAddress(cached_imp, &target);
      return std::make_shared<ThreadPlanRunToAddress>(thread, imp_addr,
                                                      stop_others);
    }
  }

  // Only claim the step if the runtime actually exports the lookup function.
  // Building the call wrapper is deliberately left to the plan's pre-resume
  // action, which is where running code in the inferior is safe.
  if (!ModuleDefinesFunction(m_objc_module_sp, "objc_msg_lookup"))
    return {};

  ValueList lookup_args;
  Value receiver_value = void_ptr_value;
  receiver_value.GetScalar() = receiver;
  lookup_args.PushValue(receiver_value);
  Value selector_value = void_ptr_value;
  selector_value.GetScalar() = selector;
  lookup_args.PushValue(selector_value);

  return std::make_shared<GNUstepThreadPlanStepThroughObjCTrampoline>(
      thread, *this, lookup_args, isa, selector);
}

std::optional<GNUstepObjCRuntime::DispatchEntryPoint>
GNUstepObjCRuntime::FindDispatchEntryPoint(lldb::addr_t pc) {
  if (!m_dispatch_entry_points_resolved) {
    m_dispatch_entry_points_resolved = true;
    // Dispatch entry points exported by libobjc2 (objc_msgSend.S,
    // sendmsg2.c). The `_super` variants are omitted: super sends compile to
    // a lookup plus a direct call, and the direct call steps normally.
    static const struct {
      const char *name;
      bool is_stret;
      bool is_sender;
    } g_entry_points[] = {
        {"objc_msgSend", false, false},
        {"objc_msgSend_fpret", false, false},
        {"objc_msgSend_stret", true, false},
        // Windows on ARM64 dispatches struct returns through this variant.
        {"objc_msgSend_stret2", true, false},
        {"objc_msg_lookup", false, false},
        {"objc_msg_lookup_sender", false, true},
    };
    Target &target = GetTargetRef();
    for (const auto &ep : g_entry_points) {
      SymbolContextList sc_list;
      target.GetImages().FindSymbolsWithNameAndType(ConstString(ep.name),
                                                    eSymbolTypeCode, sc_list);
      for (const SymbolContext &sc : sc_list) {
        if (!sc.symbol)
          continue;
        // Use the opcode address so the comparison against the PC is correct
        // on targets where the symbol address carries an ISA bit (Thumb).
        const addr_t addr =
            sc.symbol->GetAddress().GetOpcodeLoadAddress(&target);
        if (addr != LLDB_INVALID_ADDRESS)
          m_dispatch_entry_points.push_back({addr, ep.is_stret, ep.is_sender});
      }
    }
  }

  for (const DispatchEntryPoint &ep : m_dispatch_entry_points)
    if (ep.address == pc)
      return ep;
  return std::nullopt;
}

FunctionCaller *GNUstepObjCRuntime::GetMsgLookupFunctionCaller(Thread &thread) {
  // Build (once) a utility function that resolves a method implementation by
  // calling libobjc2's objc_msg_lookup, and a FunctionCaller to invoke it.
  // This mirrors AppleObjCTrampolineHandler's dispatch-lookup utility.
  static const char *g_lookup_name = "$__lldb_gnustep_objc_msg_lookup";
  static const char *g_lookup_code =
      "void *objc_msg_lookup(void *receiver, void *selector);\n"
      "void *$__lldb_gnustep_objc_msg_lookup(void *receiver, void *selector) "
      "{\n"
      "  return objc_msg_lookup(receiver, selector);\n"
      "}\n";

  std::lock_guard<std::mutex> guard(m_msg_lookup_mutex);
  if (m_msg_lookup_caller)
    return m_msg_lookup_caller;
  // Don't pay for compiling the wrapper again on every step once it is known
  // not to work in this process.
  if (m_msg_lookup_failed)
    return nullptr;

  ThreadSP thread_sp(thread.shared_from_this());
  ExecutionContext exe_ctx(thread_sp);
  Log *log = GetLog(LLDBLog::Step);

  auto utility_fn_or_error = exe_ctx.GetTargetRef().CreateUtilityFunction(
      g_lookup_code, g_lookup_name, eLanguageTypeC, exe_ctx);
  if (!utility_fn_or_error) {
    LLDB_LOG_ERROR(log, utility_fn_or_error.takeError(),
                   "failed to build objc_msg_lookup utility: {0}");
    m_msg_lookup_failed = true;
    return nullptr;
  }
  m_msg_lookup_utility_up = std::move(*utility_fn_or_error);

  TypeSystemClangSP scratch_ts_sp =
      ScratchTypeSystemClang::GetForTarget(GetTargetRef());
  if (!scratch_ts_sp) {
    m_msg_lookup_failed = true;
    return nullptr;
  }
  CompilerType void_ptr_type =
      scratch_ts_sp->GetBasicType(eBasicTypeVoid).GetPointerType();

  Value void_ptr_value;
  void_ptr_value.SetValueType(Value::ValueType::Scalar);
  void_ptr_value.SetCompilerType(void_ptr_type);
  ValueList args;
  args.PushValue(void_ptr_value);
  args.PushValue(void_ptr_value);

  Status error;
  m_msg_lookup_caller = m_msg_lookup_utility_up->MakeFunctionCaller(
      void_ptr_type, args, thread_sp, error);
  if (error.Fail()) {
    LLDB_LOG(log, "failed to make objc_msg_lookup caller: {0}",
             error.AsCString());
    m_msg_lookup_caller = nullptr;
    m_msg_lookup_failed = true;
    return nullptr;
  }
  return m_msg_lookup_caller;
}

void GNUstepObjCRuntime::UpdateISAToDescriptorMapIfNeeded() {
  if (!m_process)
    return;
  const uint32_t stop_id = m_process->GetStopID();
  if (!m_isa_map_dirty) {
    m_isa_to_descriptor_stop_id = stop_id;
    return;
  }

  // The first update has to look at everything already loaded; afterwards only
  // the modules that arrived since need scanning, so a dlopen does not re-walk
  // every symbol table in the process.
  if (m_swept_all_modules) {
    for (const ModuleSP &module_sp : m_pending_modules)
      AddClassesFromModule(module_sp);
  } else {
    const ModuleList &images = GetTargetRef().GetImages();
    std::lock_guard<std::recursive_mutex> guard(images.GetMutex());
    const size_t num_modules = images.GetSize();
    for (size_t i = 0; i < num_modules; i++)
      AddClassesFromModule(images.GetModuleAtIndex(i));
    m_swept_all_modules = true;
  }

  m_pending_modules.clear();
  m_isa_map_dirty = false;
  m_isa_to_descriptor_stop_id = stop_id;
}

bool GNUstepObjCRuntime::IsRuntimeInternalAddress(lldb::addr_t addr) {
  if (!m_objc_module_sp || addr == 0 || addr == LLDB_INVALID_ADDRESS)
    return false;
  // Statically linked, the runtime's module *is* the executable, so module
  // identity says nothing about whether an address is the runtime's own code
  // or the user's. Answering "internal" there would report every method in
  // the program as unsteppable and strand `step` at the dispatch entry point,
  // so decline to guess: stepping into the runtime's forwarding machinery is
  // a much smaller problem than never stepping into a method at all.
  if (m_objc_module_sp == GetTargetRef().GetExecutableModule())
    return false;
  Address resolved;
  if (!GetTargetRef().ResolveLoadAddress(addr, resolved))
    return false;
  return resolved.GetModule() == m_objc_module_sp;
}

llvm::StringRef GNUstepObjCRuntime::GetClassSymbolPrefix() {
  // clang mangles the public runtime symbols with a leading "._" on every
  // object format except COFF, which uses "$_" (CGObjCGNU.cpp).
  return GetTargetRef().GetArchitecture().GetTriple().isOSBinFormatCOFF()
             ? "$_OBJC_CLASS_"
             : "._OBJC_CLASS_";
}

void GNUstepObjCRuntime::AddClassesFromModule(const ModuleSP &module_sp) {
  if (!module_sp || !m_process)
    return;

  // The gnustep-2.x ABI emits every compiled class as a `<prefix>OBJC_CLASS_`
  // data symbol whose address is the class object itself (the ISA of its
  // instances), so the map can be seeded from symbol tables alone - without
  // running any code in the inferior. Classes created dynamically at runtime
  // are handled by the create-on-miss path in GetClassDescriptorFromISA.
  const llvm::StringRef prefix = GetClassSymbolPrefix();
  RegularExpression regex("^" + llvm::Regex::escape(prefix));
  SymbolContextList sc_list;
  module_sp->FindSymbolsMatchingRegExAndType(regex, eSymbolTypeAny, sc_list);

  Target &target = GetTargetRef();
  for (const SymbolContext &sc : sc_list) {
    if (!sc.symbol)
      continue;
    const addr_t isa = sc.symbol->GetAddress().GetLoadAddress(&target);
    if (isa == 0 || isa == LLDB_INVALID_ADDRESS || ISAIsCached(isa))
      continue;
    llvm::StringRef name = sc.symbol->GetName().GetStringRef();
    name.consume_front(prefix);
    auto descriptor_sp = std::make_shared<GNUstepObjCClassDescriptor>(
        m_process->shared_from_this(), isa);
    if (descriptor_sp->IsValid())
      AddClass(isa, descriptor_sp, name.str().c_str());
  }
}

bool GNUstepObjCRuntime::GetIRPasses(
    LLVMUserExpression::IRPasses &custom_passes) {
  custom_passes.EarlyPasses = std::make_shared<llvm::legacy::PassManager>();
  custom_passes.EarlyPasses->add(new GNUstepObjCSelectorRegistrationPass());
  return true;
}

ObjCLanguageRuntime::TaggedPointerVendor *
GNUstepObjCRuntime::GetTaggedPointerVendor() {
  return m_tagged_pointer_vendor_up.get();
}

ObjCLanguageRuntime::ClassDescriptorSP
GNUstepObjCRuntime::GetClassDescriptor(ValueObject &in_value) {
  const addr_t ptr = in_value.GetPointerValue().address;
  if (ptr != LLDB_INVALID_ADDRESS && m_tagged_pointer_vendor_up &&
      m_tagged_pointer_vendor_up->IsPossibleTaggedPointer(ptr))
    return m_tagged_pointer_vendor_up->GetClassDescriptor(ptr);
  return ObjCLanguageRuntime::GetClassDescriptor(in_value);
}

ObjCLanguageRuntime::ClassDescriptorSP
GNUstepObjCRuntime::GetClassDescriptorFromISA(ObjCISA isa) {
  if (ClassDescriptorSP descriptor_sp =
          ObjCLanguageRuntime::GetClassDescriptorFromISA(isa))
    return descriptor_sp;

  // The symbol sweep only sees classes with static definitions. Fall back to
  // parsing the class structure directly so classes registered at runtime
  // (e.g. via objc_allocateClassPair) resolve as well.
  if (!m_process || isa == 0 || isa == LLDB_INVALID_ADDRESS)
    return ClassDescriptorSP();
  auto descriptor_sp = std::make_shared<GNUstepObjCClassDescriptor>(
      m_process->shared_from_this(), isa);
  if (!descriptor_sp->IsValid())
    return ClassDescriptorSP();
  AddClass(isa, descriptor_sp, descriptor_sp->GetClassName().GetCString());
  return descriptor_sp;
}

bool GNUstepObjCRuntime::IsModuleObjCLibrary(const ModuleSP &module_sp) {
  return ModuleDefinesFunction(module_sp, "__objc_load");
}

bool GNUstepObjCRuntime::ReadObjCLibrary(const ModuleSP &module_sp) {
  assert(m_objc_module_sp == nullptr && "Check HasReadObjCLibrary() first");
  m_objc_module_sp = module_sp;

  // Right now we don't use this, but we might want to check for debugger
  // runtime support symbols like 'gdb_object_getClass' in the future.
  return true;
}

StructuredData::ObjectSP
GNUstepObjCRuntime::GetLanguageSpecificData(SymbolContext sc) {
  auto dict_up = std::make_unique<StructuredData::Dictionary>();
  dict_up->AddItem("Objective-C runtime version",
                   std::make_unique<StructuredData::UnsignedInteger>(2));
  return dict_up;
}

void GNUstepObjCRuntime::ModulesDidLoad(const ModuleList &module_list) {
  ReadObjCLibraryIfNeeded(module_list);

  // Everything cached from a symbol lookup can be invalidated by new modules:
  // classes to add to the map, dispatch entry points that may only now exist,
  // and negative results that may now resolve.
  {
    std::lock_guard<std::recursive_mutex> guard(module_list.GetMutex());
    const size_t num_modules = module_list.GetSize();
    for (size_t i = 0; i < num_modules; i++)
      m_pending_modules.push_back(module_list.GetModuleAtIndex(i));
  }
  m_isa_map_dirty = true;
  m_dispatch_entry_points.clear();
  m_dispatch_entry_points_resolved = false;
  m_negative_type_cache.clear();
  if (m_tagged_pointer_vendor_up)
    m_tagged_pointer_vendor_up->ModulesDidLoad();
}
