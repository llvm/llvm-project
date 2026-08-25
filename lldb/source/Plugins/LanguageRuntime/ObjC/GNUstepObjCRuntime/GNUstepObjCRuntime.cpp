//===-- GNUstepObjCRuntime.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GNUstepObjCRuntime.h"
#include "GNUstepObjCClassDescriptor.h"
#include "GNUstepObjCDeclVendor.h"
#include "GNUstepObjCTypeEncodingParser.h"
#include "GNUstepThreadPlanStepThroughObjCTrampoline.h"

#include "Plugins/Process/Utility/HistoryThread.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"

#include "lldb/Breakpoint/BreakpointList.h"
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
#include "lldb/Symbol/Symtab.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StackFrameRecognizer.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlanRunToAddress.h"
#include "lldb/Target/ThreadPlanStepOut.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegularExpression.h"
#include "lldb/Utility/StructuredData.h"
#include "lldb/ValueObject/ValueObject.h"
#include "lldb/ValueObject/ValueObjectConstResult.h"
#include "lldb/ValueObject/ValueObjectList.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(GNUstepObjCRuntime)

// An ISA chain deeper than this is not something a compiler produced. It comes
// from inferior memory, so it is bounded rather than trusted.
static constexpr uint32_t g_max_superclass_depth = 256;

// A type encoding longer than this is not something a compiler produced.
// Bounded for the same reason as the depth above.
static constexpr size_t g_max_type_encoding_length = 1024;

/// gnustep-base captures at most MAXFRAMES (NSException.m) return addresses.
static constexpr uint64_t g_max_stack_frames = 128;

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

/// Presents the object being thrown as an `exception` argument on a frame
/// stopped at libobjc2's throw entry point, which is what `thread exception`
/// and lldb-dap's exception view read.
class GNUstepObjCExceptionRecognizedStackFrame : public RecognizedStackFrame {
public:
  explicit GNUstepObjCExceptionRecognizedStackFrame(StackFrameSP frame_sp) {
    ThreadSP thread_sp = frame_sp->GetThread();
    if (!thread_sp)
      return;
    ProcessSP process_sp = thread_sp->GetProcess();
    if (!process_sp)
      return;

    const ABISP &abi = process_sp->GetABI();
    if (!abi)
      return;

    TypeSystemClangSP scratch_ts_sp =
        ScratchTypeSystemClang::GetForTarget(process_sp->GetTarget());
    if (!scratch_ts_sp)
      return;
    CompilerType void_ptr_type =
        scratch_ts_sp->GetBasicType(eBasicTypeVoid).GetPointerType();

    // `void objc_exception_throw(id object)` on every libobjc2 exception
    // back-end, so the thrown object is argument 0 regardless of how the
    // unwinder is implemented on this platform.
    ValueList args;
    Value input_value;
    input_value.SetCompilerType(void_ptr_type);
    args.PushValue(input_value);
    if (!abi->GetArgumentValues(*thread_sp, args))
      return;

    Value value(args.GetValueAtIndex(0)->GetScalar().ULongLong());
    value.SetCompilerType(void_ptr_type);
    m_exception_sp = ValueObjectConstResult::Create(frame_sp.get(), value,
                                                    ConstString("exception"));
    m_exception_sp = ValueObjectRecognizerSynthesizedValue::Create(
        *m_exception_sp, eValueTypeVariableArgument);
    m_exception_sp = m_exception_sp->GetDynamicValue(eDynamicDontRunTarget);

    m_arguments = std::make_shared<ValueObjectList>();
    m_arguments->Append(m_exception_sp);
    m_stop_desc = "hit Objective-C exception";
  }

  ValueObjectSP GetExceptionObject() override { return m_exception_sp; }

private:
  ValueObjectSP m_exception_sp;
};

class GNUstepObjCExceptionThrowFrameRecognizer : public StackFrameRecognizer {
public:
  RecognizedStackFrameSP RecognizeFrame(StackFrameSP frame) override {
    return std::make_shared<GNUstepObjCExceptionRecognizedStackFrame>(frame);
  }
  std::string GetName() override {
    return "GNUstep ObjC Exception Throw StackFrame Recognizer";
  }
};

/// The runtime library's file name differs by platform (libobjc.so.4.6,
/// objc.dll, ...), so the recognizer is registered without a module filter
/// and matched on the symbol alone; an empty module ConstString matches any
/// module (StackFrameRecognizer.cpp).
///
/// The mangled name is what has to match. libobjc2's Windows exception
/// back-end is C++ (eh_win32_msvc.cc), so where the runtime carries debug
/// info the demangled name of this extern "C" function is
/// `::objc_exception_throw(id)` rather than the bare symbol. Preferring the
/// mangled name gives `objc_exception_throw` there, and falls back to the
/// same string through the symbol when there is no debug info at all.
void RegisterGNUstepObjCExceptionRecognizer(Process *process) {
  static const std::vector<ConstString> g_symbols = {
      ConstString("objc_exception_throw")};
  static const std::string g_name =
      GNUstepObjCExceptionThrowFrameRecognizer().GetName();

  // A runtime is created per Process but recognizers live on the Target, so
  // re-running the same target would otherwise stack up a copy per run, each
  // consulted for every frame.
  StackFrameRecognizerManager &manager =
      process->GetTarget().GetFrameRecognizerManager();
  bool already_registered = false;
  manager.ForEach([&](uint32_t, bool, std::string name, std::string,
                      llvm::ArrayRef<ConstString>, Mangled::NamePreference,
                      bool) {
    if (name == g_name)
      already_registered = true;
  });
  if (already_registered)
    return;

  manager.AddRecognizer(
      std::make_shared<GNUstepObjCExceptionThrowFrameRecognizer>(),
      ConstString(), g_symbols, Mangled::NamePreference::ePreferMangled,
      /*first_instruction_only=*/true);
}
} // namespace

char GNUstepObjCRuntime::ID = 0;

void GNUstepObjCRuntime::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), "GNUstep Objective-C Language Runtime - libobjc2",
      CreateInstance, /*command_callback=*/nullptr,
      GetBreakpointExceptionPrecondition);
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

  RegisterGNUstepObjCExceptionRecognizer(process);
  return new GNUstepObjCRuntime(process);
}

GNUstepObjCRuntime::~GNUstepObjCRuntime() {
  // The encoding parser caches clang::QualTypes keyed by the TypeSystemClang
  // that minted them, and one of those ASTs belongs to the decl vendor. A
  // QualType is a tagged pointer and dropping the cache never dereferences
  // one, so this is ordering hygiene rather than a live hazard - but state it
  // here instead of leaving it to depend on member declaration order.
  m_encoding_to_type_sp.reset();
  m_decl_vendor_up.reset();
}

GNUstepObjCRuntime::GNUstepObjCRuntime(Process *process)
    : ObjCLanguageRuntime(process), m_objc_module_sp(nullptr),
      m_tagged_pointer_vendor_up(
          std::make_unique<GNUstepTaggedPointerVendor>(*process)) {
  ReadObjCLibraryIfNeeded(process->GetTarget().GetImages());
}

FunctionCaller *
GNUstepObjCRuntime::GetObjectDescriptionCaller(ExecutionContext &exe_ctx) {
  // gnustep-base's _NSPrintForDebugger (Source/NSDebug.m) is exactly:
  //
  //   if (object && [object respondsToSelector: @selector(description)])
  //     return [[object description] UTF8String];
  //   return NULL;
  //
  // Every runtime function it needs is OBJC_PUBLIC in libobjc2, so rather
  // than depending on Foundation exporting that hook - it does from
  // libgnustep-base.so but not from the MSVC gnustep-base DLL, which is why
  // this used to need a shim on Windows - reproduce it here. As a bonus this
  // makes `po` work against a bare libobjc2 with no Foundation at all, for
  // any object implementing -description.
  //
  // object_getClass() rather than a raw isa read, because it also resolves
  // tagged pointers (classForObject) and skips libobjc2's hidden classes.
  static const char *g_description_name = "$__lldb_gnustep_object_description";
  static const char *g_description_code =
      "void *object_getClass(void *object);\n"
      "void *sel_registerName(const char *name);\n"
      "signed char class_respondsToSelector(void *cls, void *sel);\n"
      "void *objc_msg_lookup(void *receiver, void *selector);\n"
      "\n"
      "const char *$__lldb_gnustep_object_description(void *object) {\n"
      "  if (!object)\n"
      "    return 0;\n"
      "  void *description_sel = sel_registerName(\"description\");\n"
      "  if (!class_respondsToSelector(object_getClass(object), "
      "description_sel))\n"
      "    return 0;\n"
      "  void *(*description_imp)(void *, void *) =\n"
      "      (void *(*)(void *, void *))objc_msg_lookup(object, "
      "description_sel);\n"
      "  if (!description_imp)\n"
      "    return 0;\n"
      "  void *description = description_imp(object, description_sel);\n"
      "  if (!description)\n"
      "    return 0;\n"
      "  void *utf8_sel = sel_registerName(\"UTF8String\");\n"
      "  if (!class_respondsToSelector(object_getClass(description), "
      "utf8_sel))\n"
      "    return 0;\n"
      "  const char *(*utf8_imp)(void *, void *) =\n"
      "      (const char *(*)(void *, void *))objc_msg_lookup(description, "
      "utf8_sel);\n"
      "  if (!utf8_imp)\n"
      "    return 0;\n"
      "  return utf8_imp(description, utf8_sel);\n"
      "}\n";

  std::lock_guard<std::mutex> guard(m_description_mutex);
  if (m_description_caller)
    return m_description_caller;
  // Don't pay for compiling the wrapper again once it is known not to work
  // in this process.
  if (m_description_failed)
    return nullptr;

  Log *log = GetLog(LLDBLog::Expressions);

  auto utility_fn_or_error = exe_ctx.GetTargetRef().CreateUtilityFunction(
      g_description_code, g_description_name, eLanguageTypeC, exe_ctx);
  if (!utility_fn_or_error) {
    LLDB_LOG_ERROR(log, utility_fn_or_error.takeError(),
                   "failed to build object description utility: {0}");
    m_description_failed = true;
    return nullptr;
  }
  m_description_utility_up = std::move(*utility_fn_or_error);

  TypeSystemClangSP scratch_ts_sp =
      ScratchTypeSystemClang::GetForTarget(GetTargetRef());
  if (!scratch_ts_sp) {
    m_description_failed = true;
    return nullptr;
  }

  Value void_ptr_value;
  void_ptr_value.SetValueType(Value::ValueType::Scalar);
  void_ptr_value.SetCompilerType(
      scratch_ts_sp->GetBasicType(eBasicTypeVoid).GetPointerType());
  ValueList args;
  args.PushValue(void_ptr_value);

  Status error;
  m_description_caller = m_description_utility_up->MakeFunctionCaller(
      scratch_ts_sp->GetCStringType(true), args, exe_ctx.GetThreadSP(), error);
  if (error.Fail()) {
    LLDB_LOG(log, "failed to make object description caller: {0}",
             error.AsCString());
    m_description_caller = nullptr;
    m_description_failed = true;
    return nullptr;
  }
  return m_description_caller;
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

  // Building the caller needs a thread, so this has to follow the frame
  // selection above. MakeFunctionCaller already compiled the wrapper into
  // the inferior, so only a fresh argument struct is needed per call.
  FunctionCaller *caller = GetObjectDescriptionCaller(exe_ctx);
  if (!caller)
    return llvm::createStringError(
        "could not build the object description function");

  if (!caller->WriteFunctionArguments(exe_ctx, wrapper_struct_addr,
                                      arg_value_list, diagnostics))
    return llvm::createStringError(
        "could not write the object description arguments");

  EvaluateExpressionOptions options;
  options.SetUnwindOnError(true);
  options.SetTryAllThreads(true);
  options.SetStopOthers(true);
  options.SetIgnoreBreakpoints(true);
  options.SetTimeout(process->GetUtilityExpressionTimeout());
  options.SetIsForUtilityExpr(true);

  ExpressionResults results = caller->ExecuteFunction(
      exe_ctx, &wrapper_struct_addr, options, diagnostics, ret);
  if (results != eExpressionCompleted)
    return llvm::createStringError(
        "could not evaluate the object description in the inferior");

  addr_t result_ptr = ret.GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
  if (result_ptr == 0 || result_ptr == LLDB_INVALID_ADDRESS)
    return llvm::createStringError("object returned no description");

  // -description returns inferior memory, so a string that never terminates
  // would otherwise be read until the host runs out of it. No real
  // description approaches this.
  static constexpr size_t g_max_description_length = 1024 * 1024;

  char buf[512];
  size_t cstr_len = 0;
  size_t full_buffer_len = sizeof(buf) - 1;
  size_t curr_len = full_buffer_len;
  while (curr_len == full_buffer_len && cstr_len < g_max_description_length) {
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
  if (type_sp) {
    class_type_or_name.SetTypeSP(type_sp);
  } else if (CompilerType runtime_type = LookupClassTypeInRuntime(class_name)) {
    // Nothing in the debug info describes this class - it was compiled -g0, or
    // registered at run time and never had a static definition at all. The
    // runtime's own metadata still does. Without this the result is a name
    // with no type, and FixUpDynamicType then pairs that name with the
    // *static* CompilerType, which ValueObjectDynamicValue::GetTypeName
    // prefers - so the value displays as its declared type, `id`.
    //
    // Last, deliberately: where debug info exists it wins, because runtime
    // metadata cannot describe members inside a struct-typed ivar.
    class_type_or_name.SetCompilerType(runtime_type);
  }

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

CompilerType
GNUstepObjCRuntime::LookupClassTypeInRuntime(ConstString class_name) {
  DeclVendor *vendor = GetDeclVendor();
  if (!vendor)
    return CompilerType();

  std::vector<CompilerDecl> decls;
  vendor->FindDecls(class_name, /*append=*/false, /*max_matches=*/1, decls);
  if (decls.empty())
    return CompilerType();

  auto *ast = llvm::dyn_cast<TypeSystemClang>(decls[0].GetTypeSystem());
  if (!ast)
    return CompilerType();
  return ast->GetTypeForDecl(decls[0].GetOpaqueDecl());
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
  std::vector<std::string> names;
  if (throw_bp)
    names.emplace_back("objc_exception_throw");

  // Unlike Apple's runtime, libobjc2 has an entry point for entering a
  // handler - but only where exceptions unwind through the Itanium ABI. The
  // MSVC build raises a native SEH exception instead (eh_win32_msvc.cc) and
  // catch is a __CxxFrameHandler3 funclet with no symbol to break on, so ask
  // the runtime module rather than the target triple.
  if (catch_bp && ModuleDefinesFunction(m_objc_module_sp, "objc_begin_catch")) {
    names.emplace_back("objc_begin_catch");
    // For gnustep-2.x on MinGW clang routes @catch through the C++ ABI
    // instead (CGObjCGNU.cpp, usesCxxExceptions), leaving libobjc2's entry
    // point exported but never reached. Which one a program calls depends on
    // how it was compiled, so offer both; the unused name never resolves.
    // That entry point is shared with C++ catch, so on MinGW this stops on
    // those too.
    //
    // isOSWindows(), not the environment: LLDB reports a MinGW PE as msvc
    // unless plugin.object-file.pe-coff.abi says otherwise.
    if (GetTargetRef().GetArchitecture().GetTriple().isOSWindows())
      names.emplace_back("__cxa_begin_catch");
  }

  if (names.empty())
    return {};

  return std::make_shared<BreakpointResolverName>(
      bkpt, names, eFunctionNameTypeBase, eLanguageTypeUnknown,
      /*offset=*/0, /*skip_prologue=*/eLazyBoolNo);
}

void GNUstepObjCRuntime::SetExceptionBreakpoints() {
  if (!m_process)
    return;

  const bool catch_bp = false;
  const bool throw_bp = true;
  const bool is_internal = true;

  if (!m_objc_exception_bp_sp) {
    m_objc_exception_bp_sp = LanguageRuntime::CreateExceptionBreakpoint(
        m_process->GetTarget(), GetLanguageType(), catch_bp, throw_bp,
        is_internal);
    if (m_objc_exception_bp_sp)
      m_objc_exception_bp_sp->SetBreakpointKind("ObjC exception");
  } else {
    m_objc_exception_bp_sp->SetEnabled(true);
  }
}

void GNUstepObjCRuntime::ClearExceptionBreakpoints() {
  if (!m_process)
    return;

  if (m_objc_exception_bp_sp)
    m_objc_exception_bp_sp->SetEnabled(false);
}

bool GNUstepObjCRuntime::ExceptionBreakpointsAreSet() {
  return m_objc_exception_bp_sp && m_objc_exception_bp_sp->IsEnabled();
}

bool GNUstepObjCRuntime::ExceptionBreakpointsExplainStop(
    StopInfoSP stop_reason) {
  if (!m_process || !m_objc_exception_bp_sp)
    return false;

  if (!stop_reason || stop_reason->GetStopReason() != eStopReasonBreakpoint)
    return false;

  const uint64_t break_site_id = stop_reason->GetValue();
  return m_process->GetBreakpointSiteList().StopPointSiteContainsBreakpoint(
      break_site_id, m_objc_exception_bp_sp->GetID());
}

ValueObjectSP
GNUstepObjCRuntime::GetExceptionObjectForThread(ThreadSP thread_sp) {
  if (!thread_sp || !thread_sp->SafeToCallFunctions())
    return {};

  // Stopped at the throw itself the object is simply argument 0, which holds
  // on every one of libobjc2's exception back-ends and is where the frame
  // recognizer already presents it. That is the path that works.
  //
  // Further into the unwind there is nothing runtime-independent to read, so
  // ask the C++ runtime - which currently recovers nothing on either
  // back-end that reaches here, for two different reasons. On ELF libobjc2
  // raises with its own exception class rather than through __cxa_throw, so
  // libstdc++ never records it and __cxa_current_exception_type() is null. On
  // MinGW it does record it, but ItaniumABIRuntime reads the word before the
  // type_info, which only locates the object for Apple's runtime - objc4
  // embeds the type_info in the same allocation. libobjc2 shares one exported
  // type_info, so that word is an unrelated global. `thread exception` is
  // therefore reliable at the throw site and empty elsewhere.
  if (StackFrameSP frame_sp = thread_sp->GetStackFrameAtIndex(0)) {
    if (RecognizedStackFrameSP recognized_sp = frame_sp->GetRecognizedFrame()) {
      if (ValueObjectSP exception_sp = recognized_sp->GetExceptionObject())
        return exception_sp;
    }
  }

  auto *cpp_runtime = m_process->GetLanguageRuntime(eLanguageTypeC_plus_plus);
  if (!cpp_runtime)
    return {};
  ValueObjectSP cpp_exception_sp =
      cpp_runtime->GetExceptionObjectForThread(thread_sp);
  if (!cpp_exception_sp)
    return {};

  // An ObjC exception raised through the Itanium ABI is indistinguishable
  // from a C++ one at this level, so confirm the thrown object really is an
  // NSException (or a subclass) before claiming it.
  ClassDescriptorSP descriptor_sp = GetClassDescriptor(*cpp_exception_sp);
  if (!descriptor_sp || !descriptor_sp->IsValid())
    return {};

  // The chain comes from inferior memory, so bound the walk rather than
  // trusting it to terminate.
  static const ConstString g_NSException("NSException");
  for (uint32_t depth = 0; descriptor_sp && depth < g_max_superclass_depth;
       ++depth, descriptor_sp = descriptor_sp->GetSuperclass()) {
    if (descriptor_sp->GetClassName() == g_NSException)
      return cpp_exception_sp;
  }
  return {};
}

std::optional<addr_t>
GNUstepObjCRuntime::GetIvarAddress(addr_t object_addr,
                                   llvm::StringRef ivar_name) {
  if (!m_process || object_addr == 0 || object_addr == LLDB_INVALID_ADDRESS)
    return std::nullopt;

  Status error;
  const addr_t isa = m_process->ReadPointerFromMemory(object_addr, error);
  if (error.Fail() || isa == 0 || isa == LLDB_INVALID_ADDRESS)
    return std::nullopt;

  const ConstString name(ivar_name);
  ClassDescriptorSP descriptor_sp = GetClassDescriptorFromISA(isa);
  // The chain comes from inferior memory, so bound the walk.
  for (uint32_t depth = 0; descriptor_sp && depth < g_max_superclass_depth;
       ++depth, descriptor_sp = descriptor_sp->GetSuperclass()) {
    const size_t num_ivars = descriptor_sp->GetNumIVars();
    for (size_t i = 0; i < num_ivars; ++i) {
      const auto &ivar = descriptor_sp->GetIVarAtIndex(i);
      if (ivar.m_name == name)
        return object_addr + ivar.m_offset;
    }
  }
  return std::nullopt;
}

ThreadSP GNUstepObjCRuntime::GetBacktraceThreadFromException(
    ValueObjectSP exception_sp) {
  if (!m_process || !exception_sp)
    return ThreadSP();

  // gnustep-base captures the stack in -[NSException raise] (NSException.m),
  // unconditionally, into a GSStackTrace held in the second slot of the
  // `_reserved` allocation:
  //
  //   #define _e_info (((id*)_reserved)[0])
  //   #define _e_stack (((id*)_reserved)[1])
  //
  // A bare `@throw` does not go through -raise, so `_reserved` is null and
  // there is simply no stack to report. That is not a failure; `thread
  // exception` prints the object and nothing more, which is what it does
  // today for every exception.
  std::optional<addr_t> reserved_ivar =
      GetIvarAddress(exception_sp->GetValueAsUnsigned(0), "_reserved");
  if (!reserved_ivar)
    return ThreadSP();

  Status error;
  const addr_t reserved =
      m_process->ReadPointerFromMemory(*reserved_ivar, error);
  if (error.Fail() || reserved == 0 || reserved == LLDB_INVALID_ADDRESS)
    return ThreadSP();

  const uint32_t ptr_size = m_process->GetAddressByteSize();
  const addr_t stack_obj =
      m_process->ReadPointerFromMemory(reserved + ptr_size, error);
  if (error.Fail() || stack_obj == 0 || stack_obj == LLDB_INVALID_ADDRESS)
    return ThreadSP();

  // GSStackTrace declares both of these @public (GSPThread.h), so they are
  // reachable by name even though the class itself is private to gnustep-base
  // and has no debug info in a normal build.
  std::optional<addr_t> returns_ivar = GetIvarAddress(stack_obj, "returns");
  std::optional<addr_t> count_ivar = GetIvarAddress(stack_obj, "numReturns");
  if (!returns_ivar || !count_ivar)
    return ThreadSP();

  const addr_t returns = m_process->ReadPointerFromMemory(*returns_ivar, error);
  if (error.Fail() || returns == 0 || returns == LLDB_INVALID_ADDRESS)
    return ThreadSP();
  const int64_t num_returns = m_process->ReadSignedIntegerFromMemory(
      *count_ivar, sizeof(int32_t), 0, error);
  if (error.Fail() || num_returns <= 0)
    return ThreadSP();

  // gnustep-base's own cap is MAXFRAMES (128); clamp rather than trust a
  // count read out of the inferior.
  const uint64_t count = std::min<uint64_t>(static_cast<uint64_t>(num_returns),
                                            g_max_stack_frames);

  std::vector<addr_t> pcs;
  pcs.reserve(count);
  for (uint64_t i = 0; i < count; ++i) {
    const addr_t pc =
        m_process->ReadPointerFromMemory(returns + i * ptr_size, error);
    if (error.Fail() || pc == 0 || pc == LLDB_INVALID_ADDRESS)
      break;
    pcs.push_back(pc);
  }
  if (pcs.empty())
    return ThreadSP();

  // Every entry came from backtrace(3), so all of them are return addresses -
  // there is no live frame 0 to except from the usual adjustment.
  //
  // No frames are skipped. gnustep-base's own -callStackReturnAddresses drops
  // the first FrameOffset (4) so that a program sees its own code first, but
  // that count is a private detail of a particular gnustep-base and dropping
  // four here would also discard the frame that raised. Apple's runtime
  // likewise leaves objc_exception_throw visible, so showing the capture
  // machinery is both safer and the more consistent choice.
  ThreadSP thread_sp = std::make_shared<HistoryThread>(
      *m_process, 0, pcs, HistoryPCType::ReturnsNoZerothFrame);
  m_process->GetExtendedThreadList().AddThread(thread_sp);
  return thread_sp;
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

  // A message to nil does not dispatch anywhere, so there is no
  // implementation to run to. Declining to provide a plan at all would leave
  // the thread stopped on the dispatch function's first instruction: harmless
  // when the runtime carries no line information for its hand-written
  // assembly, but with a runtime built from source it drops the user into
  // objc_msgSend.S. Step back out to the sender instead, which is where a nil
  // send returns to and what the user asked to step over.
  if (receiver == 0 || receiver == LLDB_INVALID_ADDRESS) {
    const bool continue_to_next_branch = true;
    const bool gather_return_value = false;
    auto step_out_sp = std::make_shared<ThreadPlanStepOut>(
        thread, stop_others, eVoteNo, eVoteNoOpinion, /*frame_idx=*/0,
        continue_to_next_branch, gather_return_value);
    // Nothing further should be stepped through on the way out; the enclosing
    // step-in plan decides where to stop once we are back in the sender.
    step_out_sp->ClearShouldStopHereCallbacks();
    return step_out_sp;
  }

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

ConstString GNUstepObjCRuntime::GetSelectorName(lldb::addr_t sel_addr) {
  if (!m_process || sel_addr == 0 || sel_addr == LLDB_INVALID_ADDRESS)
    return ConstString();

  // clang emits every selector as a global named
  // ".objc_selector_<name>_<mangled types>" (CGObjCGNU.cpp). That symbol is
  // the only place the name survives: __objc_load overwrites the selector's
  // name field with a numeric dispatch index (selector_table.cc), so it
  // cannot be read back from memory.
  Address resolved;
  if (!GetTargetRef().ResolveLoadAddress(sel_addr, resolved))
    return ConstString();
  ModuleSP module_sp = resolved.GetModule();
  if (!module_sp)
    return ConstString();
  Symtab *symtab = module_sp->GetSymtab();
  if (!symtab)
    return ConstString();

  static constexpr llvm::StringLiteral g_selector_prefix = ".objc_selector_";

  // The first selector in a section shares its address with the section's
  // start sentinel, so take every symbol at this address rather than
  // whichever one an exact lookup happens to return.
  llvm::StringRef symbol_name;
  symtab->ForEachSymbolContainingFileAddress(
      resolved.GetFileAddress(), [&](Symbol *symbol) -> bool {
        llvm::StringRef name = symbol->GetName().GetStringRef();
        if (!name.starts_with(g_selector_prefix))
          return true; // keep looking
        symbol_name = name;
        return false;
      });
  if (symbol_name.empty())
    return ConstString();

  llvm::StringRef name = symbol_name.drop_front(g_selector_prefix.size());

  // A selector name may contain both '_' and ':', so the boundary cannot be
  // found by scanning. Subtract the suffix instead: the types half of the
  // symbol is exactly the selector's own `types` string, mangled. That
  // string is still readable - only the name field was overwritten.
  const uint32_t ptr_size = m_process->GetAddressByteSize();
  Status error;
  const addr_t types_addr =
      m_process->ReadPointerFromMemory(sel_addr + ptr_size, error);
  std::string suffix("_");
  if (error.Success() && types_addr != 0 &&
      types_addr != LLDB_INVALID_ADDRESS) {
    char buffer[g_max_type_encoding_length];
    const size_t length = m_process->ReadCStringFromMemory(
        types_addr, buffer, sizeof(buffer), error);
    if (error.Success() && length < sizeof(buffer) - 1) {
      // GetSymbolNameForTypeEncoding: '@' is replaced on ELF because it is
      // reserved for symbol versioning, '=' on Windows because lld rejects
      // it in an exported name. MinGW is isOSWindows() and not ELF, so the
      // predicates are asked separately rather than derived from each other.
      const llvm::Triple &triple = GetTargetRef().GetArchitecture().GetTriple();
      std::string mangled(buffer, length);
      if (triple.isOSBinFormatELF())
        llvm::replace(mangled, '@', '\1');
      if (triple.isOSWindows())
        llvm::replace(mangled, '=', '\2');
      suffix += mangled;
    }
  }

  if (!name.consume_back(suffix))
    return ConstString();
  return ConstString(name);
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

  // Anything reached from the sweep that resolves a class by ISA or by name
  // lands back here, so publish the "up to date" state and take ownership of
  // the pending list up front: a reentrant call must not restart the sweep
  // and clear the vector the outer loop is walking.
  if (m_updating_isa_map)
    return;
  llvm::SaveAndRestore<bool> updating(m_updating_isa_map, true);
  m_isa_map_dirty = false;
  m_isa_to_descriptor_stop_id = stop_id;
  std::vector<ModuleSP> pending;
  pending.swap(m_pending_modules);

  // The first update has to look at everything already loaded; afterwards only
  // the modules that arrived since need scanning, so a dlopen does not re-walk
  // every symbol table in the process.
  if (m_swept_all_modules) {
    for (const ModuleSP &module_sp : pending)
      AddClassesFromModule(module_sp);
  } else {
    const ModuleList &images = GetTargetRef().GetImages();
    std::lock_guard<std::recursive_mutex> guard(images.GetMutex());
    const size_t num_modules = images.GetSize();
    for (size_t i = 0; i < num_modules; i++)
      AddClassesFromModule(images.GetModuleAtIndex(i));
    m_swept_all_modules = true;
  }
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

bool GNUstepObjCRuntime::ParseIvarOffsetSymbol(llvm::StringRef symbol,
                                               llvm::StringRef &class_name,
                                               llvm::StringRef &ivar_name) {
  // __objc_ivar_offset_<Class>.<ivar>.<mangled type encoding> (CGObjCGNU.cpp).
  // Neither a class nor an ivar name can contain a '.', so the first two
  // components are unambiguous however the encoding is spelled - and the
  // encoding itself is not needed to identify the ivar.
  static constexpr llvm::StringLiteral g_ivar_prefix("__objc_ivar_offset_");
  if (!symbol.consume_front(g_ivar_prefix))
    return false;

  std::tie(class_name, ivar_name) = symbol.split('.');
  // The gnustep-2.x spelling always carries the encoding, so a name with no
  // second '.' is not one: it is either the v1 ABI's two-component form or
  // the `__objc_ivar_offset_value_` variant, neither of which this runtime
  // handles. Rejecting them degrades to the behaviour before this fallback
  // existed rather than resolving to a wrong address.
  if (ivar_name.find('.') == llvm::StringRef::npos)
    return false;
  ivar_name = ivar_name.split('.').first;
  return !class_name.empty() && !ivar_name.empty();
}

addr_t GNUstepObjCRuntime::LookupRuntimeSymbol(ConstString name) {
  const llvm::StringRef symbol = name.GetStringRef();

  // An expression that touches an ivar emits a reference to the runtime's
  // offset variable. Normally the inferior exports it and the JIT resolves it
  // from the symbol table; this is the fallback for when it does not - a
  // stripped module, a @private ivar given hidden visibility, or a class built
  // at run time that never had a symbol at all.
  llvm::StringRef class_name, ivar_name;
  if (ParseIvarOffsetSymbol(symbol, class_name, ivar_name)) {
    const ConstString ivar(ivar_name);
    ClassDescriptorSP descriptor_sp =
        GetClassDescriptorFromClassName(ConstString(class_name));
    // clang emits the symbol for the class that declares the ivar, so the
    // walk is belt and braces; bound it, as the chain comes from memory.
    for (uint32_t depth = 0; descriptor_sp && depth < g_max_superclass_depth;
         ++depth, descriptor_sp = descriptor_sp->GetSuperclass()) {
      const addr_t offset_addr =
          static_cast<GNUstepObjCClassDescriptor *>(descriptor_sp.get())
              ->GetIVarOffsetAddress(ivar);
      if (offset_addr != LLDB_INVALID_ADDRESS)
        return offset_addr;
    }
    return LLDB_INVALID_ADDRESS;
  }

  // Only ivar offsets. Apple's equivalent also resolves OBJC_CLASS_$_, but
  // there is nothing here to answer with: an expression referencing a class
  // emits `._OBJC_REF_CLASS_<name>` (SymbolForClassRef, CGObjCGNU.cpp), a
  // pointer variable the JIT loads the class out of - not the class symbol
  // itself - and if that variable is absent there is no address to hand back.
  // Resolving it by name would also be circular, since the name-to-ISA map is
  // built by sweeping exactly the class symbols that would be missing.
  return LLDB_INVALID_ADDRESS;
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
    // A descriptor is a snapshot taken when it was built, and the map holds
    // it for the life of the process. Before the runtime resolves a class its
    // superclass pointer and instance size are not yet meaningful, and this
    // sweep runs at the loader's rendezvous stop - which for a dlopen'd image
    // is *before* its __objc_load. Caching one then would pin an empty class
    // permanently, so leave it to be rebuilt on demand.
    if (descriptor_sp->IsValid() && descriptor_sp->IsResolved())
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

size_t GNUstepObjCRuntime::GetByteOffsetForIvar(CompilerType &parent_qual_type,
                                                const char *ivar_name) {
  // An instance is allocated behind a reference-count word, and libobjc2
  // aligns each ivar with that word included (objc_compute_ivar_offsets,
  // ivar.c). An ivar needing more than pointer alignment therefore sits at a
  // different offset than laying the class out as a plain struct would give,
  // and only the runtime's own metadata knows which. Offsets are absolute, so
  // a superclass's ivar needs no adjustment.
  ClassDescriptorSP descriptor_sp =
      GetClassDescriptorFromClassName(parent_qual_type.GetTypeName());
  const ConstString name(ivar_name);
  for (uint32_t depth = 0; descriptor_sp && depth < g_max_superclass_depth;
       ++depth, descriptor_sp = descriptor_sp->GetSuperclass()) {
    const size_t num_ivars = descriptor_sp->GetNumIVars();
    for (size_t i = 0; i < num_ivars; ++i) {
      const auto &ivar = descriptor_sp->GetIVarAtIndex(i);
      if (ivar.m_name == name)
        return ivar.m_offset;
    }
  }
  return LLDB_INVALID_IVAR_OFFSET;
}

std::optional<uint64_t>
GNUstepObjCRuntime::GetTypeBitSize(const CompilerType &compiler_type) {
  ClassDescriptorSP descriptor_sp =
      GetClassDescriptorFromClassName(compiler_type.GetTypeName());
  if (!descriptor_sp || !descriptor_sp->IsValid())
    return std::nullopt;
  const uint64_t instance_size = descriptor_sp->GetInstanceSize();
  if (instance_size == 0)
    return std::nullopt;
  return instance_size * 8;
}

std::optional<CompilerType>
GNUstepObjCRuntime::GetRuntimeType(CompilerType base_type) {
  CompilerType class_type;
  bool is_pointer_type = false;
  if (TypeSystemClang::IsObjCObjectPointerType(base_type, &class_type))
    is_pointer_type = true;
  else if (TypeSystemClang::IsObjCObjectOrInterfaceType(base_type))
    class_type = base_type;
  else
    return std::nullopt;
  if (!class_type)
    return std::nullopt;

  ConstString class_name(class_type.GetTypeName());
  if (!class_name)
    return std::nullopt;

  if (TypeSP type_sp = LookupClassTypeInDebugInfo(class_name)) {
    if (CompilerType complete_type = type_sp->GetFullCompilerType();
        complete_type.GetCompleteType())
      return is_pointer_type ? complete_type.GetPointerType() : complete_type;
  }

  return ObjCLanguageRuntime::GetRuntimeType(base_type);
}

DeclVendor *GNUstepObjCRuntime::GetDeclVendor() {
  if (!m_decl_vendor_up)
    m_decl_vendor_up = std::make_unique<GNUstepObjCDeclVendor>(*this);
  return m_decl_vendor_up.get();
}

ObjCLanguageRuntime::EncodingToTypeSP GNUstepObjCRuntime::GetEncodingToType() {
  if (!m_encoding_to_type_sp)
    m_encoding_to_type_sp = std::make_shared<GNUstepObjCTypeEncodingParser>(
        GetTargetRef().GetArchitecture().GetTriple(), this);
  return m_encoding_to_type_sp;
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
  // libobjc2 gives a metaclass the same `name` as its class, so indexing one
  // by name would let a by-name lookup (GetISA, and through it anything that
  // resolves a class by name) hand back the metaclass: an interface whose
  // class methods look like instance methods and which has no ivars. Cache
  // metaclasses by ISA all the same, so they are not re-parsed on every
  // lookup.
  // Only a resolved class is safe to cache; see AddClassesFromModule.
  if (descriptor_sp->IsResolved()) {
    if (descriptor_sp->IsMetaclass())
      AddClass(isa, descriptor_sp);
    else
      AddClass(isa, descriptor_sp, descriptor_sp->GetClassName().GetCString());
  }
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

// FIXME: This belongs in ExceptionSearchFilter::ModulePasses, which is where
// the defect is: it asks the process for the language runtime and rejects the
// module outright when there is not one yet, and nothing revisits those
// modules. Target::ModulesDidLoad resolves breakpoints before it notifies
// language runtimes, so a breakpoint set before running misses every module
// up to the point this runtime came into existence. Fixing it there means
// re-resolving from Target once a filter's runtime appears, which would newly
// fire for every language - worth doing, but not from here.
//
// It only bites when the entry point is outside the runtime's own module,
// which on MinGW it is: libstdc++'s __cxa_begin_catch.
void GNUstepObjCRuntime::ResolveExceptionBreakpoints() {
  Target &target = GetTargetRef();
  ModuleList &modules = target.GetImages();
  for (bool internal : {false, true}) {
    BreakpointList &breakpoints = target.GetBreakpointList(internal);
    for (const BreakpointSP &bp_sp : breakpoints.Breakpoints()) {
      if (bp_sp && bp_sp->IsEnabled() && bp_sp->GetSearchFilter() &&
          bp_sp->GetSearchFilter()->GetFilterTy() == SearchFilter::Exception)
        bp_sp->ResolveBreakpointInModules(modules, /*send_event=*/true);
    }
  }
}

void GNUstepObjCRuntime::ModulesDidLoad(const ModuleList &module_list) {
  ReadObjCLibraryIfNeeded(module_list);
  if (!m_swept_exception_breakpoints) {
    m_swept_exception_breakpoints = true;
    ResolveExceptionBreakpoints();
  }

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
