//===-- GNUstepObjCRuntime.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GNUstepObjCRuntime.h"
#include "GNUstepObjCClassDescriptor.h"

#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"

#include "lldb/Core/Address.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Value.h"
#include "lldb/Expression/DiagnosticManager.h"
#include "lldb/Expression/FunctionCaller.h"
#include "lldb/Expression/UtilityFunction.h"
#include "lldb/Symbol/DeclVendor.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegularExpression.h"
#include "lldb/Utility/StructuredData.h"
#include "lldb/ValueObject/ValueObject.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(GNUstepObjCRuntime)

char GNUstepObjCRuntime::ID = 0;

void GNUstepObjCRuntime::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), "GNUstep Objective-C Language Runtime - libobjc2",
      CreateInstance);
}

void GNUstepObjCRuntime::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

static bool CanModuleBeGNUstepObjCLibrary(const ModuleSP &module_sp,
                                          const llvm::Triple &TT) {
  if (!module_sp)
    return false;
  const FileSpec &module_file_spec = module_sp->GetFileSpec();
  if (!module_file_spec)
    return false;
  llvm::StringRef filename = module_file_spec.GetFilename();
  if (TT.isOSBinFormatELF())
    return filename.starts_with("libobjc.so");
  if (TT.isOSWindows())
    return filename == "objc.dll";
  return false;
}

static bool ScanForGNUstepObjCLibraryCandidate(const ModuleList &modules,
                                               const llvm::Triple &TT) {
  std::lock_guard<std::recursive_mutex> guard(modules.GetMutex());
  size_t num_modules = modules.GetSize();
  for (size_t i = 0; i < num_modules; i++) {
    auto mod = modules.GetModuleAtIndex(i);
    if (CanModuleBeGNUstepObjCLibrary(mod, TT))
      return true;
  }
  return false;
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

  const ModuleList &images = target.GetImages();
  if (!ScanForGNUstepObjCLibraryCandidate(images, TT))
    return nullptr;

  if (TT.isOSBinFormatELF()) {
    SymbolContextList eh_pers;
    RegularExpression regex("__gnustep_objc[x]*_personality_v[0-9]+");
    images.FindSymbolsMatchingRegExAndType(regex, eSymbolTypeCode, eh_pers);
    if (eh_pers.GetSize() == 0)
      return nullptr;
  } else if (TT.isOSWindows()) {
    SymbolContextList objc_mandatory;
    images.FindSymbolsWithNameAndType(ConstString("__objc_load"),
                                      eSymbolTypeCode, objc_mandatory);
    if (objc_mandatory.GetSize() == 0)
      return nullptr;
  }

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

  ConstString class_name(objc_class_sp->GetClassName());
  if (!class_name)
    return false;

  const addr_t object_ptr = in_value.GetPointerValue().address;
  address.SetRawAddress(object_ptr);
  class_type_or_name.SetName(class_name);

  // Try to upgrade the bare name to a real type: first from the cache of
  // classes already realized from debug info, then - should a decl vendor
  // exist one day - from that.
  TypeSP type_sp(objc_class_sp->GetType());
  if (!type_sp) {
    type_sp = LookupInCompleteClassCache(class_name);
    if (type_sp)
      objc_class_sp->SetType(type_sp);
  }
  if (type_sp)
    class_type_or_name.SetTypeSP(type_sp);
  else if (auto *vendor = GetDeclVendor()) {
    auto types = vendor->FindTypes(class_name, /*max_matches*/ 1);
    if (!types.empty())
      class_type_or_name.SetCompilerType(types.front());
  }

  return !class_type_or_name.IsEmpty();
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
  // TODO: Implement this properly to avoid stepping into things like PLT stubs
  return nullptr;
}

void GNUstepObjCRuntime::UpdateISAToDescriptorMapIfNeeded() {
  if (!m_process)
    return;
  const uint32_t stop_id = m_process->GetStopID();
  if (!m_isa_map_dirty) {
    m_isa_to_descriptor_stop_id = stop_id;
    return;
  }

  // The gnustep-2.x ABI emits every compiled class as a `._OBJC_CLASS_<name>`
  // data symbol whose address is the class object itself (the ISA of its
  // instances), so the map can be seeded from symbol tables alone - without
  // running any code in the inferior. Classes created dynamically at runtime
  // are handled by the create-on-miss path in GetClassDescriptorFromISA.
  Target &target = GetTargetRef();
  const ModuleList &images = target.GetImages();

  SymbolContextList sc_list;
  RegularExpression regex(llvm::StringRef("^\\._OBJC_CLASS_"));
  images.FindSymbolsMatchingRegExAndType(regex, eSymbolTypeAny, sc_list);

  static constexpr llvm::StringLiteral g_class_prefix("._OBJC_CLASS_");
  for (const SymbolContext &sc : sc_list) {
    if (!sc.symbol)
      continue;
    const addr_t isa = sc.symbol->GetAddress().GetLoadAddress(&target);
    if (isa == 0 || isa == LLDB_INVALID_ADDRESS || ISAIsCached(isa))
      continue;
    llvm::StringRef name = sc.symbol->GetName().GetStringRef();
    name.consume_front(g_class_prefix);
    auto descriptor_sp = std::make_shared<GNUstepObjCClassDescriptor>(
        m_process->shared_from_this(), isa);
    if (descriptor_sp->IsValid())
      AddClass(isa, descriptor_sp, name.str().c_str());
  }

  m_isa_map_dirty = false;
  m_isa_to_descriptor_stop_id = stop_id;
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
  const llvm::Triple &TT = GetTargetRef().GetArchitecture().GetTriple();
  return CanModuleBeGNUstepObjCLibrary(module_sp, TT);
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
  m_isa_map_dirty = true;
}
