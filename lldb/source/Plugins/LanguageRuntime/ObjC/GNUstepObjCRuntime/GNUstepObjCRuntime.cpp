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

#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Expression/UtilityFunction.h"
#include "lldb/Symbol/DeclVendor.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
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

llvm::Error GNUstepObjCRuntime::GetObjectDescription(Stream &str,
                                                     ValueObject &valobj) {
  return llvm::createStringError(
      "LLDB's GNUStep runtime does not support object description");
}

llvm::Error
GNUstepObjCRuntime::GetObjectDescription(Stream &strm, Value &value,
                                         ExecutionContextScope *exe_scope) {
  return llvm::createStringError(
      "LLDB's GNUStep runtime does not support object description");
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
