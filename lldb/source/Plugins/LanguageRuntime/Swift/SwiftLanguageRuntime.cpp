//===-- SwiftLanguageRuntime.cpp ------------------------------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2020 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "SwiftLanguageRuntime.h"
#include "Plugins/LanguageRuntime/Swift/LLDBMemoryReader.h"
#include "SwiftLanguageRuntimeImpl.h"

#include "Plugins/Process/Utility/RegisterContext_x86.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "Utility/ARM64_DWARF_Registers.h"
#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/JITSection.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Progress.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/DataFormatters/StringPrinter.h"
#include "lldb/Host/OptionParser.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/OptionParsing.h"
#include "lldb/Utility/Timer.h"

#include "swift/AST/ASTMangler.h"
#include "swift/AST/Decl.h"
#include "swift/AST/Module.h"
#include "swift/Demangling/Demangle.h"
#include "swift/Reflection/ReflectionContext.h"
#include "swift/RemoteAST/RemoteAST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Memory.h"

// FIXME: we should not need this
#include "Plugins/Language/Swift/SwiftFormatters.h"
#include "Plugins/Language/Swift/SwiftRuntimeFailureRecognizer.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(SwiftLanguageRuntime)

namespace lldb_private {
char SwiftLanguageRuntime::ID = 0;

extern "C" unsigned long long _swift_classIsSwiftMask = 0;

const char *SwiftLanguageRuntime::GetErrorBackstopName() {
  return "swift_errorInMain";
}

const char *SwiftLanguageRuntime::GetStandardLibraryBaseName() {
  return "swiftCore";
}

static ConstString GetStandardLibraryName(Process &process) {
  // This result needs to be stored in the constructor.
  PlatformSP platform_sp(process.GetTarget().GetPlatform());
  if (platform_sp)
    return platform_sp->GetFullNameForDylib(
        ConstString(SwiftLanguageRuntime::GetStandardLibraryBaseName()));
  return {};
}

ConstString SwiftLanguageRuntime::GetStandardLibraryName() {
  return ::GetStandardLibraryName(*m_process);
}

static bool IsModuleSwiftRuntime(lldb_private::Process &process,
                                 lldb_private::Module &module) {
  return module.GetFileSpec().GetFilename() == GetStandardLibraryName(process);
}

static UnwindPlanSP
GetFollowAsyncContextUnwindPlan(RegisterContext *regctx, ArchSpec &arch,
                                bool &behaves_like_zeroth_frame);

AppleObjCRuntimeV2 *
SwiftLanguageRuntime::GetObjCRuntime(lldb_private::Process &process) {
  if (auto objc_runtime = ObjCLanguageRuntime::Get(process)) {
    if (objc_runtime->GetPluginName() ==
        AppleObjCRuntimeV2::GetPluginNameStatic())
      return (AppleObjCRuntimeV2 *)objc_runtime;
  }
  return nullptr;
}

AppleObjCRuntimeV2 *SwiftLanguageRuntime::GetObjCRuntime() {
  return GetObjCRuntime(*m_process);
}

enum class RuntimeKind { Swift, ObjC };

/// Detect a statically linked Swift runtime by looking for a well-known symbol.
static bool IsStaticSwiftRuntime(Module &image) {
  static ConstString swift_release_dealloc_sym("_swift_release_dealloc");
  return image.FindFirstSymbolWithNameAndType(swift_release_dealloc_sym);
}

/// \return the Swift or Objective-C runtime found in the loaded images.
static ModuleSP findRuntime(Process &process, RuntimeKind runtime_kind) {
  AppleObjCRuntimeV2 *objc_runtime = nullptr;
  if (runtime_kind == RuntimeKind::ObjC) {
    objc_runtime = SwiftLanguageRuntime::GetObjCRuntime(process);
    if (!objc_runtime)
      return {};
  }

  ModuleSP runtime_image;
  process.GetTarget().GetImages().ForEach([&](const ModuleSP &image) {
    if (runtime_kind == RuntimeKind::Swift && image &&
        IsModuleSwiftRuntime(process, *image)) {
      runtime_image = image;
      return false;
    }
    if (runtime_kind == RuntimeKind::ObjC &&
        objc_runtime->IsModuleObjCLibrary(image)) {
      runtime_image = image;
      return false;
    }
    return true;
  });

  if (!runtime_image && runtime_kind == RuntimeKind::Swift) {
    // Do a more expensive search for a statically linked Swift runtime.
    process.GetTarget().GetImages().ForEach([&](const ModuleSP &image) {
      if (image && IsStaticSwiftRuntime(*image)) {
        runtime_image = image;
        return false;
      }
      return true;
    });
  }
  return runtime_image;
}

static llvm::Optional<lldb::addr_t>
FindSymbolForSwiftObject(Process &process, RuntimeKind runtime_kind,
                         StringRef object, const SymbolType sym_type) {
  ModuleSP image = findRuntime(process, runtime_kind);
  Target &target = process.GetTarget();
  if (!image) {
    // Don't diagnose a missing Objective-C runtime on platforms that
    // don't have one.
    if (runtime_kind == RuntimeKind::ObjC) {
      auto *obj_file = target.GetExecutableModule()->GetObjectFile();
      bool have_objc_interop =
          obj_file && obj_file->GetPluginName().equals("mach-o");
      if (!have_objc_interop)
        return {};
    }
    target.GetDebugger().GetAsyncErrorStream()->Printf(
        "Couldn't find the %s runtime library in loaded images.\n",
        (runtime_kind == RuntimeKind::Swift) ? "Swift" : "Objective-C");
    return {};
  }

  SymbolContextList sc_list;
  image->FindSymbolsWithNameAndType(ConstString(object), sym_type, sc_list);
  if (sc_list.GetSize() != 1)
    return {};

  SymbolContext SwiftObject_Class;
  if (!sc_list.GetContextAtIndex(0, SwiftObject_Class))
    return {};
  if (!SwiftObject_Class.symbol)
    return {};
  lldb::addr_t addr =
      SwiftObject_Class.symbol->GetAddress().GetLoadAddress(&target);
  if (addr && addr != LLDB_INVALID_ADDRESS)
    return addr;

  return {};
}

static lldb::BreakpointResolverSP
CreateExceptionResolver(const lldb::BreakpointSP &bkpt, bool catch_bp, bool throw_bp) {
  BreakpointResolverSP resolver_sp;

  if (throw_bp)
    resolver_sp.reset(new BreakpointResolverName(
        bkpt, "swift_willThrow", eFunctionNameTypeBase, eLanguageTypeUnknown,
        Breakpoint::Exact, 0, eLazyBoolNo));
  // FIXME: We don't do catch breakpoints for ObjC yet.
  // Should there be some way for the runtime to specify what it can do in this
  // regard?
  return resolver_sp;
}

static const char *g_stub_log_message =
    "Swift language runtime isn't available because %s is not loaded in "
    "the process. functionality.";

/// Simple Swift programs may not actually depend on the Swift runtime
/// library (libswiftCore.dylib), but if it is missing, what we can do
/// is limited. This implementation represents that case.
class SwiftLanguageRuntimeStub {
  Process &m_process;

public:
  SwiftLanguageRuntimeStub(Process &process) : m_process(process) {}

#define STUB_LOG()                                                             \
  do {                                                                         \
    LLDB_LOGF(GetLog(LLDBLog::Expressions | LLDBLog::Types),                   \
              g_stub_log_message,                                              \
              GetStandardLibraryName(m_process).AsCString());                  \
    assert(false && "called into swift language runtime stub");                \
  } while (0)

  bool GetDynamicTypeAndAddress(ValueObject &in_value,
                                lldb::DynamicValueType use_dynamic,
                                TypeAndOrName &class_type_or_name,
                                Address &address,
                                Value::ValueType &value_type) {
    STUB_LOG();
    return false;
  }

  TypeAndOrName FixUpDynamicType(const TypeAndOrName &type_and_or_name,
                                 ValueObject &static_value) {
    STUB_LOG();
    return {};
  }

  bool IsTaggedPointer(lldb::addr_t addr, CompilerType type) {
    STUB_LOG();
    return false;
  }

  std::pair<lldb::addr_t, bool> FixupPointerValue(lldb::addr_t addr,
                                                  CompilerType type) {
    STUB_LOG();
    return {addr, false};
  }

  lldb::addr_t FixupAddress(lldb::addr_t addr, CompilerType type,
                            Status &error) {
    STUB_LOG();
    return addr;
  }

  SwiftLanguageRuntime::MetadataPromiseSP
  GetMetadataPromise(lldb::addr_t addr, ValueObject &for_object) {
    STUB_LOG();
    return {};
  }

  void ModulesDidLoad(const ModuleList &module_list) {}

  bool IsStoredInlineInBuffer(CompilerType type) {
    STUB_LOG();
    return false;
  }

  void DumpTyperef(CompilerType type, TypeSystemSwiftTypeRef *module_holder,
                   Stream *s) {
    STUB_LOG();
  }

  llvm::Optional<uint64_t> GetMemberVariableOffset(CompilerType instance_type,
                                                   ValueObject *instance,
                                                   llvm::StringRef member_name,
                                                   Status *error) {
    STUB_LOG();
    return {};
  }

  llvm::Optional<unsigned> GetNumChildren(CompilerType type,
                                          ValueObject *valobj) {
    STUB_LOG();
    return {};
  }

  llvm::Optional<std::string> GetEnumCaseName(CompilerType type,
                                              const DataExtractor &data,
                                              ExecutionContext *exe_ctx) {
    STUB_LOG();
    return {};
  }

  std::pair<bool, llvm::Optional<size_t>> GetIndexOfChildMemberWithName(
      CompilerType type, llvm::StringRef name, ExecutionContext *exe_ctx,
      bool omit_empty_base_classes, std::vector<uint32_t> &child_indexes) {
    STUB_LOG();
    return {};
  }

  CompilerType GetChildCompilerTypeAtIndex(
      CompilerType type, size_t idx, bool transparent_pointers,
      bool omit_empty_base_classes, bool ignore_array_bounds,
      std::string &child_name, uint32_t &child_byte_size,
      int32_t &child_byte_offset, uint32_t &child_bitfield_bit_size,
      uint32_t &child_bitfield_bit_offset, bool &child_is_base_class,
      bool &child_is_deref_of_parent, ValueObject *valobj,
      uint64_t &language_flags) {
    STUB_LOG();
    return {};
  }

  llvm::Optional<unsigned> GetNumFields(CompilerType type,
                                        ExecutionContext *exe_ctx) {
    STUB_LOG();
    return {};
  }

  bool GetObjectDescription(Stream &str, ValueObject &object) {
    STUB_LOG();
    return false;
  }

  void AddToLibraryNegativeCache(llvm::StringRef library_name) {}

  bool IsInLibraryNegativeCache(llvm::StringRef library_name) {
    return false;
  }

  void ReleaseAssociatedRemoteASTContext(swift::ASTContext *ctx) {}

  CompilerType BindGenericTypeParameters(StackFrame &stack_frame,
                                         CompilerType base_type) {
    STUB_LOG();
    return {};
  }

  CompilerType GetConcreteType(ExecutionContextScope *exe_scope,
                               ConstString abstract_type_name) {
    STUB_LOG();
    return {};
  }

  llvm::Optional<uint64_t> GetBitSize(CompilerType type,
                                      ExecutionContextScope *exe_scope) {
    STUB_LOG();
    return {};
  }

  llvm::Optional<uint64_t> GetByteStride(CompilerType type) {
    STUB_LOG();
    return {};
  }

  llvm::Optional<size_t> GetBitAlignment(CompilerType type,
                                         ExecutionContextScope *exe_scope) {
    STUB_LOG();
    return {};
  }

  bool IsValidErrorValue(ValueObject &in_value) {
    STUB_LOG();
    return {};
  }

  lldb::SyntheticChildrenSP
  GetBridgedSyntheticChildProvider(ValueObject &valobj) {
    STUB_LOG();
    return {};
  }

  void WillStartExecutingUserExpression(bool runs_in_playground_or_repl) {
    if (!runs_in_playground_or_repl)
      STUB_LOG();
  }

  void DidFinishExecutingUserExpression(bool runs_in_playground_or_repl) {
    if (!runs_in_playground_or_repl)
      STUB_LOG();
  }

  bool IsABIStable() {
    STUB_LOG();

    // Pick a sensible default.
    return m_process.GetTarget().GetArchitecture().GetTriple().isOSDarwin()
               ? true
               : false;
  }

  SwiftLanguageRuntimeStub(const SwiftLanguageRuntimeStub &) = delete;
  const SwiftLanguageRuntimeStub &
  operator=(const SwiftLanguageRuntimeStub &) = delete;
};

static bool HasReflectionInfo(ObjectFile *obj_file) {
  auto findSectionInObject = [&](StringRef name) {
    ConstString section_name(name);
    SectionSP section_sp =
        obj_file->GetSectionList()->FindSectionByName(section_name);
    if (section_sp)
      return true;
    return false;
  };

  StringRef field_md = obj_file->GetReflectionSectionIdentifier(
      swift::ReflectionSectionKind::fieldmd);
  StringRef assocty = obj_file->GetReflectionSectionIdentifier(
      swift::ReflectionSectionKind::assocty);
  StringRef builtin = obj_file->GetReflectionSectionIdentifier(
      swift::ReflectionSectionKind::builtin);
  StringRef capture = obj_file->GetReflectionSectionIdentifier(
      swift::ReflectionSectionKind::capture);
  StringRef typeref = obj_file->GetReflectionSectionIdentifier(
      swift::ReflectionSectionKind::typeref);
  StringRef reflstr = obj_file->GetReflectionSectionIdentifier(
      swift::ReflectionSectionKind::reflstr);

  bool hasReflectionSection = false;
  hasReflectionSection |= findSectionInObject(field_md);
  hasReflectionSection |= findSectionInObject(assocty);
  hasReflectionSection |= findSectionInObject(builtin);
  hasReflectionSection |= findSectionInObject(capture);
  hasReflectionSection |= findSectionInObject(typeref);
  hasReflectionSection |= findSectionInObject(reflstr);
  return hasReflectionSection;
}

SwiftLanguageRuntimeImpl::ReflectionContextInterface *
SwiftLanguageRuntimeImpl::GetReflectionContext() {
  if (!m_initialized_reflection_ctx)
    SetupReflection();
  return m_reflection_ctx.get();
}

void SwiftLanguageRuntimeImpl::SetupReflection() {
  LLDB_SCOPED_TIMER();
 
  // SetupABIBit() iterates of the Target's images and thus needs to
  // acquire that ModuleList's lock. We need to acquire this before
  // locking m_add_module_mutex, since ModulesDidLoad can also be
  // called from a place where that lock is already held:
  // +   lldb_private::DynamicLoaderDarwin::AddModulesUsingImageInfos()
  // +     lldb_private::ModuleList::AppendIfNeeded()
  // +       lldb_private::Target::NotifyModuleAdded()
  // +         lldb_private::Target::ModulesDidLoad()

  // The global ABI bit is read by the Swift runtime library.
  SetupABIBit();
  
  std::lock_guard<std::recursive_mutex> lock(m_add_module_mutex);
  if (m_initialized_reflection_ctx)
    return;

  auto &target = m_process.GetTarget();
  auto exe_module = target.GetExecutableModule();

  if (!exe_module) {
    LLDB_LOGF(GetLog(LLDBLog::Types), "%s: Failed to get executable module",
              LLVM_PRETTY_FUNCTION);
    m_initialized_reflection_ctx = false;
    return;
  }

  bool objc_interop = (bool)findRuntime(m_process, RuntimeKind::ObjC);
  const char *objc_interop_msg =
      objc_interop ? "with Objective-C interopability" : "Swift only";

  auto &triple = exe_module->GetArchitecture().GetTriple();
  if (triple.isArch64Bit()) {
    LLDB_LOGF(GetLog(LLDBLog::Types),
              "Initializing a 64-bit reflection context (%s) for \"%s\"",
              triple.str().c_str(), objc_interop_msg);
    m_reflection_ctx = ReflectionContextInterface::CreateReflectionContext64(
        this->GetMemoryReader(), objc_interop);
  } else if (triple.isArch32Bit()) {
    LLDB_LOGF(GetLog(LLDBLog::Types),
              "Initializing a 32-bit reflection context (%s) for \"%s\"",
              triple.str().c_str(), objc_interop_msg);
    m_reflection_ctx = ReflectionContextInterface::CreateReflectionContext32(
        this->GetMemoryReader(), objc_interop);
  } else {
    LLDB_LOGF(GetLog(LLDBLog::Types),
              "Could not initialize reflection context for \"%s\"",
              triple.str().c_str());
  }

  m_initialized_reflection_ctx = true;

  Progress progress(
      llvm::formatv("Setting up Swift reflection for '{0}'",
                    exe_module->GetFileSpec().GetFilename().AsCString()),
      m_modules_to_add.GetSize());

  size_t completion = 0;

  // Add all defered modules to reflection context that were added to
  // the target since this SwiftLanguageRuntime was created.
  m_modules_to_add.ForEach([&](const ModuleSP &module_sp) -> bool {
    AddModuleToReflectionContext(module_sp);
    progress.Increment(++completion);
    return true;
  });
  m_modules_to_add.Clear();
}

bool SwiftLanguageRuntimeImpl::IsABIStable() {
  GetReflectionContext();
  return _swift_classIsSwiftMask == 2;
}

void SwiftLanguageRuntimeImpl::SetupSwiftError() {
  m_SwiftNativeNSErrorISA =
      FindSymbolForSwiftObject(m_process, RuntimeKind::Swift,
                               "__SwiftNativeNSError", eSymbolTypeObjCClass);
}

llvm::Optional<lldb::addr_t>
SwiftLanguageRuntimeImpl::GetSwiftNativeNSErrorISA() {
  return m_SwiftNativeNSErrorISA;
}

void SwiftLanguageRuntimeImpl::SetupExclusivity() {
  m_dynamic_exclusivity_flag_addr = FindSymbolForSwiftObject(
      m_process, RuntimeKind::Swift, "_swift_disableExclusivityChecking",
      eSymbolTypeData);
  Log *log(GetLog(LLDBLog::Expressions));
  if (log)
    log->Printf(
        "SwiftLanguageRuntime: _swift_disableExclusivityChecking = %llu",
        m_dynamic_exclusivity_flag_addr ? *m_dynamic_exclusivity_flag_addr : 0);
}

llvm::Optional<lldb::addr_t>
SwiftLanguageRuntimeImpl::GetDynamicExclusivityFlagAddr() {
  return m_dynamic_exclusivity_flag_addr;
}

void SwiftLanguageRuntimeImpl::SetupABIBit() {
  if (FindSymbolForSwiftObject(m_process, RuntimeKind::ObjC,
                               "objc_debug_swift_stable_abi_bit",
                               eSymbolTypeAny))
    _swift_classIsSwiftMask = 2;
  else
    _swift_classIsSwiftMask = 1;
}

SwiftLanguageRuntimeImpl::SwiftLanguageRuntimeImpl(Process &process)
    : m_process(process) {
  // The global ABI bit is read by the Swift runtime library.
  SetupExclusivity();
  SetupSwiftError();
  Target &target = m_process.GetTarget();
  m_modules_to_add.Append(target.GetImages());
  RegisterSwiftRuntimeFailureRecognizer(m_process);
}

LanguageRuntime *
SwiftLanguageRuntime::CreateInstance(Process *process,
                                     lldb::LanguageType language) {
  if ((language != eLanguageTypeSwift) || !process)
    return nullptr;
  return new SwiftLanguageRuntime(process);
}

SwiftLanguageRuntime::SwiftLanguageRuntime(Process *process)
    : LanguageRuntime(process) {
  // It's not possible to bring up a full SwiftLanguageRuntime if the Swift
  // runtime library hasn't been loaded yet.
  if (findRuntime(*process, RuntimeKind::Swift))
    m_impl = std::make_unique<SwiftLanguageRuntimeImpl>(*process);
  else
    m_stub = std::make_unique<SwiftLanguageRuntimeStub>(*process);
}

void SwiftLanguageRuntime::ModulesDidLoad(const ModuleList &module_list) {
  assert(m_process && "modules loaded without process");
  if (m_impl) {
    m_impl->ModulesDidLoad(module_list);
    return;
  }

  bool did_load_runtime = false;
  module_list.ForEach([&](const ModuleSP &module_sp) -> bool {
    did_load_runtime |= IsModuleSwiftRuntime(*m_process, *module_sp) ||
                        IsStaticSwiftRuntime(*module_sp);
    return !did_load_runtime;
  });
  if (did_load_runtime) {
    m_impl = std::make_unique<SwiftLanguageRuntimeImpl>(*m_process);
    m_impl->ModulesDidLoad(module_list);
  }
}

static std::unique_ptr<swift::SwiftObjectFileFormat>
GetObjectFileFormat(llvm::Triple::ObjectFormatType obj_format_type) {
  std::unique_ptr<swift::SwiftObjectFileFormat> obj_file_format;
  switch (obj_format_type) {
  case llvm::Triple::MachO:
    obj_file_format = std::make_unique<swift::SwiftObjectFileFormatMachO>();
    break;
  case llvm::Triple::ELF:
    obj_file_format = std::make_unique<swift::SwiftObjectFileFormatELF>();
    break;
  case llvm::Triple::COFF:
    obj_file_format = std::make_unique<swift::SwiftObjectFileFormatCOFF>();
    break;
  default:
    if (Log *log = GetLog(LLDBLog::Types))
      log->Printf("%s: Could not find out swift reflection section names for "
                  "object format type.",
                  __FUNCTION__);
  }
  return obj_file_format;
}

bool SwiftLanguageRuntimeImpl::AddJitObjectFileToReflectionContext(
    ObjectFile &obj_file, llvm::Triple::ObjectFormatType obj_format_type) {
  assert(obj_file.GetType() == ObjectFile::eTypeJIT &&
         "Not a JIT object file!");
  auto obj_file_format = GetObjectFileFormat(obj_format_type);

  if (!obj_file_format)
    return false;

  return m_reflection_ctx->addImage(
      [&](swift::ReflectionSectionKind section_kind)
          -> std::pair<swift::remote::RemoteRef<void>, uint64_t> {
        auto section_name =
            obj_file_format->getSectionName(section_kind);
        for (auto section : *obj_file.GetSectionList()) {
          JITSection *jit_section = llvm::dyn_cast<JITSection>(section.get());
          if (jit_section && section->GetName().AsCString() == section_name) {
            DataExtractor extractor;
            auto section_size = section->GetSectionData(extractor);
            if (!section_size)
              return {};
            auto size = jit_section->getNonJitSize();
            auto data = extractor.GetData();
            if (section_size < size || !data.begin())
              return {};

            auto *Buf = malloc(size);
            std::memcpy(Buf, data.begin(), size);
            swift::remote::RemoteRef<void> remote_ref(section->GetFileAddress(),
                                                      Buf);

            return {remote_ref, size};
          }
        }
        return {};
      });
}

bool SwiftLanguageRuntimeImpl::AddObjectFileToReflectionContext(
    ModuleSP module) {
  auto obj_format_type =
      module->GetArchitecture().GetTriple().getObjectFormat();

  auto obj_file_format = GetObjectFileFormat(obj_format_type);
  if (!obj_file_format)
    return false;

  bool should_register_with_symbol_obj_file = [&]() -> bool {
    if (!m_process.GetTarget().GetSwiftReadMetadataFromDSYM())
      return false;
    auto *symbol_file = module->GetSymbolFile();
    if (!symbol_file)
      return false;
    auto *sym_obj_file = symbol_file->GetObjectFile();
    if (!sym_obj_file)
      return false;

    llvm::Optional<llvm::StringRef> maybe_segment_name =
        obj_file_format->getSymbolRichSegmentName();
    if (!maybe_segment_name)
      return false;

    llvm::StringRef segment_name = *maybe_segment_name;

    auto *section_list = sym_obj_file->GetSectionList();
    auto segment_iter = llvm::find_if(*section_list, [&](auto segment) {
      return segment->GetName() == segment_name.begin();
    });

    if (segment_iter == section_list->end())
      return false;

    auto *segment = segment_iter->get();

    auto section_iter =
        llvm::find_if(segment->GetChildren(), [&](auto section) {
          return obj_file_format->sectionContainsReflectionData(
              section->GetName().GetStringRef());
        });
    return section_iter != segment->GetChildren().end();
  }();

  llvm::Optional<llvm::StringRef> maybe_segment_name;
  llvm::Optional<llvm::StringRef> maybe_secondary_segment_name;
  ObjectFile *object_file;
  if (should_register_with_symbol_obj_file) {
    maybe_segment_name = obj_file_format->getSymbolRichSegmentName();
    maybe_secondary_segment_name = obj_file_format->getSegmentName();
    object_file = module->GetSymbolFile()->GetObjectFile();
  } else {
    maybe_segment_name = obj_file_format->getSegmentName();
    object_file = module->GetObjectFile();
  }

  if (!maybe_segment_name)
    return false;

  llvm::StringRef segment_name = *maybe_segment_name;

  auto lldb_memory_reader = GetMemoryReader();
  auto maybe_start_and_end = lldb_memory_reader->addModuleToAddressMap(
      module, should_register_with_symbol_obj_file);
  if (!maybe_start_and_end)
    return false;

  uint64_t start_address, end_address;
  std::tie(start_address, end_address) = *maybe_start_and_end;

  auto *section_list = object_file->GetSectionList();
  auto segment_iter = llvm::find_if(*section_list, [&](auto segment) {
    return segment->GetName() == segment_name.begin();
  });

  if (segment_iter == section_list->end())
    return false;

  auto *segment = segment_iter->get();
  Section *maybe_secondary_segment = nullptr;
  if (maybe_secondary_segment_name) {
    auto secondary_segment_name = *maybe_secondary_segment_name;
    auto segment_iter = llvm::find_if(*section_list, [&](auto segment) {
      return segment->GetName() == secondary_segment_name.begin();
    });

    if (segment_iter != section_list->end())
      maybe_secondary_segment = segment_iter->get();
  }
  auto find_section_with_kind = [&](Section *segment,
                                    swift::ReflectionSectionKind section_kind)
      -> std::pair<swift::remote::RemoteRef<void>, uint64_t> {
    if (!segment)
      return {};

    auto section_name = obj_file_format->getSectionName(section_kind);
    for (auto section : segment->GetChildren()) {
      // Iterate over the sections until we find the reflection section we
      // need.
      if (section->GetName().AsCString() == section_name) {
        DataExtractor extractor;
        auto size = section->GetSectionData(extractor);
        auto data = extractor.GetData();
        size = section->GetFileSize();
        if (!data.begin())
          return {};

        // Alloc a buffer and copy over the reflection section's contents.
        // This buffer will be owned by reflection context.
        auto *Buf = malloc(size);
        std::memcpy(Buf, data.begin(), size);

        // The section's address is the start address for this image
        // added with the section's virtual address. We need to use the
        // virtual address instead of the file offset because the offsets
        // encoded in the reflection section are calculated in the virtual
        // address space.
        auto address = start_address + section->GetFileAddress();
        assert(address <= end_address && "Address outside of range!");

        swift::remote::RemoteRef<void> remote_ref(address, Buf);
        return {remote_ref, size};
      }
    }
    return {};
  };
  return m_reflection_ctx->addImage(
      [&](swift::ReflectionSectionKind section_kind)
          -> std::pair<swift::remote::RemoteRef<void>, uint64_t> {
        auto pair = find_section_with_kind(segment, section_kind);
        if (pair.first)
          return pair;
        return find_section_with_kind(maybe_secondary_segment, section_kind);
      });
}

bool SwiftLanguageRuntimeImpl::AddModuleToReflectionContext(
    const lldb::ModuleSP &module_sp) {
  // This function is called from within SetupReflection so it cannot
  // call GetReflectionContext().
  assert(m_initialized_reflection_ctx);
  if (!m_reflection_ctx)
    return false;
  if (!module_sp)
    return false;
  auto *obj_file = module_sp->GetObjectFile();
  if (!obj_file)
    return false;
  auto &target = m_process.GetTarget();
  Address start_address = obj_file->GetBaseAddress();
  auto load_ptr = static_cast<uintptr_t>(
      start_address.GetLoadAddress(&target));
  if (obj_file->GetType() == ObjectFile::eTypeJIT) {
    auto object_format_type =
        module_sp->GetArchitecture().GetTriple().getObjectFormat();
    return AddJitObjectFileToReflectionContext(*obj_file, object_format_type);
  }

  if (load_ptr == 0 || load_ptr == LLDB_INVALID_ADDRESS) {
    if (obj_file->GetType() != ObjectFile::eTypeJIT)
      if (Log *log = GetLog(LLDBLog::Types))
        log->Printf("%s: failed to get start address for %s.", __FUNCTION__,
                    obj_file->GetFileSpec().GetFilename().GetCString());
    return false;
  }
  bool found = HasReflectionInfo(obj_file);
  LLDB_LOGF(GetLog(LLDBLog::Types), "%s reflection metadata in \"%s\"",
            found ? "Adding" : "No", obj_file->GetFileSpec().GetCString());
  if (!found)
    return true;

  auto read_from_file_cache =
      GetMemoryReader()->readMetadataFromFileCacheEnabled();
  // When dealing with ELF, we need to pass in the contents of the on-disk
  // file, since the Section Header Table is not present in the child process
  if (obj_file->GetPluginName().equals("elf")) {
    DataExtractor extractor;
    auto size = obj_file->GetData(0, obj_file->GetByteSize(), extractor);
    const uint8_t *file_data = extractor.GetDataStart();
    llvm::sys::MemoryBlock file_buffer((void *)file_data, size);
    m_reflection_ctx->readELF(
        swift::remote::RemoteAddress(load_ptr),
        llvm::Optional<llvm::sys::MemoryBlock>(file_buffer));
  } else if (read_from_file_cache &&
             obj_file->GetPluginName().equals("mach-o")) {
    if (!AddObjectFileToReflectionContext(module_sp))
      m_reflection_ctx->addImage(swift::remote::RemoteAddress(load_ptr));
  } else {
    m_reflection_ctx->addImage(swift::remote::RemoteAddress(load_ptr));
  }
  return true;
}

void SwiftLanguageRuntimeImpl::ModulesDidLoad(const ModuleList &module_list) {
  // If the reflection context hasn't been initialized, add them to
  // the list of deferred modules so they are added in
  // SetupReflection(), otherwise add them directly.
  std::lock_guard<std::recursive_mutex> lock(m_add_module_mutex);
  if (!m_initialized_reflection_ctx)
    m_modules_to_add.AppendIfNeeded(module_list);
  else
    module_list.ForEach([&](const ModuleSP &module_sp) -> bool {
      AddModuleToReflectionContext(module_sp);
      return true;
    });
}

std::string 
SwiftLanguageRuntimeImpl::GetObjectDescriptionExpr_Result(ValueObject &object) {
  Log *log(GetLog(LLDBLog::DataFormatters));
  std::string expr_string
      = llvm::formatv("Swift._DebuggerSupport.stringForPrintObject({0})",
                      object.GetName().GetCString()).str();
  if (log)
    log->Printf("[GetObjectDescriptionExpr_Result] expression: %s",
                expr_string.c_str());
  return expr_string;
}

std::string 
SwiftLanguageRuntimeImpl::GetObjectDescriptionExpr_Ref(ValueObject &object) {
  Log *log(GetLog(LLDBLog::DataFormatters));

  StreamString expr_string;
  std::string expr_str 
      = llvm::formatv("Swift._DebuggerSupport.stringForPrintObject(Swift."
                      "unsafeBitCast({0:x}, to: AnyObject.self))",
                      object.GetValueAsUnsigned(0)).str();

  if (log)
    log->Printf("[GetObjectDescriptionExpr_Result] expression: %s",
                expr_string.GetData());
  return expr_str;
}

static const ExecutionContextRef *GetSwiftExeCtx(ValueObject &valobj) {
  return (valobj.GetPreferredDisplayLanguage() == eLanguageTypeSwift)
             ? &valobj.GetExecutionContextRef()
             : nullptr;
}

std::string 
SwiftLanguageRuntimeImpl::GetObjectDescriptionExpr_Copy(ValueObject &object,
    lldb::addr_t &copy_location)
{
  Log *log(GetLog(LLDBLog::DataFormatters));

  ValueObjectSP static_sp(object.GetStaticValue());

  CompilerType static_type(static_sp->GetCompilerType());
  if (auto non_reference_type = static_type.GetNonReferenceType())
    static_type = non_reference_type;

  // If we are in a generic context, here the static type of the object
  // might end up being generic (i.e. <T>). We want to make sure that
  // we correctly map the type into context before asking questions or
  // printing, as IRGen requires a fully realized type to work on.
  StackFrameSP frame_sp = object.GetFrameSP();
  if (!frame_sp)
      frame_sp 
          = m_process.GetThreadList().GetSelectedThread()->GetSelectedFrame();

  auto *swift_ast_ctx =
      llvm::dyn_cast_or_null<TypeSystemSwift>(static_type.GetTypeSystem());
  if (swift_ast_ctx) {
    SwiftScratchContextLock lock(GetSwiftExeCtx(object));
    static_type = BindGenericTypeParameters(*frame_sp, static_type);
  }

  auto stride = 0;
  auto opt_stride = static_type.GetByteStride(frame_sp.get());
  if (opt_stride)
    stride = *opt_stride;

  Status error;
  copy_location = m_process.AllocateMemory(
      stride, ePermissionsReadable | ePermissionsWritable, error);
  if (copy_location == LLDB_INVALID_ADDRESS) {
    if (log)
      log->Printf("[GetObjectDescriptionExpr_Copy] copy_location invalid");
    return {};
  }

  DataExtractor data_extractor;
  if (0 == static_sp->GetData(data_extractor, error)) {
    if (log)
      log->Printf("[GetObjectDescriptionExpr_Copy] data extraction failed");
    return {};
  }

  if (0 == m_process.WriteMemory(copy_location, data_extractor.GetDataStart(),
                               data_extractor.GetByteSize(), error)) {
    if (log)
      log->Printf("[GetObjectDescriptionExpr_Copy] memory copy failed");
    return {};
  }

  std::string expr_string 
      = llvm::formatv("Swift._DebuggerSupport.stringForPrintObject(Swift."
                      "UnsafePointer<{0}>(bitPattern: {1:x})!.pointee)",
                      static_type.GetTypeName().GetCString(), copy_location).str();
  if (log)
    log->Printf("[GetObjectDescriptionExpr_Copy] expression: %s",
                expr_string.c_str());
                
  return expr_string;
}

bool 
SwiftLanguageRuntimeImpl::RunObjectDescriptionExpr(ValueObject &object, 
    std::string &expr_string, 
    Stream &result)
{
  Log *log(GetLog(LLDBLog::DataFormatters));
  ValueObjectSP result_sp;
  EvaluateExpressionOptions eval_options;
  eval_options.SetLanguage(lldb::eLanguageTypeSwift);
  eval_options.SetResultIsInternal(true);
  eval_options.SetGenerateDebugInfo(true);
  eval_options.SetTimeout(m_process.GetUtilityExpressionTimeout());
  
  StackFrameSP frame_sp = object.GetFrameSP();
  if (!frame_sp)
    frame_sp 
        = m_process.GetThreadList().GetSelectedThread()->GetSelectedFrame();
  auto eval_result = m_process.GetTarget().EvaluateExpression(
      expr_string,
      frame_sp.get(),
      result_sp, eval_options);

  if (log) {
    const char *eval_result_str 
        = m_process.ExecutionResultAsCString(eval_result);
    log->Printf("[RunObjectDescriptionExpr] %s", eval_result_str);
  }

  // Sanity check the result of the expression before moving forward
  if (!result_sp) {
    if (log)
      log->Printf(
          "[RunObjectDescriptionExpr] expression generated no result");

    result.Printf("expression produced no result");
    return false;
  }
  if (result_sp->GetError().Fail()) {
    if (log)
      log->Printf(
          "[RunObjectDescriptionExpr] expression generated error: %s",
          result_sp->GetError().AsCString());

    result.Printf("expression produced error: %s",
               result_sp->GetError().AsCString());
    return false;
  }
  if (false == result_sp->GetCompilerType().IsValid()) {
    if (log)
      log->Printf("[RunObjectDescriptionExpr] expression generated "
                  "invalid type");

    result.Printf("expression produced invalid result type");
    return false;
  }

  formatters::StringPrinter::ReadStringAndDumpToStreamOptions dump_options;
  dump_options.SetEscapeNonPrintables(false);
  dump_options.SetQuote('\0');
  dump_options.SetPrefixToken(nullptr);
  if (formatters::swift::String_SummaryProvider(
          *result_sp.get(), result,
          TypeSummaryOptions()
              .SetLanguage(lldb::eLanguageTypeSwift)
              .SetCapping(eTypeSummaryUncapped),
          dump_options)) {
    if (log)
      log->Printf("[RunObjectDescriptionExpr] expression completed "
                  "successfully");
    return true;
  } else {
    if (log)
      log->Printf("[RunObjectDescriptionExpr] expression generated "
                  "invalid string data");

    result.Printf("expression produced unprintable string");
    return false;
  }
}

static bool IsSwiftResultVariable(ConstString name) {
  if (name) {
    llvm::StringRef name_sr(name.GetStringRef());
    if (name_sr.size() > 2 &&
        (name_sr.startswith("$R") || name_sr.startswith("$E")) &&
        ::isdigit(name_sr[2]))
      return true;
  }
  return false;
}

static bool IsSwiftReferenceType(ValueObject &object) {
  CompilerType object_type(object.GetCompilerType());
  if (llvm::dyn_cast_or_null<TypeSystemSwift>(object_type.GetTypeSystem())) {
    Flags type_flags(object_type.GetTypeInfo());
    if (type_flags.AllSet(eTypeIsClass | eTypeHasValue |
                          eTypeInstanceIsPointer))
      return true;
  }
  return false;
}

bool SwiftLanguageRuntimeImpl::GetObjectDescription(Stream &str,
                                                    ValueObject &object) {
  if (object.IsUninitializedReference()) {
    str.Printf("<uninitialized>");
    return true;
  }

  std::string expr_string;
  
  if (::IsSwiftResultVariable(object.GetName())) {
    // if this thing is a Swift expression result variable, it has two
    // properties:
    // a) its name is something we can refer to in expressions for free
    // b) its type may be something we can't actually talk about in expressions
    // so, just use the result variable's name in the expression and be done
    // with it
    expr_string = GetObjectDescriptionExpr_Result(object);
  } else if (::IsSwiftReferenceType(object)) {
    // if this is a Swift class, it has two properties:
    // a) we do not need its type name, AnyObject is just as good
    // b) its value is something we can directly use to refer to it
    // so, just use the ValueObject's pointer-value and be done with it
    expr_string = GetObjectDescriptionExpr_Ref(object);
  }
  if (!expr_string.empty()) {
    StreamString probe_stream;
    if (RunObjectDescriptionExpr(object, expr_string, probe_stream)) {
      str.Printf("%s", probe_stream.GetData());
      return true;
    }
  }
  // In general, don't try to use the name of the ValueObject as it might end up
  // referring to the wrong thing.  Instead, copy the object data into the
  // target and call object description on the copy.
  lldb::addr_t copy_location = LLDB_INVALID_ADDRESS;
  expr_string = GetObjectDescriptionExpr_Copy(object, copy_location);
  if (copy_location == LLDB_INVALID_ADDRESS) {
    str.Printf("Failed to allocate memory for copy object.");
    return false;
  }

  auto cleanup =
      llvm::make_scope_exit([&]() { m_process.DeallocateMemory(copy_location); });

  if (expr_string.empty())
    return false;
  return RunObjectDescriptionExpr(object, expr_string, str);
}

StructuredDataImpl *
SwiftLanguageRuntime::GetLanguageSpecificData(StackFrame &frame) {
  auto sc = frame.GetSymbolContext(eSymbolContextFunction);
  if (!sc.function)
    return nullptr;

  auto dict_sp = std::make_shared<StructuredData::Dictionary>();
  auto symbol = sc.function->GetMangled().GetMangledName().GetStringRef();
  auto is_async = SwiftLanguageRuntime::IsAnySwiftAsyncFunctionSymbol(symbol);
  dict_sp->AddBooleanItem("IsSwiftAsyncFunction", is_async);

  auto *data = new StructuredDataImpl;
  data->SetObjectSP(dict_sp);
  return data;
}

void SwiftLanguageRuntime::FindFunctionPointersInCall(
    StackFrame &frame, std::vector<Address> &addresses, bool debug_only,
    bool resolve_thunks) {
  // Extract the mangled name from the stack frame, and realize the
  // function type in the Target's SwiftASTContext.  Then walk the
  // arguments looking for function pointers.  If we find one in the
  // FIRST argument, we can fetch the pointer value and return that.
  // FIXME: when we can ask swift/llvm for the location of function
  // arguments, then we can do this for all the function pointer
  // arguments we find.

  SymbolContext sc = frame.GetSymbolContext(eSymbolContextSymbol);
  if (sc.symbol) {
    Mangled mangled_name = sc.symbol->GetMangled();
    if (mangled_name.GuessLanguage() == lldb::eLanguageTypeSwift) {
      Status error;
      Target &target = frame.GetThread()->GetProcess()->GetTarget();
      ExecutionContext exe_ctx(frame);
      llvm::Optional<SwiftScratchContextReader> maybe_swift_ast =
          target.GetSwiftScratchContext(error, frame);
      auto scratch_ctx = maybe_swift_ast->get();
      if (scratch_ctx) {
        if (SwiftASTContext *swift_ast = scratch_ctx->GetSwiftASTContext()) {
        CompilerType function_type = swift_ast->GetTypeFromMangledTypename(
            mangled_name.GetMangledName());
        if (error.Success()) {
          if (function_type.IsFunctionType()) {
            // FIXME: For now we only check the first argument since
            // we don't know how to find the values of arguments
            // further in the argument list.
            //
            // int num_arguments = function_type.GetFunctionArgumentCount();
            // for (int i = 0; i < num_arguments; i++)

            for (int i = 0; i < 1; i++) {
              CompilerType argument_type =
                  function_type.GetFunctionArgumentTypeAtIndex(i);
              if (argument_type.IsFunctionPointerType()) {
                // We found a function pointer argument.  Try to track
                // down its value.  This is a hack for now, we really
                // should ask swift/llvm how to find the argument(s)
                // given the Swift decl for this function, and then
                // look those up in the frame.

                ABISP abi_sp(frame.GetThread()->GetProcess()->GetABI());
                ValueList argument_values;
                Value input_value;
                auto clang_ctx = ScratchTypeSystemClang::GetForTarget(target);
                if (!clang_ctx)
                  continue;

                CompilerType clang_void_ptr_type =
                    clang_ctx->GetBasicType(eBasicTypeVoid).GetPointerType();

                input_value.SetValueType(Value::ValueType::Scalar);
                input_value.SetCompilerType(clang_void_ptr_type);
                argument_values.PushValue(input_value);

                bool success = abi_sp->GetArgumentValues(
                    *(frame.GetThread().get()), argument_values);
                if (success) {
                  // Now get a pointer value from the zeroth argument.
                  Status error;
                  DataExtractor data;
                  ExecutionContext exe_ctx;
                  frame.CalculateExecutionContext(exe_ctx);
                  error = argument_values.GetValueAtIndex(0)->GetValueAsData(
                      &exe_ctx, data, NULL);
                  lldb::offset_t offset = 0;
                  lldb::addr_t fn_ptr_addr = data.GetAddress(&offset);
                  Address fn_ptr_address;
                  fn_ptr_address.SetLoadAddress(fn_ptr_addr, &target);
                  // Now check to see if this has debug info:
                  bool add_it = true;

                  if (resolve_thunks) {
                    SymbolContext sc;
                    fn_ptr_address.CalculateSymbolContext(
                        &sc, eSymbolContextEverything);
                    if (sc.comp_unit && sc.symbol) {
                      ConstString symbol_name =
                          sc.symbol->GetMangled().GetMangledName();
                      if (symbol_name) {
                        SymbolContext target_context;
                        if (GetTargetOfPartialApply(sc, symbol_name,
                                                    target_context)) {
                          if (target_context.symbol)
                            fn_ptr_address =
                                target_context.symbol->GetAddress();
                          else if (target_context.function)
                            fn_ptr_address =
                                target_context.function->GetAddressRange()
                                    .GetBaseAddress();
                        }
                      }
                    }
                  }

                  if (debug_only) {
                    LineEntry line_entry;
                    fn_ptr_address.CalculateSymbolContextLineEntry(line_entry);
                    if (!line_entry.IsValid())
                      add_it = false;
                  }
                  if (add_it)
                    addresses.push_back(fn_ptr_address);
                }
              }
            }
          }
        }
      }
      }
    }
  }
}

//------------------------------------------------------------------
// Exception breakpoint Precondition class for Swift:
//------------------------------------------------------------------
void SwiftLanguageRuntimeImpl::SwiftExceptionPrecondition::AddTypeName(
    const char *class_name) {
  m_type_names.insert(class_name);
}

void SwiftLanguageRuntimeImpl::SwiftExceptionPrecondition::AddEnumSpec(
    const char *enum_name, const char *element_name) {
  std::unordered_map<std::string, std::vector<std::string>>::value_type
      new_value(enum_name, std::vector<std::string>());
  auto result = m_enum_spec.emplace(new_value);
  result.first->second.push_back(element_name);
}

SwiftLanguageRuntimeImpl::SwiftExceptionPrecondition::
    SwiftExceptionPrecondition() {}

ValueObjectSP SwiftLanguageRuntime::CalculateErrorValueObjectFromValue(
    Value &value, ConstString name, bool persistent) {
  if (!m_process)
    return {};
  ValueObjectSP error_valobj_sp;
  auto type_system_or_err =
      m_process->GetTarget().GetScratchTypeSystemForLanguage(
          eLanguageTypeSwift);
  if (!type_system_or_err)
    return error_valobj_sp;

  auto *ast_context =
      llvm::dyn_cast_or_null<TypeSystemSwift>(&*type_system_or_err);
  if (!ast_context)
    return error_valobj_sp;

  CompilerType swift_error_proto_type = ast_context->GetErrorType();
  value.SetCompilerType(swift_error_proto_type);

  error_valobj_sp = ValueObjectConstResult::Create(m_process, value, name);

  if (error_valobj_sp && error_valobj_sp->GetError().Success()) {
    error_valobj_sp = error_valobj_sp->GetQualifiedRepresentationIfAvailable(
        lldb::eDynamicCanRunTarget, true);
    if (!IsValidErrorValue(*(error_valobj_sp.get()))) {
      error_valobj_sp.reset();
    }
  }

  if (persistent && error_valobj_sp) {
    ExecutionContext ctx =
        error_valobj_sp->GetExecutionContextRef().Lock(false);
    auto *exe_scope = ctx.GetBestExecutionContextScope();
    if (!exe_scope)
      return error_valobj_sp;
    Target &target = m_process->GetTarget();
    auto *persistent_state =
        target.GetSwiftPersistentExpressionState(*exe_scope);

    ConstString persistent_variable_name(
        persistent_state->GetNextPersistentVariableName(/*is_error*/ true));

    lldb::ValueObjectSP const_valobj_sp;

    // Check in case our value is already a constant value
    if (error_valobj_sp->GetIsConstant()) {
      const_valobj_sp = error_valobj_sp;
      const_valobj_sp->SetName(persistent_variable_name);
    } else
      const_valobj_sp =
          error_valobj_sp->CreateConstantValue(persistent_variable_name);

    lldb::ValueObjectSP live_valobj_sp = error_valobj_sp;

    error_valobj_sp = const_valobj_sp;

    ExpressionVariableSP clang_expr_variable_sp(
        persistent_state->CreatePersistentVariable(error_valobj_sp));
    clang_expr_variable_sp->m_live_sp = live_valobj_sp;
    clang_expr_variable_sp->m_flags |=
        ClangExpressionVariable::EVIsProgramReference;

    error_valobj_sp = clang_expr_variable_sp->GetValueObject();
  }
  return error_valobj_sp;
}

ValueObjectSP
SwiftLanguageRuntime::CalculateErrorValue(StackFrameSP frame_sp,
                                          ConstString variable_name) {
  ProcessSP process_sp(frame_sp->GetThread()->GetProcess());
  Status error;
  Target *target = frame_sp->CalculateTarget().get();
  ValueObjectSP error_valobj_sp;

  auto *runtime = Get(process_sp);
  if (!runtime)
    return error_valobj_sp;

  llvm::Optional<Value> arg0 =
      runtime->GetErrorReturnLocationAfterReturn(frame_sp);
  if (!arg0)
    return error_valobj_sp;

  ExecutionContext exe_ctx;
  frame_sp->CalculateExecutionContext(exe_ctx);

  auto *exe_scope = exe_ctx.GetBestExecutionContextScope();
  if (!exe_scope)
    return error_valobj_sp;

  llvm::Optional<SwiftScratchContextReader> maybe_scratch_context =
      target->GetSwiftScratchContext(error, *frame_sp);
  if (!maybe_scratch_context || error.Fail())
    return error_valobj_sp;
  auto scratch_ctx = maybe_scratch_context->get();
  if (!scratch_ctx)
    return error_valobj_sp;
  SwiftASTContext *ast_context = scratch_ctx->GetSwiftASTContext();
  if (!ast_context)
    return error_valobj_sp;


  auto buffer_up =
      std::make_unique<DataBufferHeap>(arg0->GetScalar().GetByteSize(), 0);
  arg0->GetScalar().GetBytes(buffer_up->GetData());
  lldb::DataBufferSP buffer(std::move(buffer_up));

  CompilerType swift_error_proto_type = ast_context->GetErrorType();
  if (!swift_error_proto_type.IsValid())
    return error_valobj_sp;

  error_valobj_sp = ValueObjectConstResult::Create(
      exe_scope, swift_error_proto_type, variable_name, buffer,
      endian::InlHostByteOrder(), exe_ctx.GetAddressByteSize());
  if (error_valobj_sp->GetError().Fail())
    return error_valobj_sp;

  error_valobj_sp = error_valobj_sp->GetQualifiedRepresentationIfAvailable(
      lldb::eDynamicCanRunTarget, true);
  return error_valobj_sp;
}

void SwiftLanguageRuntime::RegisterGlobalError(Target &target, ConstString name,
                                               lldb::addr_t addr) {
  auto type_system_or_err =
      target.GetScratchTypeSystemForLanguage(eLanguageTypeSwift);
  if (!type_system_or_err) {
    llvm::consumeError(type_system_or_err.takeError());
    return;
  }

  auto *ast_context = llvm::dyn_cast_or_null<SwiftASTContextForExpressions>(
      &*type_system_or_err);
  if (ast_context && !ast_context->HasFatalErrors()) {
    std::string module_name = "$__lldb_module_for_";
    module_name.append(&name.GetCString()[1]);
    SourceModule module_info;
    module_info.path.push_back(ConstString(module_name));

    Status module_creation_error;
    swift::ModuleDecl *module_decl =
        ast_context->CreateModule(module_info, module_creation_error,
                                  /*importInfo*/ {});

    if (module_creation_error.Success() && module_decl) {
      const bool is_static = false;
      const auto introducer = swift::VarDecl::Introducer::Let;

      swift::VarDecl *var_decl =
          new (*ast_context->GetASTContext()) swift::VarDecl(
              is_static, introducer, swift::SourceLoc(),
              ast_context->GetIdentifier(name.GetCString()), module_decl);
      var_decl->setInterfaceType(GetSwiftType(ast_context->GetErrorType()));
      var_decl->setDebuggerVar(true);

      SwiftPersistentExpressionState *persistent_state =
          llvm::cast<SwiftPersistentExpressionState>(
              target.GetPersistentExpressionStateForLanguage(
                  lldb::eLanguageTypeSwift));
      if (!persistent_state)
        return;

      persistent_state->RegisterSwiftPersistentDecl(var_decl);

      ConstString mangled_name;

      {
        swift::Mangle::ASTMangler mangler(true);
        mangled_name = ConstString(mangler.mangleGlobalVariableFull(var_decl));
      }

      lldb::addr_t symbol_addr;

      {
        ProcessSP process_sp(target.GetProcessSP());
        Status alloc_error;

        symbol_addr = process_sp->AllocateMemory(
            process_sp->GetAddressByteSize(),
            lldb::ePermissionsWritable | lldb::ePermissionsReadable,
            alloc_error);

        if (alloc_error.Success() && symbol_addr != LLDB_INVALID_ADDRESS) {
          Status write_error;
          process_sp->WritePointerToMemory(symbol_addr, addr, write_error);

          if (write_error.Success()) {
            persistent_state->RegisterSymbol(mangled_name, symbol_addr);
          }
        }
      }
    }
  }
}

lldb::BreakpointPreconditionSP
SwiftLanguageRuntimeImpl::GetBreakpointExceptionPrecondition(
    LanguageType language, bool throw_bp) {
  if (language != eLanguageTypeSwift)
    return lldb::BreakpointPreconditionSP();
  if (!throw_bp)
    return lldb::BreakpointPreconditionSP();
  BreakpointPreconditionSP precondition_sp(
      new SwiftLanguageRuntimeImpl::SwiftExceptionPrecondition());
  return precondition_sp;
}

bool SwiftLanguageRuntimeImpl::SwiftExceptionPrecondition::EvaluatePrecondition(
    StoppointCallbackContext &context) {
  if (!m_type_names.empty()) {
    StackFrameSP frame_sp = context.exe_ctx_ref.GetFrameSP();
    if (!frame_sp)
      return true;

    ValueObjectSP error_valobj_sp = SwiftLanguageRuntime::CalculateErrorValue(
        frame_sp, ConstString("__swift_error_var"));
    if (!error_valobj_sp || error_valobj_sp->GetError().Fail())
      return true;

    // This shouldn't fail, since at worst it will return me the object I just
    // successfully got.
    std::string full_error_name(
        error_valobj_sp->GetCompilerType().GetTypeName().AsCString());
    size_t last_dot_pos = full_error_name.rfind('.');
    std::string type_name_base;
    if (last_dot_pos == std::string::npos)
      type_name_base = full_error_name;
    else {
      if (last_dot_pos + 1 <= full_error_name.size())
        type_name_base =
            full_error_name.substr(last_dot_pos + 1, full_error_name.size());
    }

    // The type name will be the module and then the type.  If the match name
    // has a dot, we require a complete
    // match against the type, if the type name has no dot, we match it against
    // the base.

    for (std::string name : m_type_names) {
      if (name.rfind('.') != std::string::npos) {
        if (name == full_error_name)
          return true;
      } else {
        if (name == type_name_base)
          return true;
      }
    }
    return false;
  }
  return true;
}

void SwiftLanguageRuntimeImpl::SwiftExceptionPrecondition::GetDescription(
    Stream &stream, lldb::DescriptionLevel level) {
  if (level == eDescriptionLevelFull || level == eDescriptionLevelVerbose) {
    if (m_type_names.size() > 0) {
      stream.Printf("\nType Filters:");
      for (std::string name : m_type_names) {
        stream.Printf(" %s", name.c_str());
      }
      stream.Printf("\n");
    }
  }
}

Status
SwiftLanguageRuntimeImpl::SwiftExceptionPrecondition::ConfigurePrecondition(
    Args &args) {
  Status error;
  std::vector<std::string> object_typenames;
  OptionParsing::GetOptionValuesAsStrings(args, "exception-typename",
                                          object_typenames);
  for (auto type_name : object_typenames)
    AddTypeName(type_name.c_str());
  return error;
}

void SwiftLanguageRuntimeImpl::AddToLibraryNegativeCache(
    StringRef library_name) {
  std::lock_guard<std::mutex> locker(m_negative_cache_mutex);
  m_library_negative_cache.insert(library_name);
}

bool SwiftLanguageRuntimeImpl::IsInLibraryNegativeCache(
    StringRef library_name) {
  std::lock_guard<std::mutex> locker(m_negative_cache_mutex);
  return m_library_negative_cache.count(library_name) == 1;
}

class ProjectionSyntheticChildren : public SyntheticChildren {
public:
  struct FieldProjection {
    ConstString name;
    CompilerType type;
    int32_t byte_offset;

    FieldProjection(CompilerType parent_type, ExecutionContext *exe_ctx,
                    size_t idx) {
      const bool transparent_pointers = false;
      const bool omit_empty_base_classes = true;
      const bool ignore_array_bounds = false;
      bool child_is_base_class = false;
      bool child_is_deref_of_parent = false;
      std::string child_name;

      uint32_t child_byte_size;
      uint32_t child_bitfield_bit_size;
      uint32_t child_bitfield_bit_offset;
      uint64_t language_flags;

      type = parent_type.GetChildCompilerTypeAtIndex(
          exe_ctx, idx, transparent_pointers, omit_empty_base_classes,
          ignore_array_bounds, child_name, child_byte_size, byte_offset,
          child_bitfield_bit_size, child_bitfield_bit_offset,
          child_is_base_class, child_is_deref_of_parent, nullptr,
          language_flags);

      if (child_is_base_class)
        type.Clear(); // invalidate - base classes are dealt with outside of the
                      // projection
      else
        name.SetCStringWithLength(child_name.c_str(), child_name.size());
    }

    bool IsValid() { return !name.IsEmpty() && type.IsValid(); }

    explicit operator bool() { return IsValid(); }
  };

  struct TypeProjection {
    std::vector<FieldProjection> field_projections;
    ConstString type_name;
  };

  typedef std::unique_ptr<TypeProjection> TypeProjectionUP;

  bool IsScripted() override { return false; }

  std::string GetDescription() override {
    return "projection synthetic children";
  }

  ProjectionSyntheticChildren(const Flags &flags, TypeProjectionUP &&projection)
      : SyntheticChildren(flags), m_projection(std::move(projection)) {}

protected:
  TypeProjectionUP m_projection;

  class ProjectionFrontEndProvider : public SyntheticChildrenFrontEnd {
  public:
    ProjectionFrontEndProvider(ValueObject &backend,
                               TypeProjectionUP &projection)
        : SyntheticChildrenFrontEnd(backend), m_num_bases(0),
          m_projection(projection.get()) {
      lldbassert(m_projection && "need a valid projection");
      CompilerType type(backend.GetCompilerType());
      m_num_bases = type.GetNumDirectBaseClasses();
    }

    size_t CalculateNumChildren() override {
      return m_projection->field_projections.size() + m_num_bases;
    }

    lldb::ValueObjectSP GetChildAtIndex(size_t idx) override {
      if (idx < m_num_bases) {
        if (ValueObjectSP base_object_sp =
                m_backend.GetChildAtIndex(idx, true)) {
          CompilerType base_type(base_object_sp->GetCompilerType());
          ConstString base_type_name(base_type.GetTypeName());
          if (base_type_name.IsEmpty() ||
              !SwiftLanguageRuntime::IsSwiftClassName(
                  base_type_name.GetCString()))
            return base_object_sp;
          base_object_sp = m_backend.GetSyntheticBase(
              0, base_type, true,
              Mangled(base_type_name)
                  .GetDemangledName());
          return base_object_sp;
        } else
          return nullptr;
      }
      idx -= m_num_bases;
      if (idx < m_projection->field_projections.size()) {
        auto &projection(m_projection->field_projections.at(idx));
        return m_backend.GetSyntheticChildAtOffset(
            projection.byte_offset, projection.type, true, projection.name);
      }
      return nullptr;
    }

    size_t GetIndexOfChildWithName(ConstString name) override {
      for (size_t idx = 0; idx < m_projection->field_projections.size();
           idx++) {
        if (m_projection->field_projections.at(idx).name == name)
          return idx;
      }
      return UINT32_MAX;
    }

    bool Update() override { return false; }

    bool MightHaveChildren() override { return true; }

    ConstString GetSyntheticTypeName() override {
      return m_projection->type_name;
    }

  private:
    size_t m_num_bases;
    TypeProjectionUP::element_type *m_projection;
  };

public:
  SyntheticChildrenFrontEnd::AutoPointer
  GetFrontEnd(ValueObject &backend) override {
    return SyntheticChildrenFrontEnd::AutoPointer(
        new ProjectionFrontEndProvider(backend, m_projection));
  }
};

lldb::SyntheticChildrenSP
SwiftLanguageRuntimeImpl::GetBridgedSyntheticChildProvider(
    ValueObject &valobj) {
  ConstString type_name = valobj.GetCompilerType().GetTypeName();

  if (!type_name.IsEmpty()) {
    auto iter = m_bridged_synthetics_map.find(type_name.AsCString()),
         end = m_bridged_synthetics_map.end();
    if (iter != end)
      return iter->second;
  }

  ProjectionSyntheticChildren::TypeProjectionUP type_projection(
      new ProjectionSyntheticChildren::TypeProjectionUP::element_type());

  if (auto maybe_swift_ast_ctx = valobj.GetSwiftScratchContext()) {
    CompilerType swift_type =
        maybe_swift_ast_ctx->get()->GetTypeFromMangledTypename(type_name);

    if (swift_type.IsValid()) {
      ExecutionContext exe_ctx(m_process);
      bool any_projected = false;
      for (size_t idx = 0, e = swift_type.GetNumChildren(true, &exe_ctx);
           idx < e; idx++) {
        // if a projection fails, keep going - we have offsets here, so it
        // should be OK to skip some members
        if (auto projection = ProjectionSyntheticChildren::FieldProjection(
                swift_type, &exe_ctx, idx)) {
          any_projected = true;
          type_projection->field_projections.push_back(projection);
        }
      }

      if (any_projected) {
        type_projection->type_name = swift_type.GetDisplayTypeName();
        SyntheticChildrenSP synth_sp =
            SyntheticChildrenSP(new ProjectionSyntheticChildren(
                SyntheticChildren::Flags(), std::move(type_projection)));
        m_bridged_synthetics_map.insert({type_name.AsCString(), synth_sp});
        return synth_sp;
      }
    }
  }

  return nullptr;
}

void SwiftLanguageRuntimeImpl::WillStartExecutingUserExpression(
    bool runs_in_playground_or_repl) {
  if (runs_in_playground_or_repl)
    return;

  std::lock_guard<std::mutex> lock(m_active_user_expr_mutex);
  Log *log(GetLog(LLDBLog::Expressions));
  LLDB_LOG(log,
           "SwiftLanguageRuntime: starting user expression. "
           "Number active: %u",
           m_active_user_expr_count + 1);
  if (m_active_user_expr_count++ > 0)
    return;

  auto dynamic_exlusivity_flag_addr = GetDynamicExclusivityFlagAddr();
  if (!dynamic_exlusivity_flag_addr) {
    LLDB_LOG(log, "Failed to get address of disableExclusivityChecking flag");
    return;
  }

  // We're executing the first user expression. Toggle the flag.
  auto type_system_or_err =
      m_process.GetTarget().GetScratchTypeSystemForLanguage(
          eLanguageTypeC_plus_plus);
  if (!type_system_or_err) {
    LLDB_LOG_ERROR(
        log, type_system_or_err.takeError(),
        "SwiftLanguageRuntime: Unable to get pointer to type system");
    return;
  }

  ConstString BoolName("bool");
  llvm::Optional<uint64_t> bool_size =
      type_system_or_err->GetBuiltinTypeByName(BoolName).GetByteSize(nullptr);
  if (!bool_size)
    return;

  Status error;
  Scalar original_value;
  m_process.ReadScalarIntegerFromMemory(
      *dynamic_exlusivity_flag_addr, *bool_size, false, original_value, error);

  m_original_dynamic_exclusivity_flag_state = original_value.UInt() != 0;
  if (error.Fail()) {
    LLDB_LOG(log,
             "SwiftLanguageRuntime: Unable to read disableExclusivityChecking "
             "flag state: %s",
             error.AsCString());
    return;
  }

  Scalar new_value(1U);
  m_process.WriteScalarToMemory(*m_dynamic_exclusivity_flag_addr, new_value,
                                *bool_size, error);
  if (error.Fail()) {
    LLDB_LOG(log,
             "SwiftLanguageRuntime: Unable to set disableExclusivityChecking "
             "flag state: %s",
             error.AsCString());
    return;
  }

  LLDB_LOG(log,
           "SwiftLanguageRuntime: Changed disableExclusivityChecking flag "
           "state from %u to 1",
           m_original_dynamic_exclusivity_flag_state);
}

void SwiftLanguageRuntimeImpl::DidFinishExecutingUserExpression(
    bool runs_in_playground_or_repl) {
  if (runs_in_playground_or_repl)
    return;

  std::lock_guard<std::mutex> lock(m_active_user_expr_mutex);
  Log *log(GetLog(LLDBLog::Expressions));

  --m_active_user_expr_count;
  LLDB_LOG(log,
           "SwiftLanguageRuntime: finished user expression. "
           "Number active: %u",
           m_active_user_expr_count);

  if (m_active_user_expr_count > 0)
    return;

  auto dynamic_exlusivity_flag_addr = GetDynamicExclusivityFlagAddr();
  if (!dynamic_exlusivity_flag_addr) {
    LLDB_LOG(log, "Failed to get address of disableExclusivityChecking flag");
    return;
  }

  auto type_system_or_err =
      m_process.GetTarget().GetScratchTypeSystemForLanguage(
          eLanguageTypeC_plus_plus);
  if (!type_system_or_err) {
    LLDB_LOG_ERROR(
        log, type_system_or_err.takeError(),
        "SwiftLanguageRuntime: Unable to get pointer to type system");
    return;
  }

  ConstString BoolName("bool");
  llvm::Optional<uint64_t> bool_size =
      type_system_or_err->GetBuiltinTypeByName(BoolName).GetByteSize(nullptr);
  if (!bool_size)
    return;

  Status error;
  Scalar original_value(m_original_dynamic_exclusivity_flag_state ? 1U : 0U);
  m_process.WriteScalarToMemory(*dynamic_exlusivity_flag_addr, original_value,
                                *bool_size, error);
  if (error.Fail()) {
    LLDB_LOG(log,
             "SwiftLanguageRuntime: Unable to reset "
             "disableExclusivityChecking flag state: %s",
             error.AsCString());
    return;
  }
  if (log)
    LLDB_LOG(log,
             "SwiftLanguageRuntime: Changed "
             "disableExclusivityChecking flag state back to %u",
             m_original_dynamic_exclusivity_flag_state);
}

llvm::Optional<Value> SwiftLanguageRuntime::GetErrorReturnLocationAfterReturn(
    lldb::StackFrameSP frame_sp) {
  llvm::Optional<Value> error_val;

  llvm::StringRef error_reg_name;
  ArchSpec arch_spec(GetTargetRef().GetArchitecture());
  switch (arch_spec.GetMachine()) {
  case llvm::Triple::ArchType::arm:
    error_reg_name = "r6";
    break;
  case llvm::Triple::ArchType::aarch64:
    error_reg_name = "x21";
    break;
  case llvm::Triple::ArchType::x86_64:
    error_reg_name = "r12";
    break;
  default:
    break;
  }

  if (error_reg_name.empty())
    return error_val;

  RegisterContextSP reg_ctx = frame_sp->GetRegisterContext();
  const RegisterInfo *reg_info = reg_ctx->GetRegisterInfoByName(error_reg_name);
  lldbassert(reg_info &&
             "didn't get the right register name for swift error register");
  if (!reg_info)
    return error_val;

  RegisterValue reg_value;
  if (!reg_ctx->ReadRegister(reg_info, reg_value)) {
    // Do some logging here.
    return error_val;
  }

  lldb::addr_t error_addr = reg_value.GetAsUInt64();
  if (error_addr == 0)
    return error_val;

  Value val;
  if (reg_value.GetScalarValue(val.GetScalar())) {
    val.SetValueType(Value::ValueType::Scalar);
    val.SetContext(Value::ContextType::RegisterInfo,
                   const_cast<RegisterInfo *>(reg_info));
    error_val = val;
  }
  return error_val;
}

llvm::Optional<Value> SwiftLanguageRuntime::GetErrorReturnLocationBeforeReturn(
    lldb::StackFrameSP frame_sp, bool &need_to_check_after_return) {
  llvm::Optional<Value> error_val;

  if (!frame_sp) {
    need_to_check_after_return = false;
    return error_val;
  }

  // For Architectures where the error isn't returned in a register,
  // there's a magic variable that points to the value.  Check that first:

  ConstString error_location_name("$error");
  VariableListSP variables_sp = frame_sp->GetInScopeVariableList(false);
  VariableSP error_loc_var_sp = variables_sp->FindVariable(
      error_location_name, eValueTypeVariableArgument);
  if (error_loc_var_sp) {
    need_to_check_after_return = false;

    ValueObjectSP error_loc_val_sp = frame_sp->GetValueObjectForFrameVariable(
        error_loc_var_sp, eNoDynamicValues);
    if (error_loc_val_sp && error_loc_val_sp->GetError().Success())
      error_val = error_loc_val_sp->GetValue();

    return error_val;
  }

  // Otherwise, see if we know which register it lives in from the calling
  // convention. This should probably go in the ABI plugin not here, but the
  // Swift ABI can change with swiftlang versions and that would make it awkward
  // in the ABI.

  Function *func = frame_sp->GetSymbolContext(eSymbolContextFunction).function;
  if (!func) {
    need_to_check_after_return = false;
    return error_val;
  }

  need_to_check_after_return = func->CanThrow();
  return error_val;
}

lldb::BreakpointResolverSP
SwiftLanguageRuntime::CreateExceptionResolver(const lldb::BreakpointSP &bkpt, bool catch_bp,
                                              bool throw_bp) {
  return ::CreateExceptionResolver(bkpt, catch_bp, throw_bp);
}

static OptionDefinition g_swift_demangle_options[] = {
    // clang-format off
  {LLDB_OPT_SET_1, false, "expand", 'e', OptionParser::eNoArgument, nullptr, {}, 0, eArgTypeNone, "Whether LLDB should print the demangled tree"},
    // clang-format on
};

class CommandObjectSwift_Demangle : public CommandObjectParsed {
public:
  CommandObjectSwift_Demangle(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "demangle",
                            "Demangle a Swift mangled name",
                            "language swift demangle"),
        m_options() {}

  ~CommandObjectSwift_Demangle() {}

  Options *GetOptions() override { return &m_options; }

  class CommandOptions : public Options {
  public:
    CommandOptions() : Options(), m_expand(false, false) {
      OptionParsingStarting(nullptr);
    }

    virtual ~CommandOptions() {}

    Status SetOptionValue(uint32_t option_idx, llvm::StringRef option_arg,
                          ExecutionContext *execution_context) override {
      Status error;
      const int short_option = m_getopt_table[option_idx].val;
      switch (short_option) {
      case 'e':
        m_expand.SetCurrentValue(true);
        break;

      default:
        error.SetErrorStringWithFormat("invalid short option character '%c'",
                                       short_option);
        break;
      }

      return error;
    }

    void OptionParsingStarting(ExecutionContext *execution_context) override {
      m_expand.Clear();
    }

    llvm::ArrayRef<OptionDefinition> GetDefinitions() override {
      return llvm::makeArrayRef(g_swift_demangle_options);
    }

    // Options table: Required for subclasses of Options.

    OptionValueBoolean m_expand;
  };

protected:
  bool DoExecute(Args &command, CommandReturnObject &result) override {
    for (size_t i = 0; i < command.GetArgumentCount(); i++) {
      StringRef name = command.GetArgumentAtIndex(i);
      if (!name.empty()) {
        swift::Demangle::Context demangle_ctx;
        NodePointer node_ptr = nullptr;
        // Match the behavior of swift-demangle and accept Swift symbols without
        // the leading `$`. This makes symbol copy & paste more convenient.
        if (name.startswith("S") || name.startswith("s")) {
          std::string correctedName = std::string("$") + name.str();
          node_ptr = demangle_ctx.demangleSymbolAsNode(correctedName);
        } else {
          node_ptr = demangle_ctx.demangleSymbolAsNode(name);
        }
        if (node_ptr) {
          if (m_options.m_expand)
            result.GetOutputStream().PutCString(getNodeTreeAsString(node_ptr));
          result.GetOutputStream().Printf(
              "%s ---> %s\n", name.data(),
              swift::Demangle::nodeToString(node_ptr).c_str());
        }
      }
    }
    result.SetStatus(lldb::eReturnStatusSuccessFinishResult);
    return true;
  }

  CommandOptions m_options;
};

class CommandObjectSwift_RefCount : public CommandObjectRaw {
public:
  CommandObjectSwift_RefCount(CommandInterpreter &interpreter)
      : CommandObjectRaw(interpreter, "refcount",
                         "Inspect the reference count data for a Swift object",
                         "language swift refcount",
                         eCommandProcessMustBePaused | eCommandRequiresFrame) {}

  ~CommandObjectSwift_RefCount() {}

  Options *GetOptions() override { return nullptr; }

private:
  enum class ReferenceCountType {
    eReferenceStrong,
    eReferenceUnowned,
    eReferenceWeak,
  };

  llvm::Optional<uint32_t> getReferenceCount(StringRef ObjName,
                                             ReferenceCountType Type,
                                             ExecutionContext &exe_ctx,
                                             StackFrameSP &Frame) {
    std::string Kind;
    switch (Type) {
    case ReferenceCountType::eReferenceStrong:
      Kind = "";
      break;
    case ReferenceCountType::eReferenceUnowned:
      Kind = "Unowned";
      break;
    case ReferenceCountType::eReferenceWeak:
      Kind = "Weak";
      break;
    }

    EvaluateExpressionOptions eval_options;
    eval_options.SetLanguage(lldb::eLanguageTypeSwift);
    eval_options.SetResultIsInternal(true);
    ValueObjectSP result_valobj_sp;
    std::string Expr =
        (llvm::Twine("Swift._get") + Kind + llvm::Twine("RetainCount(") +
         ObjName + llvm::Twine(")"))
            .str();
    bool evalStatus = exe_ctx.GetTargetSP()->EvaluateExpression(
        Expr, Frame.get(), result_valobj_sp, eval_options);
    if (evalStatus != eExpressionCompleted)
      return llvm::None;

    bool success = false;
    uint32_t count = result_valobj_sp->GetSyntheticValue()->GetValueAsUnsigned(
        UINT32_MAX, &success);
    if (!success)
      return llvm::None;
    return count;
  }

protected:
  bool DoExecute(llvm::StringRef command,
                 CommandReturnObject &result) override {
    StackFrameSP frame_sp(m_exe_ctx.GetFrameSP());
    EvaluateExpressionOptions options;
    options.SetLanguage(lldb::eLanguageTypeSwift);
    options.SetResultIsInternal(true);
    ValueObjectSP result_valobj_sp;

    // We want to evaluate first the object we're trying to get the
    // refcount of, in order, to, e.g. see whether it's available.
    // So, given `language swift refcount patatino`, we try to
    // evaluate `expr patatino` and fail early in case there is
    // an error.
    bool evalStatus = m_exe_ctx.GetTargetSP()->EvaluateExpression(
        command, frame_sp.get(), result_valobj_sp, options);
    if (evalStatus != eExpressionCompleted) {
      result.SetStatus(lldb::eReturnStatusFailed);
      if (result_valobj_sp && result_valobj_sp->GetError().Fail())
        result.AppendError(result_valobj_sp->GetError().AsCString());
      return false;
    }

    // At this point, we're sure we're grabbing in our hands a valid
    // object and we can ask questions about it. `refcounts` are only
    // defined on class objects, so we throw an error in case we're
    // trying to look at something else.
    result_valobj_sp = result_valobj_sp->GetQualifiedRepresentationIfAvailable(
        lldb::eDynamicCanRunTarget, true);
    CompilerType result_type(result_valobj_sp->GetCompilerType());
    if (!(result_type.GetTypeInfo() & lldb::eTypeInstanceIsPointer)) {
      result.AppendError("refcount only available for class types");
      result.SetStatus(lldb::eReturnStatusFailed);
      return false;
    }

    // Ask swift debugger support in the compiler about the objects
    // reference counts, and return them to the user.
    llvm::Optional<uint32_t> strong = getReferenceCount(
        command, ReferenceCountType::eReferenceStrong, m_exe_ctx, frame_sp);
    llvm::Optional<uint32_t> unowned = getReferenceCount(
        command, ReferenceCountType::eReferenceUnowned, m_exe_ctx, frame_sp);
    llvm::Optional<uint32_t> weak = getReferenceCount(
        command, ReferenceCountType::eReferenceWeak, m_exe_ctx, frame_sp);

    std::string unavailable = "<unavailable>";

    result.AppendMessageWithFormat(
        "refcount data: (strong = %s, unowned = %s, weak = %s)\n",
        strong ? std::to_string(*strong).c_str() : unavailable.c_str(),
        unowned ? std::to_string(*unowned).c_str() : unavailable.c_str(),
        weak ? std::to_string(*weak).c_str() : unavailable.c_str());
    result.SetStatus(lldb::eReturnStatusSuccessFinishResult);
    return true;
  }
};

class CommandObjectMultiwordSwift : public CommandObjectMultiword {
public:
  CommandObjectMultiwordSwift(CommandInterpreter &interpreter)
      : CommandObjectMultiword(
            interpreter, "swift",
            "A set of commands for operating on the Swift Language Runtime.",
            "swift <subcommand> [<subcommand-options>]") {
    LoadSubCommand("demangle", CommandObjectSP(new CommandObjectSwift_Demangle(
                                   interpreter)));
    LoadSubCommand("refcount", CommandObjectSP(new CommandObjectSwift_RefCount(
                                   interpreter)));
  }

  virtual ~CommandObjectMultiwordSwift() {}
};

void SwiftLanguageRuntime::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), "Language runtime for the Swift language",
      CreateInstance,
      [](CommandInterpreter &interpreter) -> lldb::CommandObjectSP {
        return CommandObjectSP(new CommandObjectMultiwordSwift(interpreter));
      },
      SwiftLanguageRuntimeImpl::GetBreakpointExceptionPrecondition);
}

void SwiftLanguageRuntime::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

#define FORWARD(METHOD, ...)                                                   \
  assert(m_impl || m_stub);                                                    \
  return m_impl ? m_impl->METHOD(__VA_ARGS__) : m_stub->METHOD(__VA_ARGS__);

bool SwiftLanguageRuntime::GetDynamicTypeAndAddress(
    ValueObject &in_value, lldb::DynamicValueType use_dynamic,
    TypeAndOrName &class_type_or_name, Address &address,
    Value::ValueType &value_type) {
  FORWARD(GetDynamicTypeAndAddress, in_value, use_dynamic, class_type_or_name,
          address, value_type);
}

void SwiftLanguageRuntime::DumpTyperef(CompilerType type,
                                       TypeSystemSwiftTypeRef *module_holder,
                                       Stream *s) {
  FORWARD(DumpTyperef, type, module_holder, s);
}

TypeAndOrName
SwiftLanguageRuntime::FixUpDynamicType(const TypeAndOrName &type_and_or_name,
                                       ValueObject &static_value) {
  FORWARD(FixUpDynamicType, type_and_or_name, static_value);
}

bool SwiftLanguageRuntime::IsTaggedPointer(lldb::addr_t addr,
                                           CompilerType type) {
  FORWARD(IsTaggedPointer, addr, type);
}

std::pair<lldb::addr_t, bool>
SwiftLanguageRuntime::FixupPointerValue(lldb::addr_t addr, CompilerType type) {
  FORWARD(FixupPointerValue, addr, type);
}

lldb::addr_t SwiftLanguageRuntime::FixupAddress(lldb::addr_t addr,
                                                CompilerType type,
                                                Status &error) {
  FORWARD(FixupAddress, addr, type, error);
}

SwiftLanguageRuntime::MetadataPromiseSP
SwiftLanguageRuntime::GetMetadataPromise(lldb::addr_t addr,
                                         ValueObject &for_object) {
  FORWARD(GetMetadataPromise, addr, for_object);
}

bool SwiftLanguageRuntime::IsStoredInlineInBuffer(CompilerType type) {
  FORWARD(IsStoredInlineInBuffer, type);
}

llvm::Optional<uint64_t> SwiftLanguageRuntime::GetMemberVariableOffset(
    CompilerType instance_type, ValueObject *instance,
    llvm::StringRef member_name, Status *error) {
  FORWARD(GetMemberVariableOffset, instance_type, instance, member_name, error);
}

llvm::Optional<unsigned>
SwiftLanguageRuntime::GetNumChildren(CompilerType type, ValueObject *valobj) {
  FORWARD(GetNumChildren, type, valobj);
}

llvm::Optional<std::string> SwiftLanguageRuntime::GetEnumCaseName(
    CompilerType type, const DataExtractor &data, ExecutionContext *exe_ctx) {
  FORWARD(GetEnumCaseName, type, data, exe_ctx);
}

std::pair<bool, llvm::Optional<size_t>>
SwiftLanguageRuntime::GetIndexOfChildMemberWithName(
    CompilerType type, llvm::StringRef name, ExecutionContext *exe_ctx,
    bool omit_empty_base_classes, std::vector<uint32_t> &child_indexes) {
  FORWARD(GetIndexOfChildMemberWithName, type, name, exe_ctx,
          omit_empty_base_classes, child_indexes);
}

CompilerType SwiftLanguageRuntime::GetChildCompilerTypeAtIndex(
    CompilerType type, size_t idx, bool transparent_pointers,
    bool omit_empty_base_classes, bool ignore_array_bounds,
    std::string &child_name, uint32_t &child_byte_size,
    int32_t &child_byte_offset, uint32_t &child_bitfield_bit_size,
    uint32_t &child_bitfield_bit_offset, bool &child_is_base_class,
    bool &child_is_deref_of_parent, ValueObject *valobj,
    uint64_t &language_flags) {
  FORWARD(GetChildCompilerTypeAtIndex, type, idx, transparent_pointers,
          omit_empty_base_classes, ignore_array_bounds, child_name,
          child_byte_size, child_byte_offset,
          child_bitfield_bit_size, child_bitfield_bit_offset,
          child_is_base_class, child_is_deref_of_parent, valobj,
          language_flags);
}

llvm::Optional<unsigned>
SwiftLanguageRuntime::GetNumFields(CompilerType type,
                                   ExecutionContext *exe_ctx) {
  FORWARD(GetNumFields, type, exe_ctx);
}

bool SwiftLanguageRuntime::GetObjectDescription(Stream &str,
                                                ValueObject &object) {
  FORWARD(GetObjectDescription, str, object);
}

void SwiftLanguageRuntime::AddToLibraryNegativeCache(
    llvm::StringRef library_name) {
  FORWARD(AddToLibraryNegativeCache, library_name);
}

bool SwiftLanguageRuntime::IsInLibraryNegativeCache(
    llvm::StringRef library_name) {
  FORWARD(IsInLibraryNegativeCache, library_name);
}

void SwiftLanguageRuntime::ReleaseAssociatedRemoteASTContext(
    swift::ASTContext *ctx) {
  FORWARD(ReleaseAssociatedRemoteASTContext, ctx);
}

CompilerType
SwiftLanguageRuntime::BindGenericTypeParameters(StackFrame &stack_frame,
                                                CompilerType base_type) {
  FORWARD(BindGenericTypeParameters, stack_frame, base_type);
}

CompilerType
SwiftLanguageRuntime::GetConcreteType(ExecutionContextScope *exe_scope,
                                      ConstString abstract_type_name) {
  FORWARD(GetConcreteType, exe_scope, abstract_type_name);
}

llvm::Optional<uint64_t>
SwiftLanguageRuntime::GetBitSize(CompilerType type,
                                 ExecutionContextScope *exe_scope) {
  FORWARD(GetBitSize, type, exe_scope);
}

llvm::Optional<uint64_t>
SwiftLanguageRuntime::GetByteStride(CompilerType type) {
  FORWARD(GetByteStride, type);
}

llvm::Optional<size_t>
SwiftLanguageRuntime::GetBitAlignment(CompilerType type,
                                      ExecutionContextScope *exe_scope) {
  FORWARD(GetBitAlignment, type, exe_scope);
}

bool SwiftLanguageRuntime::IsValidErrorValue(ValueObject &in_value) {
  FORWARD(IsValidErrorValue, in_value);
}

lldb::SyntheticChildrenSP
SwiftLanguageRuntime::GetBridgedSyntheticChildProvider(ValueObject &valobj) {
  FORWARD(GetBridgedSyntheticChildProvider, valobj);
}

void SwiftLanguageRuntime::WillStartExecutingUserExpression(
    bool runs_in_playground_or_repl) {
  FORWARD(WillStartExecutingUserExpression, runs_in_playground_or_repl);
}

void SwiftLanguageRuntime::DidFinishExecutingUserExpression(
    bool runs_in_playground_or_repl) {
  FORWARD(DidFinishExecutingUserExpression, runs_in_playground_or_repl);
}

bool SwiftLanguageRuntime::IsABIStable() { FORWARD(IsABIStable); }

namespace {
/// The target specific register numbers used for async unwinding.
///
/// For UnwindPlans, these use eh_frame / dwarf register numbering.
struct AsyncUnwindRegisterNumbers {
  uint32_t async_ctx_regnum;
  uint32_t fp_regnum;
  uint32_t pc_regnum;
  /// A register to use as a marker to indicate how the async context is passed
  /// to the function (indirectly, or not). This needs to be communicated to the
  /// frames below us as they need to react differently. There is no good way to
  /// expose this, so we set another dummy register to communicate this state.
  uint32_t dummy_regnum;
};
} // namespace

static llvm::Optional<AsyncUnwindRegisterNumbers>
GetAsyncUnwindRegisterNumbers(llvm::Triple::ArchType triple) {
  switch (triple) {
  case llvm::Triple::x86_64: {
    AsyncUnwindRegisterNumbers regnums;
    regnums.async_ctx_regnum = dwarf_r14_x86_64;
    regnums.fp_regnum = dwarf_rbp_x86_64;
    regnums.pc_regnum = dwarf_rip_x86_64;
    regnums.dummy_regnum = dwarf_r15_x86_64;
    return regnums;
  }
  case llvm::Triple::aarch64: {
    AsyncUnwindRegisterNumbers regnums;
    regnums.async_ctx_regnum = arm64_dwarf::x22;
    regnums.fp_regnum = arm64_dwarf::fp;
    regnums.pc_regnum = arm64_dwarf::pc;
    regnums.dummy_regnum = arm64_dwarf::x23;
    return regnums;
  }
  default:
    return {};
  }
}

lldb::addr_t SwiftLanguageRuntime::GetAsyncContext(RegisterContext *regctx) {
  if (!regctx)
    return LLDB_INVALID_ADDRESS;

  auto arch = regctx->CalculateTarget()->GetArchitecture();
  if (auto regnums = GetAsyncUnwindRegisterNumbers(arch.GetMachine())) {
    auto reg = regctx->ConvertRegisterKindToRegisterNumber(
        RegisterKind::eRegisterKindDWARF, regnums->async_ctx_regnum);
    return regctx->ReadRegisterAsUnsigned(reg, LLDB_INVALID_ADDRESS);
  }

  assert(false && "swift async supports only x86_64 and arm64");
  return LLDB_INVALID_ADDRESS;
}

// Examine the register state and detect the transition from a real
// stack frame to an AsyncContext frame, or a frame in the middle of
// the AsyncContext chain, and return an UnwindPlan for these situations.
UnwindPlanSP
SwiftLanguageRuntime::GetRuntimeUnwindPlan(ProcessSP process_sp,
                                           RegisterContext *regctx,
                                           bool &behaves_like_zeroth_frame) {
  LLDB_SCOPED_TIMER();
 
  Target &target(process_sp->GetTarget());
  auto arch = target.GetArchitecture();
  llvm::Optional<AsyncUnwindRegisterNumbers> regnums =
      GetAsyncUnwindRegisterNumbers(arch.GetMachine());
  if (!regnums)
    return UnwindPlanSP();

  // If we can't fetch the fp reg, and we *can* fetch the async
  // context register, then we're in the middle of the AsyncContext
  // chain, return an UnwindPlan for that.
  addr_t fp = regctx->GetFP(LLDB_INVALID_ADDRESS);
  if (fp == LLDB_INVALID_ADDRESS) {
    if (GetAsyncContext(regctx) != LLDB_INVALID_ADDRESS)
      return GetFollowAsyncContextUnwindPlan(regctx, arch,
                                             behaves_like_zeroth_frame);
    return UnwindPlanSP();
  }

  // If we're in the prologue of a function, don't provide a Swift async
  // unwind plan.  We can be tricked by unmodified caller-registers that
  // make this look like an async frame when this is a standard ABI function
  // call, and the parent is the async frame.
  // This assumes that the frame pointer register will be modified in the
  // prologue.
  Address pc;
  pc.SetLoadAddress(regctx->GetPC(), &target);
  SymbolContext sc;
  if (pc.IsValid())
    if (!pc.CalculateSymbolContext(&sc, eSymbolContextFunction |
                                            eSymbolContextSymbol))
      return UnwindPlanSP();

  Address func_start_addr;
  uint32_t prologue_size;
  ConstString mangled_name;
  if (sc.function) {
    func_start_addr = sc.function->GetAddressRange().GetBaseAddress();
    prologue_size = sc.function->GetPrologueByteSize();
    mangled_name = sc.function->GetMangled().GetMangledName();
  } else if (sc.symbol) {
    func_start_addr = sc.symbol->GetAddress();
    prologue_size = sc.symbol->GetPrologueByteSize();
    mangled_name = sc.symbol->GetMangled().GetMangledName();
  } else {
    return UnwindPlanSP();
  }

  AddressRange prologue_range(func_start_addr, prologue_size);
  bool in_prologue = (func_start_addr == pc ||
                      prologue_range.ContainsLoadAddress(pc, &target));

  if (in_prologue) {
    if (!IsAnySwiftAsyncFunctionSymbol(mangled_name.GetStringRef()))
      return UnwindPlanSP();
  } else {
    addr_t saved_fp = LLDB_INVALID_ADDRESS;
    Status error;
    if (!process_sp->ReadMemory(fp, &saved_fp, 8, error))
      return UnwindPlanSP();

    // Get the high nibble of the dreferenced fp; if the 60th bit is set,
    // this is the transition to a swift async AsyncContext chain.
    if ((saved_fp & (0xfULL << 60)) >> 60 != 1)
      return UnwindPlanSP();
  }

  // The coroutine funclets split from an async function have 2 different ABIs:
  //  - Async suspend partial functions and the first funclet get their async
  //    context directly in the async register.
  //  - Async await resume partial functions take their context indirectly, it
  //    needs to be dereferenced to get the actual function's context.
  // The debug info for locals reflects this difference, so our unwinding of the
  // context register needs to reflect it too.
  bool indirect_context =
      IsSwiftAsyncAwaitResumePartialFunctionSymbol(mangled_name.GetStringRef());

  UnwindPlan::RowSP row(new UnwindPlan::Row);
  const int32_t ptr_size = 8;
  row->SetOffset(0);

  // A DWARF Expression to set the CFA.
  //      pushes the frame pointer register - 8
  //      dereference

  // FIXME: Row::RegisterLocation::RestoreType doesn't have a
  // deref(reg-value + offset) yet, shortcut around it with
  // a dwarf expression for now.
  // The CFA of an async frame is the address of it's associated AsyncContext.
  // In an async frame currently on the stack, this address is stored right
  // before the saved frame pointer on the stack.
  static const uint8_t g_cfa_dwarf_expression_x86_64[] = {
      llvm::dwarf::DW_OP_breg6, // DW_OP_breg6, register 6 == rbp
      0x78,                     //    sleb128 -8 (ptrsize)
      llvm::dwarf::DW_OP_deref,
  };
  static const uint8_t g_cfa_dwarf_expression_arm64[] = {
      llvm::dwarf::DW_OP_breg29, // DW_OP_breg29, register 29 == fp
      0x78,                      //    sleb128 -8 (ptrsize)
      llvm::dwarf::DW_OP_deref,
  };

  constexpr unsigned expr_size = sizeof(g_cfa_dwarf_expression_arm64);

  static_assert(sizeof(g_cfa_dwarf_expression_x86_64) ==
                    sizeof(g_cfa_dwarf_expression_arm64),
                "Code relies on DWARF  expressions being the same size");

  const uint8_t *expr = nullptr;
  if (arch.GetMachine() == llvm::Triple::x86_64)
    expr = g_cfa_dwarf_expression_x86_64;
  else if (arch.GetMachine() == llvm::Triple::aarch64)
    expr = g_cfa_dwarf_expression_arm64;
  else
    llvm_unreachable("Unsupported architecture");

  if (in_prologue) {
    if (indirect_context)
      row->GetCFAValue().SetIsRegisterDereferenced(regnums->async_ctx_regnum);
    else
      row->GetCFAValue().SetIsRegisterPlusOffset(regnums->async_ctx_regnum, 0);
  } else {
    row->GetCFAValue().SetIsDWARFExpression(expr, expr_size);
  }

  if (indirect_context) {
    if (in_prologue) {
      row->SetRegisterLocationToSame(regnums->async_ctx_regnum, false);
    } else {
      // In a "resume" coroutine, the passed context argument needs to be
      // dereferenced once to get the context. This is reflected in the debug
      // info so we need to account for it and report am async register value
      // that needs to be dereferenced to get to the context.
      // Note that the size passed for the DWARF expression is the size of the
      // array minus one. This skips the last deref for this use.
      assert(expr[expr_size - 1] == llvm::dwarf::DW_OP_deref &&
             "Should skip a deref");
      row->SetRegisterLocationToIsDWARFExpression(regnums->async_ctx_regnum,
                                                  expr, expr_size - 1, false);
    }
  } else {
    // In the first part of a split async function, the context is passed
    // directly, so we can use the CFA value directly.
    row->SetRegisterLocationToIsCFAPlusOffset(regnums->async_ctx_regnum, 0,
                                              false);
    // The fact that we are in this case needs to be communicated to the frames
    // below us as they need to react differently. There is no good way to
    // expose this, so we set another dummy register to communicate this state.
    static const uint8_t g_dummy_dwarf_expression[] = {
        llvm::dwarf::DW_OP_const1u, 0
    };
    row->SetRegisterLocationToIsDWARFExpression(
        regnums->dummy_regnum, g_dummy_dwarf_expression,
        sizeof(g_dummy_dwarf_expression), false);
  }
  row->SetRegisterLocationToAtCFAPlusOffset(regnums->pc_regnum, ptr_size,
                                            false);

  row->SetUnspecifiedRegistersAreUndefined(true);

  UnwindPlanSP plan = std::make_shared<UnwindPlan>(lldb::eRegisterKindDWARF);
  plan->AppendRow(row);
  plan->SetSourceName("Swift Transition-to-AsyncContext-Chain");
  plan->SetSourcedFromCompiler(eLazyBoolNo);
  plan->SetUnwindPlanValidAtAllInstructions(eLazyBoolYes);
  plan->SetUnwindPlanForSignalTrap(eLazyBoolYes);
  behaves_like_zeroth_frame = true;
  return plan;
}

// Creates an UnwindPlan for following the AsyncContext chain
// up the stack, from a current AsyncContext frame.
static UnwindPlanSP
GetFollowAsyncContextUnwindPlan(RegisterContext *regctx, ArchSpec &arch,
                                bool &behaves_like_zeroth_frame) {
  LLDB_SCOPED_TIMER();
 
  UnwindPlan::RowSP row(new UnwindPlan::Row);
  const int32_t ptr_size = 8;
  row->SetOffset(0);

  llvm::Optional<AsyncUnwindRegisterNumbers> regnums =
      GetAsyncUnwindRegisterNumbers(arch.GetMachine());
  if (!regnums)
    return UnwindPlanSP();

  // In the general case, the async register setup by the frame above us
  // should be dereferenced twice to get our context, except when the frame
  // above us is an async frame on the OS stack that takes its context directly
  // (see discussion in GetRuntimeUnwindPlan()). The availability of
  // dummy_regnum is used as a marker for this situation.
  if (regctx->ReadRegisterAsUnsigned(regnums->dummy_regnum, (uint64_t)-1ll) !=
      (uint64_t)-1ll) {
    row->GetCFAValue().SetIsRegisterDereferenced(regnums->async_ctx_regnum);
    row->SetRegisterLocationToSame(regnums->async_ctx_regnum, false);
  } else {
    static const uint8_t async_dwarf_expression_x86_64[] = {
        llvm::dwarf::DW_OP_regx, dwarf_r14_x86_64, // DW_OP_regx, reg
        llvm::dwarf::DW_OP_deref,                  // DW_OP_deref
        llvm::dwarf::DW_OP_deref,                  // DW_OP_deref
    };
    static const uint8_t async_dwarf_expression_arm64[] = {
        llvm::dwarf::DW_OP_regx, arm64_dwarf::x22, // DW_OP_regx, reg
        llvm::dwarf::DW_OP_deref,                  // DW_OP_deref
        llvm::dwarf::DW_OP_deref,                  // DW_OP_deref
    };

    const unsigned expr_size = sizeof(async_dwarf_expression_x86_64);
    static_assert(sizeof(async_dwarf_expression_x86_64) ==
                      sizeof(async_dwarf_expression_arm64),
                  "Expressions of different sizes");

    const uint8_t *expression = nullptr;
    if (arch.GetMachine() == llvm::Triple::x86_64)
      expression = async_dwarf_expression_x86_64;
    else if (arch.GetMachine() == llvm::Triple::aarch64)
      expression = async_dwarf_expression_arm64;
    else
      llvm_unreachable("Unsupported architecture");

    // Note how the register location gets the same expression pointer with a
    // different size. We just skip the trailing deref for it.
    assert(expression[expr_size - 1] == llvm::dwarf::DW_OP_deref &&
           "Should skip a deref");
    row->GetCFAValue().SetIsDWARFExpression(expression, expr_size);
    row->SetRegisterLocationToIsDWARFExpression(
        regnums->async_ctx_regnum, expression, expr_size - 1, false);
  }

  row->SetRegisterLocationToAtCFAPlusOffset(regnums->pc_regnum, ptr_size,
                                            false);

  row->SetUnspecifiedRegistersAreUndefined(true);

  UnwindPlanSP plan = std::make_shared<UnwindPlan>(lldb::eRegisterKindDWARF);
  plan->AppendRow(row);
  plan->SetSourceName("Swift Following-AsyncContext-Chain");
  plan->SetSourcedFromCompiler(eLazyBoolNo);
  plan->SetUnwindPlanValidAtAllInstructions(eLazyBoolYes);
  plan->SetUnwindPlanForSignalTrap(eLazyBoolYes);
  behaves_like_zeroth_frame = true;
  return plan;
}

} // namespace lldb_private
