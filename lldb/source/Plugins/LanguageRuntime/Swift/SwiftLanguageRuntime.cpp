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
#include "Plugins/TypeSystem/Swift/TypeSystemSwiftTypeRef.h"
#include "ReflectionContextInterface.h"
#include "SwiftMetadataCache.h"

#include "Plugins/ExpressionParser/Swift/SwiftPersistentExpressionState.h"
#include "Plugins/LanguageRuntime/Swift/SwiftTask.h"
#include "Plugins/Process/Utility/RegisterContext_x86.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "Plugins/TypeSystem/Swift/SwiftDemangle.h"
#include "Utility/ARM64_DWARF_Registers.h"
#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/JITSection.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Progress.h"
#include "lldb/Core/Section.h"
#include "lldb/DataFormatters/StringPrinter.h"
#include "lldb/Host/OptionParser.h"
#include "lldb/Host/SafeMachO.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Symbol/FuncUnwinders.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/UnwindLLDB.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/ErrorMessages.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/OptionParsing.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/StructuredData.h"
#include "lldb/Utility/Timer.h"
#include "lldb/ValueObject/ValueObject.h"
#include "lldb/ValueObject/ValueObjectCast.h"
#include "lldb/ValueObject/ValueObjectConstResult.h"
#include "lldb/ValueObject/ValueObjectVariable.h"

#include "lldb/lldb-enumerations.h"
#include "swift/AST/ASTMangler.h"
#include "swift/Demangling/Demangle.h"
#include "swift/RemoteAST/RemoteAST.h"
#include "swift/RemoteInspection/ReflectionContext.h"
#include "swift/Threading/ThreadLocalStorage.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/Memory.h"
#include <optional>

// FIXME: we should not need this
#include "Plugins/Language/Swift/SwiftFormatters.h"
#include "Plugins/Language/Swift/SwiftFrameRecognizers.h"

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

const char *SwiftLanguageRuntime::GetConcurrencyLibraryBaseName() {
  return "swift_Concurrency";
}

static ConstString GetStandardLibraryName(Process &process) {
  // This result needs to be stored in the constructor.
  PlatformSP platform_sp(process.GetTarget().GetPlatform());
  if (platform_sp)
    return platform_sp->GetFullNameForDylib(
        ConstString(SwiftLanguageRuntime::GetStandardLibraryBaseName()));
  return {};
}

static ConstString GetConcurrencyLibraryName(Process &process) {
  PlatformSP platform_sp = process.GetTarget().GetPlatform();
  if (platform_sp)
    return platform_sp->GetFullNameForDylib(
        ConstString(SwiftLanguageRuntime::GetConcurrencyLibraryBaseName()));
  return {};
}

ConstString SwiftLanguageRuntime::GetStandardLibraryName() {
  return ::GetStandardLibraryName(*m_process);
}

static bool IsModuleSwiftRuntime(lldb_private::Process &process,
                                 lldb_private::Module &module) {
  return module.GetFileSpec().GetFilename() == GetStandardLibraryName(process);
}

static bool IsModuleSwiftConcurrency(lldb_private::Process &process,
                                     lldb_private::Module &module) {
  return module.GetFileSpec().GetFilename() ==
         GetConcurrencyLibraryName(process);
}

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
  static ConstString swift_reflection_version_sym("swift_release");
  return image.FindFirstSymbolWithNameAndType(swift_reflection_version_sym);
}

static bool IsStaticSwiftConcurrency(Module &image) {
  static const ConstString task_switch_symbol("_swift_task_switch");
  return image.FindFirstSymbolWithNameAndType(task_switch_symbol);
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

ModuleSP SwiftLanguageRuntime::FindConcurrencyModule(Process &process) {
  ModuleSP concurrency_module;
  process.GetTarget().GetImages().ForEach([&](const ModuleSP &candidate) {
    if (candidate && IsModuleSwiftConcurrency(process, *candidate)) {
      concurrency_module = candidate;
      return false;
    }
    return true;
  });
  if (concurrency_module)
    return concurrency_module;

  // Do a more expensive search for a statically linked Swift runtime.
  process.GetTarget().GetImages().ForEach([&](const ModuleSP &candidate) {
    if (candidate && IsStaticSwiftConcurrency(*candidate)) {
      concurrency_module = candidate;
      return false;
    }
    return true;
  });
  return concurrency_module;
}

std::optional<uint32_t>
SwiftLanguageRuntime::FindConcurrencyDebugVersion(Process &process) {
  ModuleSP concurrency_module = FindConcurrencyModule(process);
  if (!concurrency_module)
    return {};

  const Symbol *version_symbol =
      concurrency_module->FindFirstSymbolWithNameAndType(
          ConstString("_swift_concurrency_debug_internal_layout_version"));
  if (!version_symbol)
    return 0;

  addr_t symbol_addr = version_symbol->GetLoadAddress(&process.GetTarget());
  if (symbol_addr == LLDB_INVALID_ADDRESS)
    return {};
  Status error;
  uint64_t version = process.ReadUnsignedIntegerFromMemory(
      symbol_addr, /*width*/ 4, /*fail_value=*/0, error);
  if (error.Fail())
    return {};
  return version;
}

static std::optional<lldb::addr_t>
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
          obj_file && obj_file->GetPluginName() == "mach-o";
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

  static const char *names[] = {"swift_willThrow", "swift_willThrowTypedImpl"};
  if (throw_bp)
    resolver_sp.reset(
        new BreakpointResolverName(bkpt, names, 2, eFunctionNameTypeBase,
                                   eLanguageTypeUnknown, 0, eLazyBoolNo));
  // FIXME: We don't do catch breakpoints for ObjC yet.
  // Should there be some way for the runtime to specify what it can do in this
  // regard?
  return resolver_sp;
}

static bool HasReflectionInfo(ObjectFile *obj_file) {
  if (!obj_file)
    return false;

  auto findSectionInObject = [&](StringRef name) {
    ConstString section_name(name);
    SectionSP section_sp =
        obj_file->GetSectionList()->FindSectionByName(section_name);
    if (section_sp)
      return true;
    return false;
  };

  const auto obj_format_type =
      obj_file->GetArchitecture().GetTriple().getObjectFormat();
  auto obj_file_format_up = GetSwiftObjectFileFormat(obj_format_type);
  if (!obj_file_format_up)
    return false;

  StringRef field_md =
      obj_file_format_up->getSectionName(swift::ReflectionSectionKind::fieldmd);
  StringRef assocty =
      obj_file_format_up->getSectionName(swift::ReflectionSectionKind::assocty);
  StringRef builtin =
      obj_file_format_up->getSectionName(swift::ReflectionSectionKind::builtin);
  StringRef capture =
      obj_file_format_up->getSectionName(swift::ReflectionSectionKind::capture);
  StringRef typeref =
      obj_file_format_up->getSectionName(swift::ReflectionSectionKind::typeref);
  StringRef reflstr =
      obj_file_format_up->getSectionName(swift::ReflectionSectionKind::reflstr);

  bool hasReflectionSection =
      findSectionInObject(field_md) || findSectionInObject(assocty) ||
      findSectionInObject(builtin) || findSectionInObject(capture) ||
      findSectionInObject(typeref) || findSectionInObject(reflstr);
  return hasReflectionSection;
}

ThreadSafeReflectionContext SwiftLanguageRuntime::GetReflectionContext() {
  m_reflection_ctx_mutex.lock();

  SetupReflection();
  // SetupReflection can potentially fail.
  if (m_initialized_reflection_ctx)
    ProcessModulesToAdd();
  return {m_reflection_ctx.get(), m_reflection_ctx_mutex};
}

void SwiftLanguageRuntime::ProcessModulesToAdd() {
  // A snapshot of the modules to be processed. This is necessary because
  // AddModuleToReflectionContext may recursively call into this function again.
  ModuleList modules_to_add_snapshot;
  modules_to_add_snapshot.Swap(m_modules_to_add);

  if (modules_to_add_snapshot.IsEmpty())
    return;

  auto &target = GetProcess().GetTarget();
  auto exe_module = target.GetExecutableModule();
  Progress progress("Setting up Swift reflection", {},
                    modules_to_add_snapshot.GetSize());
  size_t completion = 0;

  // Add all defered modules to reflection context that were added to
  // the target since this SwiftLanguageRuntime was created.
  modules_to_add_snapshot.ForEach([&](const ModuleSP &module_sp) -> bool {
    if (module_sp) {
      AddModuleToReflectionContext(module_sp);
      progress.Increment(++completion,
                         module_sp->GetFileSpec().GetFilename().GetString());
    }
    return true;
  });
}

SwiftMetadataCache *SwiftLanguageRuntime::GetSwiftMetadataCache() {
  if (!m_swift_metadata_cache.is_enabled())
    return {};
  return &m_swift_metadata_cache;
}

std::vector<std::string>
SwiftLanguageRuntime::GetConformances(llvm::StringRef mangled_name) {
  if (m_conformances.empty()) {
    using namespace swift::Demangle;
    Demangler dem;

    ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
    if (!reflection_ctx)
      return {};

    Progress progress("Parsing Swift conformances");
    swift::reflection::ConformanceCollectionResult conformances =
        reflection_ctx->GetAllConformances();

    for (auto &conformance : conformances.Conformances) {
      auto [mod, proto] = StringRef(conformance.ProtocolName).split('.');
      NodePointer n =
          swift_demangle::CreateNominal(dem, Node::Kind::Protocol, mod, proto);
      auto mangling = mangleNode(n);
      if (!mangling.isSuccess())
        return {};
      llvm::StringRef protocol =
          swift::Demangle::dropSwiftManglingPrefix(mangling.result());

      m_conformances[mangled_name].push_back(protocol.str());
    }
  }
  return m_conformances.lookup(mangled_name);
}

void SwiftLanguageRuntime::SetupReflection() {
  std::lock_guard<std::recursive_mutex> lock(m_reflection_ctx_mutex);
  if (m_initialized_reflection_ctx)
    return;

  LLDB_SCOPED_TIMER();
  
  // The global ABI bit is read by the Swift runtime library.
  SetupABIBit();
  SetupExclusivity();
  SetupSwiftError();

  auto &target = GetProcess().GetTarget();
  auto exe_module = target.GetExecutableModule();

  auto *log = GetLog(LLDBLog::Types);
  if (!exe_module) {
    LLDB_LOGF(log, "%s: Failed to get executable module",
              LLVM_PRETTY_FUNCTION);
    m_initialized_reflection_ctx = false;
    return;
  }

  bool objc_interop = (bool)findRuntime(*m_process, RuntimeKind::ObjC);
  const char *objc_interop_msg =
      objc_interop ? "with Objective-C interopability" : "Swift only";

  auto &triple = exe_module->GetArchitecture().GetTriple();
  uint32_t ptr_size = m_process->GetAddressByteSize();
  LLDB_LOG(log, "Initializing a {0}-bit reflection context ({1}) for \"{2}\"",
           ptr_size * 8, triple.str(), objc_interop_msg);
  if (ptr_size == 4 || ptr_size == 8)
    m_reflection_ctx = ReflectionContextInterface::CreateReflectionContext(
        ptr_size, this->GetMemoryReader(), objc_interop,
        GetSwiftMetadataCache());
  if (!m_reflection_ctx)
    LLDB_LOG(log, "Could not initialize reflection context for \"{0}\"",
             triple.str());
  // We set m_initialized_reflection_ctx to true here because
  // AddModuleToReflectionContext can potentially call into SetupReflection
  // again (which will early exit). This is safe to do since every other thread
  // using reflection context will have to wait until all the modules are added,
  // since the thread performing the initialization locked the mutex.
  m_initialized_reflection_ctx = true;
}

bool SwiftLanguageRuntime::IsABIStable() {
  GetReflectionContext();
  return _swift_classIsSwiftMask == 2;
}

void SwiftLanguageRuntime::SetupSwiftError() {
  if (!m_process)
    return;
  m_SwiftNativeNSErrorISA =
      FindSymbolForSwiftObject(*m_process, RuntimeKind::Swift,
                               "__SwiftNativeNSError", eSymbolTypeObjCClass);
}

std::optional<lldb::addr_t> SwiftLanguageRuntime::GetSwiftNativeNSErrorISA() {
  return m_SwiftNativeNSErrorISA;
}

void SwiftLanguageRuntime::SetupExclusivity() {
  m_dynamic_exclusivity_flag_addr = FindSymbolForSwiftObject(
      GetProcess(), RuntimeKind::Swift, "_swift_disableExclusivityChecking",
      eSymbolTypeData);
  Log *log(GetLog(LLDBLog::Expressions));
  if (log)
    log->Printf(
        "SwiftLanguageRuntime: _swift_disableExclusivityChecking = %" PRIu64,
        m_dynamic_exclusivity_flag_addr ? *m_dynamic_exclusivity_flag_addr : 0);
}

std::optional<lldb::addr_t>
SwiftLanguageRuntime::GetDynamicExclusivityFlagAddr() {
  return m_dynamic_exclusivity_flag_addr;
}

void SwiftLanguageRuntime::SetupABIBit() {
  if (FindSymbolForSwiftObject(GetProcess(), RuntimeKind::ObjC,
                               "objc_debug_swift_stable_abi_bit",
                               eSymbolTypeAny))
    _swift_classIsSwiftMask = 2;
  else
    _swift_classIsSwiftMask = 1;
}

LanguageRuntime *
SwiftLanguageRuntime::CreateInstance(Process *process,
                                     lldb::LanguageType language) {
  if ((language != eLanguageTypeSwift) || !process)
    return nullptr;
  return new SwiftLanguageRuntime(*process);
}

SwiftLanguageRuntime::SwiftLanguageRuntime(Process &process)
    : LanguageRuntime(&process) {
  Target &target = m_process->GetTarget();
  m_modules_to_add.Append(target.GetImages());
  RegisterSwiftFrameRecognizers(GetProcess());
}

void SwiftLanguageRuntime::ModulesDidLoad(const ModuleList &module_list) {
  // The modules will be lazily processed on the next call to
  // GetReflectionContext.
  m_modules_to_add.AppendIfNeeded(module_list);
  // This could be done more efficiently with a better reflection API.
  m_conformances.clear();
}

static llvm::SmallVector<llvm::StringRef, 1>
GetLikelySwiftImageNamesForModule(ModuleSP module) {
  if (!module || !module->GetFileSpec())
    return {};

  auto name =
      module->GetFileSpec().GetFileNameStrippingExtension().GetStringRef();
  if (name == "libswiftCore")
    name = "Swift";
  if (name.starts_with("libswift"))
    name = name.drop_front(8);
  if (name.starts_with("lib"))
    name = name.drop_front(3);
  return {name};
}

bool SwiftLanguageRuntime::AddJitObjectFileToReflectionContext(
    ObjectFile &obj_file, llvm::Triple::ObjectFormatType obj_format_type,
    llvm::SmallVector<llvm::StringRef, 1> likely_module_names) {
  assert(obj_file.GetType() == ObjectFile::eTypeJIT &&
         "Not a JIT object file!");
  auto obj_file_format = GetSwiftObjectFileFormat(obj_format_type);

  if (!obj_file_format)
    return false;

  auto reflection_info_id = m_reflection_ctx->AddImage(
      [&](swift::ReflectionSectionKind section_kind)
          -> std::pair<swift::remote::RemoteRef<void>, uint64_t> {
        auto section_name = obj_file_format->getSectionName(section_kind);
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
            swift::remote::RemoteRef<void> remote_ref(
                swift::remote::RemoteAddress(
                    section->GetFileAddress(),
                    swift::remote::RemoteAddress::DefaultAddressSpace),
                Buf);

            return {remote_ref, size};
          }
        }
        return {};
      },
      likely_module_names);
  // We don't care to cache modules generated by the jit, because they will
  // only be used by the current process.
  return reflection_info_id.has_value();
}

std::optional<uint32_t> SwiftLanguageRuntime::AddObjectFileToReflectionContext(
    ModuleSP module,
    llvm::SmallVector<llvm::StringRef, 1> likely_module_names) {
  auto obj_format_type =
      module->GetArchitecture().GetTriple().getObjectFormat();

  auto obj_file_format = GetSwiftObjectFileFormat(obj_format_type);
  if (!obj_file_format)
    return {};

  bool should_register_with_symbol_obj_file = [&]() -> bool {
    if (!GetProcess().GetTarget().GetSwiftReadMetadataFromDSYM())
      return false;
    auto *symbol_file = module->GetSymbolFile();
    if (!symbol_file)
      return false;
    auto *sym_obj_file = symbol_file->GetObjectFile();
    if (!sym_obj_file)
      return false;

    std::optional<llvm::StringRef> maybe_segment_name =
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

  std::optional<llvm::StringRef> maybe_segment_name;
  std::optional<llvm::StringRef> maybe_secondary_segment_name;
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
    return {};

  llvm::StringRef segment_name = *maybe_segment_name;

  auto lldb_memory_reader = GetMemoryReader();
  auto maybe_start_and_end = lldb_memory_reader->addModuleToAddressMap(
      module, should_register_with_symbol_obj_file);
  if (!maybe_start_and_end)
    return {};

  uint64_t start_address, end_address;
  std::tie(start_address, end_address) = *maybe_start_and_end;

  auto *section_list = object_file->GetSectionList();
  if (section_list->GetSize() == 0)
    return false;

  auto segment_iter = llvm::find_if(*section_list, [&](auto segment) {
    return segment->GetName() == segment_name.begin();
  });

  if (segment_iter == section_list->end())
    return {};

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
        // added with the section's virtual address subtracting the start of the
        // module's address. We need to use the virtual address instead of the
        // file offset because the offsets encoded in the reflection section are
        // calculated in the virtual address space.
        auto address = start_address + section->GetFileAddress() -
                       section_list->GetSectionAtIndex(0)->GetFileAddress();
        assert(address <= end_address && "Address outside of range!");

        swift::remote::RemoteRef<void> remote_ref(
            swift::remote::RemoteAddress(address,
                                         LLDBMemoryReader::LLDBAddressSpace),
            Buf);
        return {remote_ref, size};
      }
    }
    return {};
  };

  return m_reflection_ctx->AddImage(
      [&](swift::ReflectionSectionKind section_kind)
          -> std::pair<swift::remote::RemoteRef<void>, uint64_t> {
        auto pair = find_section_with_kind(segment, section_kind);
        if (pair.first)
          return pair;
        return find_section_with_kind(maybe_secondary_segment, section_kind);
      },
      likely_module_names);
}

bool SwiftLanguageRuntime::AddModuleToReflectionContext(
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
  auto &target = GetProcess().GetTarget();
  Address start_address = obj_file->GetBaseAddress();
  auto load_ptr = static_cast<uintptr_t>(
      start_address.GetLoadAddress(&target));
  auto likely_module_names = GetLikelySwiftImageNamesForModule(module_sp);
  if (obj_file->GetType() == ObjectFile::eTypeJIT) {
    auto object_format_type =
        module_sp->GetArchitecture().GetTriple().getObjectFormat();
    return AddJitObjectFileToReflectionContext(*obj_file, object_format_type,
                                               likely_module_names);
  }

  if (load_ptr == 0 || load_ptr == LLDB_INVALID_ADDRESS) {
    if (obj_file->GetType() != ObjectFile::eTypeJIT)
      LLDB_LOG(GetLog(LLDBLog::Types),
               "{0}: failed to get start address for \"{1}\".", __FUNCTION__,
               module_sp->GetObjectName()
                   ? module_sp->GetObjectName()
                   : obj_file->GetFileSpec().GetFilename());
    return false;
  }
  bool found = HasReflectionInfo(obj_file);
  LLDB_LOGV(GetLog(LLDBLog::Types), "{0} reflection metadata in \"{1}\"",
            found ? "Adding" : "No",
            module_sp->GetObjectName() ? module_sp->GetObjectName()
                                       : obj_file->GetFileSpec().GetFilename());
  if (!found)
    return true;

  auto read_from_file_cache =
      GetMemoryReader()->readMetadataFromFileCacheEnabled();

  std::optional<uint32_t> info_id;
  // When dealing with ELF, we need to pass in the contents of the on-disk
  // file, since the Section Header Table is not present in the child process
  if (obj_file->GetPluginName() == "elf") {
    DataExtractor extractor;
    auto size = obj_file->GetData(0, obj_file->GetByteSize(), extractor);
    const uint8_t *file_data = extractor.GetDataStart();
    llvm::sys::MemoryBlock file_buffer((void *)file_data, size);
    info_id = m_reflection_ctx->ReadELF(
        swift::remote::RemoteAddress(
            load_ptr, swift::remote::RemoteAddress::DefaultAddressSpace),
        std::optional<llvm::sys::MemoryBlock>(file_buffer),
        likely_module_names);
  } else if (read_from_file_cache &&
             obj_file->GetPluginName() == "mach-o") {
    info_id = AddObjectFileToReflectionContext(module_sp, likely_module_names);
    if (!info_id)
      info_id = m_reflection_ctx->AddImage(
          swift::remote::RemoteAddress(
              load_ptr, swift::remote::RemoteAddress::DefaultAddressSpace),
          likely_module_names);
  } else {
    info_id = m_reflection_ctx->AddImage(
        swift::remote::RemoteAddress(
            load_ptr, swift::remote::RemoteAddress::DefaultAddressSpace),
        likely_module_names);
  }

  if (!info_id) {
    LLDB_LOG(GetLog(LLDBLog::Types),
             "Error while loading reflection metadata in \"{0}\"",
             module_sp->GetObjectName());
    return false;
  }

  if (auto *swift_metadata_cache = GetSwiftMetadataCache())
      swift_metadata_cache->registerModuleWithReflectionInfoID(module_sp,
                                                               *info_id);

  return true;
}

std::string
SwiftLanguageRuntime::GetObjectDescriptionExpr_Result(ValueObject &object) {
  Log *log(GetLog(LLDBLog::DataFormatters | LLDBLog::Expressions));
  std::string expr_string
      = llvm::formatv("Swift._DebuggerSupport.stringForPrintObject({0})",
                      object.GetName().GetCString()).str();
  if (log)
    log->Printf("[GetObjectDescriptionExpr_Result] expression: %s",
                expr_string.c_str());
  return expr_string;
}

std::string
SwiftLanguageRuntime::GetObjectDescriptionExpr_Ref(ValueObject &object) {
  Log *log(GetLog(LLDBLog::DataFormatters | LLDBLog::Expressions));

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

std::string SwiftLanguageRuntime::GetObjectDescriptionExpr_Copy(
    ValueObject &object, lldb::addr_t &copy_location) {
  Log *log(GetLog(LLDBLog::DataFormatters | LLDBLog::Expressions));

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
    frame_sp =
        GetProcess().GetThreadList().GetSelectedThread()->GetSelectedFrame(
            DoNoSelectMostRelevantFrame);

  auto swift_ast_ctx =
      static_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
  if (swift_ast_ctx) {
    auto bound_type_or_err = BindGenericTypeParameters(*frame_sp, static_type);
    if (!bound_type_or_err) {
      LLDB_LOG_ERROR(log, bound_type_or_err.takeError(), "{0}");
      return {};
    }
    static_type = *bound_type_or_err;
  }

  auto stride = 0;
  auto opt_stride = static_type.GetByteStride(frame_sp.get());
  if (opt_stride)
    stride = *opt_stride;

  Status error;
  copy_location = GetProcess().AllocateMemory(
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

  if (0 == GetProcess().WriteMemory(copy_location,
                                    data_extractor.GetDataStart(),
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

llvm::Error SwiftLanguageRuntime::RunObjectDescriptionExpr(
    ValueObject &object, std::string &expr_string, Stream &result) {
  Log *log(GetLog(LLDBLog::DataFormatters | LLDBLog::Expressions));
  ValueObjectSP result_sp;
  EvaluateExpressionOptions eval_options;
  eval_options.SetUnwindOnError(true);
  eval_options.SetLanguage(lldb::eLanguageTypeSwift);
  eval_options.SetSuppressPersistentResult(true);
  eval_options.SetIgnoreBreakpoints(true);
  eval_options.SetTimeout(GetProcess().GetUtilityExpressionTimeout());

  StackFrameSP frame_sp = object.GetFrameSP();
  if (!frame_sp)
    frame_sp =
        GetProcess().GetThreadList().GetSelectedThread()->GetSelectedFrame(
            DoNoSelectMostRelevantFrame);
  if (!frame_sp)
    return llvm::createStringError("no execution context to run expression in");
  auto eval_result = GetProcess().GetTarget().EvaluateExpression(
      expr_string, frame_sp.get(), result_sp, eval_options);

  LLDB_LOG(log, "[RunObjectDescriptionExpr] {0}", toString(eval_result));

  // Sanity check the result of the expression before moving forward
  if (!result_sp) {
    LLDB_LOG(log, "[RunObjectDescriptionExpr] expression generated no result");
    return llvm::createStringError("expression produced no result");
  }
  if (result_sp->GetError().Fail()) {
    LLDB_LOG(log, "[RunObjectDescriptionExpr] expression generated error: {0}",
             result_sp->GetError().AsCString());

    return result_sp->GetError().ToError();
  }
  if (!result_sp->GetCompilerType().IsValid()) {
    LLDB_LOG(log, "[RunObjectDescriptionExpr] expression generated "
                  "invalid type");

    return llvm::createStringError("expression produced invalid result type");
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
    LLDB_LOG(log,
             "[RunObjectDescriptionExpr] expression completed successfully");
    return llvm::Error::success();
  }
  LLDB_LOG(
      log,
      "[RunObjectDescriptionExpr] expression generated invalid string data");

  return llvm::createStringError("expression produced unprintable string");
}

static bool IsVariable(ValueObject &object) {
  if (object.IsSynthetic())
    return IsVariable(*object.GetNonSyntheticValue());

  return bool(object.GetVariable());
}

static bool IsSwiftResultVariable(ConstString name) {
  if (name) {
    llvm::StringRef name_sr(name.GetStringRef());
    if (name_sr.size() > 2 &&
        (name_sr.starts_with("$R") || name_sr.starts_with("$E")) &&
        ::isdigit(name_sr[2]))
      return true;
  }
  return false;
}

static bool IsSwiftReferenceType(ValueObject &object) {
  CompilerType object_type(object.GetCompilerType());
  if (object_type.GetTypeSystem().isa_and_nonnull<TypeSystemSwift>()) {
    Flags type_flags(object_type.GetTypeInfo());
    if (type_flags.AllSet(eTypeIsClass | eTypeHasValue |
                          eTypeInstanceIsPointer))
      return true;
  }
  return false;
}

llvm::Error SwiftLanguageRuntime::GetObjectDescription(Stream &str,
                                                       ValueObject &object) {
  if (object.IsUninitializedReference())
    return llvm::createStringError("<uninitialized>");

  std::string expr_string;

  if (::IsVariable(object) || ::IsSwiftResultVariable(object.GetName())) {
    // if the object is a Swift variable, it has two properties:
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
    auto error = RunObjectDescriptionExpr(object, expr_string, probe_stream);
    if (error)
      return error;
    str.Printf("%s", probe_stream.GetData());
    return llvm::Error::success();
  }
  // In general, don't try to use the name of the ValueObject as it might end up
  // referring to the wrong thing.  Instead, copy the object data into the
  // target and call object description on the copy.
  lldb::addr_t copy_location = LLDB_INVALID_ADDRESS;
  expr_string = GetObjectDescriptionExpr_Copy(object, copy_location);
  if (copy_location == LLDB_INVALID_ADDRESS) {
    return llvm::createStringError(
        "Failed to allocate memory for copy object.");
  }

  auto cleanup = llvm::make_scope_exit(
      [&]() { GetProcess().DeallocateMemory(copy_location); });

  if (expr_string.empty())
    return llvm::createStringError("no object description");
  return RunObjectDescriptionExpr(object, expr_string, str);
}

StructuredData::ObjectSP
SwiftLanguageRuntime::GetLanguageSpecificData(SymbolContext sc) {
  if (!sc.function)
    return {};

  auto dict_sp = std::make_shared<StructuredData::Dictionary>();
  auto symbol = sc.function->GetMangled().GetMangledName().GetStringRef();
  auto is_async = SwiftLanguageRuntime::IsAnySwiftAsyncFunctionSymbol(symbol);
  dict_sp->AddBooleanItem("IsSwiftAsyncFunction", is_async);

  auto type_system_or_err =
      GetProcess().GetTarget().GetScratchTypeSystemForLanguage(
          eLanguageTypeSwift);
  if (!type_system_or_err)
    return dict_sp;

  if (auto *ts = llvm::dyn_cast_or_null<TypeSystemSwiftTypeRef>(
          type_system_or_err->get()))
    if (auto swift_ast_ctx = ts->GetSwiftASTContextOrNull(sc))
      dict_sp->AddBooleanItem("SwiftExplicitModules",
                              swift_ast_ctx->HasExplicitModules());

  return dict_sp;
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
  if (!sc.symbol)
    return;
  Mangled mangled_name = sc.symbol->GetMangled();
  if (mangled_name.GuessLanguage() != lldb::eLanguageTypeSwift)
    return;
  Target &target = frame.GetThread()->GetProcess()->GetTarget();
  ExecutionContext exe_ctx(frame);
  auto scratch_ctx = TypeSystemSwiftTypeRefForExpressions::GetForTarget(target);
  if (!scratch_ctx)
    return;
  SwiftASTContextSP swift_ast = scratch_ctx->GetSwiftASTContext(sc);
  if (!swift_ast)
    return;

  CompilerType function_type =
      swift_ast->GetTypeFromMangledTypename(mangled_name.GetMangledName());
  if (!function_type.IsFunctionType())
    return;
  // FIXME: For now we only check the first argument since
  // we don't know how to find the values of arguments
  // further in the argument list.
  //
  // int num_arguments = function_type.GetFunctionArgumentCount();
  // for (int i = 0; i < num_arguments; i++)

  for (int i = 0; i < 1; i++) {
    CompilerType argument_type =
        function_type.GetFunctionArgumentTypeAtIndex(i);
    if (!argument_type.IsFunctionPointerType())
      continue;

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

    bool success =
        abi_sp->GetArgumentValues(*(frame.GetThread().get()), argument_values);
    if (!success)
      continue;
    // Now get a pointer value from the zeroth argument.
    Status error;
    DataExtractor data;
    ExecutionContext exe_ctx;
    frame.CalculateExecutionContext(exe_ctx);
    error = argument_values.GetValueAtIndex(0)->GetValueAsData(&exe_ctx, data,
                                                               NULL);
    lldb::offset_t offset = 0;
    lldb::addr_t fn_ptr_addr = data.GetAddress(&offset);
    Address fn_ptr_address;
    fn_ptr_address.SetLoadAddress(fn_ptr_addr, &target);
    // Now check to see if this has debug info:
    bool add_it = true;

    if (resolve_thunks) {
      SymbolContext sc;
      fn_ptr_address.CalculateSymbolContext(&sc, eSymbolContextEverything);
      if (sc.comp_unit && sc.symbol) {
        ConstString symbol_name = sc.symbol->GetMangled().GetMangledName();
        if (symbol_name) {
          SymbolContext target_context;
          if (GetTargetOfPartialApply(sc, symbol_name, target_context)) {
            if (target_context.symbol)
              fn_ptr_address = target_context.symbol->GetAddress();
            else if (target_context.function)
              fn_ptr_address = target_context.function->GetAddress();
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

//------------------------------------------------------------------
// Exception breakpoint Precondition class for Swift:
//------------------------------------------------------------------
void SwiftLanguageRuntime::SwiftExceptionPrecondition::AddTypeName(
    const char *class_name) {
  m_type_names.insert(class_name);
}

void SwiftLanguageRuntime::SwiftExceptionPrecondition::AddEnumSpec(
    const char *enum_name, const char *element_name) {
  std::unordered_map<std::string, std::vector<std::string>>::value_type
      new_value(enum_name, std::vector<std::string>());
  auto result = m_enum_spec.emplace(new_value);
  result.first->second.push_back(element_name);
}

SwiftLanguageRuntime::SwiftExceptionPrecondition::SwiftExceptionPrecondition() {
}

ValueObjectSP SwiftLanguageRuntime::CalculateErrorValueObjectFromValue(
    Value &value, ConstString name, bool persistent) {
  ValueObjectSP error_valobj_sp;
  auto type_system_or_err =
      GetProcess().GetTarget().GetScratchTypeSystemForLanguage(
          eLanguageTypeSwift);
  if (!type_system_or_err)
    return error_valobj_sp;

  auto *ast_context =
      llvm::dyn_cast_or_null<TypeSystemSwift>(type_system_or_err->get());
  if (!ast_context)
    return error_valobj_sp;

  CompilerType swift_error_proto_type = ast_context->GetErrorType();
  value.SetCompilerType(swift_error_proto_type);

  error_valobj_sp = ValueObjectConstResult::Create(&GetProcess(), value, name);

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
    Target &target = GetProcess().GetTarget();
    auto *persistent_state =
        target.GetPersistentExpressionStateForLanguage(eLanguageTypeSwift);

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
  TargetSP target = frame_sp->CalculateTarget();
  ValueObjectSP error_valobj_sp;

  auto *runtime = Get(process_sp);
  if (!runtime)
    return error_valobj_sp;

  std::optional<Value> arg0 =
      runtime->GetErrorReturnLocationAfterReturn(frame_sp);
  if (!arg0)
    return error_valobj_sp;

  ExecutionContext exe_ctx;
  frame_sp->CalculateExecutionContext(exe_ctx);

  auto *exe_scope = exe_ctx.GetBestExecutionContextScope();
  if (!exe_scope)
    return error_valobj_sp;

  auto scratch_ctx = TypeSystemSwiftTypeRefForExpressions::GetForTarget(target);
  if (!scratch_ctx)
    return error_valobj_sp;

  auto buffer_up =
      std::make_unique<DataBufferHeap>(arg0->GetScalar().GetByteSize(), 0);
  arg0->GetScalar().GetBytes(buffer_up->GetData());
  lldb::DataBufferSP buffer(std::move(buffer_up));

  CompilerType swift_error_proto_type = scratch_ctx->GetErrorType();
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

  auto *swift_ast_ctx = llvm::dyn_cast_or_null<SwiftASTContextForExpressions>(
      type_system_or_err->get());
  if (!swift_ast_ctx || swift_ast_ctx->HasFatalErrors())
    return;
  std::string module_name = "$__lldb_module_for_";
  module_name.append(&name.GetCString()[1]);
  SourceModule module_info;
  module_info.path.push_back(ConstString(module_name));

  swift::ModuleDecl *module_decl = nullptr;
  auto module_decl_or_err = swift_ast_ctx->CreateEmptyModule(module_name);
  if (!module_decl_or_err)
    llvm::consumeError(module_decl_or_err.takeError());
  else
    module_decl = &*module_decl_or_err;
  if (!module_decl)
    return;
  const bool is_static = false;
  const auto introducer = swift::VarDecl::Introducer::Let;

  swift::VarDecl *var_decl = new (*swift_ast_ctx->GetASTContext())
      swift::VarDecl(is_static, introducer, swift::SourceLoc(),
                     swift_ast_ctx->GetIdentifier(name.GetCString()),
                     module_decl);
  var_decl->setInterfaceType(
      llvm::expectedToStdOptional(
          swift_ast_ctx->GetSwiftType(swift_ast_ctx->GetErrorType()))
          .value_or(swift::Type()));
  var_decl->setDebuggerVar(true);

  SwiftPersistentExpressionState *persistent_state =
      llvm::cast<SwiftPersistentExpressionState>(
          target.GetPersistentExpressionStateForLanguage(
              lldb::eLanguageTypeSwift));
  if (!persistent_state)
    return;

  persistent_state->RegisterSwiftPersistentDecl({swift_ast_ctx, var_decl});

  ConstString mangled_name;

  if (ThreadSafeASTContext ast_ctx = swift_ast_ctx->GetASTContext()) {
    swift::Mangle::ASTMangler mangler(**ast_ctx, true);
    mangled_name = ConstString(mangler.mangleGlobalVariableFull(var_decl));
  }

  if (mangled_name.IsEmpty())
    return;

  lldb::addr_t symbol_addr;

  {
    ProcessSP process_sp(target.GetProcessSP());
    Status alloc_error;

    symbol_addr = process_sp->AllocateMemory(
        process_sp->GetAddressByteSize(),
        lldb::ePermissionsWritable | lldb::ePermissionsReadable, alloc_error);

    if (alloc_error.Success() && symbol_addr != LLDB_INVALID_ADDRESS) {
      Status write_error;
      process_sp->WritePointerToMemory(symbol_addr, addr, write_error);

      if (write_error.Success()) {
        persistent_state->RegisterSymbol(mangled_name, symbol_addr);
      }
    }
  }
}

lldb::BreakpointPreconditionSP
SwiftLanguageRuntime::GetBreakpointExceptionPrecondition(LanguageType language,
                                                         bool throw_bp) {
  if (language != eLanguageTypeSwift)
    return lldb::BreakpointPreconditionSP();
  if (!throw_bp)
    return lldb::BreakpointPreconditionSP();
  BreakpointPreconditionSP precondition_sp(
      new SwiftLanguageRuntime::SwiftExceptionPrecondition());
  return precondition_sp;
}

bool SwiftLanguageRuntime::SwiftExceptionPrecondition::EvaluatePrecondition(
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

void SwiftLanguageRuntime::SwiftExceptionPrecondition::GetDescription(
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

Status SwiftLanguageRuntime::SwiftExceptionPrecondition::ConfigurePrecondition(
    Args &args) {
  Status error;
  std::vector<std::string> object_typenames;
  OptionParsing::GetOptionValuesAsStrings(args, "exception-typename",
                                          object_typenames);
  for (auto type_name : object_typenames)
    AddTypeName(type_name.c_str());
  return error;
}

void SwiftLanguageRuntime::AddToLibraryNegativeCache(StringRef library_name) {
  std::lock_guard<std::mutex> locker(m_negative_cache_mutex);
  m_library_negative_cache.insert(library_name);
}

bool SwiftLanguageRuntime::IsInLibraryNegativeCache(StringRef library_name) {
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
                    size_t idx, ValueObject *valobj) {
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

      auto type_or_err = parent_type.GetChildCompilerTypeAtIndex(
          exe_ctx, idx, transparent_pointers, omit_empty_base_classes,
          ignore_array_bounds, child_name, child_byte_size, byte_offset,
          child_bitfield_bit_size, child_bitfield_bit_offset,
          child_is_base_class, child_is_deref_of_parent, valobj,
          language_flags);
      if (!type_or_err)
        LLDB_LOG_ERROR(GetLog(LLDBLog::Types), type_or_err.takeError(),
                       "could not find child #{1}: {0}", idx);
      else
        type = *type_or_err;

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

    llvm::Expected<uint32_t> CalculateNumChildren() override {
      return m_projection->field_projections.size() + m_num_bases;
    }

    lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override {
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

    llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override {
      for (size_t idx = 0; idx < m_projection->field_projections.size();
           idx++) {
        if (m_projection->field_projections.at(idx).name == name)
          return idx;
      }
      return llvm::createStringError("Type has no child named '%s'",
                                     name.AsCString());
    }

    lldb::ChildCacheState Update() override {
      return ChildCacheState::eRefetch;
    }

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
SwiftLanguageRuntime::GetBridgedSyntheticChildProvider(ValueObject &valobj) {
  ConstString type_name = valobj.GetCompilerType().GetTypeName();

  if (!type_name.IsEmpty()) {
    auto iter = m_bridged_synthetics_map.find(type_name.AsCString()),
         end = m_bridged_synthetics_map.end();
    if (iter != end)
      return iter->second;
  }

  ProjectionSyntheticChildren::TypeProjectionUP type_projection(
      new ProjectionSyntheticChildren::TypeProjectionUP::element_type());

  if (auto swift_ast_ctx = TypeSystemSwiftTypeRefForExpressions::GetForTarget(
          valobj.GetTargetSP())) {
    CompilerType swift_type =
        swift_ast_ctx->GetTypeFromMangledTypename(type_name);

    if (swift_type.IsValid()) {
      ExecutionContext exe_ctx(GetProcess());
      bool any_projected = false;
      for (size_t idx = 0, e = llvm::expectedToStdOptional(
                                   swift_type.GetNumChildren(true, &exe_ctx))
                                   .value_or(0);
           idx < e; idx++) {
        // if a projection fails, keep going - we have offsets here, so it
        // should be OK to skip some members
        if (auto projection = ProjectionSyntheticChildren::FieldProjection(
                swift_type, &exe_ctx, idx, &valobj)) {
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

std::optional<std::pair<lldb::ValueObjectSP, bool>>
SwiftLanguageRuntime::ExtractSwiftValueObjectFromCxxWrapper(
    ValueObject &valobj) {
  ValueObjectSP swift_valobj;

  // There are three flavors of C++ wrapper classes:
  // - Reference types wrappers, which have no ivars, and have one super class
  // which contains an opaque pointer to the Swift instance.
  // - Value type wrappers which has one ivar, an opaque pointer to the Swift
  // instance.
  // - Value type wrappers, which has one ivar, a single char array with the
  // swift value embedded directly in it.
  // In all cases the value object should have exactly one child.
  if (valobj.GetNumChildrenIgnoringErrors() != 1)
    return {};

  auto child_valobj = valobj.GetChildAtIndex(0, true);
  auto child_type = child_valobj->GetCompilerType();
  auto child_name = child_type.GetMangledTypeName();

  // If this is a reference wrapper, the first child is actually the super
  // class.
  if (child_name == "swift::_impl::RefCountedClass") {
    // The super class should have exactly one ivar, the opaque pointer that
    // points to the Swift instance.
    if (child_valobj->GetNumChildrenIgnoringErrors() != 1)
      return {};

    auto opaque_ptr_valobj = child_valobj->GetChildAtIndex(0, true);

    // This is a Swift class type, which is a reference, so no need to wrap the
    // corresponding Swift type behind a pointer.
    return {{opaque_ptr_valobj, false}};
  }

  if (child_name == "swift::_impl::OpaqueStorage") {
    if (child_valobj->GetNumChildrenIgnoringErrors() != 1)
      return {};

    auto opaque_ptr_valobj = child_valobj->GetChildAtIndex(0, true);
    // This is a Swift value stored behind a pointer.
    return {{opaque_ptr_valobj, true}};
  }

  CompilerType element_type;
  if (child_type.IsArrayType(&element_type))
    if (element_type.IsCharType())
      // This is an Swift value type inlined directly into the C++ type as a
      // char[n].
      return {{valobj.GetSP(), false}};
  return {};
}

void SwiftLanguageRuntime::WillStartExecutingUserExpression(
    bool runs_in_playground_or_repl) {
  if (runs_in_playground_or_repl)
    return;

  std::lock_guard<std::mutex> lock(m_active_user_expr_mutex);
  Log *log(GetLog(LLDBLog::Expressions));
  LLDB_LOG(log,
           "SwiftLanguageRuntime: starting user expression. "
           "Number active: {0}",
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
      GetProcess().GetTarget().GetScratchTypeSystemForLanguage(
          eLanguageTypeC_plus_plus);
  if (!type_system_or_err) {
    LLDB_LOG_ERROR(
        log, type_system_or_err.takeError(),
        "SwiftLanguageRuntime: Unable to get pointer to type system: {0}");
    return;
  }

  auto ts = *type_system_or_err;
  if (!ts) {
    LLDB_LOG(log, "type system no longer live");
    return;
  }
  ConstString BoolName("bool");
  std::optional<uint64_t> bool_size = llvm::expectedToOptional(
      ts->GetBuiltinTypeByName(BoolName).GetByteSize(nullptr));
  if (!bool_size)
    return;

  Status error;
  Scalar original_value;
  GetProcess().ReadScalarIntegerFromMemory(
      *dynamic_exlusivity_flag_addr, *bool_size, false, original_value, error);

  m_original_dynamic_exclusivity_flag_state = original_value.UInt() != 0;
  if (error.Fail()) {
    LLDB_LOG(log,
             "SwiftLanguageRuntime: Unable to read disableExclusivityChecking "
             "flag state: {0}",
             error.AsCString());
    return;
  }

  Scalar new_value(1U);
  GetProcess().WriteScalarToMemory(*m_dynamic_exclusivity_flag_addr, new_value,
                                   *bool_size, error);
  if (error.Fail()) {
    LLDB_LOG(log,
             "SwiftLanguageRuntime: Unable to set disableExclusivityChecking "
             "flag state: {0}",
             error.AsCString());
    return;
  }

  LLDB_LOG(log,
           "SwiftLanguageRuntime: Changed disableExclusivityChecking flag "
           "state from {0} to 1",
           m_original_dynamic_exclusivity_flag_state);
}

void SwiftLanguageRuntime::DidFinishExecutingUserExpression(
    bool runs_in_playground_or_repl) {
  if (runs_in_playground_or_repl)
    return;

  std::lock_guard<std::mutex> lock(m_active_user_expr_mutex);
  Log *log(GetLog(LLDBLog::Expressions));

  --m_active_user_expr_count;
  LLDB_LOG(log,
           "SwiftLanguageRuntime: finished user expression. "
           "Number active: {0}",
           m_active_user_expr_count);

  if (m_active_user_expr_count > 0)
    return;

  auto dynamic_exlusivity_flag_addr = GetDynamicExclusivityFlagAddr();
  if (!dynamic_exlusivity_flag_addr) {
    LLDB_LOG(log, "Failed to get address of disableExclusivityChecking flag");
    return;
  }

  auto type_system_or_err =
      GetProcess().GetTarget().GetScratchTypeSystemForLanguage(
          eLanguageTypeC_plus_plus);
  if (!type_system_or_err) {
    LLDB_LOG_ERROR(
        log, type_system_or_err.takeError(),
        "SwiftLanguageRuntime: Unable to get pointer to type system: {0}");
    return;
  }

  auto ts = *type_system_or_err;
  if (!ts) {
    LLDB_LOG(log, "type system no longer live");
    return;
  }
  ConstString BoolName("bool");
  std::optional<uint64_t> bool_size = llvm::expectedToOptional(
      ts->GetBuiltinTypeByName(BoolName).GetByteSize(nullptr));
  if (!bool_size)
    return;

  Status error;
  Scalar original_value(m_original_dynamic_exclusivity_flag_state ? 1U : 0U);
  GetProcess().WriteScalarToMemory(*dynamic_exlusivity_flag_addr,
                                   original_value, *bool_size, error);
  if (error.Fail()) {
    LLDB_LOG(log,
             "SwiftLanguageRuntime: Unable to reset "
             "disableExclusivityChecking flag state: {0}",
             error.AsCString());
    return;
  }
  if (log)
    LLDB_LOG(log,
             "SwiftLanguageRuntime: Changed "
             "disableExclusivityChecking flag state back to {0}",
             m_original_dynamic_exclusivity_flag_state);
}

std::optional<Value> SwiftLanguageRuntime::GetErrorReturnLocationAfterReturn(
    lldb::StackFrameSP frame_sp) {
  std::optional<Value> error_val;

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

std::optional<Value> SwiftLanguageRuntime::GetErrorReturnLocationBeforeReturn(
    lldb::StackFrameSP frame_sp, bool &need_to_check_after_return) {
  std::optional<Value> error_val;

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
        m_options() {
    CommandArgumentData mangled_name_arg{eArgTypeSymbol};
    m_arguments.push_back({mangled_name_arg});
  }

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
        error = Status::FromErrorStringWithFormat(
            "invalid short option character '%c'", short_option);
        break;
      }

      return error;
    }

    void OptionParsingStarting(ExecutionContext *execution_context) override {
      m_expand.Clear();
    }

    llvm::ArrayRef<OptionDefinition> GetDefinitions() override {
      return llvm::ArrayRef(g_swift_demangle_options);
    }

    // Options table: Required for subclasses of Options.

    OptionValueBoolean m_expand;
  };

protected:
  void DoExecute(Args &command, CommandReturnObject &result) override {
    for (size_t i = 0; i < command.GetArgumentCount(); i++) {
      StringRef name = command.GetArgumentAtIndex(i);
      if (!name.empty()) {
        Context ctx;
        NodePointer node_ptr = nullptr;
        // Match the behavior of swift-demangle and accept Swift symbols without
        // the leading `$`. This makes symbol copy & paste more convenient.
        if (name.starts_with("S") || name.starts_with("s")) {
          std::string correctedName = std::string("$") + name.str();
          node_ptr =
              SwiftLanguageRuntime::DemangleSymbolAsNode(correctedName, ctx);
        } else {
          node_ptr = SwiftLanguageRuntime::DemangleSymbolAsNode(name, ctx);
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

  std::optional<uint32_t> getReferenceCount(StringRef ObjName,
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
    eval_options.SetSuppressPersistentResult(true);
    ValueObjectSP result_valobj_sp;
    std::string Expr =
        (llvm::Twine("Swift._get") + Kind + llvm::Twine("RetainCount(") +
         ObjName + llvm::Twine(")"))
            .str();
    bool evalStatus = exe_ctx.GetTargetSP()->EvaluateExpression(
        Expr, Frame.get(), result_valobj_sp, eval_options);
    if (evalStatus != eExpressionCompleted)
      return std::nullopt;

    bool success = false;
    uint32_t count = result_valobj_sp->GetSyntheticValue()->GetValueAsUnsigned(
        UINT32_MAX, &success);
    if (!success)
      return std::nullopt;
    return count;
  }

protected:
  void DoExecute(llvm::StringRef command,
                 CommandReturnObject &result) override {
    StackFrameSP frame_sp(m_exe_ctx.GetFrameSP());
    EvaluateExpressionOptions options;
    options.SetLanguage(lldb::eLanguageTypeSwift);
    options.SetSuppressPersistentResult(true);
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
      return;
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
      return;
    }

    // Ask swift debugger support in the compiler about the objects
    // reference counts, and return them to the user.
    std::optional<uint32_t> strong = getReferenceCount(
        command, ReferenceCountType::eReferenceStrong, m_exe_ctx, frame_sp);
    std::optional<uint32_t> unowned = getReferenceCount(
        command, ReferenceCountType::eReferenceUnowned, m_exe_ctx, frame_sp);
    std::optional<uint32_t> weak = getReferenceCount(
        command, ReferenceCountType::eReferenceWeak, m_exe_ctx, frame_sp);

    std::string unavailable = "<unavailable>";

    result.AppendMessageWithFormat(
        "refcount data: (strong = %s, unowned = %s, weak = %s)\n",
        strong ? std::to_string(*strong).c_str() : unavailable.c_str(),
        unowned ? std::to_string(*unowned).c_str() : unavailable.c_str(),
        weak ? std::to_string(*weak).c_str() : unavailable.c_str());
    result.SetStatus(lldb::eReturnStatusSuccessFinishResult);
  }
};

/// Construct a `ThreadTask` instance for a live (yet to be completed) Task
/// variable contained in the first argument.
static llvm::Expected<ThreadSP>
ThreadForLiveTaskArgument(Args &command, ExecutionContext &exe_ctx) {
  if (!exe_ctx.GetFramePtr())
    return llvm::createStringError("no active frame selected");

  if (command.empty() || command[0].ref().empty())
    return llvm::createStringError("missing task variable argument");

  StringRef arg = command[0].ref();

  StackFrame &frame = exe_ctx.GetFrameRef();
  uint32_t path_options =
      StackFrame::eExpressionPathOptionsAllowDirectIVarAccess;
  VariableSP var_sp;
  Status status;
  ValueObjectSP valobj_sp = frame.GetValueForVariableExpressionPath(
      arg, eDynamicDontRunTarget, path_options, var_sp, status);

  addr_t task_ptr = LLDB_INVALID_ADDRESS;
  if (status.Success() && valobj_sp) {
    if (auto task_obj_sp = valobj_sp->GetChildMemberWithName("_task"))
      task_ptr = task_obj_sp->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
    if (task_ptr == LLDB_INVALID_ADDRESS)
      return llvm::createStringError("failed to access Task pointer");
  } else {
    // The argument is not a valid variable expression, try parsing it as a
    // (task) address.
    if (arg.getAsInteger(0, task_ptr))
      return status.takeError();
  }

  if (auto *runtime = SwiftLanguageRuntime::Get(exe_ctx.GetProcessSP()))
    if (auto reflection_ctx = runtime->GetReflectionContext()) {
      auto task_info = reflection_ctx->asyncTaskInfo(task_ptr);
      if (!task_info)
        return task_info.takeError();
      if (task_info->isComplete)
        return llvm::createStringError("task has completed");

      return std::make_shared<ThreadTask>(task_info->id,
                                          task_info->resumeAsyncContext,
                                          task_info->runJob, exe_ctx);
    }

  return llvm::createStringError("failed to access Task data from runtime");
}

class CommandObjectLanguageSwiftTaskBacktrace final
    : public CommandObjectParsed {
public:
  CommandObjectLanguageSwiftTaskBacktrace(CommandInterpreter &interpreter)
      : CommandObjectParsed(
            interpreter, "backtrace",
            "Show the backtrace of Swift tasks. See `thread "
            "backtrace` for customizing backtrace output.",
            "language swift task backtrace <variable-name | address>") {
    CommandArgumentEntry arg_entry;
    arg_entry.emplace_back(eArgTypeVarName, eArgRepeatPlain, LLDB_OPT_SET_1);
    arg_entry.emplace_back(eArgTypeAddress, eArgRepeatPlain, LLDB_OPT_SET_2);
    m_arguments.push_back(arg_entry);
  }

private:
  void DoExecute(Args &command, CommandReturnObject &result) override {
    if (command.GetArgumentCount() != 1) {
      result.AppendError("missing <variable-name> or <address> argument");
      return;
    }

    llvm::Expected<ThreadSP> thread_task =
        ThreadForLiveTaskArgument(command, m_exe_ctx);
    if (auto error = thread_task.takeError()) {
      result.AppendError(toString(std::move(error)));
      return;
    }

    // GetStatus prints the backtrace.
    thread_task.get()->GetStatus(result.GetOutputStream(), 0, UINT32_MAX, 0,
                                 false, true);
    result.SetStatus(lldb::eReturnStatusSuccessFinishResult);
  }
};

class CommandObjectLanguageSwiftTaskSelect final : public CommandObjectParsed {
public:
  CommandObjectLanguageSwiftTaskSelect(CommandInterpreter &interpreter)
      : CommandObjectParsed(
            interpreter, "select",
            "Change the currently selected thread to thread representation of "
            "the given Swift Task. See `thread select`.",
            "language swift task select <variable-name | address>") {
    CommandArgumentEntry arg_entry;
    arg_entry.emplace_back(eArgTypeVarName, eArgRepeatPlain, LLDB_OPT_SET_1);
    arg_entry.emplace_back(eArgTypeAddress, eArgRepeatPlain, LLDB_OPT_SET_2);
    m_arguments.push_back(arg_entry);
  }

private:
  void DoExecute(Args &command, CommandReturnObject &result) override {
    if (command.GetArgumentCount() != 1) {
      result.AppendError("missing <variable-name> or <address> argument");
      return;
    }

    llvm::Expected<ThreadSP> thread_task =
        ThreadForLiveTaskArgument(command, m_exe_ctx);
    if (auto error = thread_task.takeError()) {
      result.AppendError(toString(std::move(error)));
      return;
    }

    auto &thread_list = m_exe_ctx.GetProcessRef().GetThreadList();
    thread_list.AddThread(thread_task.get());
    thread_list.SetSelectedThreadByID(thread_task.get()->GetID());

    result.SetStatus(lldb::eReturnStatusSuccessFinishResult);
  }
};

class CommandObjectLanguageSwiftTaskInfo final : public CommandObjectParsed {
public:
  CommandObjectLanguageSwiftTaskInfo(CommandInterpreter &interpreter)
      : CommandObjectParsed(interpreter, "info",
                            "Print info about the Task being run on the "
                            "current thread or the Task at the given address."
                            "language swift task info [<address>]") {
    AddSimpleArgumentList(eArgTypeAddress, eArgRepeatOptional);
  }

private:
  void DoExecute(Args &command, CommandReturnObject &result) override {
    addr_t task_addr = LLDB_INVALID_ADDRESS;
    StringRef task_name = "current_task";

    if (command.GetArgumentCount() == 1) {
      StringRef addr_arg = command.GetArgumentAtIndex(0);
      if (addr_arg.getAsInteger(0, task_addr)) {
        result.AppendErrorWithFormatv("invalid address format: {0}", addr_arg);
        return;
      }
      task_name = "task";
    }

    if (task_addr == LLDB_INVALID_ADDRESS) {
      if (!m_exe_ctx.GetThreadPtr()) {
        result.AppendError(
            "must be run from a running process and valid thread");
        return;
      }

      TaskInspector task_inspector;
      auto task_addr_or_err = task_inspector.GetTaskAddrFromThreadLocalStorage(
          m_exe_ctx.GetThreadRef());
      if (auto error = task_addr_or_err.takeError()) {
        result.AppendError(toString(std::move(error)));
        return;
      }

      task_addr = task_addr_or_err.get();
    }

    auto ts_or_err = m_exe_ctx.GetTargetRef().GetScratchTypeSystemForLanguage(
        eLanguageTypeSwift);
    if (auto error = ts_or_err.takeError()) {
      result.AppendErrorWithFormatv("could not get Swift type system: {0}",
                                    llvm::fmt_consume(std::move(error)));
      return;
    }

    auto *ts = llvm::dyn_cast_or_null<TypeSystemSwiftTypeRef>(ts_or_err->get());
    if (!ts) {
      result.AppendError("could not get Swift type system");
      return;
    }

    // TypeMangling for "Swift.UnsafeCurrentTask"
    CompilerType task_type =
        ts->GetTypeFromMangledTypename(ConstString("$sSctD"));
    auto task_sp = ValueObject::CreateValueObjectFromAddress(
        task_name, task_addr, m_exe_ctx, task_type, false);
    if (auto synthetic_sp = task_sp->GetSyntheticValue())
      task_sp = synthetic_sp;

    auto error = task_sp->Dump(result.GetOutputStream());
    if (error) {
      result.AppendErrorWithFormatv("failed to print current task: {0}",
                                    toString(std::move(error)));
      return;
    }

    result.SetStatus(lldb::eReturnStatusSuccessFinishResult);
  }
};

class CommandObjectLanguageSwiftTask final : public CommandObjectMultiword {
public:
  CommandObjectLanguageSwiftTask(CommandInterpreter &interpreter)
      : CommandObjectMultiword(
            interpreter, "task", "Commands for inspecting Swift Tasks.",
            "language swift task <subcommand> [<subcommand-options>]") {
    LoadSubCommand("backtrace",
                   CommandObjectSP(new CommandObjectLanguageSwiftTaskBacktrace(
                       interpreter)));
    LoadSubCommand(
        "select",
        CommandObjectSP(new CommandObjectLanguageSwiftTaskSelect(interpreter)));
    LoadSubCommand(
        "info",
        CommandObjectSP(new CommandObjectLanguageSwiftTaskInfo(interpreter)));
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
    LoadSubCommand("task", CommandObjectSP(new CommandObjectLanguageSwiftTask(
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
      SwiftLanguageRuntime::GetBreakpointExceptionPrecondition);
}

void SwiftLanguageRuntime::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

std::optional<AsyncUnwindRegisterNumbers>
GetAsyncUnwindRegisterNumbers(llvm::Triple::ArchType triple) {
  switch (triple) {
  case llvm::Triple::x86_64: {
    AsyncUnwindRegisterNumbers regnums;
    regnums.async_ctx_regnum = dwarf_r14_x86_64;
    regnums.pc_regnum = dwarf_rip_x86_64;
    return regnums;
  }
  case llvm::Triple::aarch64: {
    AsyncUnwindRegisterNumbers regnums;
    regnums.async_ctx_regnum = arm64_dwarf::x22;
    regnums.pc_regnum = arm64_dwarf::pc;
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
        regnums->GetRegisterKind(), regnums->async_ctx_regnum);
    return regctx->ReadRegisterAsUnsigned(reg, LLDB_INVALID_ADDRESS);
  }

  assert(false && "swift async supports only x86_64 and arm64");
  return LLDB_INVALID_ADDRESS;
}

/// Functional wrapper to read a register as an address.
static llvm::Expected<addr_t> ReadRegisterAsAddress(RegisterContext &regctx,
                                                    RegisterKind regkind,
                                                    unsigned regnum) {
  unsigned lldb_regnum =
      regctx.ConvertRegisterKindToRegisterNumber(regkind, regnum);
  auto reg = regctx.ReadRegisterAsUnsigned(lldb_regnum, LLDB_INVALID_ADDRESS);
  if (reg != LLDB_INVALID_ADDRESS)
    return reg;
  return llvm::createStringError(
      "SwiftLanguageRuntime: failed to read register from regctx");
}

/// Functional wrapper to read a pointer from process memory at `addr +
/// offset`.
static llvm::Expected<addr_t> ReadPtrFromAddr(Process &process, addr_t addr,
                                              int offset = 0) {
  Status error;
  addr_t ptr = process.ReadPointerFromMemory(addr + offset, error);
  if (ptr != LLDB_INVALID_ADDRESS)
    return ptr;
  return llvm::createStringError("SwiftLanguageRuntime: Failed to read ptr "
                                 "from memory address 0x%8.8" PRIx64
                                 " Error was %s",
                                 addr + offset, error.AsCString());
}

/// Computes the Canonical Frame Address (CFA) by converting the abstract
/// location of UnwindPlan::Row::FAValue into a concrete address. This is a
/// simplified version of the methods in RegisterContextUnwind, since plumbing
/// access to those here would be challenging.
static llvm::Expected<addr_t> GetCFA(Process &process, RegisterContext &regctx,
                                     addr_t pc_offset,
                                     const UnwindPlan &unwind_plan) {
  auto *row = unwind_plan.GetRowForFunctionOffset(pc_offset);
  if (!row)
    return llvm::createStringError(
        "SwiftLanguageRuntime: Invalid Unwind Row when computing CFA");

  UnwindPlan::Row::FAValue cfa_loc = row->GetCFAValue();

  using ValueType = UnwindPlan::Row::FAValue::ValueType;
  switch (cfa_loc.GetValueType()) {
  case ValueType::isRegisterPlusOffset: {
    unsigned regnum = cfa_loc.GetRegisterNumber();
    if (llvm::Expected<addr_t> regvalue = ReadRegisterAsAddress(
            regctx, unwind_plan.GetRegisterKind(), regnum))
      return *regvalue + cfa_loc.GetOffset();
    else
      return regvalue;
  }
  case ValueType::isConstant:
  case ValueType::isDWARFExpression:
  case ValueType::isRaSearch:
  case ValueType::isRegisterDereferenced:
  case ValueType::unspecified:
    break;
  }
  return llvm::createStringError(
      "SwiftLanguageRuntime: Unsupported FA location type = %d",
      cfa_loc.GetValueType());
}

static std::shared_ptr<const UnwindPlan>
GetUnwindPlanForAsyncRegister(FuncUnwinders &unwinders, Target &target,
                              Thread &thread) {
  // We cannot trust compiler emitted unwind plans, as they respect the
  // swifttail calling convention, which assumes the async register is _not_
  // restored and therefore it is not tracked by compiler plans. If LLDB uses
  // those plans, it may take "no info" to mean "register not clobbered". For
  // those reasons, always favour the assembly plan first, it will try to track
  // the async register by assuming the usual arm calling conventions.
  if (auto asm_plan = unwinders.GetAssemblyUnwindPlan(target, thread))
    return asm_plan;
  // In the unlikely case the assembly plan is not available, try all others.
  return unwinders.GetUnwindPlanAtNonCallSite(target, thread);
}

static llvm::Expected<std::shared_ptr<const UnwindPlan>>
GetAsmUnwindPlan(Address pc, SymbolContext &sc, Thread &thread) {
  FuncUnwindersSP unwinders =
      pc.GetModule()->GetUnwindTable().GetFuncUnwindersContainingAddress(pc,
                                                                         sc);
  if (!unwinders)
    return llvm::createStringError("SwiftLanguageRuntime: Failed to find "
                                   "function unwinder at address 0x%8.8" PRIx64,
                                   pc.GetFileAddress());

  auto unwind_plan = GetUnwindPlanForAsyncRegister(
      *unwinders, thread.GetProcess()->GetTarget(), thread);
  if (!unwind_plan)
    return llvm::createStringError(
        "SwiftLanguageRuntime: Failed to find non call site unwind plan at "
        "address 0x%8.8" PRIx64,
        pc.GetFileAddress());
  return unwind_plan;
}

static llvm::Expected<uint32_t>
GetFpRegisterNumber(const UnwindPlan &unwind_plan, RegisterContext &regctx) {
  uint32_t fp_unwind_regdomain;
  if (!regctx.ConvertBetweenRegisterKinds(
          lldb::eRegisterKindGeneric, LLDB_REGNUM_GENERIC_FP,
          unwind_plan.GetRegisterKind(), fp_unwind_regdomain)) {
    // This should never happen.
    // If asserts are disabled, return an error to avoid creating an invalid
    // unwind plan.
    const auto *error_msg =
        "SwiftLanguageRuntime: Failed to convert register domains";
    llvm_unreachable(error_msg);
    return llvm::createStringError(error_msg);
  }
  return fp_unwind_regdomain;
}

struct FrameSetupInfo {
  int64_t frame_setup_func_offset;
  int fp_cfa_offset;
};

/// Detect the point in the function where the prologue created a frame,
/// returning:
/// 1. The offset of the first instruction after that point. For a frameless
/// function, this offset is large positive number, so that PC can still be
/// compared against it.
/// 2. The CFA offset at which FP is stored, meaningless in the frameless case.
static llvm::Expected<FrameSetupInfo>
GetFrameSetupInfo(const UnwindPlan &unwind_plan, RegisterContext &regctx) {
  using AbstractRegisterLocation = UnwindPlan::Row::AbstractRegisterLocation;

  llvm::Expected<uint32_t> fp_unwind_regdomain =
      GetFpRegisterNumber(unwind_plan, regctx);
  if (!fp_unwind_regdomain)
    return fp_unwind_regdomain.takeError();

  // Look at the first few (12) rows of the plan and store FP's location.
  // This number is based on AAPCS, with 10 callee-saved GPRs and 8 floating
  // point registers. When STP instructions are used, the plan would have one
  // initial row, nine rows of saving callee-saved registers, and two standard
  // prologue rows (fp+lr and sp).
  const int upper_bound = std::min(12, unwind_plan.GetRowCount());
  llvm::SmallVector<AbstractRegisterLocation, 12> fp_locs;
  for (int row_idx = 0; row_idx < upper_bound; row_idx++) {
    auto *row = unwind_plan.GetRowAtIndex(row_idx);
    AbstractRegisterLocation regloc;
    if (!row->GetRegisterInfo(*fp_unwind_regdomain, regloc))
      regloc.SetSame();
    fp_locs.push_back(regloc);
  }

  // Find first location where FP is stored *at* some CFA offset.
  auto *it = llvm::find_if(
      fp_locs, [](auto fp_loc) { return fp_loc.IsAtCFAPlusOffset(); });

  // This is a frameless function, use large positive offset so that a PC can
  // still be compared against it.
  if (it == fp_locs.end())
    return FrameSetupInfo{std::numeric_limits<int64_t>::max(), 0};

  // This is an async function with a frame. The prologue roughly follows this
  // sequence of instructions:
  // adjust sp
  // save lr        @ CFA-8
  // save fp        @ CFA-16  << `it` points to this row.
  // save async_reg @ CFA-24  << subsequent row.
  // Use subsequent row, if available.
  // Pointer auth may introduce more instructions, but they don't affect the
  // unwinder rows / store to the stack.
  int row_idx = it - fp_locs.begin();
  int next_row_idx = row_idx + 1;

  // If subsequent row is invalid, approximate through current row.
  if (next_row_idx == unwind_plan.GetRowCount() ||
      next_row_idx == upper_bound ||
      !fp_locs[next_row_idx].IsAtCFAPlusOffset()) {
    LLDB_LOG(GetLog(LLDBLog::Unwind), "SwiftLanguageRuntime:: UnwindPlan did "
                                      "not contain a valid row after FP setup");
    auto *row = unwind_plan.GetRowAtIndex(row_idx);
    return FrameSetupInfo{row->GetOffset(), fp_locs[row_idx].GetOffset()};
  }

  auto *subsequent_row = unwind_plan.GetRowAtIndex(next_row_idx);
  return FrameSetupInfo{subsequent_row->GetOffset(),
                        fp_locs[next_row_idx].GetOffset()};
}

/// Reads the async register from its ABI-guaranteed stack-slot, or directly
/// from the register depending on where pc is relative to the start of the
/// function.
static llvm::Expected<addr_t> ReadAsyncContextRegisterFromUnwind(
    SymbolContext &sc, Process &process, Address pc, Address func_start_addr,
    RegisterContext &regctx, AsyncUnwindRegisterNumbers regnums) {
  llvm::Expected<std::shared_ptr<const UnwindPlan>> unwind_plan =
      GetAsmUnwindPlan(pc, sc, regctx.GetThread());
  if (!unwind_plan)
    return unwind_plan.takeError();
  llvm::Expected<FrameSetupInfo> frame_setup =
      GetFrameSetupInfo(**unwind_plan, regctx);
  if (!frame_setup)
    return frame_setup.takeError();

  // Is PC before the frame formation? If so, use async register directly.
  // This handles frameless functions, as frame_setup_func_offset is INT_MAX.
  addr_t pc_offset = pc.GetFileAddress() - func_start_addr.GetFileAddress();
  if ((int64_t)pc_offset < frame_setup->frame_setup_func_offset)
    return ReadRegisterAsAddress(regctx, regnums.GetRegisterKind(),
                                 regnums.async_ctx_regnum);

  // A frame was formed, and FP was saved at a CFA offset. Compute CFA and read
  // the location beneath where FP was saved.
  llvm::Expected<addr_t> cfa =
      GetCFA(process, regctx, pc_offset, **unwind_plan);
  if (!cfa)
    return cfa.takeError();

  addr_t async_reg_addr = process.FixDataAddress(
      *cfa + frame_setup->fp_cfa_offset - process.GetAddressByteSize());
  Status error;
  addr_t async_reg = process.ReadPointerFromMemory(async_reg_addr, error);
  if (error.Fail())
    return error.ToError();
  return async_reg;
}

static llvm::Expected<bool>
DoesContinuationPointToSameFunction(addr_t async_reg, SymbolContext &sc,
                                    Process &process) {
  llvm::Expected<addr_t> continuation_ptr = ReadPtrFromAddr(
      process, async_reg, /*offset*/ process.GetAddressByteSize());
  if (!continuation_ptr)
    return continuation_ptr.takeError();

  Address continuation_addr;
  continuation_addr.SetLoadAddress(process.FixCodeAddress(*continuation_ptr),
                                   &process.GetTarget());
  if (sc.function) {
    AddressRange unused_range;
    return sc.function->GetRangeContainingLoadAddress(
        continuation_addr.GetOffset(), process.GetTarget(), unused_range);
  }
  assert(sc.symbol);
  return sc.symbol->ContainsFileAddress(continuation_addr.GetFileAddress());
}

/// Returns true if the async register should be dereferenced once to obtain the
/// CFA of the currently executing function. This is the case at the start of
/// "Q" funclets, before the low level code changes the meaning of the async
/// register to not require the indirection.
/// The end of the prologue approximates the transition point well in non-arm64e
/// targets.
/// FIXME: In the few instructions between the end of the prologue and the
/// transition point, this approximation fails. rdar://139676623
static llvm::Expected<bool> IsIndirectContext(Process &process,
                                              StringRef mangled_name,
                                              Address pc, SymbolContext &sc,
                                              addr_t async_reg) {
  if (!SwiftLanguageRuntime::IsSwiftAsyncAwaitResumePartialFunctionSymbol(
          mangled_name))
    return false;

  // For arm64e, pointer authentication generates branches that cause stepping
  // algorithms to stop & unwind in more places. The "end of the prologue"
  // approximation fails in those; instead, check whether the continuation
  // pointer still points to the currently executing function. This works for
  // all instructions, but fails when direct recursion is involved.
  if (process.GetTarget().GetArchitecture().GetTriple().isArm64e())
    return DoesContinuationPointToSameFunction(async_reg, sc, process);

  // This is checked prior to calling this function.
  assert(sc.function || sc.symbol);
  uint32_t prologue_size = sc.function ? sc.function->GetPrologueByteSize()
                                       : sc.symbol->GetPrologueByteSize();
  Address func_start_addr =
      sc.function ? sc.function->GetAddress() : sc.symbol->GetAddress();
  // Include one instruction after the prologue. This is where breakpoints
  // by function name are set, so it's important to get this point right. This
  // instruction is exactly at address "base + prologue", so adding 1
  // in the range will do.
  AddressRange prologue_range(func_start_addr, prologue_size + 1);
  return prologue_range.ContainsLoadAddress(pc, &process.GetTarget());
}

// Examine the register state and detect the transition from a real
// stack frame to an AsyncContext frame, or a frame in the middle of
// the AsyncContext chain, and return an UnwindPlan for these situations.
UnwindPlanSP
SwiftLanguageRuntime::GetRuntimeUnwindPlan(ProcessSP process_sp,
                                           RegisterContext *regctx,
                                           bool &behaves_like_zeroth_frame) {
  auto log_expected = [](llvm::Error error) {
    Log *log = GetLog(LLDBLog::Unwind);
    LLDB_LOG_ERROR(log, std::move(error), "{0}");
    return UnwindPlanSP();
  };

  Target &target(process_sp->GetTarget());
  auto arch = target.GetArchitecture();
  std::optional<AsyncUnwindRegisterNumbers> regnums =
      GetAsyncUnwindRegisterNumbers(arch.GetMachine());
  if (!regnums)
    return UnwindPlanSP();

  // If we can't fetch the fp reg, and we *can* fetch the async
  // context register, then we're in the middle of the AsyncContext
  // chain, return an UnwindPlan for that.
  addr_t fp = regctx->GetFP(LLDB_INVALID_ADDRESS);
  if (fp == LLDB_INVALID_ADDRESS) {
    if (GetAsyncContext(regctx) != LLDB_INVALID_ADDRESS)
      return GetFollowAsyncContextUnwindPlan(process_sp, regctx, arch,
                                             behaves_like_zeroth_frame);
    return UnwindPlanSP();
  }

  Address pc;
  pc.SetLoadAddress(regctx->GetPC(), &target);
  SymbolContext sc;
  if (pc.IsValid())
    if (!pc.CalculateSymbolContext(&sc, eSymbolContextFunction |
                                            eSymbolContextSymbol))
      return UnwindPlanSP();

  Address func_start_addr;
  ConstString mangled_name;
  if (sc.function) {
    func_start_addr = sc.function->GetAddress();
    mangled_name = sc.function->GetMangled().GetMangledName();
  } else if (sc.symbol) {
    func_start_addr = sc.symbol->GetAddress();
    mangled_name = sc.symbol->GetMangled().GetMangledName();
  } else {
    return UnwindPlanSP();
  }

  if (!IsAnySwiftAsyncFunctionSymbol(mangled_name.GetStringRef()))
    return UnwindPlanSP();

  // The async register contains, at the start of the funclet:
  // 1. The async context of the async function that just finished executing,
  // for await resume ("Q") funclets ("indirect context").
  // 2. The async context for the currently executing async function, for all
  // other funclets ("Y" and "Yx" funclets, where "x" is a number).

  llvm::Expected<addr_t> async_reg = ReadAsyncContextRegisterFromUnwind(
      sc, *process_sp, pc, func_start_addr, *regctx, *regnums);
  if (!async_reg)
    return log_expected(async_reg.takeError());

  llvm::Expected<bool> maybe_indirect_context =
      IsIndirectContext(*process_sp, mangled_name, pc, sc, *async_reg);
  if (!maybe_indirect_context)
    return log_expected(maybe_indirect_context.takeError());

  llvm::Expected<addr_t> async_ctx =
      *maybe_indirect_context ? ReadPtrFromAddr(GetProcess(), *async_reg)
                              : *async_reg;
  if (!async_ctx)
    return log_expected(async_ctx.takeError());

  UnwindPlan::Row row;
  const int32_t ptr_size = 8;
  row.SetOffset(0);

  // The CFA of a funclet is its own async context.
  row.GetCFAValue().SetIsConstant(*async_ctx);

  // The value of the async register in the parent frame (which is the
  // continuation funclet) is the async context of this frame.
  row.SetRegisterLocationToIsConstant(regnums->async_ctx_regnum, *async_ctx,
                                      /*can_replace=*/false);

  if (std::optional<addr_t> pc_after_prologue =
          TrySkipVirtualParentProlog(*async_ctx, *process_sp))
    row.SetRegisterLocationToIsConstant(regnums->pc_regnum, *pc_after_prologue,
                                        false);
  else
    row.SetRegisterLocationToAtCFAPlusOffset(regnums->pc_regnum, ptr_size,
                                             false);
  row.SetUnspecifiedRegistersAreUndefined(true);

  UnwindPlanSP plan = std::make_shared<UnwindPlan>(lldb::eRegisterKindDWARF);
  plan->AppendRow(row);
  plan->SetSourceName("Swift Transition-to-AsyncContext-Chain");
  // Make this plan more authoritative, so that the unwinding fallback
  // mechanisms don't kick in and produce a physical backtrace instead.
  plan->SetSourcedFromCompiler(eLazyBoolYes);
  plan->SetUnwindPlanValidAtAllInstructions(eLazyBoolYes);
  plan->SetUnwindPlanForSignalTrap(eLazyBoolYes);
  return plan;
}

UnwindPlanSP SwiftLanguageRuntime::GetFollowAsyncContextUnwindPlan(
    ProcessSP process_sp, RegisterContext *regctx, ArchSpec &arch,
    bool &behaves_like_zeroth_frame) {
  UnwindPlan::Row row;
  const int32_t ptr_size = 8;
  row.SetOffset(0);

  std::optional<AsyncUnwindRegisterNumbers> regnums =
      GetAsyncUnwindRegisterNumbers(arch.GetMachine());
  if (!regnums)
    return UnwindPlanSP();

  row.GetCFAValue().SetIsRegisterDereferenced(regnums->async_ctx_regnum);
  // The value of the async register in the parent frame (which is the
  // continuation funclet) is the async context of this frame.
  row.SetRegisterLocationToIsCFAPlusOffset(regnums->async_ctx_regnum,
                                           /*offset*/ 0, false);

  const unsigned num_indirections = 1;
  if (std::optional<addr_t> pc_after_prologue = TrySkipVirtualParentProlog(
          GetAsyncContext(regctx), *process_sp, num_indirections))
    row.SetRegisterLocationToIsConstant(regnums->pc_regnum, *pc_after_prologue,
                                        false);
  else
    row.SetRegisterLocationToAtCFAPlusOffset(regnums->pc_regnum, ptr_size,
                                             false);

  row.SetUnspecifiedRegistersAreUndefined(true);

  UnwindPlanSP plan = std::make_shared<UnwindPlan>(lldb::eRegisterKindDWARF);
  plan->AppendRow(row);
  plan->SetSourceName("Swift Following-AsyncContext-Chain");
  // Make this plan more authoritative, so that the unwinding fallback
  // mechanisms don't kick in and produce a physical backtrace instead.
  plan->SetSourcedFromCompiler(eLazyBoolYes);
  plan->SetUnwindPlanValidAtAllInstructions(eLazyBoolYes);
  plan->SetUnwindPlanForSignalTrap(eLazyBoolYes);
  behaves_like_zeroth_frame = true;
  return plan;
}

std::optional<lldb::addr_t> SwiftLanguageRuntime::TrySkipVirtualParentProlog(
    lldb::addr_t async_reg_val, Process &process, unsigned num_indirections) {
  assert(num_indirections <= 2 &&
         "more than two dereferences should not be needed");
  if (async_reg_val == LLDB_INVALID_ADDRESS || async_reg_val == 0)
    return {};

  const auto ptr_size = process.GetAddressByteSize();
  Status error;

  // Compute the CFA of this frame.
  addr_t cfa = async_reg_val;
  for (; num_indirections != 0; --num_indirections) {
    process.ReadMemory(cfa, &cfa, ptr_size, error);
    if (error.Fail())
      return {};
  }

  // The last funclet will have a zero CFA, we don't want to read that.
  if (cfa == 0)
    return {};

  // Get the PC of the parent frame, i.e. the continuation pointer, which is
  // the second field of the CFA.
  addr_t pc_location = cfa + ptr_size;
  addr_t pc_value = process.ReadPointerFromMemory(pc_location, error);
  if (error.Fail())
    return {};
  // Clear any high order bits of this code address so that SetLoadAddress works
  // properly.
  pc_value = process.FixCodeAddress(pc_value);

  Address pc;
  Target &target = process.GetTarget();
  pc.SetLoadAddress(pc_value, &target);
  if (!pc.IsValid())
    return {};

  SymbolContext sc;
  bool sc_ok = pc.CalculateSymbolContext(&sc, eSymbolContextFunction |
                                                  eSymbolContextSymbol);
  if (!sc_ok || (!sc.symbol && !sc.function)) {
    Log *log = GetLog(LLDBLog::Unwind);
    LLDB_LOGF(log,
              "SwiftLanguageRuntime::%s Failed to find a symbol context for "
              "address 0x%" PRIx64,
              __FUNCTION__, pc_value);
    return {};
  }

  auto prologue_size = sc.symbol ? sc.symbol->GetPrologueByteSize()
                                 : sc.function->GetPrologueByteSize();
  return pc_value + prologue_size;
}

/// Attempts to read the memory location at `task_addr_location`, producing
/// the Task pointer if possible.
static llvm::Expected<lldb::addr_t>
ReadTaskAddr(lldb::addr_t task_addr_location, Process &process) {
  Status status;
  addr_t task_addr = process.ReadPointerFromMemory(task_addr_location, status);
  if (status.Fail())
    return llvm::joinErrors(
        llvm::createStringError("could not get current task from thread"),
        status.takeError());
  return task_addr;
}

/// Compute the location where the Task pointer for `real_thread` is stored by
/// the runtime.
static llvm::Expected<lldb::addr_t>
ComputeTaskAddrLocationFromThreadLocalStorage(Thread &real_thread) {
#if !SWIFT_THREADING_USE_RESERVED_TLS_KEYS
  return llvm::createStringError(
      "getting the current task from a thread is not supported");
#else
  // Compute the thread local storage address for this thread.
  addr_t tsd_addr = LLDB_INVALID_ADDRESS;

  if (auto info_sp = real_thread.GetExtendedInfo())
    if (auto *info_dict = info_sp->GetAsDictionary())
      info_dict->GetValueForKeyAsInteger("tsd_address", tsd_addr);

  if (tsd_addr == LLDB_INVALID_ADDRESS)
    return llvm::createStringError("could not read current task from thread");

  // Offset of the Task pointer in a Thread's local storage.
  Process &process = *real_thread.GetProcess();
  size_t ptr_size = process.GetAddressByteSize();
  uint64_t task_ptr_offset_in_tls =
      swift::tls_get_key(swift::tls_key::concurrency_task) * ptr_size;
  return tsd_addr + task_ptr_offset_in_tls;
#endif
}

llvm::Expected<lldb::addr_t>
TaskInspector::GetTaskAddrFromThreadLocalStorage(Thread &thread) {
  // Look through backing threads when inspecting TLS.
  Thread &real_thread =
      thread.GetBackingThread() ? *thread.GetBackingThread() : thread;

  if (auto it = m_tid_to_task_addr_location.find(real_thread.GetID());
      it != m_tid_to_task_addr_location.end()) {
#ifndef NDEBUG
    // In assert builds, check that caching did not produce incorrect results.
    llvm::Expected<lldb::addr_t> task_addr_location =
        ComputeTaskAddrLocationFromThreadLocalStorage(real_thread);
    assert(task_addr_location);
    assert(it->second == *task_addr_location);
#endif
    llvm::Expected<lldb::addr_t> task_addr =
        ReadTaskAddr(it->second, *thread.GetProcess());
    if (task_addr)
      return task_addr;
    // If the cached task addr location became invalid, invalidate the cache.
    m_tid_to_task_addr_location.erase(it);
    LLDB_LOG_ERROR(GetLog(LLDBLog::OS), task_addr.takeError(),
                   "TaskInspector: evicted task location address due to "
                   "invalid memory read: {0}");
  }

  llvm::Expected<lldb::addr_t> task_addr_location =
      ComputeTaskAddrLocationFromThreadLocalStorage(real_thread);
  if (!task_addr_location)
    return task_addr_location;

  llvm::Expected<lldb::addr_t> task_addr =
      ReadTaskAddr(*task_addr_location, *thread.GetProcess());

  // If the read from this TLS address is successful, cache the TLS address.
  // Caching without a valid read is dangerous: earlier in the thread
  // lifetime, the result of GetExtendedInfo can be invalid.
  if (task_addr &&
      real_thread.GetProcess()->GetTarget().GetSwiftCacheTaskPointerLocation())
    m_tid_to_task_addr_location.try_emplace(real_thread.GetID(),
                                            *task_addr_location);
  return task_addr;
}

namespace {

/// Lightweight wrapper around TaskStatusRecord pointers, providing:
///   * traversal over the embedded linnked list of status records
///   * information contained within records
///
/// Currently supports TaskNameStatusRecord. See swift/ABI/TaskStatus.h
struct TaskStatusRecord {
  Process &process;
  addr_t addr;
  size_t addr_size;

  TaskStatusRecord(Process &process, addr_t addr)
      : process(process), addr(addr) {
    addr_size = process.GetAddressByteSize();
  }

  operator bool() const { return addr && addr != LLDB_INVALID_ADDRESS; }

  // The offset of TaskStatusRecord members. The unit is pointers, and must be
  // converted to bytes based on the target's address size.
  static constexpr unsigned FlagsPointerOffset = 0;
  static constexpr unsigned ParentPointerOffset = 1;
  static constexpr unsigned TaskNamePointerOffset = 2;

  enum Kind : uint64_t {
    TaskName = 6,
  };

  uint64_t getKind(Status &status) {
    const offset_t flagsByteOffset = FlagsPointerOffset * addr_size;
    if (status.Success())
      return process.ReadUnsignedIntegerFromMemory(
          addr + flagsByteOffset, addr_size, UINT64_MAX, status);
    return UINT64_MAX;
  }

  std::optional<std::string> getName(Status &status) {
    if (getKind(status) != Kind::TaskName)
      return {};

    const offset_t taskNameByteOffset = TaskNamePointerOffset * addr_size;
    addr_t name_addr =
        process.ReadPointerFromMemory(addr + taskNameByteOffset, status);
    if (!status.Success())
      return {};

    std::string name;
    process.ReadCStringFromMemory(name_addr, name, status);
    if (status.Success())
      return name;

    return {};
  }

  addr_t getParent(Status &status) {
    const offset_t parentByteOffset = ParentPointerOffset * addr_size;
    addr_t parent = LLDB_INVALID_ADDRESS;
    if (*this && status.Success())
      parent = process.ReadPointerFromMemory(addr + parentByteOffset, status);
    return parent;
  }
};

/// Lightweight wrapper around Task pointers, providing access to a Task's
/// active status record. See swift/ABI/Task.h
struct Task {
  Process &process;
  addr_t addr;

  operator bool() const { return addr && addr != LLDB_INVALID_ADDRESS; }

  // The offset of the active TaskStatusRecord pointer. The unit is pointers,
  // and must be converted to bytes based on the target's address size.
  static constexpr unsigned ActiveTaskStatusRecordPointerOffset = 13;

  TaskStatusRecord getActiveTaskStatusRecord(Status &status) {
    const offset_t activeTaskStatusRecordByteOffset =
        ActiveTaskStatusRecordPointerOffset * process.GetAddressByteSize();
    addr_t status_record = LLDB_INVALID_ADDRESS;
    if (status.Success())
      status_record = process.ReadPointerFromMemory(
          addr + activeTaskStatusRecordByteOffset, status);
    return {process, status_record};
  }
};

}; // namespace

llvm::Expected<std::optional<std::string>> GetTaskName(lldb::addr_t task_addr,
                                                       Process &process) {
  Status status;
  Task task{process, task_addr};
  auto status_record = task.getActiveTaskStatusRecord(status);
  while (status_record) {
    if (auto name = status_record.getName(status))
      return *name;
    status_record.addr = status_record.getParent(status);
  }
  if (status.Success())
    return std::nullopt;
  return status.takeError();
}

} // namespace lldb_private
