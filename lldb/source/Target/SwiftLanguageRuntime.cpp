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

#include "lldb/Target/SwiftLanguageRuntime.h"
#include "SwiftLanguageRuntimeImpl.h"

#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
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

#include "swift/AST/ASTMangler.h"
#include "swift/AST/Decl.h"
#include "swift/AST/Module.h"
#include "swift/Reflection/ReflectionContext.h"
#include "swift/RemoteAST/RemoteAST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Memory.h"

// FIXME: we should not need this
#include "Plugins/Language/Swift/SwiftFormatters.h"
#include "Plugins/Language/Swift/SwiftRuntimeFailureRecognizer.h"

using namespace lldb;
using namespace lldb_private;

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

/// \return the Swift or Objective-C runtime found in the loaded images.
static ModuleSP findRuntime(Process &process, RuntimeKind runtime_kind) {
  AppleObjCRuntimeV2 *objc_runtime = nullptr;
  if (runtime_kind == RuntimeKind::ObjC) {
    objc_runtime = SwiftLanguageRuntime::GetObjCRuntime(process);
    if (!objc_runtime)
      return {};
  }

  ModuleList images = process.GetTarget().GetImages();
  for (unsigned i = 0, e = images.GetSize(); i < e; ++i) {
    ModuleSP image = images.GetModuleAtIndex(i);
    if (!image)
      continue;
    if (runtime_kind == RuntimeKind::Swift &&
        IsModuleSwiftRuntime(process, *image))
      return image;
    if (runtime_kind == RuntimeKind::ObjC &&
        objc_runtime->IsModuleObjCLibrary(image))
      return image;
  }
  return {};
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
          obj_file && obj_file->GetPluginName().GetStringRef().equals("mach-o");
      if (!have_objc_interop)
        return {};
    }
    target.GetDebugger().GetAsyncErrorStream()->Printf(
        "Couldn't find the %s runtime library in loaded images.\n",
        (runtime_kind == RuntimeKind::Swift) ? "Swift" : "Objective-C");
    lldbassert(image.get() && "couldn't find runtime library in loaded images");
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
    LLDB_LOGF(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_EXPRESSIONS | \
                                                     LIBLLDB_LOG_TYPES),       \
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

SwiftLanguageRuntimeImpl::NativeReflectionContext *
SwiftLanguageRuntimeImpl::GetReflectionContext() {
  if (!m_initialized_reflection_ctx)
    SetupReflection();
  return m_reflection_ctx.get();
}

void SwiftLanguageRuntimeImpl::SetupReflection() {
  std::lock_guard<std::recursive_mutex> lock(m_add_module_mutex);
  if (m_initialized_reflection_ctx)
    return;

  m_reflection_ctx.reset(new NativeReflectionContext(this->GetMemoryReader()));
  m_initialized_reflection_ctx = true;

  auto &target = m_process.GetTarget();
  auto exe_module = target.GetExecutableModule();
  if (!AddModuleToReflectionContext(exe_module)) {
    m_reflection_ctx.reset();
    return;
  }

  // Add all defered modules to reflection context that were added to
  // the target since this SwiftLanguageRuntime was created.
  m_modules_to_add.ForEach([&](const ModuleSP &module_sp) -> bool {
    AddModuleToReflectionContext(module_sp);
    return true;
  });
  m_modules_to_add.Clear();

  // The global ABI bit is read by the Swift runtime library.
  SetupABIBit();
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
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));
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
    did_load_runtime |= IsModuleSwiftRuntime(*m_process, *module_sp);
    return !did_load_runtime;
  });
  if (did_load_runtime) {
    m_impl = std::make_unique<SwiftLanguageRuntimeImpl>(*m_process);
    m_impl->ModulesDidLoad(module_list);
  }
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
  Address start_address = obj_file->GetBaseAddress();
  auto load_ptr = static_cast<uintptr_t>(
      start_address.GetLoadAddress(&(m_process.GetTarget())));
  if (load_ptr == 0 || load_ptr == LLDB_INVALID_ADDRESS) {
    if (obj_file->GetType() != ObjectFile::eTypeJIT)
      if (Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_TYPES))
        log->Printf("%s: failed to get start address for %s.", __FUNCTION__,
                    obj_file->GetFileSpec().GetFilename().GetCString());
    return false;
  }
  if (HasReflectionInfo(obj_file)) {
    // When dealing with ELF, we need to pass in the contents of the on-disk
    // file, since the Section Header Table is not present in the child process
    if (obj_file->GetPluginName().GetStringRef().equals("elf")) {
      DataExtractor extractor;
      auto size = obj_file->GetData(0, obj_file->GetByteSize(), extractor);
      const uint8_t *file_data = extractor.GetDataStart();
      llvm::sys::MemoryBlock file_buffer((void *)file_data, size);
      m_reflection_ctx->readELF(swift::remote::RemoteAddress(load_ptr),
          llvm::Optional<llvm::sys::MemoryBlock>(file_buffer));
    } else {
      m_reflection_ctx->addImage(swift::remote::RemoteAddress(load_ptr));
    }
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

static bool GetObjectDescription_ResultVariable(Process &process, Stream &str,
                                                ValueObject &object) {
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_DATAFORMATTERS));

  StreamString expr_string;
  expr_string.Printf("Swift._DebuggerSupport.stringForPrintObject(%s)",
                     object.GetName().GetCString());

  if (log)
    log->Printf("[GetObjectDescription_ResultVariable] expression: %s",
                expr_string.GetData());

  ValueObjectSP result_sp;
  EvaluateExpressionOptions eval_options;
  eval_options.SetLanguage(lldb::eLanguageTypeSwift);
  eval_options.SetResultIsInternal(true);
  eval_options.SetGenerateDebugInfo(true);
  eval_options.SetTimeout(process.GetUtilityExpressionTimeout());
  auto eval_result = process.GetTarget().EvaluateExpression(
      expr_string.GetData(),
      process.GetThreadList().GetSelectedThread()->GetSelectedFrame().get(),
      result_sp, eval_options);

  if (log) {
    switch (eval_result) {
    case eExpressionCompleted:
      log->Printf("[GetObjectDescription_ResultVariable] eExpressionCompleted");
      break;
    case eExpressionSetupError:
      log->Printf(
          "[GetObjectDescription_ResultVariable] eExpressionSetupError");
      break;
    case eExpressionParseError:
      log->Printf(
          "[GetObjectDescription_ResultVariable] eExpressionParseError");
      break;
    case eExpressionDiscarded:
      log->Printf("[GetObjectDescription_ResultVariable] eExpressionDiscarded");
      break;
    case eExpressionInterrupted:
      log->Printf(
          "[GetObjectDescription_ResultVariable] eExpressionInterrupted");
      break;
    case eExpressionHitBreakpoint:
      log->Printf(
          "[GetObjectDescription_ResultVariable] eExpressionHitBreakpoint");
      break;
    case eExpressionTimedOut:
      log->Printf("[GetObjectDescription_ResultVariable] eExpressionTimedOut");
      break;
    case eExpressionResultUnavailable:
      log->Printf(
          "[GetObjectDescription_ResultVariable] eExpressionResultUnavailable");
      break;
    case eExpressionStoppedForDebug:
      log->Printf(
          "[GetObjectDescription_ResultVariable] eExpressionStoppedForDebug");
      break;
    case eExpressionThreadVanished:
      log->Printf(
          "[GetObjectDescription_ResultVariable] eExpressionThreadVanished");
      break;
    }
  }

  // sanitize the result of the expression before moving forward
  if (!result_sp) {
    if (log)
      log->Printf("[GetObjectDescription_ResultVariable] expression generated "
                  "no result");
    return false;
  }
  if (result_sp->GetError().Fail()) {
    if (log)
      log->Printf("[GetObjectDescription_ResultVariable] expression generated "
                  "error: %s",
                  result_sp->GetError().AsCString());
    return false;
  }
  if (false == result_sp->GetCompilerType().IsValid()) {
    if (log)
      log->Printf("[GetObjectDescription_ResultVariable] expression generated "
                  "invalid type");
    return false;
  }

  formatters::StringPrinter::ReadStringAndDumpToStreamOptions dump_options;
  dump_options.SetEscapeNonPrintables(false);
  dump_options.SetQuote('\0');
  dump_options.SetPrefixToken(nullptr);
  if (formatters::swift::String_SummaryProvider(
          *result_sp.get(), str,
          TypeSummaryOptions()
              .SetLanguage(lldb::eLanguageTypeSwift)
              .SetCapping(eTypeSummaryUncapped),
          dump_options)) {
    if (log)
      log->Printf("[GetObjectDescription_ResultVariable] expression completed "
                  "successfully");
    return true;
  } else {
    if (log)
      log->Printf("[GetObjectDescription_ResultVariable] expression generated "
                  "invalid string data");
    return false;
  }
}

static bool GetObjectDescription_ObjectReference(Process &process, Stream &str,
                                                 ValueObject &object) {
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_DATAFORMATTERS));

  StreamString expr_string;
  expr_string.Printf("Swift._DebuggerSupport.stringForPrintObject(Swift."
                     "unsafeBitCast(0x%" PRIx64 ", to: AnyObject.self))",
                     object.GetValueAsUnsigned(0));

  if (log)
    log->Printf("[GetObjectDescription_ObjectReference] expression: %s",
                expr_string.GetData());

  ValueObjectSP result_sp;
  EvaluateExpressionOptions eval_options;
  eval_options.SetLanguage(lldb::eLanguageTypeSwift);
  eval_options.SetResultIsInternal(true);
  eval_options.SetGenerateDebugInfo(true);
  eval_options.SetTimeout(process.GetUtilityExpressionTimeout());
  auto eval_result = process.GetTarget().EvaluateExpression(
      expr_string.GetData(),
      process.GetThreadList().GetSelectedThread()->GetSelectedFrame().get(),
      result_sp, eval_options);

  if (log) {
    switch (eval_result) {
    case eExpressionCompleted:
      log->Printf(
          "[GetObjectDescription_ObjectReference] eExpressionCompleted");
      break;
    case eExpressionSetupError:
      log->Printf(
          "[GetObjectDescription_ObjectReference] eExpressionSetupError");
      break;
    case eExpressionParseError:
      log->Printf(
          "[GetObjectDescription_ObjectReference] eExpressionParseError");
      break;
    case eExpressionDiscarded:
      log->Printf(
          "[GetObjectDescription_ObjectReference] eExpressionDiscarded");
      break;
    case eExpressionInterrupted:
      log->Printf(
          "[GetObjectDescription_ObjectReference] eExpressionInterrupted");
      break;
    case eExpressionHitBreakpoint:
      log->Printf(
          "[GetObjectDescription_ObjectReference] eExpressionHitBreakpoint");
      break;
    case eExpressionTimedOut:
      log->Printf("[GetObjectDescription_ObjectReference] eExpressionTimedOut");
      break;
    case eExpressionResultUnavailable:
      log->Printf("[GetObjectDescription_ObjectReference] "
                  "eExpressionResultUnavailable");
      break;
    case eExpressionStoppedForDebug:
      log->Printf(
          "[GetObjectDescription_ObjectReference] eExpressionStoppedForDebug");
      break;
    case eExpressionThreadVanished:
      log->Printf(
          "[GetObjectDescription_ObjectReference] eExpressionThreadVanished");
      break;
    }
  }

  // sanitize the result of the expression before moving forward
  if (!result_sp) {
    if (log)
      log->Printf("[GetObjectDescription_ObjectReference] expression generated "
                  "no result");
    return false;
  }
  if (result_sp->GetError().Fail()) {
    if (log)
      log->Printf("[GetObjectDescription_ObjectReference] expression generated "
                  "error: %s",
                  result_sp->GetError().AsCString());
    return false;
  }
  if (false == result_sp->GetCompilerType().IsValid()) {
    if (log)
      log->Printf("[GetObjectDescription_ObjectReference] expression generated "
                  "invalid type");
    return false;
  }

  formatters::StringPrinter::ReadStringAndDumpToStreamOptions dump_options;
  dump_options.SetEscapeNonPrintables(false);
  dump_options.SetQuote('\0');
  dump_options.SetPrefixToken(nullptr);
  if (formatters::swift::String_SummaryProvider(
          *result_sp.get(), str,
          TypeSummaryOptions()
              .SetLanguage(lldb::eLanguageTypeSwift)
              .SetCapping(eTypeSummaryUncapped),
          dump_options)) {
    if (log)
      log->Printf("[GetObjectDescription_ObjectReference] expression completed "
                  "successfully");
    return true;
  } else {
    if (log)
      log->Printf("[GetObjectDescription_ObjectReference] expression generated "
                  "invalid string data");
    return false;
  }
}

static const ExecutionContextRef *GetSwiftExeCtx(ValueObject &valobj) {
  return (valobj.GetPreferredDisplayLanguage() == eLanguageTypeSwift)
             ? &valobj.GetExecutionContextRef()
             : nullptr;
}

static bool GetObjectDescription_ObjectCopy(SwiftLanguageRuntimeImpl *runtime,
                                            Process &process, Stream &str,
                                            ValueObject &object) {
  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_DATAFORMATTERS));

  ValueObjectSP static_sp(object.GetStaticValue());

  CompilerType static_type(static_sp->GetCompilerType());
  if (auto non_reference_type = static_type.GetNonReferenceType())
    static_type = non_reference_type;

  Status error;

  // If we are in a generic context, here the static type of the object
  // might end up being generic (i.e. <T>). We want to make sure that
  // we correctly map the type into context before asking questions or
  // printing, as IRGen requires a fully realized type to work on.
  auto frame_sp =
      process.GetThreadList().GetSelectedThread()->GetSelectedFrame();
  auto *swift_ast_ctx =
      llvm::dyn_cast_or_null<TypeSystemSwift>(static_type.GetTypeSystem());
  if (swift_ast_ctx) {
    SwiftASTContextLock lock(GetSwiftExeCtx(object));
    static_type = runtime->BindGenericTypeParameters(*frame_sp, static_type);
  }

  auto stride = 0;
  auto opt_stride = static_type.GetByteStride(frame_sp.get());
  if (opt_stride)
    stride = *opt_stride;

  lldb::addr_t copy_location = process.AllocateMemory(
      stride, ePermissionsReadable | ePermissionsWritable, error);
  if (copy_location == LLDB_INVALID_ADDRESS) {
    if (log)
      log->Printf("[GetObjectDescription_ObjectCopy] copy_location invalid");
    return false;
  }
  auto cleanup =
      llvm::make_scope_exit([&]() { process.DeallocateMemory(copy_location); });

  DataExtractor data_extractor;
  if (0 == static_sp->GetData(data_extractor, error)) {
    if (log)
      log->Printf("[GetObjectDescription_ObjectCopy] data extraction failed");
    return false;
  }

  if (0 == process.WriteMemory(copy_location, data_extractor.GetDataStart(),
                               data_extractor.GetByteSize(), error)) {
    if (log)
      log->Printf("[GetObjectDescription_ObjectCopy] memory copy failed");
    return false;
  }

  StreamString expr_string;
  expr_string.Printf("Swift._DebuggerSupport.stringForPrintObject(Swift."
                     "UnsafePointer<%s>(bitPattern: 0x%" PRIx64 ")!.pointee)",
                     static_type.GetTypeName().GetCString(), copy_location);

  if (log)
    log->Printf("[GetObjectDescription_ObjectCopy] expression: %s",
                expr_string.GetData());

  ValueObjectSP result_sp;
  EvaluateExpressionOptions eval_options;
  eval_options.SetLanguage(lldb::eLanguageTypeSwift);
  eval_options.SetResultIsInternal(true);
  eval_options.SetGenerateDebugInfo(true);
  eval_options.SetTimeout(process.GetUtilityExpressionTimeout());
  auto eval_result = process.GetTarget().EvaluateExpression(
      expr_string.GetData(),
      process.GetThreadList().GetSelectedThread()->GetSelectedFrame().get(),
      result_sp, eval_options);

  if (log) {
    switch (eval_result) {
    case eExpressionCompleted:
      log->Printf("[GetObjectDescription_ObjectCopy] eExpressionCompleted");
      break;
    case eExpressionSetupError:
      log->Printf("[GetObjectDescription_ObjectCopy] eExpressionSetupError");
      break;
    case eExpressionParseError:
      log->Printf("[GetObjectDescription_ObjectCopy] eExpressionParseError");
      break;
    case eExpressionDiscarded:
      log->Printf("[GetObjectDescription_ObjectCopy] eExpressionDiscarded");
      break;
    case eExpressionInterrupted:
      log->Printf("[GetObjectDescription_ObjectCopy] eExpressionInterrupted");
      break;
    case eExpressionHitBreakpoint:
      log->Printf("[GetObjectDescription_ObjectCopy] eExpressionHitBreakpoint");
      break;
    case eExpressionTimedOut:
      log->Printf("[GetObjectDescription_ObjectCopy] eExpressionTimedOut");
      break;
    case eExpressionResultUnavailable:
      log->Printf(
          "[GetObjectDescription_ObjectCopy] eExpressionResultUnavailable");
      break;
    case eExpressionStoppedForDebug:
      log->Printf(
          "[GetObjectDescription_ObjectCopy] eExpressionStoppedForDebug");
      break;
    case eExpressionThreadVanished:
      log->Printf(
          "[GetObjectDescription_ObjectCopy] eExpressionThreadVanished");
      break;
    }
  }

  // sanitize the result of the expression before moving forward
  if (!result_sp) {
    if (log)
      log->Printf(
          "[GetObjectDescription_ObjectCopy] expression generated no result");

    str.Printf("expression produced no result");
    return true;
  }
  if (result_sp->GetError().Fail()) {
    if (log)
      log->Printf(
          "[GetObjectDescription_ObjectCopy] expression generated error: %s",
          result_sp->GetError().AsCString());

    str.Printf("expression produced error: %s",
               result_sp->GetError().AsCString());
    return true;
  }
  if (false == result_sp->GetCompilerType().IsValid()) {
    if (log)
      log->Printf("[GetObjectDescription_ObjectCopy] expression generated "
                  "invalid type");

    str.Printf("expression produced invalid result type");
    return true;
  }

  formatters::StringPrinter::ReadStringAndDumpToStreamOptions dump_options;
  dump_options.SetEscapeNonPrintables(false);
  dump_options.SetQuote('\0');
  dump_options.SetPrefixToken(nullptr);
  if (formatters::swift::String_SummaryProvider(
          *result_sp.get(), str,
          TypeSummaryOptions()
              .SetLanguage(lldb::eLanguageTypeSwift)
              .SetCapping(eTypeSummaryUncapped),
          dump_options)) {
    if (log)
      log->Printf("[GetObjectDescription_ObjectCopy] expression completed "
                  "successfully");
  } else {
    if (log)
      log->Printf("[GetObjectDescription_ObjectCopy] expression generated "
                  "invalid string data");

    str.Printf("expression produced unprintable string");
  }
  return true;
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

  if (::IsSwiftResultVariable(object.GetName())) {
    // if this thing is a Swift expression result variable, it has two
    // properties:
    // a) its name is something we can refer to in expressions for free
    // b) its type may be something we can't actually talk about in expressions
    // so, just use the result variable's name in the expression and be done
    // with it
    StreamString probe_stream;
    if (GetObjectDescription_ResultVariable(m_process, probe_stream, object)) {
      str.Printf("%s", probe_stream.GetData());
      return true;
    }
  } else if (::IsSwiftReferenceType(object)) {
    // if this is a Swift class, it has two properties:
    // a) we do not need its type name, AnyObject is just as good
    // b) its value is something we can directly use to refer to it
    // so, just use the ValueObject's pointer-value and be done with it
    StreamString probe_stream;
    if (GetObjectDescription_ObjectReference(m_process, probe_stream, object)) {
      str.Printf("%s", probe_stream.GetData());
      return true;
    }
  }

  // in general, don't try to use the name of the ValueObject as it might end up
  // referring to the wrong thing
  return GetObjectDescription_ObjectCopy(this, m_process, str, object);
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
      llvm::Optional<SwiftASTContextReader> maybe_swift_ast =
          target.GetScratchSwiftASTContext(error, frame);
      if (maybe_swift_ast) {
        SwiftASTContext *swift_ast = maybe_swift_ast->get();
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
                auto clang_ctx = TypeSystemClang::GetScratch(target);
                if (!clang_ctx)
                  continue;

                CompilerType clang_void_ptr_type =
                    clang_ctx->GetBasicType(eBasicTypeVoid).GetPointerType();

                input_value.SetValueType(Value::eValueTypeScalar);
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

  llvm::Optional<SwiftASTContextReader> maybe_ast_context =
      target->GetScratchSwiftASTContext(error, *frame_sp);
  if (!maybe_ast_context || error.Fail())
    return error_valobj_sp;
  SwiftASTContext *ast_context = maybe_ast_context->get();

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

  bool IsScripted() { return false; }

  std::string GetDescription() { return "projection synthetic children"; }

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
  SyntheticChildrenFrontEnd::AutoPointer GetFrontEnd(ValueObject &backend) {
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

  if (auto maybe_swift_ast_ctx = valobj.GetScratchSwiftASTContext()) {
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
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));
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
  Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_EXPRESSIONS));

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
    val.SetValueType(Value::eValueTypeScalar);
    val.SetContext(Value::eContextTypeRegisterInfo,
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

static const char *
SwiftDemangleNodeKindToCString(const swift::Demangle::Node::Kind node_kind) {
#define NODE(e)                                                                \
  case swift::Demangle::Node::Kind::e:                                         \
    return #e;

  switch (node_kind) {
#include "swift/Demangling/DemangleNodes.def"
  }
  return "swift::Demangle::Node::Kind::???";
#undef NODE
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

  virtual Options *GetOptions() { return &m_options; }

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
  void PrintNode(swift::Demangle::NodePointer node_ptr, Stream &stream,
                 int depth = 0) {
    if (!node_ptr)
      return;

    std::string indent(2 * depth, ' ');

    stream.Printf("%s", indent.c_str());

    stream.Printf("kind=%s",
                  SwiftDemangleNodeKindToCString(node_ptr->getKind()));
    if (node_ptr->hasText()) {
      std::string Text = node_ptr->getText().str();
      stream.Printf(", text=\"%s\"", Text.c_str());
    }
    if (node_ptr->hasIndex())
      stream.Printf(", index=%" PRIu64, node_ptr->getIndex());

    stream.Printf("\n");

    for (auto &&child : *node_ptr) {
      PrintNode(child, stream, depth + 1);
    }
  }

  bool DoExecute(Args &command, CommandReturnObject &result) {
    for (size_t i = 0; i < command.GetArgumentCount(); i++) {
      const char *arg = command.GetArgumentAtIndex(i);
      if (arg && *arg) {
        swift::Demangle::Context demangle_ctx;
        auto node_ptr = demangle_ctx.demangleSymbolAsNode(llvm::StringRef(arg));
        if (node_ptr) {
          if (m_options.m_expand) {
            PrintNode(node_ptr, result.GetOutputStream());
          }
          result.GetOutputStream().Printf(
              "%s ---> %s\n", arg,
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

  virtual Options *GetOptions() { return nullptr; }

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
  bool DoExecute(llvm::StringRef command, CommandReturnObject &result) {
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

lldb_private::ConstString SwiftLanguageRuntime::GetPluginNameStatic() {
  static ConstString g_name("swift");
  return g_name;
}

lldb_private::ConstString SwiftLanguageRuntime::GetPluginName() {
  return GetPluginNameStatic();
}

uint32_t SwiftLanguageRuntime::GetPluginVersion() { return 1; }

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

} // namespace lldb_private
