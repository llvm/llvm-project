//===-- SwiftLanguageRuntimeDynamicTypeResolution.cpp ---------------------===//
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

#include "LLDBMemoryReader.h"
#include "SwiftLanguageRuntimeImpl.h"
#include "SwiftLanguageRuntime.h"

#include "Plugins/ExpressionParser/Clang/ClangUtil.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/ProcessStructReader.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Timer.h"
#include "swift/AST/Types.h"

#include "swift/AST/ASTContext.h"
#include "swift/AST/ASTMangler.h"
#include "swift/AST/ASTWalker.h"
#include "swift/AST/Decl.h"
#include "swift/Demangling/Demangle.h"
#include "swift/Reflection/ReflectionContext.h"
#include "swift/Reflection/TypeRefBuilder.h"
#include "swift/Remote/MemoryReader.h"
#include "swift/RemoteAST/RemoteAST.h"

#include <sstream>

using namespace lldb;
using namespace lldb_private;

namespace lldb_private {
swift::Type GetSwiftType(CompilerType type) {
  auto *ts = type.GetTypeSystem();
  if (auto *tr = llvm::dyn_cast_or_null<TypeSystemSwiftTypeRef>(ts))
    return tr->GetSwiftType(type);
  if (auto *ast = llvm::dyn_cast_or_null<SwiftASTContext>(ts))
    return ast->GetSwiftType(type);
  return {};
}

swift::CanType GetCanonicalSwiftType(CompilerType type) {
  swift::Type swift_type = nullptr;
  auto *ts = type.GetTypeSystem();
  if (auto *tr = llvm::dyn_cast_or_null<TypeSystemSwiftTypeRef>(ts))
    swift_type = tr->GetSwiftType(type);
  if (auto *ast = llvm::dyn_cast_or_null<SwiftASTContext>(ts))
    swift_type = ast->GetSwiftType(type);
  return swift_type ? swift_type->getCanonicalType() : swift::CanType();
}

static lldb::addr_t
MaskMaybeBridgedPointer(Process &process, lldb::addr_t addr,
                        lldb::addr_t *masked_bits = nullptr) {
  const ArchSpec &arch_spec(process.GetTarget().GetArchitecture());
  const llvm::Triple &triple = arch_spec.GetTriple();
  bool is_arm = false;
  bool is_intel = false;
  bool is_s390x = false;
  bool is_32 = false;
  bool is_64 = false;
  if (triple.isAArch64() || triple.isARM())
    is_arm = true;
  else if (triple.isX86())
    is_intel = true;
  else if (triple.isSystemZ())
    is_s390x = true;
  else // this is a really random CPU core to be running on - just get out fast
    return addr;

  switch (arch_spec.GetAddressByteSize()) {
  case 4:
    is_32 = true;
    break;
  case 8:
    is_64 = true;
    break;
  default:
    // this is a really random pointer size to be running on - just get out fast
    return addr;
  }

  lldb::addr_t mask = 0;

  if (is_arm && is_64)
    mask = SWIFT_ABI_ARM64_SWIFT_SPARE_BITS_MASK;
  else if (is_arm && is_32)
    mask = SWIFT_ABI_ARM_SWIFT_SPARE_BITS_MASK;
  else if (is_intel && is_64)
    mask = SWIFT_ABI_X86_64_SWIFT_SPARE_BITS_MASK;
  else if (is_intel && is_32)
    mask = SWIFT_ABI_I386_SWIFT_SPARE_BITS_MASK;
  else if (is_s390x && is_64)
    mask = SWIFT_ABI_S390X_SWIFT_SPARE_BITS_MASK;

  if (masked_bits)
    *masked_bits = addr & mask;
  return addr & ~mask;
}

lldb::addr_t
SwiftLanguageRuntime::MaskMaybeBridgedPointer(lldb::addr_t addr,
                                              lldb::addr_t *masked_bits) {
  return m_process ? ::MaskMaybeBridgedPointer(*m_process, addr, masked_bits)
                   : addr;
}

lldb::addr_t SwiftLanguageRuntime::MaybeMaskNonTrivialReferencePointer(
    lldb::addr_t addr,
    SwiftASTContext::NonTriviallyManagedReferenceStrategy strategy) {

  if (addr == 0)
    return addr;

  AppleObjCRuntime *objc_runtime = GetObjCRuntime();

  if (objc_runtime) {
    // tagged pointers don't perform any masking
    if (objc_runtime->IsTaggedPointer(addr))
      return addr;
  }

  if (!m_process)
    return addr;
  const ArchSpec &arch_spec(m_process->GetTarget().GetArchitecture());
  const llvm::Triple &triple = arch_spec.GetTriple();
  bool is_arm = false;
  bool is_intel = false;
  bool is_64 = false;
  if (triple.isAArch64() || triple.isARM())
    is_arm = true;
  else if (triple.isX86())
    is_intel = true;
  else // this is a really random CPU core to be running on - just get out fast
    return addr;

  switch (arch_spec.GetAddressByteSize()) {
  case 4:
    break;
  case 8:
    is_64 = true;
    break;
  default:
    // this is a really random pointer size to be running on - just get out fast
    return addr;
  }

  lldb::addr_t mask = 0;

  if (strategy ==
      SwiftASTContext::NonTriviallyManagedReferenceStrategy::eWeak) {
    bool is_indirect = true;

    // On non-objc platforms, the weak reference pointer always pointed to a
    // runtime structure.
    // For ObjC platforms, the masked value determines whether it is indirect.

    uint32_t value = 0;

    if (objc_runtime) {

      if (is_intel) {
        if (is_64) {
          mask = SWIFT_ABI_X86_64_OBJC_WEAK_REFERENCE_MARKER_MASK;
          value = SWIFT_ABI_X86_64_OBJC_WEAK_REFERENCE_MARKER_VALUE;
        } else {
          mask = SWIFT_ABI_I386_OBJC_WEAK_REFERENCE_MARKER_MASK;
          value = SWIFT_ABI_I386_OBJC_WEAK_REFERENCE_MARKER_VALUE;
        }
      } else if (is_arm) {
        if (is_64) {
          mask = SWIFT_ABI_ARM64_OBJC_WEAK_REFERENCE_MARKER_MASK;
          value = SWIFT_ABI_ARM64_OBJC_WEAK_REFERENCE_MARKER_VALUE;
        } else {
          mask = SWIFT_ABI_ARM_OBJC_WEAK_REFERENCE_MARKER_MASK;
          value = SWIFT_ABI_ARM_OBJC_WEAK_REFERENCE_MARKER_VALUE;
        }
      }
    } else {
      // This name is a little confusing. The "DEFAULT" marking in System.h
      // is supposed to mean: the value for non-ObjC platforms.  So
      // DEFAULT_OBJC here actually means "non-ObjC".
      mask = SWIFT_ABI_DEFAULT_OBJC_WEAK_REFERENCE_MARKER_MASK;
      value = SWIFT_ABI_DEFAULT_OBJC_WEAK_REFERENCE_MARKER_VALUE;
    }

    is_indirect = ((addr & mask) == value);

    if (!is_indirect)
      return addr;

    // The masked value of address is a pointer to the runtime structure.
    // The first field of the structure is the actual pointer.
    Process *process = GetProcess();
    Status error;

    lldb::addr_t masked_addr = addr & ~mask;
    lldb::addr_t isa_addr = process->ReadPointerFromMemory(masked_addr, error);
    if (error.Fail()) {
      LLDB_LOG(GetLog(LLDBLog::Expressions | LLDBLog::Types),
               "Couldn't deref masked pointer");
      return addr;
    }
    return isa_addr;

  }

  if (is_arm && is_64)
    mask = SWIFT_ABI_ARM64_OBJC_NUM_RESERVED_LOW_BITS;
  else if (is_intel && is_64)
    mask = SWIFT_ABI_X86_64_OBJC_NUM_RESERVED_LOW_BITS;
  else
    mask = SWIFT_ABI_DEFAULT_OBJC_NUM_RESERVED_LOW_BITS;

  mask = (1 << mask) | (1 << (mask + 1));

  return addr & ~mask;
}

namespace {

/// An implementation of the generic ReflectionContextInterface that
/// is templatized on target pointer width and specialized to either
/// 32-bit or 64-bit pointers, with and without ObjC interoperability.
template <typename ReflectionContext>
class TargetReflectionContext
    : public SwiftLanguageRuntimeImpl::ReflectionContextInterface {
  ReflectionContext m_reflection_ctx;

public:
  TargetReflectionContext(
      std::shared_ptr<swift::reflection::MemoryReader> reader)
      : m_reflection_ctx(reader) {}

  bool addImage(
      llvm::function_ref<std::pair<swift::remote::RemoteRef<void>, uint64_t>(
          swift::ReflectionSectionKind)>
          find_section,
      llvm::SmallVector<llvm::StringRef, 1> likely_module_names) override {
    return m_reflection_ctx.addImage(find_section, likely_module_names);
  }

  bool
  addImage(swift::remote::RemoteAddress image_start,
           llvm::SmallVector<llvm::StringRef, 1> likely_module_names) override {
    return m_reflection_ctx.addImage(image_start, likely_module_names);
  }

  bool readELF(
      swift::remote::RemoteAddress ImageStart,
      llvm::Optional<llvm::sys::MemoryBlock> FileBuffer,
      llvm::SmallVector<llvm::StringRef, 1> likely_module_names = {}) override {
    return m_reflection_ctx.readELF(ImageStart, FileBuffer,
                                    likely_module_names);
  }

  const swift::reflection::TypeInfo *
  getTypeInfo(const swift::reflection::TypeRef *type_ref,
              swift::remote::TypeInfoProvider *provider) override {
    return m_reflection_ctx.getTypeInfo(type_ref, provider);
  }

  swift::reflection::MemoryReader &getReader() override {
    return m_reflection_ctx.getReader();
  }

  bool ForEachSuperClassType(
      LLDBTypeInfoProvider *tip, lldb::addr_t pointer,
      std::function<bool(SwiftLanguageRuntimeImpl::SuperClassType)> fn)
      override {
    // Guard against faulty self-referential metadata.
    unsigned limit = 256;
    auto md_ptr = m_reflection_ctx.readMetadataFromInstance(pointer);
    if (!md_ptr)
      return false;

    // Class object.
    while (md_ptr && *md_ptr && --limit) {
      // Reading metadata is potentially expensive since (in a remote
      // debugging scenario it may even incur network traffic) so we
      // just return closures that the caller can use to query details
      // if they need them.'
      auto metadata = *md_ptr;
      if (fn({[=]() -> const swift::reflection::RecordTypeInfo * {
                auto *ti = m_reflection_ctx.getMetadataTypeInfo(metadata, tip);
                return llvm::dyn_cast_or_null<
                    swift::reflection::RecordTypeInfo>(ti);
              },
              [=]() -> const swift::reflection::TypeRef * {
                return m_reflection_ctx.readTypeFromMetadata(metadata);
              }}))
        return true;

      // Continue with the base class.
      md_ptr = m_reflection_ctx.readSuperClassFromClassMetadata(metadata);
    }
    return false;
  }

  llvm::Optional<std::pair<const swift::reflection::TypeRef *,
                           swift::reflection::RemoteAddress>>
  projectExistentialAndUnwrapClass(
      swift::reflection::RemoteAddress existential_address,
      const swift::reflection::TypeRef &existential_tr) override {
    return m_reflection_ctx.projectExistentialAndUnwrapClass(
        existential_address, existential_tr);
  }

  const swift::reflection::TypeRef *
  readTypeFromMetadata(lldb::addr_t metadata_address,
                       bool skip_artificial_subclasses) override {
    return m_reflection_ctx.readTypeFromMetadata(metadata_address,
                                                 skip_artificial_subclasses);
  }

  const swift::reflection::TypeRef *
  readTypeFromInstance(lldb::addr_t instance_address,
                       bool skip_artificial_subclasses) override {
    auto metadata_address =
        m_reflection_ctx.readMetadataFromInstance(instance_address);
    if (!metadata_address) {
      LLDB_LOGF(GetLog(LLDBLog::Types),
                "could not read heap metadata for object at %llu\n",
                instance_address);
      return nullptr;
    }

    return m_reflection_ctx.readTypeFromMetadata(*metadata_address,
                                                 skip_artificial_subclasses);
  }

  swift::reflection::TypeRefBuilder &getBuilder() override {
    return m_reflection_ctx.getBuilder();
  }

  llvm::Optional<bool> isValueInlinedInExistentialContainer(
      swift::remote::RemoteAddress existential_address) override {
    return m_reflection_ctx.isValueInlinedInExistentialContainer(
        existential_address);
  }

  swift::remote::RemoteAbsolutePointer
  stripSignedPointer(swift::remote::RemoteAbsolutePointer pointer) override {
    return m_reflection_ctx.stripSignedPointer(pointer);
  }
};

} // namespace

std::unique_ptr<SwiftLanguageRuntimeImpl::ReflectionContextInterface>
SwiftLanguageRuntimeImpl::ReflectionContextInterface::CreateReflectionContext32(
    std::shared_ptr<swift::remote::MemoryReader> reader, bool ObjCInterop) {
  using ReflectionContext32ObjCInterop =
      TargetReflectionContext<swift::reflection::ReflectionContext<
          swift::External<swift::WithObjCInterop<swift::RuntimeTarget<4>>>>>;
  using ReflectionContext32NoObjCInterop =
      TargetReflectionContext<swift::reflection::ReflectionContext<
          swift::External<swift::NoObjCInterop<swift::RuntimeTarget<4>>>>>;
  if (ObjCInterop)    
    return std::make_unique<ReflectionContext32ObjCInterop>(reader);
  return std::make_unique<ReflectionContext32NoObjCInterop>(reader);
}

std::unique_ptr<SwiftLanguageRuntimeImpl::ReflectionContextInterface>
SwiftLanguageRuntimeImpl::ReflectionContextInterface::CreateReflectionContext64(
    std::shared_ptr<swift::remote::MemoryReader> reader, bool ObjCInterop) {
  using ReflectionContext64ObjCInterop =
      TargetReflectionContext<swift::reflection::ReflectionContext<
          swift::External<swift::WithObjCInterop<swift::RuntimeTarget<8>>>>>;
  using ReflectionContext64NoObjCInterop =
      TargetReflectionContext<swift::reflection::ReflectionContext<
          swift::External<swift::NoObjCInterop<swift::RuntimeTarget<8>>>>>;
  if (ObjCInterop)
    return std::make_unique<ReflectionContext64ObjCInterop>(reader);
  return std::make_unique<ReflectionContext64NoObjCInterop>(reader);
}

SwiftLanguageRuntimeImpl::ReflectionContextInterface::
    ~ReflectionContextInterface() {}

const CompilerType &SwiftLanguageRuntimeImpl::GetBoxMetadataType() {
  if (m_box_metadata_type.IsValid())
    return m_box_metadata_type;

  static ConstString g_type_name("__lldb_autogen_boxmetadata");
  const bool is_packed = false;
  if (TypeSystemClang *ast_ctx =
          ScratchTypeSystemClang::GetForTarget(m_process.GetTarget())) {
    CompilerType voidstar =
        ast_ctx->GetBasicType(lldb::eBasicTypeVoid).GetPointerType();
    CompilerType uint32 = ast_ctx->GetIntTypeFromBitSize(32, false);

    m_box_metadata_type = ast_ctx->GetOrCreateStructForIdentifier(
        g_type_name, {{"kind", voidstar}, {"offset", uint32}}, is_packed);
  }

  return m_box_metadata_type;
}

std::shared_ptr<LLDBMemoryReader>
SwiftLanguageRuntimeImpl::GetMemoryReader() {
  if (!m_memory_reader_sp) {
    m_memory_reader_sp.reset(new LLDBMemoryReader(
        m_process, [&](swift::remote::RemoteAbsolutePointer pointer) {
          auto *reflection_context = GetReflectionContext();
          return reflection_context->stripSignedPointer(pointer);
        }));
  }

  return m_memory_reader_sp;
}

void SwiftLanguageRuntimeImpl::PushLocalBuffer(uint64_t local_buffer,
                                               uint64_t local_buffer_size) {
  ((LLDBMemoryReader *)GetMemoryReader().get())
      ->pushLocalBuffer(local_buffer, local_buffer_size);
}

void SwiftLanguageRuntimeImpl::PopLocalBuffer() {
  ((LLDBMemoryReader *)GetMemoryReader().get())->popLocalBuffer();
}

SwiftLanguageRuntime::MetadataPromise::MetadataPromise(
    ValueObject &for_object, SwiftLanguageRuntimeImpl &runtime,
    lldb::addr_t location)
    : m_for_object_sp(for_object.GetSP()), m_swift_runtime(runtime),
      m_metadata_location(location) {}

CompilerType
SwiftLanguageRuntime::MetadataPromise::FulfillTypePromise(Status *error) {
  if (error)
    error->Clear();

  Log *log(GetLog(LLDBLog::Types));

  if (log)
    log->Printf("[MetadataPromise] asked to fulfill type promise at location "
                "0x%" PRIx64,
                m_metadata_location);

  if (m_compiler_type.hasValue())
    return m_compiler_type.getValue();

  llvm::Optional<SwiftScratchContextReader> maybe_swift_scratch_ctx =
      m_for_object_sp->GetSwiftScratchContext();
  if (!maybe_swift_scratch_ctx) {
    error->SetErrorString("couldn't get Swift scratch context");
    return CompilerType();
  }
  auto scratch_ctx = maybe_swift_scratch_ctx->get();
  if (!scratch_ctx) {
    error->SetErrorString("couldn't get Swift scratch context");
    return CompilerType();
  }
  SwiftASTContext *swift_ast_ctx = scratch_ctx->GetSwiftASTContext();
  if (!swift_ast_ctx) {
    error->SetErrorString("couldn't get Swift scratch context");
    return CompilerType();
  }
  auto &remote_ast = m_swift_runtime.GetRemoteASTContext(*swift_ast_ctx);
  swift::remoteAST::Result<swift::Type> result =
      remote_ast.getTypeForRemoteTypeMetadata(
          swift::remote::RemoteAddress(m_metadata_location));

  if (result) {
    m_compiler_type = {swift_ast_ctx, result.getValue().getPointer()};
    if (log)
      log->Printf("[MetadataPromise] result is type %s",
                  m_compiler_type->GetTypeName().AsCString());
    return m_compiler_type.getValue();
  } else {
    const auto &failure = result.getFailure();
    if (error)
      error->SetErrorStringWithFormat("error in resolving type: %s",
                                      failure.render().c_str());
    if (log)
      log->Printf("[MetadataPromise] failure: %s", failure.render().c_str());
    return (m_compiler_type = CompilerType()).getValue();
  }
}

SwiftLanguageRuntime::MetadataPromiseSP
SwiftLanguageRuntimeImpl::GetMetadataPromise(lldb::addr_t addr,
                                             ValueObject &for_object) {
  llvm::Optional<SwiftScratchContextReader> maybe_swift_scratch_ctx =
      for_object.GetSwiftScratchContext();
  if (!maybe_swift_scratch_ctx)
    return nullptr;
  auto scratch_ctx = maybe_swift_scratch_ctx->get();
  if (!scratch_ctx)
    return nullptr;
  SwiftASTContext *swift_ast_ctx = scratch_ctx->GetSwiftASTContext();
  if (swift_ast_ctx->HasFatalErrors())
    return nullptr;
  if (addr == 0 || addr == LLDB_INVALID_ADDRESS)
    return nullptr;

  auto key = std::make_pair(swift_ast_ctx->GetASTContext(), addr);
  auto iter = m_promises_map.find(key);
  if (iter != m_promises_map.end())
    return iter->second;

  SwiftLanguageRuntime::MetadataPromiseSP promise_sp(
      new SwiftLanguageRuntime::MetadataPromise(for_object, *this, addr));
  m_promises_map.insert({key, promise_sp});
  return promise_sp;
}

swift::remoteAST::RemoteASTContext &
SwiftLanguageRuntimeImpl::GetRemoteASTContext(SwiftASTContext &swift_ast_ctx) {
  // If we already have a remote AST context for this AST context,
  // return it.
  auto known = m_remote_ast_contexts.find(swift_ast_ctx.GetASTContext());
  if (known != m_remote_ast_contexts.end())
    return *known->second;

  // Initialize a new remote AST context.
  (void)GetReflectionContext();
  auto remote_ast_up = std::make_unique<swift::remoteAST::RemoteASTContext>(
      *swift_ast_ctx.GetASTContext(), GetMemoryReader());
  auto &remote_ast = *remote_ast_up;
  m_remote_ast_contexts.insert(
      {swift_ast_ctx.GetASTContext(), std::move(remote_ast_up)});
  return remote_ast;
}

void SwiftLanguageRuntimeImpl::ReleaseAssociatedRemoteASTContext(
    swift::ASTContext *ctx) {
  m_remote_ast_contexts.erase(ctx);
}

namespace {
class ASTVerifier : public swift::ASTWalker {
  bool hasMissingPatterns = false;

  bool walkToDeclPre(swift::Decl *D) override {
    if (auto *PBD = llvm::dyn_cast<swift::PatternBindingDecl>(D)) {
      if (PBD->getPatternList().empty()) {
        hasMissingPatterns = true;
        return false;
      }
    }
    return true;
  }

public:
  /// Detect (one form of) incomplete types. These may appear if
  /// member variables have Clang-imported types that couldn't be
  /// resolved.
  static bool Verify(swift::Decl *D) {
    if (!D)
      return false;

    ASTVerifier verifier;
    D->walk(verifier);
    return !verifier.hasMissingPatterns;
  }
};

} // namespace

class LLDBTypeInfoProvider : public swift::remote::TypeInfoProvider {
  SwiftLanguageRuntimeImpl &m_runtime;
  TypeSystemSwift &m_typesystem;

public:
  LLDBTypeInfoProvider(SwiftLanguageRuntimeImpl &runtime,
                       TypeSystemSwift &typesystem)
      : m_runtime(runtime), m_typesystem(typesystem) {}

  const swift::reflection::TypeInfo *
  getTypeInfo(llvm::StringRef mangledName) override {
    // TODO: Should we cache the mangled name -> compiler type lookup, too?
    Log *log(GetLog(LLDBLog::Types));
    if (log)
      log->Printf("[LLDBTypeInfoProvider] Looking up debug type info for %s",
                  mangledName.str().c_str());

    // Materialize a Clang type from the debug info.
    assert(swift::Demangle::getManglingPrefixLength(mangledName) == 0);
    std::string wrapped;
    // The mangled name passed in is bare. Add global prefix ($s) and type (D).
    llvm::raw_string_ostream(wrapped) << "$s" << mangledName << 'D';
#ifndef NDEBUG
    {
      // Check that our hardcoded mangling wrapper is still up-to-date.
      swift::Demangle::Context dem;
      auto node = dem.demangleSymbolAsNode(wrapped);
      assert(node && node->getKind() == swift::Demangle::Node::Kind::Global);
      assert(node->getNumChildren() == 1);
      node = node->getChild(0);
      assert(node->getKind() == swift::Demangle::Node::Kind::TypeMangling);
      assert(node->getNumChildren() == 1);
      node = node->getChild(0);
      assert(node->getKind() == swift::Demangle::Node::Kind::Type);
      assert(node->getNumChildren() == 1);
      node = node->getChild(0);
      assert(node->getKind() != swift::Demangle::Node::Kind::Type);
    }
#endif
    ConstString mangled(wrapped);
    CompilerType swift_type = m_typesystem.GetTypeFromMangledTypename(mangled);
    auto *ts = llvm::dyn_cast_or_null<TypeSystemSwift>(swift_type.GetTypeSystem());
    if (!ts)
      return nullptr;
    CompilerType clang_type;
    bool is_imported =
        ts->IsImportedType(swift_type.GetOpaqueQualType(), &clang_type);
    if (!is_imported || !clang_type) {
      if (log)
        log->Printf("[LLDBTypeInfoProvider] Could not find clang debug type info for %s",
                    mangledName.str().c_str());
      return nullptr;
    }
    
    return GetOrCreateTypeInfo(clang_type);
  }
  
  const swift::reflection::TypeInfo *
  GetOrCreateTypeInfo(CompilerType clang_type) {
    if (auto ti = m_runtime.lookupClangTypeInfo(clang_type))
      return *ti;

    // Build a TypeInfo for the Clang type.
    auto size = clang_type.GetByteSize(nullptr);
    auto bit_align = clang_type.GetTypeBitAlign(nullptr);
    std::vector<swift::reflection::FieldInfo> fields;
    if (clang_type.IsAggregateType()) {
      // Recursively collect TypeInfo records for all fields.
      for (uint32_t i = 0, e = clang_type.GetNumFields(); i != e; ++i) {
        std::string name;
        uint64_t bit_offset_ptr = 0;
        uint32_t bitfield_bit_size_ptr = 0;
        bool is_bitfield_ptr = false;
        CompilerType field_type = clang_type.GetFieldAtIndex(
            i, name, &bit_offset_ptr, &bitfield_bit_size_ptr, &is_bitfield_ptr);
        if (is_bitfield_ptr) {
          Log *log(GetLog(LLDBLog::Types));
          if (log)
            log->Printf("[LLDBTypeInfoProvider] bitfield support is not yet "
                        "implemented");
          continue;
        }
        swift::reflection::FieldInfo field_info = {
            name, (unsigned)bit_offset_ptr / 8, 0, nullptr,
            *GetOrCreateTypeInfo(field_type)};
        fields.push_back(field_info);
      }
    }
    return m_runtime.emplaceClangTypeInfo(clang_type, size, bit_align, fields);
  }
};

llvm::Optional<const swift::reflection::TypeInfo *>
SwiftLanguageRuntimeImpl::lookupClangTypeInfo(CompilerType clang_type) {
  std::lock_guard<std::recursive_mutex> locker(m_clang_type_info_mutex);
  {
    auto it = m_clang_type_info.find(clang_type.GetOpaqueQualType());
    if (it != m_clang_type_info.end()) {
      if (it->second)
        return &*it->second;
      return nullptr;
    }
  }
  {
    auto it = m_clang_record_type_info.find(clang_type.GetOpaqueQualType());
    if (it != m_clang_record_type_info.end()) {
      if (it->second)
        return &*it->second;
      return nullptr;
    }
  }
  return {};
}

const swift::reflection::TypeInfo *
SwiftLanguageRuntimeImpl::emplaceClangTypeInfo(
    CompilerType clang_type, llvm::Optional<uint64_t> byte_size,
    llvm::Optional<size_t> bit_align,
    llvm::ArrayRef<swift::reflection::FieldInfo> fields) {
  std::lock_guard<std::recursive_mutex> locker(m_clang_type_info_mutex);
  if (!byte_size || !bit_align) {
    m_clang_type_info.insert({clang_type.GetOpaqueQualType(), llvm::None});
    return nullptr;
  }
  assert(*bit_align % 8 == 0 && "Bit alignment no a multiple of 8!");
  auto byte_align = *bit_align / 8;
  // The stride is the size rounded up to alignment.
  size_t byte_stride = llvm::alignTo(*byte_size, byte_align);
  if (fields.empty()) {
    auto it_b = m_clang_type_info.insert(
        {clang_type.GetOpaqueQualType(),
         swift::reflection::TypeInfo(swift::reflection::TypeInfoKind::Builtin,
                                     *byte_size, byte_align, byte_stride, 0,
                                     true)});
    return &*it_b.first->second;
  }
  auto it_b = m_clang_record_type_info.insert(
      {clang_type.GetOpaqueQualType(),
       swift::reflection::RecordTypeInfo(
           *byte_size, byte_align, byte_stride, 0, false,
           swift::reflection::RecordKind::Struct, fields)});
  return &*it_b.first->second;
}

llvm::Optional<uint64_t>
SwiftLanguageRuntimeImpl::GetMemberVariableOffsetRemoteAST(
    CompilerType instance_type, ValueObject *instance,
    llvm::StringRef member_name) {
  auto *scratch_ctx =
      llvm::cast<SwiftASTContext>(instance_type.GetTypeSystem());
  if (scratch_ctx->HasFatalErrors())
    return {};

  auto *remote_ast = &GetRemoteASTContext(*scratch_ctx);
  // Check whether we've already cached this offset.
  swift::TypeBase *swift_type =
      GetCanonicalSwiftType(instance_type).getPointer();

  // Perform the cache lookup.
  MemberID key{swift_type, ConstString(member_name).GetCString()};
  auto it = m_member_offsets.find(key);
  if (it != m_member_offsets.end())
    return it->second;

  // Dig out metadata describing the type, if it's easy to find.
  // FIXME: the Remote AST library should make this easier.
  swift::remote::RemoteAddress optmeta(nullptr);
  const swift::TypeKind type_kind = swift_type->getKind();
  switch (type_kind) {
  case swift::TypeKind::Class:
  case swift::TypeKind::BoundGenericClass: {
    LLDB_LOGF(GetLog(LLDBLog::Types),
              "[MemberVariableOffsetResolver] type is a class - trying to "
              "get metadata for valueobject %s",
              (instance ? instance->GetName().AsCString() : "<null>"));
    if (instance) {
      lldb::addr_t pointer = instance->GetPointerValue();
      if (!pointer || pointer == LLDB_INVALID_ADDRESS)
        break;
      swift::remote::RemoteAddress address(pointer);
      if (auto metadata = remote_ast->getHeapMetadataForObject(address))
        optmeta = metadata.getValue();
    }
    LLDB_LOGF(GetLog(LLDBLog::Types),
              "[MemberVariableOffsetResolver] optmeta = 0x%" PRIx64,
              optmeta.getAddressData());
    break;
  }

  default:
    // Bind generic parameters if necessary.
    if (instance && swift_type->hasTypeParameter())
      if (auto *frame = instance->GetExecutionContextRef().GetFrameSP().get())
        if (auto bound = BindGenericTypeParameters(*frame, instance_type)) {
          LLDB_LOGF(
              GetLog(LLDBLog::Types),
              "[MemberVariableOffsetResolver] resolved non-class type = %s",
              bound.GetTypeName().AsCString());

          swift_type = GetCanonicalSwiftType(bound).getPointer();
          MemberID key{swift_type, ConstString(member_name).GetCString()};
          auto it = m_member_offsets.find(key);
          if (it != m_member_offsets.end())
            return it->second;
        }
  }

  // Try to determine whether it is safe to use RemoteAST.  RemoteAST
  // is faster than RemoteMirrors, but can't do dynamic types (checked
  // inside RemoteAST) or incomplete types (checked here).
  bool safe_to_use_remote_ast = true;
  if (swift::Decl *type_decl = swift_type->getNominalOrBoundGenericNominal())
    safe_to_use_remote_ast &= ASTVerifier::Verify(type_decl);

  // Use RemoteAST to determine the member offset.
  if (safe_to_use_remote_ast) {
    swift::remoteAST::Result<uint64_t> result =
        remote_ast->getOffsetOfMember(swift_type, optmeta, member_name);
    if (result) {
      LLDB_LOGF(GetLog(LLDBLog::Types),
                "[MemberVariableOffsetResolver] offset discovered = %" PRIu64,
                (uint64_t)result.getValue());

      // Cache this result.
      MemberID key{swift_type, ConstString(member_name).GetCString()};
      m_member_offsets.insert({key, result.getValue()});
      return result.getValue();
    }

    const auto &failure = result.getFailure();
    LLDB_LOGF(GetLog(LLDBLog::Types),
              "[MemberVariableOffsetResolver] failure: %s",
              failure.render().c_str());
  }
  return {};
}

llvm::Optional<uint64_t> SwiftLanguageRuntimeImpl::GetMemberVariableOffsetRemoteMirrors(
    CompilerType instance_type, ValueObject *instance, llvm::StringRef member_name,
    Status *error) {
  LLDB_LOGF(GetLog(LLDBLog::Types), "using remote mirrors");
  auto *ts = llvm::dyn_cast_or_null<TypeSystemSwiftTypeRef>(
      instance_type.GetTypeSystem());
  if (!ts) {
    if (error)
      error->SetErrorString("not a Swift type");
    return {};
  }

  // Try the static type metadata.
  auto frame = instance ? instance->GetExecutionContextRef().GetFrameSP().get()
                        : nullptr;
  if (auto *ti = llvm::dyn_cast_or_null<swift::reflection::RecordTypeInfo>(
          GetSwiftRuntimeTypeInfo(instance_type, frame))) {
    auto fields = ti->getFields();

    // Handle tuples.
    if (ti->getRecordKind() == swift::reflection::RecordKind::Tuple) {
      unsigned tuple_idx;
      if (member_name.getAsInteger(10, tuple_idx) ||
          tuple_idx >= ti->getNumFields()) {
        if (error)
          error->SetErrorString("tuple index out of bounds");
        return {};
      }
      return fields[tuple_idx].Offset;
    }

    // Handle other record types.
   for (auto &field : fields)
      if (StringRef(field.Name) == member_name)
        return field.Offset;
  }

  // Try the instance type metadata.
  llvm::Optional<uint64_t> result;
  if (!instance)
    return result;
  ForEachSuperClassType(*instance, [&](SuperClassType super_class) -> bool {
    auto *ti = super_class.get_record_type_info();
    if (!ti)
      return false;

    for (auto &field : ti->getFields())
      if (StringRef(field.Name) == member_name) {
        result = field.Offset;
        return true;
      }
    return false;
  });
  return result;
}

llvm::Optional<uint64_t> SwiftLanguageRuntimeImpl::GetMemberVariableOffset(
    CompilerType instance_type, ValueObject *instance,
    llvm::StringRef member_name, Status *error) {
  LLDB_SCOPED_TIMER();
  llvm::Optional<uint64_t> offset;

  if (!instance_type.IsValid())
    return {};

  LLDB_LOGF(GetLog(LLDBLog::Types),
            "[GetMemberVariableOffset] asked to resolve offset for member %s",
            member_name.str().c_str());

  // Using the module context for RemoteAST is cheaper bit only safe
  // when there is no dynamic type resolution involved.
  // If this is already in the expression context, ask RemoteAST.
  if (llvm::isa<SwiftASTContext>(instance_type.GetTypeSystem()))
    offset =
        GetMemberVariableOffsetRemoteAST(instance_type, instance, member_name);
  if (!offset) {
    // Convert to a TypeRef-type, if necessary.
    if (auto *module_ctx = llvm::dyn_cast_or_null<SwiftASTContext>(
            instance_type.GetTypeSystem()))
      instance_type =
          module_ctx->GetTypeRefType(instance_type.GetOpaqueQualType());

    offset = GetMemberVariableOffsetRemoteMirrors(instance_type, instance,
                                                  member_name, error);
#ifndef NDEBUG
    {
      // Convert to an AST type, if necessary.
      if (auto *ts = llvm::dyn_cast_or_null<TypeSystemSwiftTypeRef>(
              instance_type.GetTypeSystem()))
        instance_type = ts->ReconstructType(instance_type);
      auto reference = GetMemberVariableOffsetRemoteAST(instance_type, instance,
                                                        member_name);
      if (reference.hasValue() && offset != reference) {
        instance_type.dump();
        llvm::dbgs() << "member_name = " << member_name << "\n";
        llvm::dbgs() << "remote mirrors: " << offset << "\n";
        llvm::dbgs() << "remote AST: " << reference << "\n";
        //      assert(offset == reference && "RemoteAST and Remote Mirrors
        //      diverge");
      }
    }
#endif
  }
  if (offset) {
    LLDB_LOGF(GetLog(LLDBLog::Types),
              "[GetMemberVariableOffset] offset of %s is %lld",
              member_name.str().c_str(), *offset);
  } else {
    LLDB_LOGF(GetLog(LLDBLog::Types), "[GetMemberVariableOffset] failed for %s",
              member_name.str().c_str());
    if (error)
      error->SetErrorStringWithFormat("could not resolve member offset");
  }
  return offset;
}

static CompilerType GetWeakReferent(TypeSystemSwiftTypeRef &ts,
                                    CompilerType type) {
  // FIXME: This is very similar to TypeSystemSwiftTypeRef::GetReferentType().
  using namespace swift::Demangle;
  Demangler dem;
  auto mangled = type.GetMangledTypeName().GetStringRef();
  NodePointer n = dem.demangleSymbol(mangled);
  if (!n || n->getKind() != Node::Kind::Global || !n->hasChildren())
    return {};
  n = n->getFirstChild();
  if (!n || n->getKind() != Node::Kind::TypeMangling || !n->hasChildren())
    return {};
  n = n->getFirstChild();
  if (!n || n->getKind() != Node::Kind::Type || !n->hasChildren())
    return {};
  n = n->getFirstChild();
  if (!n ||
      (n->getKind() != Node::Kind::Weak &&
       n->getKind() != Node::Kind::Unowned &&
       n->getKind() != Node::Kind::Unmanaged) ||
      !n->hasChildren())
    return {};
  n = n->getFirstChild();
  if (!n || n->getKind() != Node::Kind::Type || !n->hasChildren())
    return {};
  // FIXME: We only need to canonicalize this node, not the entire type.
  n = ts.CanonicalizeSugar(dem, n->getFirstChild());
  if (!n || n->getKind() != Node::Kind::SugaredOptional || !n->hasChildren())
    return {};
  n = n->getFirstChild();
  return ts.RemangleAsType(dem, n);
}

llvm::Optional<unsigned>
SwiftLanguageRuntimeImpl::GetNumChildren(CompilerType type,
                                         ValueObject *valobj) {
  LLDB_SCOPED_TIMER();
   auto *ts = llvm::dyn_cast_or_null<TypeSystemSwiftTypeRef>(
      type.GetTypeSystem());
  if (!ts)
    return {};

  // Try the static type metadata.
  auto frame =
      valobj ? valobj->GetExecutionContextRef().GetFrameSP().get() : nullptr;
  const swift::reflection::TypeRef *tr = nullptr;
  auto *ti = GetSwiftRuntimeTypeInfo(type, frame, &tr);
  if (!ti)
    return {};
  // Structs and Tuples.
  if (auto *rti = llvm::dyn_cast<swift::reflection::RecordTypeInfo>(ti)) {
    switch (rti->getRecordKind()) {
    case swift::reflection::RecordKind::ExistentialMetatype:
    case swift::reflection::RecordKind::ThickFunction:
      // There are two fields, `function` and `context`, but they're not exposed
      // by lldb.
      return 0;
    case swift::reflection::RecordKind::OpaqueExistential:
      // `OpaqueExistential` is documented as:
      //     An existential is a three-word buffer followed by value metadata...
      // The buffer is exposed as children named `payload_data_{0,1,2}`, and
      // the number of fields are increased to match.
      return rti->getNumFields() + 3;
    default:
      return rti->getNumFields();
    }
  }
  if (auto *eti = llvm::dyn_cast<swift::reflection::EnumTypeInfo>(ti)) {
    return eti->getNumPayloadCases();
  }
  // Objects.
  if (auto *rti = llvm::dyn_cast<swift::reflection::ReferenceTypeInfo>(ti)) {
    switch (rti->getReferenceKind()) {
    case swift::reflection::ReferenceKind::Weak:
    case swift::reflection::ReferenceKind::Unowned:
    case swift::reflection::ReferenceKind::Unmanaged:
      // Weak references are implicitly Optionals, so report the one
      // child of Optional here.
      if (GetWeakReferent(*ts, type))
        return 1;
      break;
    default:
      break;
    }

    if (!tr)
      return {};

    auto *reflection_ctx = GetReflectionContext();
    auto &builder = reflection_ctx->getBuilder();
    auto tc = swift::reflection::TypeConverter(builder);
    LLDBTypeInfoProvider tip(*this, *ts);
    auto *cti = tc.getClassInstanceTypeInfo(tr, 0, &tip);
    if (auto *rti =
            llvm::dyn_cast_or_null<swift::reflection::RecordTypeInfo>(cti)) {
      // The superclass, if any, is an extra child.
      if (builder.lookupSuperclass(tr))
        return rti->getNumFields() + 1;
      return rti->getNumFields();
    }

    return {};
  }
  // FIXME: Implement more cases.
  return {};
}

llvm::Optional<unsigned>
SwiftLanguageRuntimeImpl::GetNumFields(CompilerType type,
                                       ExecutionContext *exe_ctx) {
  auto *ts =
      llvm::dyn_cast_or_null<TypeSystemSwiftTypeRef>(type.GetTypeSystem());
  if (!ts)
    return {};

  using namespace swift::reflection;
  // Try the static type metadata.
  const TypeRef *tr = nullptr;
  auto *ti = GetSwiftRuntimeTypeInfo(type, exe_ctx->GetFramePtr(), &tr);
  if (!ti)
    return {};
  // Structs and Tuples.
  switch (ti->getKind()) {
  case TypeInfoKind::Record: {
    // Structs and Tuples.
    auto *rti = llvm::cast<RecordTypeInfo>(ti);
    switch (rti->getRecordKind()) {
    case RecordKind::ExistentialMetatype:
    case RecordKind::ThickFunction:
      // There are two fields, `function` and `context`, but they're not exposed
      // by lldb.
      return 0;
    case RecordKind::OpaqueExistential:
      // `OpaqueExistential` is documented as:
      //     An existential is a three-word buffer followed by value metadata...
      // The buffer is exposed as fields named `payload_data_{0,1,2}`, and
      // the number of fields are increased to match.
      return rti->getNumFields() + 3;
    default:
      return rti->getNumFields();
    }
  }
  case TypeInfoKind::Enum: {
    auto *eti = llvm::cast<EnumTypeInfo>(ti);
    return eti->getNumPayloadCases();
  }
  case TypeInfoKind::Reference: {
    // Objects.
    auto *rti = llvm::cast<ReferenceTypeInfo>(ti);
    switch (rti->getReferenceKind()) {
    case ReferenceKind::Weak:
    case ReferenceKind::Unowned:
    case ReferenceKind::Unmanaged:
      if (auto referent = GetWeakReferent(*ts, type))
        return referent.GetNumFields(exe_ctx);
      return 0;
    case ReferenceKind::Strong:
      TypeConverter tc(GetReflectionContext()->getBuilder());
      LLDBTypeInfoProvider tip(*this, *ts);
      auto *cti = tc.getClassInstanceTypeInfo(tr, 0, &tip);
      if (auto *rti = llvm::dyn_cast_or_null<RecordTypeInfo>(cti)) {
        return rti->getNumFields();
      }

      return {};
    }
  }
  default:
    // FIXME: Implement more cases.
    return {};
  }
}

static CompilerType
GetTypeFromTypeRef(TypeSystemSwiftTypeRef &ts,
                   const swift::reflection::TypeRef *type_ref) {
  if (!type_ref)
    return {};
  swift::Demangle::Demangler dem;
  swift::Demangle::NodePointer node = type_ref->getDemangling(dem);
  return ts.RemangleAsType(dem, node);
}

static std::pair<bool, llvm::Optional<size_t>>
findFieldWithName(const std::vector<swift::reflection::FieldInfo> &fields,
                  llvm::StringRef name, bool is_enum,
                  std::vector<uint32_t> &child_indexes, uint32_t offset = 0) {
  uint32_t index = 0;
  bool is_nonpayload_enum_case = false;
  auto it = std::find_if(fields.begin(), fields.end(), [&](const auto &field) {
    if (name != field.Name) {
      // A nonnull TypeRef is required for enum cases, where it represents cases
      // that have a payload. In other types it will be true anyway.
      if (field.TR)
        ++index;
      return false;
    }
    if (is_enum)
      is_nonpayload_enum_case = (field.TR == nullptr);
    return true;
  });
  // Not found.
  if (it == fields.end())
    return {false, {}};
  // Found, but no index to report.
  if (is_nonpayload_enum_case)
    return {true, {}};
  child_indexes.push_back(offset + index);
  return {true, child_indexes.size()};
}

static llvm::Optional<std::string>
GetMultiPayloadEnumCaseName(const swift::reflection::EnumTypeInfo *eti,
                            const DataExtractor &data) {
  assert(eti->getEnumKind() == swift::reflection::EnumKind::MultiPayloadEnum);
  auto payload_capacity = eti->getPayloadSize();
  if (data.GetByteSize() == payload_capacity + 1) {
    auto tag = data.GetDataStart()[payload_capacity];
    const auto &cases = eti->getCases();
    if (tag >= 0 && tag < cases.size())
      return cases[tag].Name;
  }
  return {};
}

llvm::Optional<std::string> SwiftLanguageRuntimeImpl::GetEnumCaseName(
    CompilerType type, const DataExtractor &data, ExecutionContext *exe_ctx) {
  using namespace swift::reflection;
  using namespace swift::remote;
  auto *ti = GetSwiftRuntimeTypeInfo(type, exe_ctx->GetFramePtr());
  if (!ti)
    return {};
  if (ti->getKind() != TypeInfoKind::Enum)
    return {};

  auto *eti = llvm::cast<EnumTypeInfo>(ti);
  PushLocalBuffer((int64_t)data.GetDataStart(), data.GetByteSize());
  auto defer = llvm::make_scope_exit([&] { PopLocalBuffer(); });
  RemoteAddress addr(data.GetDataStart());
  int case_index;
  if (eti->projectEnumValue(*GetMemoryReader(), addr, &case_index))
    return eti->getCases()[case_index].Name;

  // Temporary workaround.
  if (eti->getEnumKind() == EnumKind::MultiPayloadEnum &&
      type.GetMangledTypeName().GetStringRef().startswith(
          "$s10Foundation9IndexPathV7Storage"))
    return GetMultiPayloadEnumCaseName(eti, data);

  return {};
}

std::pair<bool, llvm::Optional<size_t>>
SwiftLanguageRuntimeImpl::GetIndexOfChildMemberWithName(
    CompilerType type, llvm::StringRef name, ExecutionContext *exe_ctx,
    bool omit_empty_base_classes, std::vector<uint32_t> &child_indexes) {
  LLDB_SCOPED_TIMER();
  auto *ts =
      llvm::dyn_cast_or_null<TypeSystemSwiftTypeRef>(type.GetTypeSystem());
  if (!ts)
    return {false, {}};

  using namespace swift::reflection;
  // Try the static type metadata.
  const TypeRef *tr = nullptr;
  auto *ti = GetSwiftRuntimeTypeInfo(type, exe_ctx->GetFramePtr(), &tr);
  if (!ti)
    return {false, {}};
  switch (ti->getKind()) {
  case TypeInfoKind::Record: {
    // Structs and Tuples.
    auto *rti = llvm::cast<RecordTypeInfo>(ti);
    switch (rti->getRecordKind()) {
    case RecordKind::ExistentialMetatype:
    case RecordKind::ThickFunction:
      // There are two fields, `function` and `context`, but they're not exposed
      // by lldb.
      return {true, {0}};
    case RecordKind::OpaqueExistential:
      // `OpaqueExistential` is documented as:
      //     An existential is a three-word buffer followed by value metadata...
      // The buffer is exposed as children named `payload_data_{0,1,2}`, and
      // the number of fields are increased to match.
      if (name.startswith("payload_data_")) {
        uint32_t index;
        if (name.take_back().getAsInteger(10, index) && index < 3) {
          child_indexes.push_back(index);
          return {true, child_indexes.size()};
        }
      }
      return findFieldWithName(rti->getFields(), name, false, child_indexes, 3);
    default:
      return findFieldWithName(rti->getFields(), name, false, child_indexes);
    }
  }
  case TypeInfoKind::Enum: {
    auto *eti = llvm::cast<EnumTypeInfo>(ti);
    return findFieldWithName(eti->getCases(), name, true, child_indexes);
  }
  case TypeInfoKind::Reference: {
    // Objects.
    auto *rti = llvm::cast<ReferenceTypeInfo>(ti);
    switch (rti->getReferenceKind()) {
    case ReferenceKind::Weak:
    case ReferenceKind::Unowned:
    case ReferenceKind::Unmanaged:
      return GetIndexOfChildMemberWithName(GetWeakReferent(*ts, type), name,
                                           exe_ctx, omit_empty_base_classes,
                                           child_indexes);
    case ReferenceKind::Strong: {
      auto *reflection_ctx = GetReflectionContext();
      auto &builder = reflection_ctx->getBuilder();
      TypeConverter tc(builder);
      LLDBTypeInfoProvider tip(*this, *ts);
      // `current_tr` iterates the class hierarchy, from the current class, each
      // superclass, and ends on null.
      auto *current_tr = tr;
      while (current_tr) {
        auto *record_ti = llvm::dyn_cast_or_null<RecordTypeInfo>(
            tc.getClassInstanceTypeInfo(current_tr, 0, &tip));
        if (!record_ti)
          break;
        auto *super_tr = builder.lookupSuperclass(current_tr);
        uint32_t offset = super_tr ? 1 : 0;
        auto found_size = findFieldWithName(record_ti->getFields(), name, false,
                                            child_indexes, offset);
        if (found_size.first)
          return found_size;
        current_tr = super_tr;
        child_indexes.push_back(0);
      }
      child_indexes.clear();
      return {false, {}};
    }
    }
  }
  default:
    // FIXME: Implement more cases.
    return {false, {}};
  }
}

CompilerType SwiftLanguageRuntimeImpl::GetChildCompilerTypeAtIndex(
    CompilerType type, size_t idx, bool transparent_pointers,
    bool omit_empty_base_classes, bool ignore_array_bounds,
    std::string &child_name, uint32_t &child_byte_size,
    int32_t &child_byte_offset, uint32_t &child_bitfield_bit_size,
    uint32_t &child_bitfield_bit_offset, bool &child_is_base_class,
    bool &child_is_deref_of_parent, ValueObject *valobj,
    uint64_t &language_flags) {
  auto *ts = llvm::dyn_cast_or_null<TypeSystemSwiftTypeRef>(
      type.GetTypeSystem());
  if (!ts)
    return {};

  // The actual conversion from the FieldInfo record.
  auto get_from_field_info =
      [&](const swift::reflection::FieldInfo &field,
          llvm::Optional<TypeSystemSwift::TupleElement> tuple,
          bool hide_existentials) -> CompilerType {
    child_name = tuple ? tuple->element_name.GetStringRef().str() : field.Name;
    child_byte_size = field.TI.getSize();
    child_byte_offset = field.Offset;
    child_bitfield_bit_size = 0;
    child_bitfield_bit_offset = 0;
    child_is_base_class = false;
    child_is_deref_of_parent = false;
    language_flags = 0;
    // SwiftASTContext hardcodes the members of protocols as raw
    // pointers. Remote Mirrors reports them as UnknownObject instead.
    if (hide_existentials && ts->IsExistentialType(type.GetOpaqueQualType()))
      return ts->GetRawPointerType();
    CompilerType result =
        tuple ? tuple->element_type : GetTypeFromTypeRef(*ts, field.TR);
    // Bug-for-bug compatibility. See comment in SwiftASTContext::GetBitSize().
    if (result.IsFunctionType())
      child_byte_size = ts->GetPointerByteSize();
    return result;
  };

  // Try the static type metadata.
  auto frame =
      valobj ? valobj->GetExecutionContextRef().GetFrameSP().get() : nullptr;
  auto *ti = GetSwiftRuntimeTypeInfo(type, frame);
  if (!ti)
    return {};
  // Structs and Tuples.
  if (auto *rti =
          llvm::dyn_cast_or_null<swift::reflection::RecordTypeInfo>(ti)) {
    auto fields = rti->getFields();

    // Handle tuples.
    if (idx > rti->getNumFields())
      LLDB_LOGF(GetLog(LLDBLog::Types), "index %zu is out of bounds (%d)", idx,
                rti->getNumFields());
    llvm::Optional<TypeSystemSwift::TupleElement> tuple;
    if (rti->getRecordKind() == swift::reflection::RecordKind::Tuple)
      tuple = ts->GetTupleElement(type.GetOpaqueQualType(), idx);
    if (rti->getRecordKind() ==
        swift::reflection::RecordKind::OpaqueExistential) {
      // Compatibility with SwiftASTContext.
      if (idx < 3) {
        child_name = "payload_data_";
        child_name += ('0' + idx);
        child_byte_size = ts->GetPointerByteSize();
        child_byte_offset = ts->GetPointerByteSize() * idx;
        child_bitfield_bit_size = 0;
        child_bitfield_bit_offset = 0;
        child_is_base_class = false;
        child_is_deref_of_parent = false;
        language_flags = 0;
        return ts->GetRawPointerType();
      }
      return get_from_field_info(fields[idx - 3], tuple, false);
    }
    if (rti->getRecordKind() ==
        swift::reflection::RecordKind::ClassExistential) {
      // Compatibility with SwiftASTContext.
      if (idx == 0) {
        child_name = fields[idx].Name;
        child_byte_size = ts->GetPointerByteSize();
        child_byte_offset = ts->GetPointerByteSize() * idx;
        child_bitfield_bit_size = 0;
        child_bitfield_bit_offset = 0;
        child_is_base_class = false;
        child_is_deref_of_parent = false;
        language_flags = 0;
          return ts->GetRawPointerType();
      }
    }
    return get_from_field_info(fields[idx], tuple, true);
  }
  // Enums.
  if (auto *eti = llvm::dyn_cast_or_null<swift::reflection::EnumTypeInfo>(ti)) {
    unsigned i = 0;
    for (auto &enum_case : eti->getCases()) {
      // Skip non-payload cases.
      if (!enum_case.TR)
        continue;
      if (i++ == idx) {
        auto is_indirect = [](const swift::reflection::FieldInfo &field) {
          // FIXME: This is by observation. What's the correct condition?
          if (auto *tr =
                  llvm::dyn_cast_or_null<swift::reflection::BuiltinTypeRef>(
                      field.TR))
            return llvm::StringRef(tr->getMangledName()).equals("Bo");
          return false;
        };
        auto result = get_from_field_info(enum_case, {}, true);
        if (is_indirect(enum_case))
          language_flags |= TypeSystemSwift::LanguageFlags::eIsIndirectEnumCase;
        return result;
      }
    }
    LLDB_LOGF(GetLog(LLDBLog::Types), "index %zu is out of bounds (%d)", idx,
              eti->getNumPayloadCases());
    return {};
  }
  if (auto *rti =
          llvm::dyn_cast_or_null<swift::reflection::ReferenceTypeInfo>(ti)) {
    // Objects.
    // Try the instance type metadata.
    if (!valobj)
      return {};
    bool found_start = false;
    using namespace swift::Demangle;
    Demangler dem;
    auto mangled = type.GetMangledTypeName().GetStringRef();
    NodePointer type_node = dem.demangleSymbol(mangled);
    llvm::StringRef type_name = TypeSystemSwiftTypeRef::GetBaseName(
        ts->CanonicalizeSugar(dem, type_node));

    auto *reflection_ctx = GetReflectionContext();
    if (!reflection_ctx)
      return {};
    CompilerType instance_type = valobj->GetCompilerType();
    auto *instance_ts =
        llvm::dyn_cast_or_null<TypeSystemSwift>(instance_type.GetTypeSystem());
    if (!instance_ts)
      return {};

    // LLDBTypeInfoProvider needs to kept alive until as long as supers gets accessed.
    llvm::SmallVector<SuperClassType, 2> supers;
    LLDBTypeInfoProvider tip(*this, *instance_ts);
    lldb::addr_t pointer = valobj->GetPointerValue();
    reflection_ctx->ForEachSuperClassType(
        &tip, pointer, [&](SuperClassType sc) -> bool {
          if (!found_start) {
            // The ValueObject always points to the same class instance,
            // even when querying base classes. Drop base classes until we
            // reach the requested type.
            if (auto *tr = sc.get_typeref()) {
              NodePointer base_class = tr->getDemangling(dem);
              if (TypeSystemSwiftTypeRef::GetBaseName(base_class) != type_name)
                return false;
              found_start = true;
            }
          }
          supers.push_back(sc);
          return supers.size() >= 2;
        });

    if (supers.size() == 0) {
      LLDB_LOGF(GetLog(LLDBLog::Types),
                "Couldn't find the type metadata for %s in instance",
                type.GetTypeName().AsCString());
      return {};
    }

    switch (rti->getReferenceKind()) {
    case swift::reflection::ReferenceKind::Weak:
    case swift::reflection::ReferenceKind::Unowned:
    case swift::reflection::ReferenceKind::Unmanaged:
      // Weak references are implicitly Optionals, so report the one
      // child of Optional here.
      if (idx != 0)
        break; // Maybe assert that type is not an Optional?
      child_name = "some";
      child_byte_size = ts->GetPointerByteSize();
      child_byte_offset = 0;
      child_bitfield_bit_size = 0;
      child_bitfield_bit_offset = 0;
      child_is_base_class = false;
      child_is_deref_of_parent = false;
      language_flags = 0;
      if (CompilerType optional = GetWeakReferent(*ts, type))
        return optional;
      break;
    default:
      break;
    }

    // Handle the artificial base class fields.
    unsigned i = 0;
    if (supers.size() > 1) {
      auto *type_ref = supers[1].get_typeref();
      auto *objc_tr =
          llvm::dyn_cast_or_null<swift::reflection::ObjCClassTypeRef>(type_ref);
      // SwiftASTContext hides the ObjC base class for Swift classes.
      if (!objc_tr || objc_tr->getName() != "_TtCs12_SwiftObject")
        if (i++ == idx) {
          // A synthetic field for the base class itself.  Only the direct
          // base class gets injected. Its parent will be a nested
          // field in the base class.
          if (!type_ref) {
            child_name = "<base class>";
            return {};
          }
          CompilerType super_type = GetTypeFromTypeRef(*ts, type_ref);
          child_name = super_type.GetTypeName().GetStringRef().str();
          // FIXME: This should be fixed in GetDisplayTypeName instead!
          if (child_name == "__C.NSObject")
            child_name = "ObjectiveC.NSObject";
          if (auto *rti = supers[1].get_record_type_info())
            child_byte_size = rti->getSize();
          // FIXME: This seems wrong in SwiftASTContext.
          child_byte_size = ts->GetPointerByteSize();
          child_byte_offset = 0;
          child_bitfield_bit_size = 0;
          child_bitfield_bit_offset = 0;
          child_is_base_class = true;
          child_is_deref_of_parent = false;
          language_flags = 0;
          return super_type;
        }
    }

    // Handle the "real" fields.
    auto *object = supers[0].get_record_type_info();
    if (!object)
      return {};
    for (auto &field : object->getFields())
      if (i++ == idx)
        return get_from_field_info(field, {}, true);

    LLDB_LOGF(GetLog(LLDBLog::Types), "index %zu is out of bounds (%d)", idx,
              i);
    return {};
  }
  LLDB_LOGF(GetLog(LLDBLog::Types), "Cannot retrieve type information for %s",
            type.GetTypeName().AsCString());
  return {};
}

bool SwiftLanguageRuntimeImpl::ForEachSuperClassType(
    ValueObject &instance, std::function<bool(SuperClassType)> fn) {
  auto *reflection_ctx = GetReflectionContext();
  if (!reflection_ctx)
    return false;
  CompilerType instance_type = instance.GetCompilerType();
  auto *ts =
      llvm::dyn_cast_or_null<TypeSystemSwift>(instance_type.GetTypeSystem());
  if (!ts)
    return false;

  LLDBTypeInfoProvider tip(*this, *ts);
  lldb::addr_t pointer = instance.GetPointerValue();
  return reflection_ctx->ForEachSuperClassType(&tip, pointer, fn);
}

bool SwiftLanguageRuntime::IsSelf(Variable &variable) {
  // A variable is self if its name if "self", and it's either a
  // function argument or a local variable and it's scope is a
  // constructor. These checks are sorted from cheap to expensive.
  if (variable.GetUnqualifiedName().GetStringRef() != "self")
    return false;

  if (variable.GetScope() == lldb::eValueTypeVariableArgument)
    return true;

  if (variable.GetScope() != lldb::eValueTypeVariableLocal)
    return false;

  SymbolContextScope *sym_ctx_scope = variable.GetSymbolContextScope();
  if (!sym_ctx_scope)
    return false;
  Function *function = sym_ctx_scope->CalculateSymbolContextFunction();
  if (!function)
    return false;
  StringRef func_name = function->GetMangled().GetMangledName().GetStringRef();
  swift::Demangle::Context demangle_ctx;
  swift::Demangle::NodePointer node_ptr =
      demangle_ctx.demangleSymbolAsNode(func_name);
  if (!node_ptr)
    return false;
  if (node_ptr->getKind() != swift::Demangle::Node::Kind::Global)
    return false;
  if (node_ptr->getNumChildren() != 1)
    return false;
  node_ptr = node_ptr->getFirstChild();
  return node_ptr->getKind() == swift::Demangle::Node::Kind::Constructor ||
         node_ptr->getKind() == swift::Demangle::Node::Kind::Allocator;
}

/// Determine whether the scratch SwiftASTContext has been locked.
static bool IsScratchContextLocked(Target &target) {
  if (target.GetSwiftScratchContextLock().try_lock()) {
    target.GetSwiftScratchContextLock().unlock();
    return false;
  }
  return true;
}

/// Determine whether the scratch SwiftASTContext has been locked.
static bool IsScratchContextLocked(TargetSP target) {
  return target ? IsScratchContextLocked(*target) : true;
}

static bool IsPrivateNSClass(NodePointer node) {
  if (!node || node->getKind() != Node::Kind::Type ||
      node->getNumChildren() == 0)
    return false;
  NodePointer classNode = node->getFirstChild();
  if (!classNode || classNode->getKind() != Node::Kind::Class ||
      classNode->getNumChildren() < 2)
    return false;
  for (NodePointer child : *classNode)
    if (child->getKind() == Node::Kind::Identifier && child->hasText())
      return child->getText().startswith("__NS") ||
             child->getText().startswith("NSTaggedPointer");
  return false;
}

bool SwiftLanguageRuntimeImpl::GetDynamicTypeAndAddress_Class(
    ValueObject &in_value, CompilerType class_type,
    lldb::DynamicValueType use_dynamic, TypeAndOrName &class_type_or_name,
    Address &address) {
  AddressType address_type;
  lldb::addr_t instance_ptr = in_value.GetPointerValue(&address_type);
  if (instance_ptr == LLDB_INVALID_ADDRESS || instance_ptr == 0)
    return false;

  auto *tss =
      llvm::dyn_cast_or_null<TypeSystemSwift>(class_type.GetTypeSystem());
  if (!tss)
    return false;
  address.SetRawAddress(instance_ptr);
  auto &ts = tss->GetTypeSystemSwiftTypeRef();
  // Ask the Objective-C runtime about Objective-C types.
  if (tss->IsImportedType(class_type.GetOpaqueQualType(), nullptr))
    if (auto *objc_runtime = SwiftLanguageRuntime::GetObjCRuntime(m_process)) {
      Value::ValueType value_type;
      if (objc_runtime->GetDynamicTypeAndAddress(
              in_value, use_dynamic, class_type_or_name, address, value_type)) {
        bool found = false;
        // Return the most specific class which we can get the typeref.
        ForEachSuperClassType(in_value, [&](SuperClassType sc) -> bool {
          if (auto *tr = sc.get_typeref()) {
            swift::Demangle::Demangler dem;
            swift::Demangle::NodePointer node = tr->getDemangling(dem);
            // Skip private Foundation types since it's unlikely that would be 
            // useful to users.
            if (IsPrivateNSClass(node))
              return false;
            class_type_or_name.SetCompilerType(ts.RemangleAsType(dem, node));
            found = true;
            return true;
          }
          return false;
        });
        return found;
      }
      return false;
    }
  Log *log(GetLog(LLDBLog::Types));
  auto *reflection_ctx = GetReflectionContext();
  const auto *typeref = reflection_ctx->readTypeFromInstance(instance_ptr);
  if (!typeref)
    return false;
  swift::Demangle::Demangler dem;
  swift::Demangle::NodePointer node = typeref->getDemangling(dem);
  class_type_or_name.SetCompilerType(ts.RemangleAsType(dem, node));

#ifndef NDEBUG
  // Dynamic type resolution in RemoteAST might pull in other Swift modules, so
  // use the scratch context where such operations are legal and safe.
  llvm::Optional<SwiftScratchContextReader> maybe_scratch_ctx =
      in_value.GetSwiftScratchContext();
  if (!maybe_scratch_ctx)
    return false;
  auto scratch_ctx = maybe_scratch_ctx->get();
  if (!scratch_ctx)
    return false;
  SwiftASTContext *swift_ast_ctx = scratch_ctx->GetSwiftASTContext();
  if (!swift_ast_ctx)
    return true;

  auto &remote_ast = GetRemoteASTContext(*swift_ast_ctx);
  auto remote_ast_metadata_address = remote_ast.getHeapMetadataForObject(
      swift::remote::RemoteAddress(instance_ptr));
  if (remote_ast_metadata_address) {
    auto instance_type = remote_ast.getTypeForRemoteTypeMetadata(
        remote_ast_metadata_address.getValue(),
        /*skipArtificial=*/true);
    if (instance_type) {
      auto ref_type = ToCompilerType(instance_type.getValue());
      ConstString a = ref_type.GetMangledTypeName();
      ConstString b = class_type_or_name.GetCompilerType().GetMangledTypeName();
      if (a != b)
        llvm::dbgs() << "RemoteAST and runtime diverge " << a << " != " << b
                     << "\n";
    } else {
      if (log) {
        log->Printf("could not get type metadata: %s\n",
                    instance_type.getFailure().render().c_str());
      }
    }
  }

#endif
  return true;
}

bool SwiftLanguageRuntimeImpl::IsValidErrorValue(ValueObject &in_value) {
  CompilerType var_type = in_value.GetStaticValue()->GetCompilerType();
  SwiftASTContext::ProtocolInfo protocol_info;
  if (!SwiftASTContext::GetProtocolTypeInfo(var_type, protocol_info))
    return false;
  if (!protocol_info.m_is_errortype)
    return false;

  unsigned index = SwiftASTContext::ProtocolInfo::error_instance_index;
  ValueObjectSP instance_type_sp(
      in_value.GetStaticValue()->GetChildAtIndex(index, true));
  if (!instance_type_sp)
    return false;
  lldb::addr_t metadata_location = instance_type_sp->GetValueAsUnsigned(0);
  if (metadata_location == 0 || metadata_location == LLDB_INVALID_ADDRESS)
    return false;

  if (auto swift_native_nserror_isa = GetSwiftNativeNSErrorISA()) {
    if (auto objc_runtime = SwiftLanguageRuntime::GetObjCRuntime(m_process)) {
      if (auto descriptor =
              objc_runtime->GetClassDescriptor(*instance_type_sp)) {
        if (descriptor->GetISA() != *swift_native_nserror_isa) {
          // not a __SwiftNativeNSError - but statically typed as ErrorType
          // return true here
          return true;
        }
      }
    }
  }

  if (SwiftLanguageRuntime::GetObjCRuntime(m_process)) {
    // this is a swift native error but it can be bridged to ObjC
    // so it needs to be layout compatible

    size_t ptr_size = m_process.GetAddressByteSize();
    size_t metadata_offset =
        ptr_size + 4 + (ptr_size == 8 ? 4 : 0);        // CFRuntimeBase
    metadata_offset += ptr_size + ptr_size + ptr_size; // CFIndex + 2*CFRef

    metadata_location += metadata_offset;
    Status error;
    lldb::addr_t metadata_ptr_value =
        m_process.ReadPointerFromMemory(metadata_location, error);
    if (metadata_ptr_value == 0 || metadata_ptr_value == LLDB_INVALID_ADDRESS ||
        error.Fail())
      return false;
  } else {
    // this is a swift native error and it has no way to be bridged to ObjC
    // so it adopts a more compact layout

    Status error;

    size_t ptr_size = m_process.GetAddressByteSize();
    size_t metadata_offset = 2 * ptr_size;
    metadata_location += metadata_offset;
    lldb::addr_t metadata_ptr_value =
        m_process.ReadPointerFromMemory(metadata_location, error);
    if (metadata_ptr_value == 0 || metadata_ptr_value == LLDB_INVALID_ADDRESS ||
        error.Fail())
      return false;
  }

  return true;
}

bool SwiftLanguageRuntimeImpl::GetDynamicTypeAndAddress_Protocol(
    ValueObject &in_value, CompilerType protocol_type,
    lldb::DynamicValueType use_dynamic, TypeAndOrName &class_type_or_name,
    Address &address) {
  auto remote_ast_impl = [&](bool use_local_buffer,
                             lldb::addr_t existential_address)
      -> llvm::Optional<std::pair<CompilerType, Address>> {
    // Dynamic type resolution in RemoteAST might pull in other Swift modules,
    // so
    // use the scratch context where such operations are legal and safe.
    llvm::Optional<SwiftScratchContextReader> maybe_scratch_ctx =
        in_value.GetSwiftScratchContext();
    if (!maybe_scratch_ctx)
      return {};
    auto scratch_ctx = maybe_scratch_ctx->get();
    if (!scratch_ctx)
      return {};
    SwiftASTContext *swift_ast_ctx = scratch_ctx->GetSwiftASTContext();
    if (!swift_ast_ctx)
      return {};

    swift::remote::RemoteAddress remote_existential(existential_address);
    auto &remote_ast = GetRemoteASTContext(*swift_ast_ctx);
    auto swift_type = GetSwiftType(protocol_type);
    if (!swift_type)
      return {};
    if (use_local_buffer)
      PushLocalBuffer(existential_address,
                      in_value.GetByteSize().getValueOr(0));

    auto result = remote_ast.getDynamicTypeAndAddressForExistential(
        remote_existential, swift_type);
    if (use_local_buffer)
      PopLocalBuffer();

    if (!result.isSuccess())
      return {};

    auto type_and_address = result.getValue();

    CompilerType type = ToCompilerType(type_and_address.InstanceType);
    Address address;
    address.SetRawAddress(type_and_address.PayloadAddress.getAddressData());
    return {{type, address}};
  };

  Log *log(GetLog(LLDBLog::Types));
  auto *tss =
      llvm::dyn_cast_or_null<TypeSystemSwift>(protocol_type.GetTypeSystem());
  if (!tss) {
    if (log)
      log->Printf("Could not get type system swift");
    return false;
  }

  const swift::reflection::TypeRef *protocol_typeref =
      GetTypeRef(protocol_type, &tss->GetTypeSystemSwiftTypeRef());
  if (!protocol_typeref) {
    if (log)
      log->Printf("Could not get protocol typeref");
    return false;
  }

  lldb::addr_t existential_address;
  bool use_local_buffer = false;

  if (in_value.GetValueType() == eValueTypeConstResult &&
      in_value.GetValue().GetValueType() ==
          lldb_private::Value::ValueType::HostAddress) {
    if (log)
      log->Printf("existential value is a const result");

    // We have a locally materialized value that is a host address;
    // register it with MemoryReader so it does not treat it as a load
    // address.  Note that this assumes that any address at that host
    // address is also a load address. If this assumption breaks there
    // will be a crash in readBytes().
    existential_address = in_value.GetValue().GetScalar().ULongLong();
    use_local_buffer = true;
  } else {
    existential_address = in_value.GetAddressOf();
  }

  if (log)
    log->Printf("existential address is 0x%llx", existential_address);

  if (!existential_address || existential_address == LLDB_INVALID_ADDRESS) {
    if (log)
      log->Printf("Existential address is invalid");
    return false;
  }


  if (use_local_buffer)
    PushLocalBuffer(existential_address, in_value.GetByteSize().getValueOr(0));

  swift::remote::RemoteAddress remote_existential(existential_address);

  auto *reflection_ctx = GetReflectionContext();
  auto pair = reflection_ctx->projectExistentialAndUnwrapClass(
      remote_existential, *protocol_typeref);
  if (use_local_buffer)
    PopLocalBuffer();

  if (!pair) {
    if (log)
      log->Printf("Runtime failed to get dynamic type of existential");
    return false;
  }

  const swift::reflection::TypeRef *typeref;
  swift::remote::RemoteAddress out_address(nullptr);
  std::tie(typeref, out_address) = *pair;
  auto &ts = tss->GetTypeSystemSwiftTypeRef();
  swift::Demangle::Demangler dem;
  swift::Demangle::NodePointer node = typeref->getDemangling(dem);
  class_type_or_name.SetCompilerType(ts.RemangleAsType(dem, node));
  address.SetRawAddress(out_address.getAddressData());

#ifndef NDEBUG
  auto reference_pair = remote_ast_impl(use_local_buffer, existential_address);
  assert(pair.hasValue() >= reference_pair.hasValue() &&
         "RemoteAST and runtime diverge");

  if (reference_pair) {
    CompilerType ref_type = std::get<CompilerType>(*reference_pair);
    Address ref_address = std::get<Address>(*reference_pair);
    ConstString a = class_type_or_name.GetCompilerType().GetMangledTypeName();
    ConstString b = ref_type.GetMangledTypeName();
    if (a != b)
      llvm::dbgs() << "RemoteAST and runtime diverge " << a << " != " << b
                   << "\n";
  }
#endif
  return true;
}

llvm::Optional<lldb::addr_t>
SwiftLanguageRuntimeImpl::GetTypeMetadataForTypeNameAndFrame(
    StringRef mdvar_name, StackFrame &frame) {
  VariableList *var_list = frame.GetVariableList(false);
  if (!var_list)
    return {};

  VariableSP var_sp(var_list->FindVariable(ConstString(mdvar_name)));
  if (!var_sp)
    return {};

  ValueObjectSP metadata_ptr_var_sp(
      frame.GetValueObjectForFrameVariable(var_sp, lldb::eNoDynamicValues));
  if (!metadata_ptr_var_sp ||
      metadata_ptr_var_sp->UpdateValueIfNeeded() == false)
    return {};

  lldb::addr_t metadata_location(metadata_ptr_var_sp->GetValueAsUnsigned(0));
  if (metadata_location == 0 || metadata_location == LLDB_INVALID_ADDRESS)
    return {};

  return metadata_location;
}

  SwiftLanguageRuntime::MetadataPromiseSP
SwiftLanguageRuntimeImpl::GetPromiseForTypeNameAndFrame(const char *type_name,
                                                        StackFrame *frame) {
  if (!frame || !type_name || !type_name[0])
    return nullptr;

  StreamString type_metadata_ptr_var_name;
  type_metadata_ptr_var_name.Printf("$%s", type_name);
  VariableList *var_list = frame->GetVariableList(false);
  if (!var_list)
    return nullptr;

  VariableSP var_sp(var_list->FindVariable(
      ConstString(type_metadata_ptr_var_name.GetData())));
  if (!var_sp)
    return nullptr;

  ValueObjectSP metadata_ptr_var_sp(
      frame->GetValueObjectForFrameVariable(var_sp, lldb::eNoDynamicValues));
  if (!metadata_ptr_var_sp ||
      metadata_ptr_var_sp->UpdateValueIfNeeded() == false)
    return nullptr;

  lldb::addr_t metadata_location(metadata_ptr_var_sp->GetValueAsUnsigned(0));
  if (metadata_location == 0 || metadata_location == LLDB_INVALID_ADDRESS)
    return nullptr;
  return GetMetadataPromise(metadata_location, *metadata_ptr_var_sp);
}

static void
ForEachGenericParameter(swift::Demangle::NodePointer node,
                        std::function<void(unsigned, unsigned)> callback) {
  if (!node)
    return;

  using namespace swift::Demangle;
  switch (node->getKind()) {
  case Node::Kind::DependentGenericParamType: {
    if (node->getNumChildren() != 2)
      return;
    NodePointer depth_node = node->getChild(0);
    NodePointer index_node = node->getChild(1);
    if (!depth_node || !depth_node->hasIndex() || !index_node ||
        !index_node->hasIndex())
      return;
    callback(depth_node->getIndex(), index_node->getIndex());
    break;
  }
  default:
    // Visit the child nodes.
    for (unsigned i = 0; i < node->getNumChildren(); ++i)
      ForEachGenericParameter(node->getChild(i), callback);
  }
}

CompilerType
SwiftLanguageRuntimeImpl::BindGenericTypeParameters(StackFrame &stack_frame,
                                                    TypeSystemSwiftTypeRef &ts,
                                                    ConstString mangled_name) {
  LLDB_SCOPED_TIMER();
  using namespace swift::Demangle;

  Status error;
  auto *reflection_ctx = GetReflectionContext();
  if (!reflection_ctx) {
    LLDB_LOG(GetLog(LLDBLog::Expressions | LLDBLog::Types),
             "No reflection context available.");
    return ts.GetTypeFromMangledTypename(mangled_name);
  }

  Demangler dem;
  NodePointer canonical = TypeSystemSwiftTypeRef::Transform(
      dem, dem.demangleSymbol(mangled_name.GetStringRef()),
      [](NodePointer node) {
        if (node->getKind() != Node::Kind::DynamicSelf)
          return node;
        // Substitute the static type for dynamic self.
        assert(node->getNumChildren() == 1);
        if (node->getNumChildren() != 1)
          return node;
        NodePointer type = node->getChild(0);
        if (type->getKind() != Node::Kind::Type || type->getNumChildren() != 1)
          return node;
        return type->getChild(0);
      });

  // Build the list of type substitutions.
  swift::reflection::GenericArgumentMap substitutions;
  ForEachGenericParameter(canonical, [&](unsigned depth, unsigned index) {
    if (substitutions.count({depth, index}))
      return;
    StreamString mdvar_name;
    mdvar_name.Printf(u8"$\u03C4_%d_%d", depth, index);

    llvm::Optional<lldb::addr_t> metadata_location =
        GetTypeMetadataForTypeNameAndFrame(mdvar_name.GetString(), stack_frame);
    if (!metadata_location)
      return;

    const swift::reflection::TypeRef *type_ref =
        reflection_ctx->readTypeFromMetadata(*metadata_location);
    if (!type_ref)
      return;
    substitutions.insert({{depth, index}, type_ref});
  });

  // Nothing to do if there are no type parameters.
  auto get_canonical = [&]() {
    auto mangling = mangleNode(canonical);
    if (!mangling.isSuccess())
      return CompilerType();
    return ts.GetTypeFromMangledTypename(ConstString(mangling.result()));
  };
  if (substitutions.empty())
    return get_canonical();

  // Build a TypeRef from the demangle tree.
  auto type_ref_or_err =
      decodeMangledType(reflection_ctx->getBuilder(), canonical);
  if (type_ref_or_err.isError()) {
    LLDB_LOG(GetLog(LLDBLog::Expressions | LLDBLog::Types),
             "Couldn't get TypeRef");
    return get_canonical();
  }
  const swift::reflection::TypeRef *type_ref = type_ref_or_err.getType();

  // Apply the substitutions.
  const swift::reflection::TypeRef *bound_type_ref =
      type_ref->subst(reflection_ctx->getBuilder(), substitutions);
  NodePointer node = bound_type_ref->getDemangling(dem);

  // Import the type into the scratch context. Subsequent conversions
  // to Swift types must be performed in the scratch context, since
  // the bound type may combine types from different
  // lldb::Modules. Contrary to the AstContext variant of this
  // function, we don't want to do this earlier, because the
  // canonicalization in GetCanonicalDemangleTree() must be performed in
  // the original context as to resolve type aliases correctly.
  auto &target = m_process.GetTarget();
  auto maybe_scratch_ctx = target.GetSwiftScratchContext(error, stack_frame);
  if (!maybe_scratch_ctx) {
    LLDB_LOG(GetLog(LLDBLog::Expressions | LLDBLog::Types),
             "No scratch context available.");
    return ts.GetTypeFromMangledTypename(mangled_name);
  }
  auto scratch_ctx = maybe_scratch_ctx->get();
  if (!scratch_ctx) {
    LLDB_LOG(GetLog(LLDBLog::Expressions | LLDBLog::Types),
             "No scratch context available.");
    return ts.GetTypeFromMangledTypename(mangled_name);
  }
  CompilerType bound_type =  scratch_ctx->RemangleAsType(dem, node);
  LLDB_LOG(GetLog(LLDBLog::Expressions | LLDBLog::Types), "Bound {0} -> {1}.",
           mangled_name, bound_type.GetMangledTypeName());
  return bound_type;
}

CompilerType
SwiftLanguageRuntimeImpl::BindGenericTypeParameters(StackFrame &stack_frame,
                                                    CompilerType base_type) {
  LLDB_SCOPED_TIMER();

  // If this is a TypeRef type, bind that.
  auto sc = stack_frame.GetSymbolContext(lldb::eSymbolContextEverything);
  if (auto *ts = llvm::dyn_cast_or_null<TypeSystemSwiftTypeRef>(
          base_type.GetTypeSystem()))
    return BindGenericTypeParameters(stack_frame, *ts,
                                     base_type.GetMangledTypeName());

  Status error;
  auto &target = m_process.GetTarget();
  assert(IsScratchContextLocked(target) &&
         "Swift scratch context not locked ahead of archetype binding");

  // A failing Clang import in a module context permanently damages
  // that module context.  Binding archetypes can trigger an import of
  // another module, so switch to a scratch context where such an
  // operation is safe.
  llvm::Optional<SwiftScratchContextReader> maybe_scratch_ctx =
      target.GetSwiftScratchContext(error, stack_frame);
  if (!maybe_scratch_ctx)
    return base_type;
  auto scratch_ctx = maybe_scratch_ctx->get();
  if (!scratch_ctx)
    return base_type;
  SwiftASTContext *swift_ast_ctx = scratch_ctx->GetSwiftASTContext();
  if (!swift_ast_ctx)
    return base_type;
  base_type = swift_ast_ctx->ImportType(base_type, error);

  if (base_type.GetTypeInfo() & lldb::eTypeIsSwift) {
    swift::Type target_swift_type(GetSwiftType(base_type));
    if (target_swift_type->hasArchetype())
      target_swift_type = target_swift_type->mapTypeOutOfContext().getPointer();

    // FIXME: This is wrong, but it doesn't actually matter right now since
    // all conformances are always visible
    auto *module_decl = swift_ast_ctx->GetASTContext()->getStdlibModule();

    // Replace opaque types with their underlying types when possible.
    swift::Mangle::ASTMangler mangler(true);

    // Rewrite all dynamic self types to their static self types.
    target_swift_type =
        target_swift_type.transform([](swift::Type type) -> swift::Type {
          if (auto *dynamic_self =
                  llvm::dyn_cast<swift::DynamicSelfType>(type.getPointer()))
            return dynamic_self->getSelfType();
          return type;
        });

    // Thicken generic metatypes. Once substituted, they should always
    // be thick. TypeRef::subst() does the same transformation.
    target_swift_type =
        target_swift_type.transform([](swift::Type type) -> swift::Type {
          using namespace swift;
          const auto thin = MetatypeRepresentation::Thin;
          const auto thick = MetatypeRepresentation::Thick;
          if (auto *metatype = dyn_cast<AnyMetatypeType>(type.getPointer()))
            if (metatype->hasRepresentation() &&
                metatype->getRepresentation() == thin &&
                metatype->getInstanceType()->hasTypeParameter())
              return MetatypeType::get(metatype->getInstanceType(), thick);
          return type;
        });

    while (target_swift_type->hasOpaqueArchetype()) {
      auto old_type = target_swift_type;
      target_swift_type = target_swift_type.subst(
          [&](swift::SubstitutableType *type) -> swift::Type {
            auto opaque_type =
                llvm::dyn_cast<swift::OpaqueTypeArchetypeType>(type);
            if (!opaque_type ||
                !opaque_type->getInterfaceType()
                  ->is<swift::GenericTypeParamType>())
              return type;

            // Try to find the symbol for the opaque type descriptor in the
            // process.
            auto mangled_name = ConstString(
                mangler.mangleOpaqueTypeDescriptor(opaque_type->getDecl()));

            SymbolContextList found;
            target.GetImages().FindSymbolsWithNameAndType(
                mangled_name, eSymbolTypeData, found);

            if (found.GetSize() == 0)
              return type;

            swift::Type result_type;

            for (unsigned i = 0, e = found.GetSize(); i < e; ++i) {
              SymbolContext found_sc;
              if (!found.GetContextAtIndex(i, found_sc))
                continue;

              // See if the symbol has an address.
              if (!found_sc.symbol)
                continue;

              auto addr = found_sc.symbol->GetAddress().GetLoadAddress(&target);
              if (!addr || addr == LLDB_INVALID_ADDRESS)
                continue;

              // Ask RemoteAST to get the underlying type out of the descriptor.
              auto &remote_ast = GetRemoteASTContext(*swift_ast_ctx);
              auto genericParam = opaque_type->getInterfaceType()
                  ->getAs<swift::GenericTypeParamType>();
              auto underlying_type_result =
                  remote_ast.getUnderlyingTypeForOpaqueType(
                      swift::remote::RemoteAddress(addr),
                      opaque_type->getSubstitutions(),
                      genericParam->getIndex());

              if (!underlying_type_result)
                continue;

              // If we haven't yet gotten an underlying type, use this as our
              // possible result.
              if (!result_type) {
                result_type = underlying_type_result.getValue();
              }
              // If we have two possibilities, they should match.
              else if (!result_type->isEqual(
                           underlying_type_result.getValue())) {
                return type;
              }
            }

            if (!result_type)
              return type;

            return result_type;
          },
          swift::LookUpConformanceInModule(module_decl),
          swift::SubstFlags::DesugarMemberTypes |
              swift::SubstFlags::SubstituteOpaqueArchetypes);

      // Stop if we've reached a fixpoint where we can't further resolve opaque
      // types.
      if (old_type->isEqual(target_swift_type))
        break;
    }

    target_swift_type = target_swift_type.subst(
        [this, &stack_frame,
         &swift_ast_ctx](swift::SubstitutableType *type) -> swift::Type {
          StreamString type_name;
          if (!SwiftLanguageRuntime::GetAbstractTypeName(type_name, type))
            return type;
          CompilerType concrete_type = this->GetConcreteType(
              &stack_frame, ConstString(type_name.GetString()));
          Status import_error;
          CompilerType target_concrete_type =
              swift_ast_ctx->ImportType(concrete_type, import_error);

          if (target_concrete_type.IsValid())
            return swift::Type(GetSwiftType(target_concrete_type));

          return type;
        },
        swift::LookUpConformanceInModule(module_decl),
        swift::SubstFlags::DesugarMemberTypes);
    assert(target_swift_type);

    return ToCompilerType({target_swift_type.getPointer()});
  }
  return base_type;
}

bool SwiftLanguageRuntime::GetAbstractTypeName(StreamString &name,
                                               swift::Type swift_type) {
  auto *generic_type_param = swift_type->getAs<swift::GenericTypeParamType>();
  if (!generic_type_param)
    return false;

  name.Printf(u8"\u03C4_%d_%d", generic_type_param->getDepth(),
              generic_type_param->getIndex());
  return true;
}

bool SwiftLanguageRuntimeImpl::GetDynamicTypeAndAddress_Value(
    ValueObject &in_value, CompilerType &bound_type,
    lldb::DynamicValueType use_dynamic, TypeAndOrName &class_type_or_name,
    Address &address) {
  class_type_or_name.SetCompilerType(bound_type);

  llvm::Optional<uint64_t> size = bound_type.GetByteSize(
      in_value.GetExecutionContextRef().GetFrameSP().get());
  if (!size)
    return false;
  lldb::addr_t val_address = in_value.GetAddressOf(true, nullptr);
  if (*size && (!val_address || val_address == LLDB_INVALID_ADDRESS))
    return false;

  address.SetLoadAddress(val_address, in_value.GetTargetSP().get());
  return true;
}

bool SwiftLanguageRuntimeImpl::GetDynamicTypeAndAddress_IndirectEnumCase(
    ValueObject &in_value, lldb::DynamicValueType use_dynamic,
    TypeAndOrName &class_type_or_name, Address &address) {
  static ConstString g_offset("offset");

  DataExtractor data;
  Status error;
  if (!(in_value.GetParent() && in_value.GetParent()->GetData(data, error) &&
        error.Success()))
    return false;

  bool has_payload;
  bool is_indirect;
  CompilerType payload_type;
  if (!SwiftASTContext::GetSelectedEnumCase(
          in_value.GetParent()->GetCompilerType(), data, nullptr, &has_payload,
          &payload_type, &is_indirect))
    return false;

  if (has_payload && is_indirect && payload_type)
    class_type_or_name.SetCompilerType(payload_type);

  lldb::addr_t box_addr = in_value.GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
  if (box_addr == LLDB_INVALID_ADDRESS)
    return false;

  box_addr = MaskMaybeBridgedPointer(m_process, box_addr);
  lldb::addr_t box_location = m_process.ReadPointerFromMemory(box_addr, error);
  if (box_location == LLDB_INVALID_ADDRESS)
    return false;

  box_location = MaskMaybeBridgedPointer(m_process, box_location);
  ProcessStructReader reader(&m_process, box_location, GetBoxMetadataType());
  uint32_t offset = reader.GetField<uint32_t>(g_offset);
  lldb::addr_t box_value = box_addr + offset;

  // try to read one byte at the box value
  m_process.ReadUnsignedIntegerFromMemory(box_value, 1, 0, error);
  if (error.Fail()) // and if that fails, then we're off in no man's land
    return false;

  Flags type_info(payload_type.GetTypeInfo());
  if (type_info.AllSet(eTypeIsSwift | eTypeIsClass)) {
    lldb::addr_t old_box_value = box_value;
    box_value = m_process.ReadPointerFromMemory(box_value, error);
    if (box_value == LLDB_INVALID_ADDRESS)
      return false;

    DataExtractor data(&box_value, m_process.GetAddressByteSize(),
                       m_process.GetByteOrder(),
                       m_process.GetAddressByteSize());
    ValueObjectSP valobj_sp(ValueObject::CreateValueObjectFromData(
        "_", data, m_process, payload_type));
    if (!valobj_sp)
      return false;

    Value::ValueType value_type;
    if (!GetDynamicTypeAndAddress(*valobj_sp, use_dynamic, class_type_or_name,
                                  address, value_type))
      return false;

    address.SetRawAddress(old_box_value);
    return true;
  } else if (type_info.AllSet(eTypeIsSwift | eTypeIsProtocol)) {
    SwiftASTContext::ProtocolInfo protocol_info;
    if (!SwiftASTContext::GetProtocolTypeInfo(payload_type, protocol_info))
      return false;
    auto ptr_size = m_process.GetAddressByteSize();
    std::vector<uint8_t> buffer(ptr_size * protocol_info.m_num_storage_words,
                                0);
    for (uint32_t idx = 0; idx < protocol_info.m_num_storage_words; idx++) {
      lldb::addr_t word = m_process.ReadUnsignedIntegerFromMemory(
          box_value + idx * ptr_size, ptr_size, 0, error);
      if (error.Fail())
        return false;
      memcpy(&buffer[idx * ptr_size], &word, ptr_size);
    }
    DataExtractor data(&buffer[0], buffer.size(), m_process.GetByteOrder(),
                       m_process.GetAddressByteSize());
    ValueObjectSP valobj_sp(ValueObject::CreateValueObjectFromData(
        "_", data, m_process, payload_type));
    if (!valobj_sp)
      return false;

    Value::ValueType value_type;
    if (!GetDynamicTypeAndAddress(*valobj_sp, use_dynamic, class_type_or_name,
                                  address, value_type))
      return false;

    address.SetRawAddress(box_value);
    return true;
  } else {
    // This is most likely a statically known type.
    address.SetLoadAddress(box_value, &m_process.GetTarget());
    return true;
  }
}

void SwiftLanguageRuntimeImpl::DumpTyperef(
    CompilerType type, TypeSystemSwiftTypeRef *module_holder, Stream *s) {
  if (!s)
    return;

  const auto *typeref = GetTypeRef(type, module_holder);
  if (!typeref)
    return;

  std::ostringstream string_stream;
  typeref->dump(string_stream);
  s->PutCString(string_stream.str());
}

// Dynamic type resolution tends to want to generate scalar data - but there are
// caveats
// Per original comment here
// "Our address is the location of the dynamic type stored in memory.  It isn't
// a load address,
//  because we aren't pointing to the LOCATION that stores the pointer to us,
//  we're pointing to us..."
// See inlined comments for exceptions to this general rule.
Value::ValueType
SwiftLanguageRuntimeImpl::GetValueType(ValueObject &in_value,
                                       CompilerType dynamic_type,
                                       bool is_indirect_enum_case) {
  Value::ValueType static_value_type = in_value.GetValue().GetValueType();
  CompilerType static_type = in_value.GetCompilerType();
  Flags static_type_flags(static_type.GetTypeInfo());
  Flags dynamic_type_flags(dynamic_type.GetTypeInfo());

  if (dynamic_type_flags.AllSet(eTypeIsSwift)) {
    // for a protocol object where does the dynamic data live if the target
    // object is a struct? (for a class, it's easy)
    if (static_type_flags.AllSet(eTypeIsSwift | eTypeIsProtocol) &&
        dynamic_type_flags.AnySet(eTypeIsStructUnion | eTypeIsEnumeration)) {
      if (auto *ts = llvm::dyn_cast_or_null<TypeSystemSwiftTypeRef>(
              static_type.GetTypeSystem()))
        static_type = ts->ReconstructType(static_type);
      SwiftASTContext *swift_ast_ctx =
          llvm::dyn_cast_or_null<SwiftASTContext>(static_type.GetTypeSystem());
      if (!swift_ast_ctx)
        return {};
      if (swift_ast_ctx->IsErrorType(static_type.GetOpaqueQualType())) {
        // ErrorType values are always a pointer
        return Value::ValueType::LoadAddress;
      }

      lldb::addr_t existential_address;
      bool use_local_buffer = false;

      if (in_value.GetValueType() == eValueTypeConstResult &&
          // We have a locally materialized value that is a host address;
          // register it with MemoryReader so it does not treat it as a load
          // address.  Note that this assumes that any address at that host
          // address is also a load address. If this assumption breaks there
          // will be a crash in readBytes().
          static_value_type == lldb_private::Value::ValueType::HostAddress) {
        existential_address = in_value.GetValue().GetScalar().ULongLong();
        use_local_buffer = true;
      } else {
        existential_address = in_value.GetAddressOf();
      }

      if (use_local_buffer)
        PushLocalBuffer(existential_address,
                        in_value.GetByteSize().getValueOr(0));

      // Read the value witness table and check if the data is inlined in
      // the existential container or not.
      swift::remote::RemoteAddress remote_existential(existential_address);
      auto *reflection_ctx = GetReflectionContext();
      llvm::Optional<bool> is_inlined =
          reflection_ctx->isValueInlinedInExistentialContainer(
              remote_existential);

      if (use_local_buffer)
        PopLocalBuffer();

      // An error has occurred when trying to read value witness table,
      // default to treating it as pointer.
      if (!is_inlined.hasValue())
        return Value::ValueType::LoadAddress;

      // Inlined data, same as static data.
      if (*is_inlined)
        return static_value_type;

      // If the data is not inlined, we have a pointer.
      return Value::ValueType::LoadAddress;
    }

    if (static_type_flags.AllSet(eTypeIsSwift | eTypeIsGenericTypeParam)) {
      // if I am handling a non-pointer Swift type obtained from an archetype,
      // then the runtime vends the location
      // of the object, not the object per se (since the object is not a pointer
      // itself, this is way easier to achieve)
      // hence, it's a load address, not a scalar containing a pointer as for
      // ObjC classes
      if (dynamic_type_flags.AllClear(eTypeIsPointer | eTypeIsReference |
                                      eTypeInstanceIsPointer))
        return Value::ValueType::LoadAddress;
    }

    if (static_type_flags.AllSet(eTypeIsSwift | eTypeIsPointer) &&
        static_type_flags.AllClear(eTypeIsGenericTypeParam)) {
      // FIXME: This branch is not covered by any testcases in the test suite.
      if (is_indirect_enum_case || static_type_flags.AllClear(eTypeIsBuiltIn))
        return Value::ValueType::LoadAddress;
    }
  }

  // Enabling this makes the inout_variables test hang.
  //  return Value::eValueTypeScalar;
  if (static_type_flags.AllSet(eTypeIsSwift) &&
      dynamic_type_flags.AllSet(eTypeIsSwift) &&
      dynamic_type_flags.AllClear(eTypeIsPointer | eTypeInstanceIsPointer))
    return static_value_type;
  return Value::ValueType::Scalar;
}

bool SwiftLanguageRuntimeImpl::GetDynamicTypeAndAddress_ClangType(
    ValueObject &in_value, lldb::DynamicValueType use_dynamic,
    TypeAndOrName &class_type_or_name, Address &address,
    Value::ValueType &value_type) {
  AppleObjCRuntime *objc_runtime =
      SwiftLanguageRuntime::GetObjCRuntime(m_process);
  if (!objc_runtime)
    return false;

  // This is a Clang type, which means it must have been an
  // Objective-C protocol. Protocols are not represented in DWARF and
  // LLDB's ObjC runtime implementation doesn't know how to deal with
  // them either.  Use the Objective-C runtime to perform dynamic type
  // resolution first, and then map the dynamic Objective-C type back
  // into Swift.
  TypeAndOrName dyn_class_type_or_name = class_type_or_name;
  if (!objc_runtime->GetDynamicTypeAndAddress(
          in_value, use_dynamic, dyn_class_type_or_name, address, value_type))
    return false;

  StringRef dyn_name = dyn_class_type_or_name.GetName().GetStringRef();
  // If this is an Objective-C runtime value, skip; this is handled elsewhere.
  if (swift::Demangle::isOldFunctionTypeMangling(dyn_name) ||
      dyn_name.startswith("__NS"))
    return false;

  std::string remangled;
  {
    // Create a mangle tree for __C.dyn_name?.
    using namespace swift::Demangle;
    NodeFactory factory;
    NodePointer global = factory.createNode(Node::Kind::Global);
    NodePointer tm = factory.createNode(Node::Kind::TypeMangling);
    global->addChild(tm, factory);
    NodePointer bge = factory.createNode(Node::Kind::BoundGenericEnum);
    tm->addChild(bge, factory);
    NodePointer ety = factory.createNode(Node::Kind::Type);
    bge->addChild(ety, factory);
    NodePointer e = factory.createNode(Node::Kind::Enum);
    e->addChild(factory.createNode(Node::Kind::Module, "Swift"), factory);
    e->addChild(factory.createNode(Node::Kind::Identifier, "Optional"),
                factory);
    ety->addChild(e, factory);
    NodePointer list = factory.createNode(Node::Kind::TypeList);
    bge->addChild(list, factory);
    NodePointer cty = factory.createNode(Node::Kind::Type);
    list->addChild(cty, factory);
    NodePointer c = factory.createNode(Node::Kind::Class);
    c->addChild(factory.createNode(Node::Kind::Module, "__C"), factory);
    c->addChild(factory.createNode(Node::Kind::Identifier, dyn_name), factory);
    cty->addChild(c, factory);

    auto mangling = mangleNode(global);
    if (!mangling.isSuccess())
      return false;
    remangled = mangling.result();
  }

  // Import the remangled dynamic name into the scratch context.
  assert(IsScratchContextLocked(in_value.GetTargetSP()) &&
         "Swift scratch context not locked ahead of dynamic type resolution");
  llvm::Optional<SwiftScratchContextReader> maybe_scratch_ctx =
      in_value.GetSwiftScratchContext();
  if (!maybe_scratch_ctx)
    return false;
  CompilerType swift_type =
      maybe_scratch_ctx->get()->GetTypeFromMangledTypename(
          ConstString(remangled));

  // Roll back the ObjC dynamic type resolution.
  if (!swift_type)
    return false;
  class_type_or_name = dyn_class_type_or_name;
  class_type_or_name.SetCompilerType(swift_type);
  value_type = Value::ValueType::Scalar;
  return true;
}

static bool IsIndirectEnumCase(ValueObject &valobj) {
  return (valobj.GetLanguageFlags() &
          SwiftASTContext::LanguageFlags::eIsIndirectEnumCase) ==
         SwiftASTContext::LanguageFlags::eIsIndirectEnumCase;
}

static bool CouldHaveDynamicValue(ValueObject &in_value) {
  if (IsIndirectEnumCase(in_value))
    return true;
  CompilerType var_type(in_value.GetCompilerType());
  Flags var_type_flags(var_type.GetTypeInfo());
  if (var_type_flags.AllSet(eTypeIsSwift | eTypeInstanceIsPointer)) {
    // Swift class instances are actually pointers, but base class instances
    // are inlined at offset 0 in the class data. If we just let base classes
    // be dynamic, it would cause an infinite recursion. So we would usually
    // disable it.
    return !in_value.IsBaseClass();
  }
  return var_type.IsPossibleDynamicType(nullptr, false, false);
}

bool SwiftLanguageRuntimeImpl::GetDynamicTypeAndAddress(
    ValueObject &in_value, lldb::DynamicValueType use_dynamic,
    TypeAndOrName &class_type_or_name, Address &address,
    Value::ValueType &value_type) {
  class_type_or_name.Clear();
  if (use_dynamic == lldb::eNoDynamicValues)
    return false;

  LLDB_SCOPED_TIMER();

  // Try to import a Clang type into Swift.
  if (in_value.GetObjectRuntimeLanguage() == eLanguageTypeObjC)
    return GetDynamicTypeAndAddress_ClangType(
        in_value, use_dynamic, class_type_or_name, address, value_type);

  if (!CouldHaveDynamicValue(in_value))
    return false;

  CompilerType val_type(in_value.GetCompilerType());
  Flags type_info(val_type.GetTypeInfo());
  if (!type_info.AnySet(eTypeIsSwift))
    return false;

  bool success = false;
  bool is_indirect_enum_case = IsIndirectEnumCase(in_value);
  // Type kinds with instance metadata don't need generic type resolution.
  if (is_indirect_enum_case)
    success = GetDynamicTypeAndAddress_IndirectEnumCase(
        in_value, use_dynamic, class_type_or_name, address);
  else if (type_info.AnySet(eTypeIsClass) ||
           type_info.AllSet(eTypeIsBuiltIn | eTypeIsPointer | eTypeHasValue))
    success = GetDynamicTypeAndAddress_Class(in_value, val_type, use_dynamic,
                                             class_type_or_name, address);
  else if (type_info.AnySet(eTypeIsProtocol))
    success = GetDynamicTypeAndAddress_Protocol(in_value, val_type, use_dynamic,
                                                class_type_or_name, address);
  else {
    // Perform generic type resolution.
    StackFrameSP frame = in_value.GetExecutionContextRef().GetFrameSP();
    if (!frame)
      return false;

    CompilerType bound_type = BindGenericTypeParameters(*frame.get(), val_type);
    if (!bound_type)
      return false;

    Flags subst_type_info(bound_type.GetTypeInfo());
    if (subst_type_info.AnySet(eTypeIsClass)) {
      success = GetDynamicTypeAndAddress_Class(
          in_value, bound_type, use_dynamic, class_type_or_name, address);
    } else if (subst_type_info.AnySet(eTypeIsProtocol)) {
      success = GetDynamicTypeAndAddress_Protocol(
          in_value, bound_type, use_dynamic, class_type_or_name, address);
    } else {
      success = GetDynamicTypeAndAddress_Value(
          in_value, bound_type, use_dynamic, class_type_or_name, address);
    }
  }

  if (success)
    value_type = GetValueType(in_value, class_type_or_name.GetCompilerType(),
                              is_indirect_enum_case);
  return success;
}

TypeAndOrName SwiftLanguageRuntimeImpl::FixUpDynamicType(
    const TypeAndOrName &type_and_or_name, ValueObject &static_value) {
  CompilerType static_type = static_value.GetCompilerType();
  CompilerType dynamic_type = type_and_or_name.GetCompilerType();
  // The logic in this function only applies to static/dynamic Swift types.
  if (llvm::isa<TypeSystemClang>(static_type.GetTypeSystem()))
    return type_and_or_name;

  bool should_be_made_into_ref = false;
  bool should_be_made_into_ptr = false;
  Flags type_flags = static_type.GetTypeInfo();
  Flags type_andor_name_flags = dynamic_type.GetTypeInfo();

  // if the static type is a pointer or reference, so should the
  // dynamic type caveat: if the static type is a Swift class
  // instance, the dynamic type could either be a Swift type (no need
  // to change anything), or an ObjC type in which case it needs to be
  // made into a pointer
  if (type_flags.AnySet(eTypeIsPointer))
    should_be_made_into_ptr =
        (type_flags.AllClear(eTypeIsGenericTypeParam | eTypeIsBuiltIn) &&
         !IsIndirectEnumCase(static_value));
  else if (type_flags.AnySet(eTypeInstanceIsPointer))
    should_be_made_into_ptr = !type_andor_name_flags.AllSet(eTypeIsSwift);
  else if (type_flags.AnySet(eTypeIsReference))
    should_be_made_into_ref = true;
  else if (type_flags.AllSet(eTypeIsSwift | eTypeIsProtocol))
    should_be_made_into_ptr =
        dynamic_type.IsRuntimeGeneratedType() && !dynamic_type.IsPointerType();

  if (type_and_or_name.HasType()) {
    // The type will always be the type of the dynamic object.  If our
    // parent's type was a pointer, then our type should be a pointer
    // to the type of the dynamic object.  If a reference, then the
    // original type should be okay...
    CompilerType corrected_type = dynamic_type;
    if (should_be_made_into_ptr)
      corrected_type = dynamic_type.GetPointerType();
    else if (should_be_made_into_ref)
      corrected_type = dynamic_type.GetLValueReferenceType();
    TypeAndOrName result = type_and_or_name;
    result.SetCompilerType(corrected_type);
    return result;
  }
  return type_and_or_name;
}

bool SwiftLanguageRuntimeImpl::IsTaggedPointer(lldb::addr_t addr,
                                               CompilerType type) {
  swift::CanType swift_can_type = GetCanonicalSwiftType(type);
  if (!swift_can_type)
    return false;
  switch (swift_can_type->getKind()) {
  case swift::TypeKind::UnownedStorage: {
    Target &target = m_process.GetTarget();
    llvm::Triple triple = target.GetArchitecture().GetTriple();
    // On Darwin the Swift runtime stores unowned references to
    // Objective-C objects as a pointer to a struct that has the
    // actual object pointer at offset zero. The least significant bit
    // of the reference pointer indicates whether the reference refers
    // to an Objective-C or Swift object.
    //
    // This is a property of the Swift runtime(!). In the future it
    // may be necessary to check for the version of the Swift runtime
    // (or indirectly by looking at the version of the remote
    // operating system) to determine how to interpret references.
    if (triple.isOSDarwin())
      // Check whether this is a reference to an Objective-C object.
      if ((addr & 1) == 1)
        return true;
  } break;
  default:
    break;
  }
  return false;
}

std::pair<lldb::addr_t, bool>
SwiftLanguageRuntimeImpl::FixupPointerValue(lldb::addr_t addr,
                                            CompilerType type) {
  // Check for an unowned Darwin Objective-C reference.
  if (IsTaggedPointer(addr, type)) {
    // Clear the discriminator bit to get at the pointer to Objective-C object.
    bool needs_deref = true;
    return {addr & ~1ULL, needs_deref};
  }

  // Adjust the pointer to strip away the spare bits.
  Target &target = m_process.GetTarget();
  llvm::Triple triple = target.GetArchitecture().GetTriple();
  switch (triple.getArch()) {
  case llvm::Triple::ArchType::aarch64:
    return {addr & ~SWIFT_ABI_ARM64_SWIFT_SPARE_BITS_MASK, false};
  case llvm::Triple::ArchType::arm:
    return {addr & ~SWIFT_ABI_ARM_SWIFT_SPARE_BITS_MASK, false};
  case llvm::Triple::ArchType::x86:
    return {addr & ~SWIFT_ABI_I386_SWIFT_SPARE_BITS_MASK, false};
  case llvm::Triple::ArchType::x86_64:
    return {addr & ~SWIFT_ABI_X86_64_SWIFT_SPARE_BITS_MASK, false};
  case llvm::Triple::ArchType::systemz:
    return {addr & ~SWIFT_ABI_S390X_SWIFT_SPARE_BITS_MASK, false};
  case llvm::Triple::ArchType::ppc64le:
    return {addr & ~SWIFT_ABI_POWERPC64_SWIFT_SPARE_BITS_MASK, false};
  default:
    break;
  }
  return {addr, false};
}

lldb::addr_t SwiftLanguageRuntimeImpl::FixupAddress(lldb::addr_t addr,
                                                    CompilerType type,
                                                    Status &error) {
  swift::CanType swift_can_type = GetCanonicalSwiftType(type);
  if (!swift_can_type)
    return addr;
  switch (swift_can_type->getKind()) {
  case swift::TypeKind::UnownedStorage: {
    // Peek into the reference to see whether it needs an extra deref.
    // If yes, return the fixed-up address we just read.
    Target &target = m_process.GetTarget();
    size_t ptr_size = m_process.GetAddressByteSize();
    lldb::addr_t refd_addr = LLDB_INVALID_ADDRESS;
    target.ReadMemory(addr, &refd_addr, ptr_size, error, true);
    if (error.Success()) {
      bool extra_deref;
      std::tie(refd_addr, extra_deref) = FixupPointerValue(refd_addr, type);
      if (extra_deref)
        return refd_addr;
    }
  } break;
  default:
    break;
  }
  return addr;
}

const swift::reflection::TypeRef *
SwiftLanguageRuntimeImpl::GetTypeRef(CompilerType type,
                                     TypeSystemSwiftTypeRef *module_holder) {
  // Demangle the mangled name.
  swift::Demangle::Demangler dem;
  ConstString mangled_name = type.GetMangledTypeName();
  auto *ts = llvm::dyn_cast_or_null<TypeSystemSwift>(type.GetTypeSystem());
  if (!ts)
    return nullptr;
  swift::Demangle::NodePointer node =
      module_holder->GetCanonicalDemangleTree(dem, mangled_name.GetStringRef());
  if (!node)
    return nullptr;

  // Build a TypeRef.
  auto *reflection_ctx = GetReflectionContext();
  if (!reflection_ctx)
    return nullptr;

  auto type_ref_or_err =
    swift::Demangle::decodeMangledType(reflection_ctx->getBuilder(), node);
  if (type_ref_or_err.isError())
    return nullptr;
  const swift::reflection::TypeRef *type_ref = type_ref_or_err.getType();
  return type_ref;
}

const swift::reflection::TypeInfo *
SwiftLanguageRuntimeImpl::GetSwiftRuntimeTypeInfo(
    CompilerType type, ExecutionContextScope *exe_scope,
    swift::reflection::TypeRef const **out_tr) {
  auto *ts = llvm::dyn_cast_or_null<TypeSystemSwift>(type.GetTypeSystem());
  if (!ts)
    return nullptr;

  // Resolve all type aliases.
  type = type.GetCanonicalType();

  // Resolve all generic type parameters in the type for the current
  // frame. Generic parameter binding has to happen in the scratch
  // context, so we lock it while we are in this function.
  std::unique_ptr<SwiftScratchContextLock> lock;
  if (exe_scope)
    if (StackFrame *frame = exe_scope->CalculateStackFrame().get()) {
      ExecutionContext exe_ctx;
      frame->CalculateExecutionContext(exe_ctx);
      lock = std::make_unique<SwiftScratchContextLock>(&exe_ctx);
      type = BindGenericTypeParameters(*frame, type);
    }

  // BindGenericTypeParameters imports the type into the scratch
  // context, but we need to resolve (any DWARF links in) the typeref
  // in the original module.
  const swift::reflection::TypeRef *type_ref =
      GetTypeRef(type, &ts->GetTypeSystemSwiftTypeRef());
  if (!type_ref)
    return nullptr;

  if (out_tr)
    *out_tr = type_ref;

  auto *reflection_ctx = GetReflectionContext();
  if (!reflection_ctx)
    return nullptr;

  LLDBTypeInfoProvider provider(*this, *ts);
  return reflection_ctx->getTypeInfo(type_ref, &provider);
}

bool SwiftLanguageRuntimeImpl::IsStoredInlineInBuffer(CompilerType type) {
  if (auto *type_info = GetSwiftRuntimeTypeInfo(type, nullptr))
    return type_info->isBitwiseTakable() && type_info->getSize() <= 24;
  return true;
}

llvm::Optional<uint64_t>
SwiftLanguageRuntimeImpl::GetBitSize(CompilerType type,
                                     ExecutionContextScope *exe_scope) {
  if (auto *type_info = GetSwiftRuntimeTypeInfo(type, exe_scope))
    return type_info->getSize() * 8;
  return {};
}

llvm::Optional<uint64_t>
SwiftLanguageRuntimeImpl::GetByteStride(CompilerType type) {
  if (auto *type_info = GetSwiftRuntimeTypeInfo(type, nullptr))
    return type_info->getStride();
  return {};
}

llvm::Optional<size_t>
SwiftLanguageRuntimeImpl::GetBitAlignment(CompilerType type,
                                          ExecutionContextScope *exe_scope) {
  if (auto *type_info = GetSwiftRuntimeTypeInfo(type, exe_scope))
    return type_info->getAlignment() * 8;
  return {};
}

bool SwiftLanguageRuntime::IsAllowedRuntimeValue(ConstString name) {
  return name.GetStringRef() == "self";
}

bool SwiftLanguageRuntime::CouldHaveDynamicValue(ValueObject &in_value) {
  return ::CouldHaveDynamicValue(in_value);
}

CompilerType
SwiftLanguageRuntimeImpl::GetConcreteType(ExecutionContextScope *exe_scope,
                                          ConstString abstract_type_name) {
  if (!exe_scope)
    return CompilerType();

  StackFrame *frame(exe_scope->CalculateStackFrame().get());
  if (!frame)
    return CompilerType();

  SwiftLanguageRuntime::MetadataPromiseSP promise_sp(
      GetPromiseForTypeNameAndFrame(abstract_type_name.GetCString(), frame));
  if (!promise_sp)
    return CompilerType();

  return promise_sp->FulfillTypePromise();
}

} // namespace lldb_private
