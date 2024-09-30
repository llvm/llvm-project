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
#include "ReflectionContextInterface.h"
#include "SwiftLanguageRuntime.h"
#include "SwiftLanguageRuntimeImpl.h"
#include "SwiftMetadataCache.h"

#include "Plugins/ExpressionParser/Clang/ClangUtil.h"
#include "Plugins/Language/Swift/LogChannelSwift.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "Plugins/TypeSystem/Swift/SwiftDemangle.h"
#include "lldb/Core/ValueObjectMemory.h"
#include "lldb/Host/SafeMachO.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/ProcessStructReader.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Timer.h"
#include "llvm/ADT/STLExtras.h"

#include "swift/AST/ASTContext.h"
#include "swift/AST/ASTMangler.h"
#include "swift/AST/ASTWalker.h"
#include "swift/Demangling/Demangle.h"
#include "swift/RemoteInspection/ReflectionContext.h"
#include "swift/RemoteInspection/TypeRefBuilder.h"
#include "swift/Strings.h"

#include <sstream>

#define HEALTH_LOG(FMT, ...)                                                   \
  do {                                                                         \
    LLDB_LOG(GetLog(LLDBLog::Types), FMT, ##__VA_ARGS__);                      \
    LLDB_LOG(lldb_private::GetSwiftHealthLog(), FMT, ##__VA_ARGS__);           \
  } while (0)

using namespace lldb;
using namespace lldb_private;

namespace lldb_private {

std::string toString(const swift::reflection::TypeRef *tr) {
  if (!tr)
    return "null typeref";
  std::stringstream s;
  tr->dump(s);
  return s.str();
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
    lldb::addr_t addr, TypeSystemSwift::NonTriviallyManagedReferenceKind kind) {

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

  if (kind == TypeSystemSwift::NonTriviallyManagedReferenceKind::eWeak) {
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

const CompilerType &SwiftLanguageRuntimeImpl::GetBoxMetadataType() {
  if (m_box_metadata_type.IsValid())
    return m_box_metadata_type;

  static ConstString g_type_name("__lldb_autogen_boxmetadata");
  const bool is_packed = false;
  if (TypeSystemClangSP clang_ts_sp =
          ScratchTypeSystemClang::GetForTarget(m_process.GetTarget())) {
    CompilerType voidstar =
        clang_ts_sp->GetBasicType(lldb::eBasicTypeVoid).GetPointerType();
    CompilerType uint32 = clang_ts_sp->GetIntTypeFromBitSize(32, false);

    m_box_metadata_type = clang_ts_sp->GetOrCreateStructForIdentifier(
        g_type_name, {{"kind", voidstar}, {"offset", uint32}}, is_packed);
  }

  return m_box_metadata_type;
}

std::shared_ptr<LLDBMemoryReader>
SwiftLanguageRuntimeImpl::GetMemoryReader() {
  if (!m_memory_reader_sp) {
    m_memory_reader_sp.reset(new LLDBMemoryReader(
        m_process, [&](swift::remote::RemoteAbsolutePointer pointer) {
          ThreadSafeReflectionContext reflection_context =
              GetReflectionContext();
          if (!reflection_context)
            return pointer;
          return reflection_context->StripSignedPointer(pointer);
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

class LLDBTypeInfoProvider : public swift::remote::TypeInfoProvider {
  SwiftLanguageRuntimeImpl &m_runtime;
  Status m_error;
  std::optional<SwiftScratchContextReader> m_reader;

public:
  LLDBTypeInfoProvider(SwiftLanguageRuntimeImpl &runtime,
                       ExecutionContextScope *exe_scope)
      : m_runtime(runtime),
        m_reader(m_runtime.GetProcess().GetTarget().GetSwiftScratchContext(
            m_error,
            exe_scope ? *exe_scope : m_runtime.GetProcess().GetTarget())) {}
  LLDBTypeInfoProvider(SwiftLanguageRuntimeImpl &runtime,
                       ExecutionContext *exe_ctx)
      : m_runtime(runtime),
        m_reader(m_runtime.GetProcess().GetTarget().GetSwiftScratchContext(
            m_error, exe_ctx ? *exe_ctx->GetBestExecutionContextScope()
                             : m_runtime.GetProcess().GetTarget())) {}

  swift::remote::TypeInfoProvider::IdType getId() override {
    if (m_reader)
      return (void *)((char *)m_reader->get() +
                      m_reader->get()->GetGeneration() +
                      m_runtime.GetGeneration());
    return (void *)0;
  }

  const swift::reflection::TypeInfo *
  getTypeInfo(llvm::StringRef mangledName) override {
    // TODO: Should we cache the mangled name -> compiler type lookup, too?
    LLDB_LOG(GetLog(LLDBLog::Types),
             "[LLDBTypeInfoProvider] Looking up debug type info for {0}",
             mangledName);

    if (!m_reader) {
      LLDB_LOG(GetLog(LLDBLog::Types),
               "[LLDBTypeInfoProvider] no scratch context");
      return nullptr;
    }
    TypeSystemSwiftTypeRef &typesystem = *m_reader->get();

    // Materialize a Clang type from the debug info.
    assert(swift::Demangle::getManglingPrefixLength(mangledName) == 0);
    std::string wrapped;
    // The mangled name passed in is bare. Add global prefix ($s) and type (D).
    llvm::raw_string_ostream(wrapped) << "$s" << mangledName << 'D';
    swift::Demangle::Demangler dem;
    auto *node = dem.demangleSymbol(wrapped);
    if (!node) {
      // Try `mangledName` as plain ObjC class name. Ex: NSObject, NSView, etc.
      auto maybeMangled = swift_demangle::mangleClass(
          dem, swift::MANGLING_MODULE_OBJC, mangledName);
      if (!maybeMangled.isSuccess()) {
        LLDB_LOG(GetLog(LLDBLog::Types),
                 "[LLDBTypeInfoProvider] invalid mangled name: {0}",
                 mangledName);
        return nullptr;
      }
      wrapped = maybeMangled.result();
      LLDB_LOG(GetLog(LLDBLog::Types),
               "[LLDBTypeInfoProvider] using mangled ObjC class name: {0}",
               wrapped);
    } else {
#ifndef NDEBUG
      // Check that our hardcoded mangling wrapper is still up-to-date.
      assert(node->getKind() == swift::Demangle::Node::Kind::Global);
      assert(node->getNumChildren() == 1);
      node = node->getChild(0);
      assert(node->getKind() == swift::Demangle::Node::Kind::TypeMangling);
      assert(node->getNumChildren() == 1);
      node = node->getChild(0);
      assert(node->getKind() == swift::Demangle::Node::Kind::Type);
      assert(node->getNumChildren() == 1);
      node = node->getChild(0);
      assert(node->getKind() != swift::Demangle::Node::Kind::Type);
#endif
    }

    ConstString mangled(wrapped);
    CompilerType swift_type = typesystem.GetTypeFromMangledTypename(mangled);
    auto ts = swift_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
    if (!ts)
      return nullptr;
    bool is_imported = true;
    CompilerType clang_type =
        m_runtime.LookupAnonymousClangType(mangled.AsCString());
    if (!clang_type)
      is_imported =
          ts->IsImportedType(swift_type.GetOpaqueQualType(), &clang_type);
    if (!is_imported || !clang_type) {
      HEALTH_LOG("[LLDBTypeInfoProvider] Could not find clang debug type info "
                 "for {0}",
                 mangledName);
      return nullptr;
    }

    return GetOrCreateTypeInfo(clang_type);
  }

  const swift::reflection::TypeInfo *
  GetOrCreateTypeInfo(CompilerType clang_type) {
    if (auto ti = m_runtime.lookupClangTypeInfo(clang_type))
      return *ti;

    auto &process = m_runtime.GetProcess();
    ExecutionContext exe_ctx;
    process.CalculateExecutionContext(exe_ctx);
    auto *exe_scope = exe_ctx.GetBestExecutionContextScope();
    TypeSystemSwiftTypeRef &typesystem = *m_reader->get();
    // Build a TypeInfo for the Clang type.
    auto size = clang_type.GetByteSize(exe_scope);
    auto bit_align = clang_type.GetTypeBitAlign(exe_scope);
    std::vector<swift::reflection::FieldInfo> fields;
    if (clang_type.IsAggregateType()) {
      // Recursively collect TypeInfo records for all fields.
      for (uint32_t i = 0, e = clang_type.GetNumFields(&exe_ctx); i != e; ++i) {
        std::string name;
        uint64_t bit_offset_ptr = 0;
        uint32_t bitfield_bit_size_ptr = 0;
        bool is_bitfield_ptr = false;
        CompilerType field_type = clang_type.GetFieldAtIndex(
            i, name, &bit_offset_ptr, &bitfield_bit_size_ptr, &is_bitfield_ptr);
        if (is_bitfield_ptr) {
          LLDB_LOG(
              GetLog(LLDBLog::Types),
              "[LLDBTypeInfoProvider] bitfield support is not yet implemented");
          continue;
        }
        CompilerType swift_type;
        if (field_type.IsAnonymousType()) {
          // Store anonymous tuples in a side table, to solve the
          // problem that they cannot be looked up by name.
          static unsigned m_num_anonymous_clang_types = 0;
          std::string unique_swift_name(field_type.GetTypeName());
          llvm::raw_string_ostream(unique_swift_name)
              << '#' << m_num_anonymous_clang_types++;
          swift_type = typesystem.CreateClangStructType(unique_swift_name);
          auto *key = swift_type.GetMangledTypeName().AsCString();
          m_runtime.RegisterAnonymousClangType(key, field_type);
        } else {
          swift_type = typesystem.ConvertClangTypeToSwiftType(field_type);
        }
        auto *typeref = m_runtime.GetTypeRef(swift_type, &typesystem);
        swift::reflection::FieldInfo field_info = {
            name, (unsigned)bit_offset_ptr / 8, 0, typeref,
            *GetOrCreateTypeInfo(field_type)};
        fields.push_back(field_info);
        }
    }
    return m_runtime.emplaceClangTypeInfo(clang_type, size, bit_align, fields);
  }
};

void SwiftLanguageRuntimeImpl::RegisterAnonymousClangType(
    const char *key, CompilerType clang_type) {
  const std::lock_guard<std::recursive_mutex> locker(m_clang_type_info_mutex);
  m_anonymous_clang_types.insert({key, clang_type});
}

CompilerType
SwiftLanguageRuntimeImpl::LookupAnonymousClangType(const char *key) {
  const std::lock_guard<std::recursive_mutex> locker(m_clang_type_info_mutex);
  return m_anonymous_clang_types.lookup(key);
}

std::optional<const swift::reflection::TypeInfo *>
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
    CompilerType clang_type, std::optional<uint64_t> byte_size,
    std::optional<size_t> bit_align,
    llvm::ArrayRef<swift::reflection::FieldInfo> fields) {
  const std::lock_guard<std::recursive_mutex> locker(m_clang_type_info_mutex);
  if (!byte_size || !bit_align) {
    m_clang_type_info.insert({clang_type.GetOpaqueQualType(), std::nullopt});
    return nullptr;
  }
  assert(*bit_align % 8 == 0 && "Bit alignment no a multiple of 8!");
  auto byte_align = *bit_align / 8;
  // The stride is the size rounded up to alignment.
  const size_t byte_stride = llvm::alignTo(*byte_size, byte_align);
  unsigned extra_inhabitants = 0;
  if (clang_type.IsPointerType())
    extra_inhabitants = swift::swift_getHeapObjectExtraInhabitantCount();

  if (fields.empty()) {
    auto it_b = m_clang_type_info.insert(
        {clang_type.GetOpaqueQualType(),
         swift::reflection::TypeInfo(swift::reflection::TypeInfoKind::Builtin,
                                     *byte_size, byte_align, byte_stride,
                                     extra_inhabitants, true)});
    return &*it_b.first->second;
  }
  auto it_b = m_clang_record_type_info.insert(
      {clang_type.GetOpaqueQualType(),
       swift::reflection::RecordTypeInfo(
           *byte_size, byte_align, byte_stride, extra_inhabitants, false,
           swift::reflection::RecordKind::Struct, fields)});
  return &*it_b.first->second;
}

std::optional<uint64_t>
SwiftLanguageRuntimeImpl::GetMemberVariableOffsetRemoteMirrors(
    CompilerType instance_type, ValueObject *instance,
    llvm::StringRef member_name, Status *error) {
  LLDB_LOG(GetLog(LLDBLog::Types), "using remote mirrors");
  auto ts =
      instance_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwiftTypeRef>();
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
  std::optional<uint64_t> result;
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

std::optional<uint64_t> SwiftLanguageRuntimeImpl::GetMemberVariableOffset(
    CompilerType instance_type, ValueObject *instance,
    llvm::StringRef member_name, Status *error) {
  LLDB_SCOPED_TIMER();
  std::optional<uint64_t> offset;

  if (!instance_type.IsValid())
    return {};

  LLDB_LOG(GetLog(LLDBLog::Types),
           "[GetMemberVariableOffset] asked to resolve offset for member {0}",
           member_name);

  // Using the module context for RemoteAST is cheaper but only safe
  // when there is no dynamic type resolution involved.
  // If this is already in the expression context, ask RemoteAST.
  if (instance_type.GetTypeSystem().isa_and_nonnull<SwiftASTContext>())
    offset =
        GetMemberVariableOffsetRemoteAST(instance_type, instance, member_name);
  if (!offset) {
    // Convert to a TypeRef-type, if necessary.
    if (auto module_ctx =
            instance_type.GetTypeSystem().dyn_cast_or_null<SwiftASTContext>())
      instance_type =
          module_ctx->GetTypeRefType(instance_type.GetOpaqueQualType());

    offset = GetMemberVariableOffsetRemoteMirrors(instance_type, instance,
                                                  member_name, error);
#ifndef NDEBUG
    if (ModuleList::GetGlobalModuleListProperties()
            .GetSwiftValidateTypeSystem()) {
      // Convert to an AST type, if necessary.
      if (auto ts = instance_type.GetTypeSystem()
                        .dyn_cast_or_null<TypeSystemSwiftTypeRef>()) {
        ExecutionContext exe_ctx = instance->GetExecutionContextRef().Lock(false);
        instance_type = ts->ReconstructType(instance_type, &exe_ctx);
      }
      auto reference = GetMemberVariableOffsetRemoteAST(instance_type, instance,
                                                        member_name);
      if (reference.has_value() && offset != reference) {
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
    LLDB_LOG(GetLog(LLDBLog::Types),
             "[GetMemberVariableOffset] offset of {0} is {1}",
             member_name, *offset);
  } else {
    LLDB_LOG(GetLog(LLDBLog::Types), "[GetMemberVariableOffset] failed for {0}",
             member_name);
    if (error)
      error->SetErrorStringWithFormat("could not resolve member offset");
  }
  return offset;
}

namespace {
CompilerType GetWeakReferent(TypeSystemSwiftTypeRef &ts, CompilerType type) {
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

CompilerType GetTypeFromTypeRef(TypeSystemSwiftTypeRef &ts,
                                const swift::reflection::TypeRef *type_ref) {
  if (!type_ref)
    return {};
  swift::Demangle::Demangler dem;
  swift::Demangle::NodePointer node = type_ref->getDemangling(dem);
  return ts.RemangleAsType(dem, node);
}

struct ExistentialSyntheticChild {
  std::string name;
  std::function<CompilerType(void)> get_type;
};

/// Return the synthetic children of an Existential type.
/// The closure in get_type will depend on ts and tr.
/// Roughly corresponds to GetExistentialTypeChild() in SwiftASTContext.cpp
llvm::SmallVector<ExistentialSyntheticChild, 4>
GetExistentialSyntheticChildren(std::shared_ptr<TypeSystemSwiftTypeRef> ts,
                                const swift::reflection::TypeRef *tr,
                                const swift::reflection::TypeInfo *ti) {
  llvm::SmallVector<ExistentialSyntheticChild, 4> children;
  if (!ts)
    return children;
  auto *protocol_composition_tr =
      llvm::dyn_cast<swift::reflection::ProtocolCompositionTypeRef>(tr);
  if (!protocol_composition_tr)
    return children;
  if (ti && (llvm::isa<swift::reflection::ReferenceTypeInfo>(ti) ||
             llvm::isa<swift::reflection::RecordTypeInfo>(ti))) {
    children.push_back({"object", [=]() {
                          if (auto *super_class_tr =
                                  protocol_composition_tr->getSuperclass())
                            return GetTypeFromTypeRef(*ts, super_class_tr);
                          else
                            return ts->GetRawPointerType();
    }});
  }
  assert(children.size());
  return children;
}

/// Log the fact that a type kind is not supported.
void LogUnimplementedTypeKind(const char *function, CompilerType type) {
  // When running the test suite assert that all cases are covered.
  HEALTH_LOG("{0}: unimplemented type info in {1}", type.GetMangledTypeName(),
             function);
#ifndef NDEBUG
  llvm::dbgs() << function << ": unimplemented type info in"
               << type.GetMangledTypeName() << "\n";
  if (ModuleList::GetGlobalModuleListProperties().GetSwiftValidateTypeSystem())
    assert(false && "not implemented");
#endif
}

} // namespace

llvm::Expected<uint32_t>
SwiftLanguageRuntimeImpl::GetNumChildren(CompilerType type,
                                         ExecutionContextScope *exe_scope) {
  LLDB_SCOPED_TIMER();

  auto ts = type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwiftTypeRef>();
  if (!ts)
    return llvm::createStringError("no Swift typesystem");
  if (!type)
    return llvm::createStringError("invalid type");

  // Deal with the LLDB-only SILPackType variant.
  if (auto pack_type = ts->IsSILPackType(type))
    if (pack_type->expanded)
      return pack_type->count;

  // Deal with Clang types.
  {
    CompilerType clang_type =
        LookupAnonymousClangType(type.GetMangledTypeName().AsCString());
    if (!clang_type)
      ts->IsImportedType(type.GetOpaqueQualType(), &clang_type);
    if (clang_type) {
      bool is_signed;
      if (clang_type.IsEnumerationType(is_signed))
        return 1;
      ExecutionContext exe_ctx;
      if (exe_scope)
        exe_scope->CalculateExecutionContext(exe_ctx);
      return clang_type.GetNumChildren(true, exe_scope ? &exe_ctx : nullptr);
    }
  }

  // Try the static type metadata.
  const swift::reflection::TypeRef *tr = nullptr;
  auto *ti = GetSwiftRuntimeTypeInfo(type, exe_scope, &tr);
  if (!ti)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "could not get Swift runtime type info for type " +
            type.GetMangledTypeName().GetString());
  if (llvm::isa<swift::reflection::BuiltinTypeInfo>(ti)) {
    // This logic handles Swift Builtin types. By handling them now, the cost of
    // unnecessarily loading ASTContexts can be avoided. Builtin types are
    // assumed to be internal "leaf" types, having no children. Or,
    // alternatively, opaque types.
    //
    // However, some imported Clang types (specifically enums) will also produce
    // `BuiltinTypeInfo` instances. These types are not to be handled here.
    LLDB_LOG(GetLog(LLDBLog::Types),
             "{0}: unrecognized builtin type info or this is a Clang type "
             "without DWARF debug info",
             type.GetMangledTypeName());
    return llvm::createStringError("missing debug info for Clang type \"" +
                                   type.GetDisplayTypeName().GetString() +
                                   "\"");
  }
  // Structs and Tuples.
  if (auto *rti = llvm::dyn_cast<swift::reflection::RecordTypeInfo>(ti)) {
    LLDB_LOG(GetLog(LLDBLog::Types), "{0}: RecordTypeInfo(num_fields={1})",
             type.GetMangledTypeName(), rti->getNumFields());
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
    LLDB_LOG(GetLog(LLDBLog::Types), "{0}: EnumTypeInfo(num_payload_cases={1})",
             type.GetMangledTypeName(), eti->getNumPayloadCases());
    return eti->getNumPayloadCases();
  }
  // Objects.
  if (auto *rti = llvm::dyn_cast<swift::reflection::ReferenceTypeInfo>(ti)) {
    LLDB_LOG(GetLog(LLDBLog::Types), "{0}: ReferenceTypeInfo()",
             type.GetMangledTypeName().GetCString());
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
      return llvm::createStringError("could not typeref for " +
                                     type.GetMangledTypeName().GetString());

    // Existentials.
    if (size_t n = GetExistentialSyntheticChildren(ts, tr, ti).size())
      return n;

    ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
    if (!reflection_ctx)
      return llvm::createStringError("no reflection context");

    LLDBTypeInfoProvider tip(*this, exe_scope);
    auto *cti = reflection_ctx->GetClassInstanceTypeInfo(
        tr, &tip, ts->GetDescriptorFinder());
    if (auto *rti =
            llvm::dyn_cast_or_null<swift::reflection::RecordTypeInfo>(cti)) {
      LLDB_LOG(GetLog(LLDBLog::Types),
               "{0}: class RecordTypeInfo(num_fields={1})",
               type.GetMangledTypeName(), rti->getNumFields());

      // The superclass, if any, is an extra child.
      if (reflection_ctx->LookupSuperclass(tr,
                                           ts->GetDescriptorFinder()))
        return rti->getNumFields() + 1;
      return rti->getNumFields();
    }
    return llvm::createStringError("No Swift runtime type info for " +
                                   type.GetMangledTypeName().GetString());
  }

  LogUnimplementedTypeKind(__FUNCTION__, type);
  return llvm::createStringError("GetNumChildren unimplemented for type " +
                                 type.GetMangledTypeName().GetString());
}

std::optional<unsigned>
SwiftLanguageRuntimeImpl::GetNumFields(CompilerType type,
                                       ExecutionContext *exe_ctx) {
  auto ts = type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwiftTypeRef>();
  if (!ts)
    return {};

  using namespace swift::reflection;
  // Try the static type metadata.
  const TypeRef *tr = nullptr;
  auto *ti = GetSwiftRuntimeTypeInfo(
      type, exe_ctx ? exe_ctx->GetBestExecutionContextScope() : nullptr, &tr);
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
      ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
      if (!reflection_ctx)
        return {};

      LLDBTypeInfoProvider tip(*this, exe_ctx);
      auto *cti = reflection_ctx->GetClassInstanceTypeInfo(
          tr, &tip, ts->GetDescriptorFinder());
      if (auto *rti = llvm::dyn_cast_or_null<RecordTypeInfo>(cti)) {
        return rti->getNumFields();
      }

      return {};
    }
  }
  default:
    LogUnimplementedTypeKind(__FUNCTION__, type);
    return {};
  }
}

static std::pair<SwiftLanguageRuntime::LookupResult, std::optional<size_t>>
findFieldWithName(const std::vector<swift::reflection::FieldInfo> &fields,
                  const swift::reflection::TypeRef *tr, llvm::StringRef name,
                  bool is_enum, std::vector<uint32_t> &child_indexes,
                  uint32_t offset = 0) {
  uint32_t index = 0;
  uint32_t name_as_index = 0;
  bool name_is_index = false;
  auto *tuple_tr = llvm::dyn_cast<swift::reflection::TupleTypeRef>(tr);
  if (tuple_tr)
    name_is_index = !name.getAsInteger(10, name_as_index);
  bool is_nonpayload_enum_case = false;

  auto it = std::find_if(fields.begin(), fields.end(), [&](const auto &field) {
    if (name_is_index && name_as_index == index)
      return true;

    // In some situations the cached TI for a tuple type is missing the names,
    // but the type_ref has them.
    StringRef field_name = field.Name;
    if (!field.Name.size())
      if (tuple_tr && tuple_tr->getLabels().size() > index)
        field_name = tuple_tr->getLabels().at(index);
    if (name != field_name) {
      // A nonnull TypeRef is required for enum cases, where it represents cases
      // that have a payload. In other types it will be true anyway.
      if (!is_enum || field.TR)
        ++index;
      return false;
    }
    if (is_enum)
      is_nonpayload_enum_case = (field.TR == nullptr);
    return true;
  });

  // Not found.
  if (it == fields.end())
    return {SwiftLanguageRuntime::eNotFound, {}};
  // Found, but no index to report.
  if (is_nonpayload_enum_case)
    return {SwiftLanguageRuntime::eFound, {}};
  child_indexes.push_back(offset + index);
  return {SwiftLanguageRuntime::eFound, child_indexes.size()};
}

std::optional<std::string> SwiftLanguageRuntimeImpl::GetEnumCaseName(
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

  LogUnimplementedTypeKind(__FUNCTION__, type);
  return {};
}

std::pair<SwiftLanguageRuntime::LookupResult, std::optional<size_t>>
SwiftLanguageRuntimeImpl::GetIndexOfChildMemberWithName(
    CompilerType type, llvm::StringRef name, ExecutionContext *exe_ctx,
    bool omit_empty_base_classes, std::vector<uint32_t> &child_indexes) {
  LLDB_SCOPED_TIMER();
  auto ts = type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwiftTypeRef>();
  if (!ts)
    return {SwiftLanguageRuntime::eError, {}};

  using namespace swift::reflection;
  // Try the static type metadata.
  const TypeRef *tr = nullptr;
  auto *ti = GetSwiftRuntimeTypeInfo(type, exe_ctx->GetFramePtr(), &tr);
  if (!ti)
    return {SwiftLanguageRuntime::eError, {}};
  switch (ti->getKind()) {
  case TypeInfoKind::Record: {
    // Structs and Tuples.
    auto *rti = llvm::cast<RecordTypeInfo>(ti);
    switch (rti->getRecordKind()) {
    case RecordKind::ExistentialMetatype:
    case RecordKind::ThickFunction:
      // There are two fields, `function` and `context`, but they're not exposed
      // by lldb.
      return {SwiftLanguageRuntime::eFound, {0}};
    case RecordKind::OpaqueExistential:
      // `OpaqueExistential` is documented as:
      //     An existential is a three-word buffer followed by value metadata...
      // The buffer is exposed as children named `payload_data_{0,1,2}`, and
      // the number of fields are increased to match.
      if (name.starts_with("payload_data_")) {
        uint32_t index;
        if (name.take_back().getAsInteger(10, index) && index < 3) {
          child_indexes.push_back(index);
          return {SwiftLanguageRuntime::eFound, child_indexes.size()};
        }
      }
      return findFieldWithName(rti->getFields(), tr, name, false, child_indexes,
                               3);
    default:
      return findFieldWithName(rti->getFields(), tr, name, false, child_indexes);
    }
  }
  case TypeInfoKind::Enum: {
    auto *eti = llvm::cast<EnumTypeInfo>(ti);
    return findFieldWithName(eti->getCases(), tr, name, true, child_indexes);
  }
  case TypeInfoKind::Reference: {
    // Objects.
    auto *rti = llvm::cast<ReferenceTypeInfo>(ti);
    switch (rti->getReferenceKind()) {
    case ReferenceKind::Weak:
    case ReferenceKind::Unowned:
    case ReferenceKind::Unmanaged:
      // Weak references are implicitly optional.
      child_indexes.push_back(0);
      if (name == "some")
        return {SwiftLanguageRuntime::eFound, child_indexes.size()};
      return GetIndexOfChildMemberWithName(GetWeakReferent(*ts, type), name,
                                           exe_ctx, omit_empty_base_classes,
                                           child_indexes);
    case ReferenceKind::Strong: {
      ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
      if (!reflection_ctx)
        return {SwiftLanguageRuntime::eError, {}};

      size_t idx = 0;
      for (auto &protocol_child : GetExistentialSyntheticChildren(ts, tr, ti)) {
        if (protocol_child.name == name) {
          child_indexes.push_back(idx);
          return {SwiftLanguageRuntime::eFound, child_indexes.size()};
        }
        ++idx;
      }

      LLDBTypeInfoProvider tip(*this, exe_ctx);
      // `current_tr` iterates the class hierarchy, from the current class, each
      // superclass, and ends on null.
      auto *current_tr = tr;
      while (current_tr) {
        auto *record_ti = llvm::dyn_cast_or_null<RecordTypeInfo>(
            reflection_ctx->GetClassInstanceTypeInfo(
                current_tr, &tip, ts->GetDescriptorFinder()));
        if (!record_ti) {
          child_indexes.clear();
          return {SwiftLanguageRuntime::eError, {}};
        }
        auto *super_tr = reflection_ctx->LookupSuperclass(
            current_tr, ts->GetDescriptorFinder());
        uint32_t offset = super_tr ? 1 : 0;
        auto found_size = findFieldWithName(record_ti->getFields(), current_tr,
                                            name, false, child_indexes, offset);
        if (found_size.first == SwiftLanguageRuntime::eError ||
            found_size.first == SwiftLanguageRuntime::eFound)
          return found_size;
        current_tr = super_tr;
        child_indexes.push_back(0);
      }
      child_indexes.clear();
      return {SwiftLanguageRuntime::eNotFound, {}};
    }
    }
  }
  case TypeInfoKind::Builtin: {
    // Clang enums have an artificial rawValue property.
    CompilerType clang_type;
    if (ts->IsImportedType(type.GetOpaqueQualType(), &clang_type)) {
      bool is_signed;
      if (clang_type.IsEnumerationType(is_signed) && name == "rawValue") {
        child_indexes.push_back(0);
        return {SwiftLanguageRuntime::eFound, {1}};
      }
      // ReflectionContext returns unknown types as
      // Builtins. ClangImported builtins are rare, and if we ask for
      // a child of a builtin, most likely this means this is a type
      // we couldn't find in DWARF.
      return {SwiftLanguageRuntime::eError, {0}};
    }
    // SIMD types have an artificial _value.
    if (name == "_value" && TypeSystemSwiftTypeRef::IsSIMDType(type)) {
      child_indexes.push_back(0);
      return {SwiftLanguageRuntime::eFound, {1}};
    }
    return {SwiftLanguageRuntime::eNotFound, {0}};
  }
  default:
    LogUnimplementedTypeKind(__FUNCTION__, type);
    return {SwiftLanguageRuntime::eError, {}};
  }
}

llvm::Expected<CompilerType>
SwiftLanguageRuntimeImpl::GetChildCompilerTypeAtIndex(
    CompilerType type, size_t idx, bool transparent_pointers,
    bool omit_empty_base_classes, bool ignore_array_bounds,
    std::string &child_name, uint32_t &child_byte_size,
    int32_t &child_byte_offset, uint32_t &child_bitfield_bit_size,
    uint32_t &child_bitfield_bit_offset, bool &child_is_base_class,
    bool &child_is_deref_of_parent, ValueObject *valobj,
    uint64_t &language_flags) {
  auto ts = type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwiftTypeRef>();
  if (!ts)
    return llvm::createStringError("no type system");

  lldb::addr_t pointer = LLDB_INVALID_ADDRESS;
  ExecutionContext exe_ctx;
  if (valobj) {
    exe_ctx = valobj->GetExecutionContextRef();
    pointer = valobj->GetPointerValue();
  }

  // Deal with the LLDB-only SILPackType variant.
  if (auto pack_element_type = ts->GetSILPackElementAtIndex(type, idx)) {
    llvm::raw_string_ostream os(child_name);
    os << '.' << idx;
    child_byte_size =
        GetBitSize(pack_element_type, exe_ctx.GetBestExecutionContextScope())
            .value_or(0);
    int stack_dir = -1;
    child_byte_offset = ts->GetPointerByteSize() * idx * stack_dir;
    child_bitfield_bit_size = 0;
    child_bitfield_bit_offset = 0;
    child_is_base_class = false;
    child_is_deref_of_parent = true;
    return pack_element_type;
  }

  // The actual conversion from the FieldInfo record.
  auto get_from_field_info =
      [&](const swift::reflection::FieldInfo &field,
          std::optional<TypeSystemSwift::TupleElement> tuple,
          bool hide_existentials) -> CompilerType {
    bool is_indirect_enum =
        !field.Offset && field.TR &&
        llvm::isa<swift::reflection::BuiltinTypeRef>(field.TR) &&
        llvm::isa<swift::reflection::ReferenceTypeInfo>(field.TI) &&
        llvm::cast<swift::reflection::ReferenceTypeInfo>(field.TI)
                .getReferenceKind() == swift::reflection::ReferenceKind::Strong;
    child_name = tuple ? tuple->element_name.GetStringRef().str() : field.Name;
    child_byte_size = field.TI.getSize();
    child_byte_offset = field.Offset;
    child_bitfield_bit_size = 0;
    child_bitfield_bit_offset = 0;
    child_is_base_class = false;
    child_is_deref_of_parent = false;
    language_flags = 0;
    if (is_indirect_enum)
      language_flags |= TypeSystemSwift::LanguageFlags::eIsIndirectEnumCase;
    // SwiftASTContext hardcodes the members of protocols as raw
    // pointers. Remote Mirrors reports them as UnknownObject instead.
    if (hide_existentials && ts->IsExistentialType(type.GetOpaqueQualType()))
      return ts->GetRawPointerType();
    CompilerType result;
    if (tuple)
      result = tuple->element_type;
    else if (is_indirect_enum) {
      ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
      if (!reflection_ctx)
        return {};
      // The indirect enum field should point to a closure context.
      LLDBTypeInfoProvider tip(*this, &exe_ctx);
      lldb::addr_t instance = MaskMaybeBridgedPointer(m_process, pointer);
      auto *ti = reflection_ctx->GetTypeInfoFromInstance(
          instance, &tip, ts->GetDescriptorFinder());
      if (!ti)
        return {};
      auto *rti = llvm::dyn_cast_or_null<swift::reflection::RecordTypeInfo>(ti);
      if (rti->getFields().size() < 1)
        return {};
      auto &field = rti->getFields()[0];
      auto *type_ref = field.TR;
      result = GetTypeFromTypeRef(*ts, type_ref);
      child_byte_offset = field.Offset;
    } else
      result = GetTypeFromTypeRef(*ts, field.TR);
    // Bug-for-bug compatibility. See comment in
    // SwiftASTContext::GetBitSize().
    if (result.IsFunctionType())
      child_byte_size = ts->GetPointerByteSize();
    return result;
  };

  // Try the static type metadata.
  const swift::reflection::TypeRef *tr = nullptr;
  auto *ti = GetSwiftRuntimeTypeInfo(
      type, exe_ctx.GetBestExecutionContextScope(), &tr);
  if (!ti)
    return llvm::createStringError("could not get runtime type info for" +
                                   toString(tr));

  // Structs and Tuples.
  if (auto *rti =
          llvm::dyn_cast_or_null<swift::reflection::RecordTypeInfo>(ti)) {
    auto fields = rti->getFields();

    // Handle tuples.
    std::optional<TypeSystemSwift::TupleElement> tuple;
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
      if (idx - 3 >= fields.size())
        return llvm::createStringError(llvm::Twine("index") + llvm::Twine(idx) +
                                       "is out of bounds (" +
                                       llvm::Twine(fields.size()) + ")");
      return get_from_field_info(fields[idx - 3], tuple, false);
    }
    if (rti->getRecordKind() ==
        swift::reflection::RecordKind::ClassExistential) {
      // Compatibility with SwiftASTContext.
      size_t i = 0;
      for (auto &protocol_child : GetExistentialSyntheticChildren(ts, tr, ti))
        if (i++ == idx) {
          child_name = protocol_child.name;
          child_byte_size = ts->GetPointerByteSize();
          child_byte_offset = ts->GetPointerByteSize() * idx;
          child_bitfield_bit_size = 0;
          child_bitfield_bit_offset = 0;
          child_is_base_class = false;
          child_is_deref_of_parent = false;
          language_flags = 0;
          return protocol_child.get_type();
        }
    }
    if (idx >= fields.size())
      return llvm::createStringError(llvm::Twine("index") + llvm::Twine(idx) +
                                     "is out of bounds (" +
                                     llvm::Twine(fields.size()) + ")");
    return get_from_field_info(fields[idx], tuple, true);
  }
  // Enums.
  if (auto *eti = llvm::dyn_cast_or_null<swift::reflection::EnumTypeInfo>(ti)) {
    unsigned i = 0;
    for (auto &enum_case : eti->getCases()) {
      // Skip non-payload cases.
      if (!enum_case.TR)
        continue;
      if (i++ == idx)
        return get_from_field_info(enum_case, {}, true);
    }
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        llvm::Twine("index") + llvm::Twine(idx) + "is out of bounds (" +
            llvm::Twine(eti->getNumPayloadCases()) + ")");
  }
  if (auto *rti =
          llvm::dyn_cast_or_null<swift::reflection::ReferenceTypeInfo>(ti)) {
    // Is this an Existential?
    size_t i = 0;
    for (auto &protocol_child : GetExistentialSyntheticChildren(ts, tr, ti))
      if (i++ == idx) {
        child_name = protocol_child.name;
        child_byte_size =  ts->GetPointerByteSize();
        child_byte_offset = ts->GetPointerByteSize() * idx;
        child_bitfield_bit_size = 0;
        child_bitfield_bit_offset = 0;
        child_is_base_class = false;
        child_is_deref_of_parent = false;
        language_flags = 0;
        return protocol_child.get_type();
      }
    if (i) {
      return llvm::createStringError(llvm::Twine("index") + llvm::Twine(idx) +
                                     "is out of bounds (" + llvm::Twine(i - 1) +
                                     ")");
    }

    // Objects.
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

    // Try the instance type metadata.
    if (!valobj)
      return llvm::createStringError("object has no address");

    bool found_start = false;
    using namespace swift::Demangle;
    Demangler dem;
    auto mangled = type.GetMangledTypeName().GetStringRef();
    NodePointer type_node = dem.demangleSymbol(mangled);
    llvm::StringRef type_name = TypeSystemSwiftTypeRef::GetBaseName(
        ts->CanonicalizeSugar(dem, type_node));

    ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
    if (!reflection_ctx)
      return llvm::createStringError("no reflection context");

    CompilerType instance_type = valobj->GetCompilerType();
    auto instance_ts =
        instance_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
    if (!instance_ts)
      return llvm::createStringError("no typesystem");

    // LLDBTypeInfoProvider needs to be kept alive while supers gets accessed.
    llvm::SmallVector<SuperClassType, 2> supers;
    auto superclass_finder = [&](SuperClassType sc) -> bool {
      // If the typeref is invalid, we don't want to process it (for
      // example, this could be an artifical ObjC class).
      if (!sc.get_typeref())
        return false;

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
    };

    LLDBTypeInfoProvider tip(*this, &exe_ctx);
    // Try out the instance pointer based super class traversal first, as its
    // usually faster.
    reflection_ctx->ForEachSuperClassType(&tip, ts->GetDescriptorFinder(),
                                          pointer, superclass_finder);

    if (supers.empty())
      // If the pointer based super class traversal failed (this may happen
      // when metadata is not present in the binary, for example: embedded
      // Swift), try the typeref based one next.
      reflection_ctx->ForEachSuperClassType(&tip, ts->GetDescriptorFinder(), tr,
                                            superclass_finder);

    if (supers.empty()) {
      LLDB_LOG(GetLog(LLDBLog::Types),
               "Couldn't find the type metadata for {0} in instance",
               type.GetTypeName());

      auto *cti = reflection_ctx->GetClassInstanceTypeInfo(
          tr, &tip, ts->GetDescriptorFinder());
      if (auto *rti =
              llvm::dyn_cast_or_null<swift::reflection::RecordTypeInfo>(cti)) {
        auto fields = rti->getFields();
        if (idx < fields.size()) {
          std::optional<TypeSystemSwift::TupleElement> tuple;
          return get_from_field_info(fields[idx], tuple, true);
        }
      }
      return llvm::createStringError(llvm::Twine("index") + llvm::Twine(idx) +
                                     "is out of bounds");
    }

    // Handle the artificial base class fields.
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
            return CompilerType();
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
      return llvm::createStringError("no record type info");
    for (auto &field : object->getFields())
      if (i++ == idx)
        return get_from_field_info(field, {}, true);

    return llvm::createStringError(llvm::Twine("index") + llvm::Twine(idx) +
                                   "is out of bounds (" + llvm::Twine(i - 1) +
                                   ")");
  }
  if (llvm::dyn_cast_or_null<swift::reflection::BuiltinTypeInfo>(ti)) {
    // Clang enums have an artificial rawValue property. We could
    // consider handling them here, but
    // TypeSystemSwiftTypeRef::GetChildCompilerTypeAtIndex can also
    // handle them and without a Process.
    return CompilerType();
  }
  LogUnimplementedTypeKind(__FUNCTION__, type);
  return llvm::createStringError("not implemented");
}

bool SwiftLanguageRuntimeImpl::ForEachSuperClassType(
    ValueObject &instance, std::function<bool(SuperClassType)> fn) {
  ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
  if (!reflection_ctx)
    return false;
  CompilerType instance_type = instance.GetCompilerType();
  auto ts = instance_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
  if (!ts)
    return false;

  ExecutionContext exe_ctx(instance.GetExecutionContextRef());
  LLDBTypeInfoProvider tip(*this, &exe_ctx);
  lldb::addr_t pointer = instance.GetPointerValue();
  return reflection_ctx->ForEachSuperClassType(
      &tip, ts->GetTypeSystemSwiftTypeRef().GetDescriptorFinder(),
      pointer, fn);
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
  Context ctx;
  auto *node_ptr = SwiftLanguageRuntime::DemangleSymbolAsNode(func_name, ctx);
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

static swift::Demangle::NodePointer
CreatePackType(swift::Demangle::Demangler &dem, TypeSystemSwiftTypeRef &ts,
               llvm::ArrayRef<TypeSystemSwift::TupleElement> elements) {
  auto *pack = dem.createNode(Node::Kind::Pack);
  for (const auto &element : elements) {
    auto *type = dem.createNode(Node::Kind::Type);
    auto *element_type = swift_demangle::GetDemangledType(
        dem, element.element_type.GetMangledTypeName().GetStringRef());
    if (!element_type)
      return {};
    type->addChild(element_type, dem);
    pack->addChild(type, dem);
  }
  return pack;
}

bool SwiftLanguageRuntimeImpl::GetDynamicTypeAndAddress_Pack(
    ValueObject &in_value, CompilerType pack_type,
    lldb::DynamicValueType use_dynamic, TypeAndOrName &pack_type_or_name,
    Address &address, Value::ValueType &value_type) {
  Log *log(GetLog(LLDBLog::Types));
  ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
  if (!reflection_ctx)
    return false;

  // Return a tuple type, with one element per pack element and its
  // type has all DependentGenericParamType that appear in type packs
  // substituted.

  StackFrameSP frame = in_value.GetExecutionContextRef().GetFrameSP();
  if (!frame)
    return false;
  ConstString func_name = frame->GetSymbolContext(eSymbolContextFunction)
                              .GetFunctionName(Mangled::ePreferMangled);

  // Extract the generic signature from the function symbol.
  auto ts =
      pack_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwiftTypeRef>();
  if (!ts)
    return false;
  auto signature =
    SwiftLanguageRuntime::GetGenericSignature(func_name.GetStringRef(), *ts);
  if (!signature) {
    LLDB_LOG(log, "cannot decode pack_expansion type: failed to decode generic "
                  "signature from function name");
    return false;
  }
  // This type has already been resolved?
  if (auto info = ts->IsSILPackType(pack_type))
    if (info->expanded)
      return false;

  Target &target = m_process.GetTarget();
  size_t ptr_size = m_process.GetAddressByteSize();

  swift::Demangle::Demangler dem;

  auto expand_pack_type = [&](ConstString mangled_pack_type,
                              bool indirect) -> swift::Demangle::NodePointer {
    // Find pack_type in the pack_expansions.
    unsigned i = 0;
    SwiftLanguageRuntime::GenericSignature::PackExpansion *pack_expansion =
        nullptr;
    for (auto &pe : signature->pack_expansions) {
      if (pe.mangled_type == mangled_pack_type) {
        pack_expansion = &pe;
        break;
      }
      ++i;
    }
    if (!pack_expansion) {
      LLDB_LOG(log, "cannot decode pack_expansion type: failed to find a "
                    "matching type in the function signature");
      return {};
    }

    // Extract the count.
    llvm::SmallString<16> buf;
    llvm::raw_svector_ostream os(buf);
    os << "$pack_count_" << signature->GetCountForValuePack(i);
    StringRef count_var = os.str();
    std::optional<lldb::addr_t> count =
        GetTypeMetadataForTypeNameAndFrame(count_var, *frame);
    if (!count) {
      LLDB_LOG(log,
               "cannot decode pack_expansion type: failed to find count "
               "argument \"%s\" in frame",
               count_var.str());
      return {};
    }

    // Extract the metadata for the type packs in this value pack.
    llvm::SmallDenseMap<std::pair<unsigned, unsigned>, lldb::addr_t> type_packs;
    swift::Demangle::NodePointer dem_pack_type =
        dem.demangleSymbol(mangled_pack_type.GetStringRef());
    auto shape = signature->generic_params[pack_expansion->shape];
    // Filter out all type packs in this value pack.
    bool error = false;
    ForEachGenericParameter(dem_pack_type, [&](unsigned depth, unsigned index) {
      if (type_packs.count({depth, index}))
        return;
      for (auto p : shape.same_shape.set_bits()) {
        // If a generic parameter that shows up in the
        // pack_expansion has the same shape as the pack expansion
        // it's a type pack.
        auto &generic_param = signature->generic_params[p];
        if (generic_param.depth == depth && generic_param.index == index) {
          llvm::SmallString<16> buf;
          llvm::raw_svector_ostream os(buf);
          os << u8"$\u03C4_" << shape.depth << '_' << shape.index;
          StringRef mds_var = os.str();
          std::optional<lldb::addr_t> mds_ptr =
              GetTypeMetadataForTypeNameAndFrame(mds_var, *frame);
          if (!mds_ptr) {
            LLDB_LOG(log,
                      "cannot decode pack_expansion type: failed to find "
                      "metadata "
                      "for \"{0}\" in frame",
                      mds_var.str());
            error = true;
            return;
          }
          type_packs.insert({{depth, index}, *mds_ptr});
        }
      }
    });
    if (error)
      return {};

    // Walk the type packs.
    std::vector<TypeSystemSwift::TupleElement> elements;
    for (unsigned j = 0; j < *count; ++j) {

      // Build the list of type substitutions.
      swift::reflection::GenericArgumentMap substitutions;
      for (auto it : type_packs) {
        unsigned depth = it.first.first;
        unsigned index = it.first.second;
        lldb::addr_t md_ptr = it.second + j * ptr_size;

        // Read the type metadata pointer.
        Status status;
        lldb::addr_t md = LLDB_INVALID_ADDRESS;
        target.ReadMemory(md_ptr, &md, ptr_size, status, true);
        if (!status.Success()) {
          LLDB_LOGF(log,
                    "cannot decode pack_expansion type: failed to read type "
                    "pack for type %d/%d of type pack with shape %d %d",
                    j, (unsigned)*count, depth, index);
          return {};
        }

        auto *type_ref = reflection_ctx->ReadTypeFromMetadata(
            md, ts->GetDescriptorFinder());
        if (!type_ref) {
          LLDB_LOGF(log,
                    "cannot decode pack_expansion type: failed to decode type "
                    "metadata for type %d/%d of type pack with shape %d %d",
                    j, (unsigned)*count, depth, index);
          return {};
        }
        substitutions.insert({{depth, index}, type_ref});
      }
      if (substitutions.empty())
        return {};

      // Replace all pack expansions with a singular type. Otherwise the
      // reflection context won't accept them.
      NodePointer pack_element = TypeSystemSwiftTypeRef::Transform(
          dem, dem_pack_type, [](NodePointer node) {
            if (node->getKind() != Node::Kind::PackExpansion)
              return node;
            assert(node->getNumChildren() == 2);
            if (node->getNumChildren() != 2)
              return node;
            return node->getChild(0);
          });

      // Build a TypeRef from the demangle tree.
      auto type_ref = reflection_ctx->GetTypeRefOrNull(
          dem, pack_element, ts->GetDescriptorFinder());
      if (!type_ref)
        return {};

      // Apply the substitutions.
      auto bound_typeref = reflection_ctx->ApplySubstitutions(
          type_ref, substitutions, ts->GetDescriptorFinder());
      swift::Demangle::NodePointer node = bound_typeref->getDemangling(dem);
      CompilerType type = ts->RemangleAsType(dem, node);

      // Add the substituted type to the tuple.
      elements.push_back({{}, type});
    }
    if (indirect) {
      // Create a tuple type with all the concrete types in the pack.
      CompilerType tuple = ts->CreateTupleType(elements);
      // TODO: Remove unnecessary mangling roundtrip.
      // Wrap the type inside a SILPackType to mark it for GetChildAtIndex.
      CompilerType sil_pack_type = ts->CreateSILPackType(tuple, indirect);
      swift::Demangle::NodePointer global =
          dem.demangleSymbol(sil_pack_type.GetMangledTypeName().GetStringRef());
      using Kind = Node::Kind;
      auto *dem_sil_pack_type =
          swift_demangle::ChildAtPath(global, {Kind::TypeMangling, Kind::Type});
      return dem_sil_pack_type;
    } else {
      return CreatePackType(dem, *ts, elements);
    }
  };

  swift::Demangle::Context dem_ctx;
  auto node = dem_ctx.demangleSymbolAsNode(
      pack_type.GetMangledTypeName().GetStringRef());

  // Expand all the pack types that appear in the incoming type,
  // either at the root level or as arguments of bound generic types.
  bool indirect = false;
  auto transformed = TypeSystemSwiftTypeRef::Transform(
      dem, node, [&](swift::Demangle::NodePointer node) {
        if (node->getKind() == swift::Demangle::Node::Kind::SILPackIndirect)
          indirect = true;
        if (node->getKind() != swift::Demangle::Node::Kind::SILPackIndirect &&
            node->getKind() != swift::Demangle::Node::Kind::SILPackDirect &&
            node->getKind() != swift::Demangle::Node::Kind::Pack)
          return node;

        if (node->getNumChildren() != 1)
          return node;
        node = node->getChild(0);
        CompilerType pack_type = ts->RemangleAsType(dem, node);
        ConstString mangled_pack_type = pack_type.GetMangledTypeName();
        LLDB_LOG(log, "decoded pack_expansion type: {0}", mangled_pack_type);
        auto result = expand_pack_type(mangled_pack_type, indirect);
        if (!result) {
          LLDB_LOG(log, "failed to expand pack type: {0}", mangled_pack_type);
          return node;
        }
        return result;
      });

  CompilerType expanded_type = ts->RemangleAsType(dem, transformed);
  pack_type_or_name.SetCompilerType(expanded_type);

  AddressType address_type;
  lldb::addr_t addr = in_value.GetAddressOf(true, &address_type);
  value_type = Value::GetValueTypeFromAddressType(address_type);
  if (indirect) {
    Status status;
    addr = m_process.ReadPointerFromMemory(addr, status);
    if (status.Fail()) {
      LLDB_LOG(log, "failed to dereference indirect pack: {0}",
               expanded_type.GetMangledTypeName());
      return false;
    }
  }
  address.SetRawAddress(addr);
  return true;
}

#ifndef NDEBUG
bool SwiftLanguageRuntimeImpl::IsScratchContextLocked(Target &target) {
  if (target.GetSwiftScratchContextLock().try_lock()) {
    target.GetSwiftScratchContextLock().unlock();
    return false;
  }
  return true;
}

bool SwiftLanguageRuntimeImpl::IsScratchContextLocked(TargetSP target) {
  return target ? IsScratchContextLocked(*target) : true;
}
#endif

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
      return child->getText().starts_with("__NS") ||
             child->getText().starts_with("NSTaggedPointer");
  return false;
}

bool SwiftLanguageRuntimeImpl::GetDynamicTypeAndAddress_Class(
    ValueObject &in_value, CompilerType class_type,
    lldb::DynamicValueType use_dynamic, TypeAndOrName &class_type_or_name,
    Address &address, Value::ValueType &value_type) {
  AddressType address_type;
  lldb::addr_t instance_ptr = in_value.GetPointerValue(&address_type);
  value_type = Value::GetValueTypeFromAddressType(address_type);

  if (instance_ptr == LLDB_INVALID_ADDRESS || instance_ptr == 0)
    return false;

  // Unwrap reference types.
  Status error;
  instance_ptr = FixupAddress(instance_ptr, class_type, error);
  if (!error.Success())
    return false;

  auto tss = class_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
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
  ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
  if (!reflection_ctx)
    return false;

  const auto *typeref = reflection_ctx->ReadTypeFromInstance(
      instance_ptr, ts.GetDescriptorFinder(), true);

  // If we couldn't find the typeref from the instance, the best we can do is
  // use the static type. This is a valid use case when the binary doesn't
  // contain any metadata (for example, embedded Swift).
  if (!typeref) 
    typeref = reflection_ctx->GetTypeRefOrNull(class_type.GetMangledTypeName(),
                                               ts.GetDescriptorFinder());

  if (!typeref) {
    HEALTH_LOG("could not read typeref for type: {0} (instance_ptr = {0:x})",
               class_type.GetMangledTypeName(), instance_ptr);
    return false;
  }
  swift::Demangle::Demangler dem;
  swift::Demangle::NodePointer node = typeref->getDemangling(dem);
  CompilerType dynamic_type = ts.RemangleAsType(dem, node);
  LLDB_LOG(log, "dynamic type of instance_ptr {0:x} is {1}", instance_ptr,
           class_type.GetMangledTypeName());
  class_type_or_name.SetCompilerType(dynamic_type);

#ifndef NDEBUG
  if (ModuleList::GetGlobalModuleListProperties()
          .GetSwiftValidateTypeSystem()) {
    ConstString a = class_type_or_name.GetCompilerType().GetMangledTypeName();
    ConstString b = SwiftLanguageRuntimeImpl::GetDynamicTypeName_ClassRemoteAST(
        in_value, instance_ptr);
    if (b && a != b)
      llvm::dbgs() << "RemoteAST and runtime diverge " << a << " != " << b
                   << "\n";
  }
#endif
  return true;
}

bool SwiftLanguageRuntimeImpl::IsValidErrorValue(ValueObject &in_value) {
  CompilerType var_type = in_value.GetStaticValue()->GetCompilerType();
  auto tss = var_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
  if (!tss)
    return false;
  if (!tss->IsErrorType(var_type.GetOpaqueQualType()))
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
  Log *log(GetLog(LLDBLog::Types));
  auto tss =
      protocol_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
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
    log->Printf("existential address is 0x%" PRIx64, existential_address);

  if (!existential_address || existential_address == LLDB_INVALID_ADDRESS) {
    if (log)
      log->Printf("Existential address is invalid");
    return false;
  }


  if (use_local_buffer)
    PushLocalBuffer(existential_address, in_value.GetByteSize().value_or(0));

  swift::remote::RemoteAddress remote_existential(existential_address);

  ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
  if (!reflection_ctx)
    return false;

  auto pair = reflection_ctx->ProjectExistentialAndUnwrapClass(
      remote_existential, *protocol_typeref,
      tss->GetTypeSystemSwiftTypeRef().GetDescriptorFinder());
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
  if (ModuleList::GetGlobalModuleListProperties()
          .GetSwiftValidateTypeSystem()) {
    auto reference_pair = GetDynamicTypeAndAddress_ProtocolRemoteAST(
        in_value, protocol_type, use_local_buffer, existential_address);
    assert(pair.has_value() >= reference_pair.has_value() &&
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
  }
#endif
  return true;
}

bool SwiftLanguageRuntimeImpl::GetDynamicTypeAndAddress_ExistentialMetatype(
    ValueObject &in_value, CompilerType meta_type,
    lldb::DynamicValueType use_dynamic, TypeAndOrName &class_type_or_name,
    Address &address) {
  // Resolve the dynamic type of the metatype.
  AddressType address_type;
  lldb::addr_t ptr = in_value.GetPointerValue(&address_type);
  if (ptr == LLDB_INVALID_ADDRESS || ptr == 0)
    return false;

  ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
  if (!reflection_ctx)
    return false;

  auto tss = meta_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
  if (!tss)
    return false;

  const swift::reflection::TypeRef *type_ref =
      reflection_ctx->ReadTypeFromMetadata(
          ptr, tss->GetTypeSystemSwiftTypeRef().GetDescriptorFinder());

  if (!type_ref)
    return false;

  using namespace swift::Demangle;
  Demangler dem;
  NodePointer node = type_ref->getDemangling(dem);
  // Wrap the resolved type in a metatype again for the data formatter to
  // recognize.
  if (!node || node->getKind() != Node::Kind::Type)
    return false;
  NodePointer wrapped = dem.createNode(Node::Kind::Type);
  NodePointer meta = dem.createNode(Node::Kind::Metatype);
  meta->addChild(node, dem);
  wrapped->addChild(meta,dem);

  meta_type = tss->GetTypeSystemSwiftTypeRef().RemangleAsType(dem, wrapped);
  class_type_or_name.SetCompilerType(meta_type);
  address.SetRawAddress(ptr);
  return true;
}

CompilerType SwiftLanguageRuntimeImpl::GetTypeFromMetadata(TypeSystemSwift &ts,
                                                           Address address) {
  lldb::addr_t ptr = address.GetLoadAddress(&GetProcess().GetTarget());
  if (ptr == LLDB_INVALID_ADDRESS)
    return {};

  ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
  if (!reflection_ctx)
    return {};

  const swift::reflection::TypeRef *type_ref =
      reflection_ctx->ReadTypeFromMetadata(
          ptr, ts.GetTypeSystemSwiftTypeRef().GetDescriptorFinder());

  using namespace swift::Demangle;
  Demangler dem;
  NodePointer node = type_ref->getDemangling(dem);
  return ts.GetTypeSystemSwiftTypeRef().RemangleAsType(dem, node);
}

std::optional<lldb::addr_t>
SwiftLanguageRuntimeImpl::GetTypeMetadataForTypeNameAndFrame(
    StringRef mdvar_name, StackFrame &frame) {
  VariableList *var_list = frame.GetVariableList(false, nullptr);
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

void SwiftLanguageRuntime::ForEachGenericParameter(
    swift::Demangle::NodePointer node,
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

CompilerType SwiftLanguageRuntimeImpl::BindGenericTypeParameters(
    CompilerType unbound_type,
    std::function<CompilerType(unsigned, unsigned)> type_resolver) {
  LLDB_SCOPED_TIMER();
  using namespace swift::Demangle;

  auto ts =
      unbound_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
  Status error;
  ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
  if (!reflection_ctx) {
    LLDB_LOG(GetLog(LLDBLog::Types),
             "No reflection context available.");
    return unbound_type;
  }

  Demangler dem;
  NodePointer unbound_node =
      dem.demangleSymbol(unbound_type.GetMangledTypeName().GetStringRef());
  auto type_ref = reflection_ctx->GetTypeRefOrNull(
      dem, unbound_node,
      ts->GetTypeSystemSwiftTypeRef().GetDescriptorFinder());
  if (!type_ref) {
    LLDB_LOG(GetLog(LLDBLog::Expressions | LLDBLog::Types),
             "Couldn't get TypeRef of unbound type");
    return {};
  }

  swift::reflection::GenericArgumentMap substitutions;
  bool failure = false;
  ForEachGenericParameter(unbound_node, [&](unsigned depth, unsigned index) {
    if (failure)
      return;
    if (substitutions.count({depth, index}))
      return;

    auto type = type_resolver(depth, index);
    if (!type) {
      LLDB_LOG(GetLog(LLDBLog::Expressions | LLDBLog::Types),
               "type_finder function failed to find type.");
      failure = true;
      return;
    }

    const auto *type_ref = reflection_ctx->GetTypeRefOrNull(
        type.GetMangledTypeName().GetStringRef(),
        ts->GetTypeSystemSwiftTypeRef().GetDescriptorFinder());
    if (!type_ref) {
      LLDB_LOG(GetLog(LLDBLog::Expressions | LLDBLog::Types),
               "Couldn't get TypeRef when binding generic type parameters.");
      failure = true;
      return;
    }

    substitutions.insert({{depth, index}, type_ref});
  });

  if (failure)
    return {};

  // Apply the substitutions.
  const swift::reflection::TypeRef *bound_type_ref =
      reflection_ctx->ApplySubstitutions(
          type_ref, substitutions,
          ts->GetTypeSystemSwiftTypeRef().GetDescriptorFinder());
  NodePointer node = bound_type_ref->getDemangling(dem);
  return ts->GetTypeSystemSwiftTypeRef().RemangleAsType(dem, node);
}

CompilerType
SwiftLanguageRuntimeImpl::BindGenericTypeParameters(StackFrame &stack_frame,
                                                    TypeSystemSwiftTypeRef &ts,
                                                    ConstString mangled_name) {
  LLDB_SCOPED_TIMER();
  using namespace swift::Demangle;

  Status error;
  ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
  if (!reflection_ctx) {
    LLDB_LOG(GetLog(LLDBLog::Expressions | LLDBLog::Types),
             "No reflection context available.");
    return ts.GetTypeFromMangledTypename(mangled_name);
  }

  Demangler dem;

  NodePointer canonical = TypeSystemSwiftTypeRef::GetStaticSelfType(
      dem, dem.demangleSymbol(mangled_name.GetStringRef()));

  // Build the list of type substitutions.
  swift::reflection::GenericArgumentMap substitutions;
  ForEachGenericParameter(canonical, [&](unsigned depth, unsigned index) {
    if (substitutions.count({depth, index}))
      return;
    StreamString mdvar_name;
    mdvar_name.Printf(u8"$\u03C4_%d_%d", depth, index);

    std::optional<lldb::addr_t> metadata_location =
        GetTypeMetadataForTypeNameAndFrame(mdvar_name.GetString(), stack_frame);
    if (!metadata_location)
      return;
    const swift::reflection::TypeRef *type_ref =
        reflection_ctx->ReadTypeFromMetadata(*metadata_location,
                                             ts.GetDescriptorFinder());
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
  const swift::reflection::TypeRef *type_ref = reflection_ctx->GetTypeRefOrNull(
      dem, canonical, ts.GetDescriptorFinder());
  if (!type_ref)
    return get_canonical();

  // Apply the substitutions.
  const swift::reflection::TypeRef *bound_type_ref =
      reflection_ctx->ApplySubstitutions(type_ref, substitutions,
                                         ts.GetDescriptorFinder());
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
  CompilerType bound_type = scratch_ctx->RemangleAsType(dem, node);
  LLDB_LOG(GetLog(LLDBLog::Expressions | LLDBLog::Types), "Bound {0} -> {1}.",
           mangled_name, bound_type.GetMangledTypeName());
  return bound_type;
}

CompilerType
SwiftLanguageRuntimeImpl::BindGenericTypeParameters(StackFrame &stack_frame,
                                                    CompilerType base_type) {
  // If this is a TypeRef type, bind that.
  auto sc = stack_frame.GetSymbolContext(lldb::eSymbolContextEverything);
  if (auto ts =
          base_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwiftTypeRef>())
    return BindGenericTypeParameters(stack_frame, *ts,
                                     base_type.GetMangledTypeName());
  return BindGenericTypeParametersRemoteAST(stack_frame, base_type);
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
    Address &address, Value::ValueType &value_type) {
  value_type = Value::ValueType::Invalid;
  class_type_or_name.SetCompilerType(bound_type);

  ExecutionContext exe_ctx = in_value.GetExecutionContextRef().Lock(true);
  std::optional<uint64_t> size =
      bound_type.GetByteSize(exe_ctx.GetBestExecutionContextScope());
  if (!size)
    return false;
  AddressType address_type;
  lldb::addr_t val_address = in_value.GetAddressOf(true, &address_type);
  if (*size && (!val_address || val_address == LLDB_INVALID_ADDRESS))
    return false;

  value_type = Value::GetValueTypeFromAddressType(address_type);
  address.SetLoadAddress(val_address, in_value.GetTargetSP().get());
  return true;
}

bool SwiftLanguageRuntimeImpl::GetDynamicTypeAndAddress_IndirectEnumCase(
    ValueObject &in_value, lldb::DynamicValueType use_dynamic,
    TypeAndOrName &class_type_or_name, Address &address,
    Value::ValueType &value_type) {
  Status error;
  CompilerType child_type = in_value.GetCompilerType();
  class_type_or_name.SetCompilerType(child_type);

  auto *enum_obj = in_value.GetParent();
  lldb::addr_t box_addr = enum_obj->GetPointerValue();
  if (box_addr == LLDB_INVALID_ADDRESS)
    return false;

  box_addr =
    MaskMaybeBridgedPointer(m_process, box_addr);
  lldb::addr_t box_location = m_process.ReadPointerFromMemory(box_addr, error);
  if (box_location == LLDB_INVALID_ADDRESS)
    return false;

  ABISP abi_sp = m_process.GetABI();
  if (abi_sp)
    box_location = abi_sp->FixCodeAddress(box_location);

  box_location = MaskMaybeBridgedPointer(m_process, box_location);
  lldb::addr_t box_value = box_addr + in_value.GetByteOffset();
  Flags type_info(child_type.GetTypeInfo());
  if (type_info.AllSet(eTypeIsSwift) &&
      type_info.AnySet(eTypeIsClass | eTypeIsProtocol)) {
    ExecutionContext exe_ctx = in_value.GetExecutionContextRef();
    ValueObjectSP valobj_sp = ValueObjectMemory::Create(
        exe_ctx.GetBestExecutionContextScope(), "_", box_value, child_type);
    if (!valobj_sp)
      return false;

    return GetDynamicTypeAndAddress(*valobj_sp, use_dynamic, class_type_or_name,
                                    address, value_type);
  } else {
    // This is most likely a statically known type.
    address.SetLoadAddress(box_value, &m_process.GetTarget());
    value_type = Value::GetValueTypeFromAddressType(eAddressTypeLoad);
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


Process &SwiftLanguageRuntimeImpl::GetProcess() const {
  return m_process;
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
                                       Value::ValueType static_value_type,
                                       bool is_indirect_enum_case) {
  CompilerType static_type = in_value.GetCompilerType();
  Flags static_type_flags(static_type.GetTypeInfo());
  Flags dynamic_type_flags(dynamic_type.GetTypeInfo());

  if (dynamic_type_flags.AllSet(eTypeIsSwift)) {
    // for a protocol object where does the dynamic data live if the target
    // object is a struct? (for a class, it's easy)
    if (static_type_flags.AllSet(eTypeIsSwift | eTypeIsProtocol) &&
        dynamic_type_flags.AnySet(eTypeIsStructUnion | eTypeIsEnumeration)) {
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
                        in_value.GetByteSize().value_or(0));

      // Read the value witness table and check if the data is inlined in
      // the existential container or not.
      swift::remote::RemoteAddress remote_existential(existential_address);
      if (ThreadSafeReflectionContext reflection_ctx = GetReflectionContext()) {
        std::optional<bool> is_inlined =
            reflection_ctx->IsValueInlinedInExistentialContainer(
                remote_existential);

        if (use_local_buffer)
          PopLocalBuffer();

        // An error has occurred when trying to read value witness table,
        // default to treating it as pointer.
        if (!is_inlined.has_value())
          return Value::ValueType::LoadAddress;

        // Inlined data, same as static data.
        if (*is_inlined)
          return static_value_type;
      }

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

namespace {
struct SwiftNominalType {
  std::string module;
  std::string identifier;
};

// Find the Swift class that backs an ObjC type.
//
// A Swift class that uses the @objc(<ClassName>) attribute will emit ObjC
// metadata into the binary. Typically, ObjC classes have a symbol in the form
// of OBJC_CLASS_$_<ClassName>, however for Swift classes, there are two symbols
// that both point to the ObjC class metadata, where the second symbol is a
// Swift mangled name.
std::optional<SwiftNominalType> GetSwiftClass(ValueObject &valobj,
                                              AppleObjCRuntime &objc_runtime) {
  // To find the Swift symbol, the following preparation steps are taken:
  //   1. Get the value's ISA pointer
  //   2. Resolve the ISA load address into an Address instance
  //   3. Get the Module that contains the Address
  auto descriptor_sp = objc_runtime.GetClassDescriptor(valobj);
  if (!descriptor_sp)
    return {};

  auto isa_load_addr = descriptor_sp->GetISA();
  Address isa;
  const auto &sections = objc_runtime.GetTargetRef().GetSectionLoadList();
  if (!sections.ResolveLoadAddress(isa_load_addr, isa))
    return {};

  // Next, iterate over the Module's symbol table, looking for a symbol with
  // following criteria:
  //   1. The symbol address is the ISA address
  //   2. The symbol name is a Swift mangled name
  std::optional<StringRef> swift_symbol;
  auto find_swift_symbol_for_isa = [&](Symbol *symbol) {
    if (symbol->GetAddress() == isa) {
      StringRef symbol_name =
          symbol->GetMangled().GetMangledName().GetStringRef();
      if (SwiftLanguageRuntime::IsSwiftMangledName(symbol_name)) {
        swift_symbol = symbol_name;
        return false;
      }
    }
    return true;
  };

  isa.GetModule()->GetSymtab()->ForEachSymbolContainingFileAddress(
      isa.GetFileAddress(), find_swift_symbol_for_isa);
  if (!swift_symbol)
    return {};

  // Once the Swift symbol is found, demangle it into a node tree. The node tree
  // provides the final data, the name of the class and the name of its module.
  swift::Demangle::Context ctx;
  auto *global = ctx.demangleSymbolAsNode(*swift_symbol);
  using Kind = Node::Kind;
  auto *class_node = swift_demangle::ChildAtPath(
      global, {Kind::TypeMetadata, Kind::Type, Kind::Class});
  if (class_node && class_node->getNumChildren() == 2) {
    auto module_node = class_node->getFirstChild();
    auto ident_node = class_node->getLastChild();
    if (module_node->getKind() == Kind::Module && module_node->hasText() &&
        ident_node->getKind() == Kind::Identifier && ident_node->hasText()) {
      auto module_name = module_node->getText();
      auto class_name = ident_node->getText();
      return SwiftNominalType{module_name.str(), class_name.str()};
    }
  }

  return {};
}

} // namespace

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
      dyn_name.starts_with("__NS"))
    return false;

  SwiftNominalType swift_class;

  if (auto maybe_swift_class = GetSwiftClass(in_value, *objc_runtime)) {
    swift_class = *maybe_swift_class;
    std::string type_name =
        (llvm::Twine(swift_class.module) + "." + swift_class.identifier).str();
    dyn_class_type_or_name.SetName(type_name.data());
    address.SetRawAddress(in_value.GetPointerValue());
  } else {
    swift_class.module = swift::MANGLING_MODULE_OBJC;
    swift_class.identifier = dyn_name;
  }

  std::string remangled;
  {
    // Create a mangle tree for Swift.Optional<$module.$class>
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
    e->addChild(factory.createNode(Node::Kind::Module, swift::STDLIB_NAME),
                factory);
    e->addChild(factory.createNode(Node::Kind::Identifier, "Optional"),
                factory);
    ety->addChild(e, factory);
    NodePointer list = factory.createNode(Node::Kind::TypeList);
    bge->addChild(list, factory);
    NodePointer cty = factory.createNode(Node::Kind::Type);
    list->addChild(cty, factory);
    NodePointer c = factory.createNode(Node::Kind::Class);
    c->addChild(factory.createNode(Node::Kind::Module, swift_class.module),
                factory);
    c->addChild(
        factory.createNode(Node::Kind::Identifier, swift_class.identifier),
        factory);
    cty->addChild(c, factory);

    auto mangling = mangleNode(global);
    if (!mangling.isSuccess())
      return false;
    remangled = mangling.result();
  }

  // Import the remangled dynamic name into the scratch context.
  assert(IsScratchContextLocked(in_value.GetTargetSP()) &&
         "Swift scratch context not locked ahead of dynamic type resolution");
  std::optional<SwiftScratchContextReader> maybe_scratch_ctx =
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
          TypeSystemSwift::LanguageFlags::eIsIndirectEnumCase) ==
         TypeSystemSwift::LanguageFlags::eIsIndirectEnumCase;
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

  Value::ValueType static_value_type = Value::ValueType::Invalid;
  bool success = false;
  bool is_indirect_enum_case = IsIndirectEnumCase(in_value);
  // Type kinds with instance metadata don't need generic type resolution.
  if (is_indirect_enum_case)
    success = GetDynamicTypeAndAddress_IndirectEnumCase(
        in_value, use_dynamic, class_type_or_name, address, static_value_type);
  else if (type_info.AnySet(eTypeIsPack))
    success = GetDynamicTypeAndAddress_Pack(in_value, val_type, use_dynamic,
                                            class_type_or_name, address,
                                            static_value_type);
  else if (type_info.AnySet(eTypeIsClass) ||
           type_info.AllSet(eTypeIsBuiltIn | eTypeIsPointer | eTypeHasValue))
    success = GetDynamicTypeAndAddress_Class(in_value, val_type, use_dynamic,
                                             class_type_or_name, address,
                                             static_value_type);
  else if (type_info.AllSet(eTypeIsMetatype | eTypeIsProtocol))
    success = GetDynamicTypeAndAddress_ExistentialMetatype(
        in_value, val_type, use_dynamic, class_type_or_name, address);
  else if (type_info.AnySet(eTypeIsProtocol))
    success = GetDynamicTypeAndAddress_Protocol(in_value, val_type, use_dynamic,
                                                class_type_or_name, address);
  else {
    CompilerType bound_type;
    if (type_info.AnySet(eTypeHasUnboundGeneric | eTypeHasDynamicSelf)) {
      // Perform generic type resolution.
      StackFrameSP frame = in_value.GetExecutionContextRef().GetFrameSP();
      if (!frame)
        return false;

      bound_type = BindGenericTypeParameters(*frame.get(), val_type);
      if (!bound_type)
        return false;
    } else {
      bound_type = val_type;
    }

    Flags subst_type_info(bound_type.GetTypeInfo());
    if (subst_type_info.AnySet(eTypeIsClass)) {
      success = GetDynamicTypeAndAddress_Class(in_value, bound_type,
                                               use_dynamic, class_type_or_name,
                                               address, static_value_type);
    } else if (subst_type_info.AnySet(eTypeIsProtocol)) {
      success = GetDynamicTypeAndAddress_Protocol(
          in_value, bound_type, use_dynamic, class_type_or_name, address);
    } else {
      success = GetDynamicTypeAndAddress_Value(in_value, bound_type,
                                               use_dynamic, class_type_or_name,
                                               address, static_value_type);
    }
  }

  if (success) {
    // If we haven't found a better static value type, use the value object's
    // one.
    if (static_value_type == Value::ValueType::Invalid)
      static_value_type = in_value.GetValue().GetValueType();

    value_type = GetValueType(in_value, class_type_or_name.GetCompilerType(),
                              static_value_type, is_indirect_enum_case);
  }
  return success;
}

TypeAndOrName SwiftLanguageRuntimeImpl::FixUpDynamicType(
    const TypeAndOrName &type_and_or_name, ValueObject &static_value) {
  CompilerType static_type = static_value.GetCompilerType();
  CompilerType dynamic_type = type_and_or_name.GetCompilerType();
  // The logic in this function only applies to static/dynamic Swift types.
  if (static_type.GetTypeSystem().isa_and_nonnull<TypeSystemClang>())
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
  Demangler dem;
  auto *root = dem.demangleSymbol(type.GetMangledTypeName().GetStringRef());
  using Kind = Node::Kind;
  auto *unowned_node = swift_demangle::ChildAtPath(
      root, {Kind::TypeMangling, Kind::Type, Kind::Unowned});
  if (!unowned_node)
    return false;

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
  // Peek into the reference to see whether it needs an extra deref.
  // If yes, return the fixed-up address we just read.
  lldb::addr_t stripped_addr = LLDB_INVALID_ADDRESS;
  bool extra_deref;
  std::tie(stripped_addr, extra_deref) = FixupPointerValue(addr, type);
  if (extra_deref) {
    Target &target = m_process.GetTarget();
    size_t ptr_size = m_process.GetAddressByteSize();
    lldb::addr_t refd_addr = LLDB_INVALID_ADDRESS;
    target.ReadMemory(stripped_addr, &refd_addr, ptr_size, error, true);
    return refd_addr;
  }
  return addr;
}

const swift::reflection::TypeRef *
SwiftLanguageRuntimeImpl::GetTypeRef(CompilerType type,
                                     TypeSystemSwiftTypeRef *module_holder) {
  Log *log(GetLog(LLDBLog::Types));
  if (log && log->GetVerbose())
    LLDB_LOGF(log, "[SwiftLanguageRuntimeImpl::GetTypeRef] Getting typeref for "
                "type: %s\n",
                type.GetMangledTypeName().GetCString());

  // Demangle the mangled name.
  swift::Demangle::Demangler dem;
  llvm::StringRef mangled_name = type.GetMangledTypeName().GetStringRef();
  auto ts = type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
  if (!ts)
    return nullptr;

  // List of commonly used types known to have been been annotated with
  // @_originallyDefinedIn to a different module.
  static llvm::StringMap<llvm::StringRef> known_types_with_redefined_modules = {
      {"$s14CoreFoundation7CGFloatVD", "$s12CoreGraphics7CGFloatVD"}};

  auto it = known_types_with_redefined_modules.find(mangled_name);
  if (it != known_types_with_redefined_modules.end())
    mangled_name = it->second;

  swift::Demangle::NodePointer node =
      module_holder->GetCanonicalDemangleTree(dem, mangled_name);
  if (!node)
    return nullptr;

  // Build a TypeRef.
  ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
  if (!reflection_ctx)
    return nullptr;

  const auto *type_ref = reflection_ctx->GetTypeRefOrNull(
      dem, node, module_holder->GetDescriptorFinder());
  if (!type_ref)
    return nullptr;
  if (log && log->GetVerbose()) {
    std::stringstream ss;
    type_ref->dump(ss);
    LLDB_LOG(log,
             "[SwiftLanguageRuntimeImpl::GetTypeRef] Found typeref for "
             "type: {0}:\n{0}",
             type.GetMangledTypeName(), ss.str());
  }
  return type_ref;
}

const swift::reflection::TypeInfo *
SwiftLanguageRuntimeImpl::GetSwiftRuntimeTypeInfo(
    CompilerType type, ExecutionContextScope *exe_scope,
    swift::reflection::TypeRef const **out_tr) {
  Log *log(GetLog(LLDBLog::Types));

  if (log && log->GetVerbose())
    LLDB_LOG(log, "[SwiftLanguageRuntimeImpl::GetSwiftRuntimeTypeInfo] Getting "
             "type info for type: {0}",
             type.GetMangledTypeName());

  auto ts = type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
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

  ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
  if (!reflection_ctx)
    return nullptr;

  LLDBTypeInfoProvider provider(*this, exe_scope);
  return reflection_ctx->GetTypeInfo(
      type_ref, &provider,
      ts->GetTypeSystemSwiftTypeRef().GetDescriptorFinder());
}

bool SwiftLanguageRuntimeImpl::IsStoredInlineInBuffer(CompilerType type) {
  if (auto *type_info = GetSwiftRuntimeTypeInfo(type, nullptr))
    return type_info->isBitwiseTakable() && type_info->getSize() <= 24;
  return true;
}

std::optional<uint64_t>
SwiftLanguageRuntimeImpl::GetBitSize(CompilerType type,
                                     ExecutionContextScope *exe_scope) {
  if (auto *type_info = GetSwiftRuntimeTypeInfo(type, exe_scope))
    return type_info->getSize() * 8;
  return {};
}

std::optional<uint64_t>
SwiftLanguageRuntimeImpl::GetByteStride(CompilerType type) {
  if (auto *type_info = GetSwiftRuntimeTypeInfo(type, nullptr))
    return type_info->getStride();
  return {};
}

std::optional<size_t>
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

  SwiftLanguageRuntimeImpl::MetadataPromiseSP promise_sp(
      GetPromiseForTypeNameAndFrame(abstract_type_name.GetCString(), frame));
  if (!promise_sp)
    return CompilerType();

  const SymbolContext &sc = frame->GetSymbolContext(eSymbolContextFunction);
  return promise_sp->FulfillTypePromise(sc);
}

} // namespace lldb_private
