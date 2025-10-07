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
#include "Plugins/TypeSystem/Swift/TypeSystemSwiftTypeRef.h"
#include "ReflectionContextInterface.h"
#include "SwiftLanguageRuntime.h"
#include "SwiftMetadataCache.h"

#include "Plugins/ExpressionParser/Clang/ClangUtil.h"
#include "Plugins/Language/Swift/LogChannelSwift.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "Plugins/TypeSystem/Swift/SwiftDemangle.h"
#include "lldb/Host/SafeMachO.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/ProcessStructReader.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Timer.h"
#include "lldb/ValueObject/ValueObjectCast.h"
#include "lldb/ValueObject/ValueObjectMemory.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include "swift/AST/ASTContext.h"
#include "swift/AST/ASTMangler.h"
#include "swift/AST/ASTWalker.h"
#include "swift/Demangling/Demangle.h"
#include "swift/Demangling/ManglingFlavor.h"
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
    Status error;

    lldb::addr_t masked_addr = addr & ~mask;
    lldb::addr_t isa_addr =
        GetProcess().ReadPointerFromMemory(masked_addr, error);
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

const CompilerType &SwiftLanguageRuntime::GetBoxMetadataType() {
  if (m_box_metadata_type.IsValid())
    return m_box_metadata_type;

  static ConstString g_type_name("__lldb_autogen_boxmetadata");
  const bool is_packed = false;
  if (TypeSystemClangSP clang_ts_sp =
          ScratchTypeSystemClang::GetForTarget(GetProcess().GetTarget())) {
    CompilerType voidstar =
        clang_ts_sp->GetBasicType(lldb::eBasicTypeVoid).GetPointerType();
    CompilerType uint32 = clang_ts_sp->GetIntTypeFromBitSize(32, false);

    m_box_metadata_type = clang_ts_sp->GetOrCreateStructForIdentifier(
        g_type_name, {{"kind", voidstar}, {"offset", uint32}}, is_packed);
  }

  return m_box_metadata_type;
}

std::shared_ptr<LLDBMemoryReader> SwiftLanguageRuntime::GetMemoryReader() {
  if (!m_memory_reader_sp) {
    m_memory_reader_sp.reset(new LLDBMemoryReader(
        GetProcess(), [&](swift::remote::RemoteAbsolutePointer pointer) {
          ThreadSafeReflectionContext reflection_context =
              GetReflectionContext();
          if (!reflection_context)
            return pointer;
          return reflection_context->StripSignedPointer(pointer);
        }));
  }

  return m_memory_reader_sp;
}

MemoryReaderLocalBufferHolder SwiftLanguageRuntime::PushLocalBuffer(uint64_t local_buffer,
                                           uint64_t local_buffer_size) {
  return ((LLDBMemoryReader *)GetMemoryReader().get())
      ->pushLocalBuffer(local_buffer, local_buffer_size);
}


class LLDBTypeInfoProvider : public swift::remote::TypeInfoProvider {
  SwiftLanguageRuntime &m_runtime;
  TypeSystemSwiftTypeRef &m_ts;

public:
  LLDBTypeInfoProvider(SwiftLanguageRuntime &runtime,
                       TypeSystemSwiftTypeRef &ts)
      : m_runtime(runtime), m_ts(ts) {}

  swift::remote::TypeInfoProvider::IdType getId() override {
    return (void *)((char *)&m_ts + m_runtime.GetGeneration());
  }

  const swift::reflection::TypeInfo *
  getTypeInfo(llvm::StringRef mangledName) override {
    // TODO: Should we cache the mangled name -> compiler type lookup, too?
    LLDB_LOG(GetLog(LLDBLog::Types),
             "[LLDBTypeInfoProvider] Looking up debug type info for {0}",
             mangledName);

    TypeSystemSwiftTypeRef &typesystem = m_ts;

    // Materialize a Clang type from the debug info.
    assert(swift::Demangle::getManglingPrefixLength(mangledName) == 0);
    std::string wrapped;
    // The mangled name passed in is bare. Add global prefix ($s) and type (D).
    llvm::raw_string_ostream(wrapped) << "$s" << mangledName << 'D';
    swift::Demangle::Demangler dem;
    auto *node = dem.demangleSymbol(wrapped);
    if (!node) {
      // Try `mangledName` as plain ObjC class name. Ex: NSObject, NSView, etc.
      // Since this looking up an ObjC type, the default mangling falvor should
      // be used.
      auto maybeMangled = swift_demangle::MangleClass(
          dem, swift::MANGLING_MODULE_OBJC, mangledName,
          swift::Mangle::ManglingFlavor::Default);
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
    TypeSystemSwiftTypeRef &typesystem = m_ts;
    // Build a TypeInfo for the Clang type.
    std::optional<uint64_t> size =
        llvm::expectedToOptional(clang_type.GetByteSize(exe_scope));
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
        const swift::reflection::TypeRef *typeref = nullptr;
        auto typeref_or_err = m_runtime.GetTypeRef(swift_type, &typesystem);
        if (!typeref_or_err)
          LLDB_LOG_ERRORV(GetLog(LLDBLog::Types), typeref_or_err.takeError(),
                          "{0}");
        else
          typeref = &*typeref_or_err;

        swift::reflection::FieldInfo field_info = {
            name, (unsigned)bit_offset_ptr / 8, 0, typeref,
            *GetOrCreateTypeInfo(field_type)};
        fields.push_back(field_info);
        }
    }
    return m_runtime.emplaceClangTypeInfo(clang_type, size, bit_align, fields);
  }
};

void SwiftLanguageRuntime::RegisterAnonymousClangType(const char *key,
                                                      CompilerType clang_type) {
  const std::lock_guard<std::recursive_mutex> locker(m_clang_type_info_mutex);
  m_anonymous_clang_types.insert({key, clang_type});
}

CompilerType SwiftLanguageRuntime::LookupAnonymousClangType(const char *key) {
  const std::lock_guard<std::recursive_mutex> locker(m_clang_type_info_mutex);
  return m_anonymous_clang_types.lookup(key);
}

std::optional<const swift::reflection::TypeInfo *>
SwiftLanguageRuntime::lookupClangTypeInfo(CompilerType clang_type) {
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

const swift::reflection::TypeInfo *SwiftLanguageRuntime::emplaceClangTypeInfo(
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
SwiftLanguageRuntime::GetMemberVariableOffsetRemoteMirrors(
    CompilerType instance_type, ValueObject *instance,
    llvm::StringRef member_name, Status *error) {
  LLDB_LOG(GetLog(LLDBLog::Types), "using remote mirrors");
  auto ts =
      instance_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwiftTypeRef>();
  if (!ts) {
    if (error)
      *error = Status::FromErrorString("not a Swift type");
    return {};
  }

  // Try the static type metadata.
  auto frame = instance ? instance->GetExecutionContextRef().GetFrameSP().get()
                        : nullptr;
  auto ti_or_err = GetSwiftRuntimeTypeInfo(instance_type, frame);
  if (!ti_or_err) {
    LLDB_LOG_ERRORV(GetLog(LLDBLog::Types), ti_or_err.takeError(), "{0}");
    return {};
  }
  auto *ti = &*ti_or_err;
  if (auto *rti =
          llvm::dyn_cast_or_null<swift::reflection::RecordTypeInfo>(ti)) {
    auto fields = rti->getFields();

    // Handle tuples.
    if (rti->getRecordKind() == swift::reflection::RecordKind::Tuple) {
      unsigned tuple_idx;
      if (member_name.getAsInteger(10, tuple_idx) ||
          tuple_idx >= rti->getNumFields()) {
        if (error)
          *error = Status::FromErrorString("tuple index out of bounds");
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

std::optional<uint64_t> SwiftLanguageRuntime::GetMemberVariableOffset(
    CompilerType instance_type, ValueObject *instance,
    llvm::StringRef member_name, Status *error) {
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
      *error = Status::FromErrorString("could not resolve member offset");
  }
  return offset;
}

namespace {

CompilerType GetTypeFromTypeRef(TypeSystemSwiftTypeRef &ts,
                                const swift::reflection::TypeRef *type_ref,
                                swift::Mangle::ManglingFlavor flavor) {
  if (!type_ref)
    return {};
  swift::Demangle::Demangler dem;
  swift::Demangle::NodePointer node = type_ref->getDemangling(dem);
  // TODO: the mangling flavor should come from the TypeRef.
  return ts.RemangleAsType(dem, node, flavor);
}

struct ExistentialSyntheticChild {
  std::string name;
  std::function<CompilerType(void)> get_type;
};

/// Return the synthetic children of an Existential type.
/// The closure in get_type will depend on ts and tr.
/// Roughly corresponds to GetExistentialTypeChild() in SwiftASTContext.cpp
llvm::SmallVector<ExistentialSyntheticChild, 4>
GetExistentialSyntheticChildren(TypeSystemSwiftTypeRef &ts,
                                const swift::reflection::TypeRef *tr,
                                const swift::reflection::TypeInfo *ti,
                                swift::Mangle::ManglingFlavor flavor) {
  llvm::SmallVector<ExistentialSyntheticChild, 4> children;
  auto *protocol_composition_tr =
      llvm::dyn_cast<swift::reflection::ProtocolCompositionTypeRef>(tr);
  if (!protocol_composition_tr)
    return children;
  if (!ti)
    return children;
  auto *rti = llvm::dyn_cast<swift::reflection::RecordTypeInfo>(ti);
  if (rti || llvm::isa<swift::reflection::ReferenceTypeInfo>(ti)) {
    TypeSystemSwiftTypeRefSP ts_sp = ts.GetTypeSystemSwiftTypeRef();
    children.push_back({"object", [=]() {
                          if (auto *super_class_tr =
                                  protocol_composition_tr->getSuperclass())
                            return GetTypeFromTypeRef(*ts_sp, super_class_tr,
                                                      flavor);
                          else
                            return rti ? ts_sp->GetBuiltinUnknownObjectType()
                                       : ts_sp->GetBuiltinRawPointerType();
                        }});
    if (rti) {
      auto &fields = rti->getFields();
      // We replaced "object" with a more specific type.
      for (unsigned i = 1; i < fields.size(); ++i) {
        TypeSystemSwiftTypeRefSP ts_sp = ts.GetTypeSystemSwiftTypeRef();
        auto *type_ref = fields[i].TR;
        children.push_back(
            {fields[i].Name,
             [=]() { return GetTypeFromTypeRef(*ts_sp, type_ref, flavor); }});
      }
    }
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
  llvm::dbgs() << function << ": unimplemented type info in "
               << type.GetMangledTypeName() << "\n";
  if (ModuleList::GetGlobalModuleListProperties().GetSwiftValidateTypeSystem())
    assert(false && "not implemented");
#endif
}

/// Resolve a (chain of) typedefs.
CompilerType GetTypedefedTypeRecursive(CompilerType type) {
  // Guard against cycles.
  for (unsigned i = 0; i < 128; ++i) {
    if (!type.IsTypedefType())
      return type;
    type = type.GetTypedefedType();
  }
  return type;
}

} // namespace

/// This class exists to unify iteration over runtime type
/// information. The visitor callback has closure parameters that can
/// be called to make additional expensive queries on a child.
///
/// TODO: This is not the final evolution step.
///
/// - We probably should remove the "depth" parameter entirely and
///   implement the access path computation for
///   GetIndexOfChildMemberWithName at a different layer.
///
/// -  We could cache the results for the last execution context.
class SwiftRuntimeTypeVisitor {
  SwiftLanguageRuntime &m_runtime;
  ExecutionContext m_exe_ctx;
  swift::Mangle::ManglingFlavor m_flavor =
      swift::Mangle::ManglingFlavor::Default;
  CompilerType m_type;
  ValueObject *m_valobj = nullptr;
  bool m_hide_superclass = false;
  bool m_include_clang_types = false;
  bool m_visit_superclass = false;

  void SetFlavor() {
    if (auto ts_sp =
            m_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwiftTypeRef>())
      m_flavor = ts_sp->GetManglingFlavor(&m_exe_ctx);
  }

public:
  struct ChildInfo {
    uint32_t byte_size = 0;
    int32_t byte_offset = 0;
    uint32_t bitfield_bit_size = 0;
    uint32_t bitfield_bit_offset = 0;
    bool is_base_class = false;
    bool is_deref_of_parent = false;
    bool is_enum = false;
    uint64_t language_flags = 0;
  };
  using GetChildInfoClosure = std::function<llvm::Expected<ChildInfo>(void)>;
  using GetChildNameClosure = std::function<std::string(void)>;

  /// If callback returns an error the visitor is cancelled and the
  /// error is returned. The \c type parameter is the type of the
  /// visited child, the \c depth parameter is the nesting depth.
  using VisitCallback = std::function<llvm::Error(
      CompilerType, unsigned, GetChildNameClosure, GetChildInfoClosure)>;
  SwiftRuntimeTypeVisitor(SwiftLanguageRuntime &runtime, CompilerType type,
                          ValueObject *valobj)
      : m_runtime(runtime), m_type(type), m_valobj(valobj) {
    if (valobj)
      m_exe_ctx = valobj->GetExecutionContextRef();
    SetFlavor();
  }
  SwiftRuntimeTypeVisitor(SwiftLanguageRuntime &runtime, CompilerType type,
                          ExecutionContextScope *exe_scope,
                          bool hide_superclass, bool include_clang_types)
      : m_runtime(runtime), m_type(type), m_hide_superclass(hide_superclass),
        m_include_clang_types(include_clang_types) {
    if (exe_scope)
      exe_scope->CalculateExecutionContext(m_exe_ctx);
    SetFlavor();
  }
  SwiftRuntimeTypeVisitor(SwiftLanguageRuntime &runtime, CompilerType type,
                          ExecutionContext *exe_ctx, bool hide_superclass,
                          bool include_clang_types, bool visit_superclass)
      : m_runtime(runtime), m_type(type), m_hide_superclass(hide_superclass),
        m_include_clang_types(include_clang_types),
        m_visit_superclass(visit_superclass) {
    if (exe_ctx)
      m_exe_ctx = *exe_ctx;
    SetFlavor();
  }
  llvm::Error VisitAllChildren(VisitCallback callback) {
    return VisitImpl({}, callback).takeError();
  }
  llvm::Expected<unsigned> CountChildren() { return VisitImpl({}, {}); }
  llvm::Error VisitChildAtIndex(unsigned idx, VisitCallback callback) {
    return VisitImpl(idx, callback).takeError();
  }

private:
  /// Implements all three kinds of traversals in one function to
  /// centralize the logic. If not counting (= visit_callback exists)
  /// the function returns 0 on success.
  llvm::Expected<unsigned> VisitImpl(std::optional<unsigned> visit_only,
                                     VisitCallback visit_callback);
};

llvm::Expected<unsigned>
SwiftRuntimeTypeVisitor::VisitImpl(std::optional<unsigned> visit_only,
                                   VisitCallback visit_callback)

{
  if (!m_type)
    return llvm::createStringError("invalid type");

  const unsigned success = 0;
  bool count_only = !visit_callback;
  auto ts_or_err =
      SwiftLanguageRuntime::GetReflectionTypeSystem(m_type, m_exe_ctx);
  if (!ts_or_err)
    return ts_or_err.takeError();
  auto &ts = *ts_or_err->get();

  // Deal with the LLDB-only SILPackType variant.
  if (auto pack_type_info = ts.IsSILPackType(m_type)) {
    if (!pack_type_info->expanded)
      return llvm::createStringError("unimplemented kind of SIL pack type");
    if (count_only)
      return pack_type_info->count;

    auto visit_pack_element = [&](CompilerType pack_element_type,
                                  unsigned idx) {
      auto get_name = [&]() -> std::string {
        std::string name;
        llvm::raw_string_ostream os(name);
        os << '.' << idx;
        return name;
      };
      auto get_info = [&]() -> llvm::Expected<ChildInfo> {
        ChildInfo child;
        auto size_or_err = m_runtime.GetBitSize(
            pack_element_type, m_exe_ctx.GetBestExecutionContextScope());
        if (!size_or_err)
          return size_or_err.takeError();
        child.byte_size = *size_or_err;
        int stack_dir = -1;
        child.byte_offset = ts.GetPointerByteSize() * idx * stack_dir;
        child.bitfield_bit_size = 0;
        child.bitfield_bit_offset = 0;
        child.is_base_class = false;
        child.is_deref_of_parent = true;
        return child;
      };
      return visit_callback(pack_element_type, 0, get_name, get_info);
    };
    for (unsigned i = 0; i < pack_type_info->count; ++i)
      if (!visit_only || *visit_only == i)
        if (auto err =
                visit_pack_element(ts.GetSILPackElementAtIndex(m_type, i), i))
          return err;
    return success;
  }

  // FIXME: Remove this entire mode.
  assert(!m_include_clang_types || (m_include_clang_types && count_only));
  if (m_include_clang_types && count_only) {
    CompilerType clang_type = m_runtime.LookupAnonymousClangType(
        m_type.GetMangledTypeName().AsCString());
    if (!clang_type)
      ts.IsImportedType(m_type.GetOpaqueQualType(), &clang_type);
    if (clang_type) {
      clang_type = GetTypedefedTypeRecursive(clang_type);
      bool is_signed;
      if (clang_type.IsEnumerationType(is_signed))
        return 1;
      return clang_type.GetNumChildren(true, &m_exe_ctx);
    }
  }

  // The actual conversion from the FieldInfo record.
  auto visit_field_info =
      [&](const swift::reflection::FieldInfo &field,
          std::optional<TypeSystemSwift::TupleElement> tuple,
          bool hide_existentials, bool is_enum,
          unsigned depth = 0) -> llvm::Expected<unsigned> {
    auto get_name = [&]() -> std::string {
      return tuple ? tuple->element_name.GetStringRef().str() : field.Name;
    };
    // SwiftASTContext hardcodes the members of protocols as raw
    // pointers. Remote Mirrors reports them as UnknownObject instead.
    if (hide_existentials && ts.IsExistentialType(m_type.GetOpaqueQualType())) {
      auto get_info = [&]() -> llvm::Expected<ChildInfo> {
        ChildInfo child;
        child.byte_size = field.TI.getSize();
        child.byte_offset = field.Offset;
        child.bitfield_bit_size = 0;
        child.bitfield_bit_offset = 0;
        child.is_base_class = false;
        child.is_deref_of_parent = false;
        child.language_flags = 0;
        return child;
      };
      CompilerType type = ts.GetRawPointerType();
      if (auto err = visit_callback(type, depth, get_name, get_info))
        return err;
      return success;
    }
    CompilerType field_type;
    if (tuple)
      field_type = tuple->element_type;
    else {
      if (!field_type)
        field_type = GetTypeFromTypeRef(ts, field.TR, m_flavor);
    }
    auto get_info = [&]() -> llvm::Expected<ChildInfo> {
      ChildInfo child;
      child.byte_size = field.TI.getSize();
      // Bug-for-bug compatibility. See comment in
      // SwiftASTContext::GetBitSize().
      if (field_type.IsFunctionType())
        child.byte_size = ts.GetPointerByteSize();
      child.byte_offset = field.Offset;
      child.bitfield_bit_size = 0;
      child.bitfield_bit_offset = 0;
      child.is_base_class = false;
      child.is_deref_of_parent = false;
      child.language_flags = 0;
      return child;
    };
    if (auto err = visit_callback(field_type, depth, get_name, get_info))
      return err;
    return success;
  };

  // Try the static type metadata.
  const swift::reflection::TypeRef *tr = nullptr;
  auto ti_or_err = m_runtime.GetSwiftRuntimeTypeInfo(
      m_type, m_exe_ctx.GetBestExecutionContextScope(), &tr);
  if (!ti_or_err)
    return ti_or_err.takeError();
  auto *ti = &*ti_or_err;

  // Structs and Tuples.
  if (auto *rti =
          llvm::dyn_cast_or_null<swift::reflection::RecordTypeInfo>(ti)) {
    auto fields = rti->getFields();

    // Handle tuples.
    std::optional<TypeSystemSwift::TupleElement> tuple;
    if (rti->getRecordKind() == swift::reflection::RecordKind::Tuple) {
      if (count_only)
        return fields.size();
      for (unsigned i = 0; i < fields.size(); ++i)
        if (!visit_only || *visit_only == i) {
          tuple = ts.GetTupleElement(m_type.GetOpaqueQualType(), i);
          auto result = visit_field_info(fields[i], tuple,
                                         /*hide_existentials=*/true,
                                         /*is_enum=*/false);
          if (!result)
            return result.takeError();
        }
      return success;
    }

    if (rti->getRecordKind() ==
        swift::reflection::RecordKind::OpaqueExistential) {
      auto visit_existential = [&](unsigned idx) -> llvm::Expected<unsigned> {
        // Compatibility with SwiftASTContext.
        if (idx < 3) {
          auto get_name = [&]() -> std::string {
            std::string child_name = "payload_data_";
            child_name += ('0' + idx);
            return child_name;
          };
          auto get_info = [&]() -> llvm::Expected<ChildInfo> {
            ChildInfo child;
            child.byte_size = ts.GetPointerByteSize();
            child.byte_offset = ts.GetPointerByteSize() * idx;
            child.bitfield_bit_size = 0;
            child.bitfield_bit_offset = 0;
            child.is_base_class = false;
            child.is_deref_of_parent = false;
            child.language_flags = 0;
            return child;
          };
          if (auto err =
                  visit_callback(ts.GetRawPointerType(), 0, get_name, get_info))
            return err;
          return success;
        }
        return visit_field_info(fields[idx - 3], tuple,
                                /*hide_existentials=*/false,
                                /*is_enum=*/false);
      };
      if (count_only)
        return fields.size() + 3;

      for (unsigned i = 0; i < fields.size() + 3; ++i)
        if (!visit_only || *visit_only == i) {
          auto result = visit_existential(i);
          if (!result)
            return result.takeError();
        }
      return success;
    }

    if (rti->getRecordKind() ==
        swift::reflection::RecordKind::ClassExistential) {
      // Compatibility with SwiftASTContext.
      auto children = GetExistentialSyntheticChildren(ts, tr, ti, m_flavor);
      if (count_only)
        return children.size();
      auto visit_existential = [&](ExistentialSyntheticChild c, unsigned idx) {
        auto get_name = [&]() -> std::string { return c.name; };
        auto get_info = [&]() -> llvm::Expected<ChildInfo> {
          ChildInfo child;
          child.byte_size = ts.GetPointerByteSize();
          child.byte_offset = ts.GetPointerByteSize() * idx;
          child.bitfield_bit_size = 0;
          child.bitfield_bit_offset = 0;
          child.is_base_class = false;
          child.is_deref_of_parent = false;
          child.language_flags = 0;
          return child;
        };
        return visit_callback(c.get_type(), 0, get_name, get_info);
      };
      for (unsigned i = 0; i < children.size(); ++i)
        if (!visit_only || *visit_only == i)
          if (auto err = visit_existential(children[i], i))
            return err;
      return success;
    }

    // Structs.
    if (count_only)
      return fields.size();
    for (unsigned i = 0; i < fields.size(); ++i)
      if (!visit_only || *visit_only == i) {
        auto result = visit_field_info(fields[i], tuple,
                                       /*hide_existentials=*/true,
                                       /*is_enum=*/false);
        if (!result)
          return result.takeError();
      }
    return success;
  }

  // Enums.
  if (llvm::dyn_cast_or_null<swift::reflection::EnumTypeInfo>(ti)) {
    // The dynamic "child" of a payload-carrying enum is provided by
    // the Enum synthetic child provider.
    return success;
  }

  // Objects.
  if (auto *rti =
          llvm::dyn_cast_or_null<swift::reflection::ReferenceTypeInfo>(ti)) {
    // Is this an Existential?
    unsigned i = 0;
    auto children = GetExistentialSyntheticChildren(ts, tr, ti, m_flavor);
    if (children.size()) {
      if (count_only)
        return children.size();
      auto visit_existential = [&](ExistentialSyntheticChild c, unsigned idx) {
        auto get_name = [&]() -> std::string { return c.name; };
        auto get_info = [&]() -> llvm::Expected<ChildInfo> {
          ChildInfo child;
          child.byte_size = ts.GetPointerByteSize();
          child.byte_offset = ts.GetPointerByteSize() * idx;
          child.bitfield_bit_size = 0;
          child.bitfield_bit_offset = 0;
          child.is_base_class = false;
          child.is_deref_of_parent = false;
          child.language_flags = 0;
          return child;
        };
        return visit_callback(c.get_type(), 0, get_name, get_info);
      };
      for (unsigned i = 0; i < children.size(); ++i)
        if (!visit_only || *visit_only == i)
          if (auto err = visit_existential(children[i], i))
            return err;
      return success;
    }

    // Objects.
    switch (rti->getReferenceKind()) {
    case swift::reflection::ReferenceKind::Weak:
    case swift::reflection::ReferenceKind::Unowned:
    case swift::reflection::ReferenceKind::Unmanaged:
      // Weak references are implicitly Optionals, optionals have only synthetic
      // children.
      if (count_only)
        return 0;
      if (!visit_only || *visit_only == 0) {
        return success;
      }
      break;
    default:
      break;
    }

    bool found_start = false;
    using namespace swift::Demangle;
    Demangler dem;
    auto mangled = m_type.GetMangledTypeName().GetStringRef();
    NodePointer type_node = dem.demangleSymbol(mangled);
    llvm::StringRef type_name = TypeSystemSwiftTypeRef::GetBaseName(
        ts.CanonicalizeSugar(dem, type_node));

    ThreadSafeReflectionContext reflection_ctx =
        m_runtime.GetReflectionContext();
    if (!reflection_ctx)
      return llvm::createStringError("no reflection context");

    // Try the instance type metadata.
    if (!m_valobj) {
      LLDBTypeInfoProvider tip(m_runtime, ts);
      auto cti_or_err = reflection_ctx->GetClassInstanceTypeInfo(
          *tr, &tip, ts.GetDescriptorFinder());
      if (!cti_or_err)
        return cti_or_err.takeError();
      if (auto *rti = llvm::dyn_cast_or_null<swift::reflection::RecordTypeInfo>(
              &*cti_or_err)) {
        LLDB_LOG(GetLog(LLDBLog::Types),
                 "{0}: class RecordTypeInfo(num_fields={1})",
                 m_type.GetMangledTypeName(), rti->getNumFields());

        if (count_only) {
          // The superclass, if any, is an extra child.
          if (!m_hide_superclass &&
              reflection_ctx->LookupSuperclass(*tr, ts.GetDescriptorFinder()))
            return rti->getNumFields() + 1;
          return rti->getNumFields();
        }
        if (m_visit_superclass) {
          unsigned depth = 0;
          reflection_ctx->ForEachSuperClassType(
              &tip, ts.GetDescriptorFinder(), tr, [&](SuperClassType sc) {
                auto *tr = sc.get_typeref();
                if (!tr || llvm::isa<swift::reflection::ObjCClassTypeRef>(tr))
                  return true;
                auto *cti = sc.get_record_type_info();
                if (!cti)
                  return true;

                if (auto *super_tr = reflection_ctx->LookupSuperclass(
                        *tr, ts.GetDescriptorFinder()))
                  if (auto error = visit_callback(
                          GetTypeFromTypeRef(ts, super_tr, m_flavor), depth,
                          []() -> std::string { return "<base class>"; },
                          []() -> llvm::Expected<ChildInfo> {
                            return ChildInfo();
                          })) {
                    LLDB_LOG_ERRORV(GetLog(LLDBLog::Types), std::move(error),
                                    "{0}");
                    return true;
                  }

                auto &fields = cti->getFields();
                for (unsigned i = 0; i < fields.size(); ++i)
                  if (!visit_only || *visit_only == i) {
                    auto result = visit_field_info(fields[i], {},
                                                   /*hide_existentials=*/true,
                                                   /*is_enum=*/false, depth);
                    if (!result) {
                      LLDB_LOG_ERRORV(GetLog(LLDBLog::Types),
                                      result.takeError(), "{0}");
                      return true;
                    }
                  }
                ++depth;
                return false;
              });
          return success;
        }
        if (auto *super_tr = reflection_ctx->LookupSuperclass(
                *tr, ts.GetDescriptorFinder())) {
          auto get_name = []() -> std::string { return "<base class>"; };
          auto get_info = []() -> llvm::Expected<ChildInfo> {
            return ChildInfo();
          };

          if (auto error =
                  visit_callback(GetTypeFromTypeRef(ts, super_tr, m_flavor), 0,
                                 get_name, get_info))
            return error;
        }

        auto &fields = rti->getFields();
        for (unsigned i = 0; i < fields.size(); ++i)
          if (!visit_only || *visit_only == i) {
            auto result = visit_field_info(fields[i], {},
                                           /*hide_existentials=*/true,
                                           /*is_enum=*/false);
            if (!result)
              return result.takeError();
          }
        return success;
      }
      return llvm::createStringError("class instance is not a record: " +
                                     m_type.GetMangledTypeName().GetString());
    }

    CompilerType instance_type = m_valobj->GetCompilerType();
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

    LLDBTypeInfoProvider tip(m_runtime, ts);
    lldb::addr_t instance = ::MaskMaybeBridgedPointer(
        m_runtime.GetProcess(), m_valobj->GetPointerValue().address);

    // Try out the instance pointer based super class traversal first, as its
    // usually faster.
    reflection_ctx->ForEachSuperClassType(&tip, ts.GetDescriptorFinder(),
                                          instance, superclass_finder);

    if (supers.empty())
      // If the pointer based super class traversal failed (this may happen
      // when metadata is not present in the binary, for example: embedded
      // Swift), try the typeref based one next.
      reflection_ctx->ForEachSuperClassType(&tip, ts.GetDescriptorFinder(), tr,
                                            superclass_finder);

    if (supers.empty() && tr) {
      LLDB_LOG(GetLog(LLDBLog::Types),
               "Couldn't find the type metadata for {0} in instance",
               m_type.GetTypeName());

      auto cti_or_err = reflection_ctx->GetClassInstanceTypeInfo(
          *tr, &tip, ts.GetDescriptorFinder());
      const swift::reflection::TypeInfo *cti = nullptr;
      if (cti_or_err)
        cti = &*cti_or_err;
      else
        LLDB_LOG_ERRORV(GetLog(LLDBLog::Types), cti_or_err.takeError(), "{0}");
      if (auto *rti =
              llvm::dyn_cast_or_null<swift::reflection::RecordTypeInfo>(cti)) {
        auto fields = rti->getFields();

        if (count_only)
          return fields.size();

        for (unsigned i = 0; i < fields.size(); ++i)
          if (!visit_only || *visit_only == i) {
            auto result = visit_field_info(fields[i], {},
                                           /*hide_existentials=*/true,
                                           /*is_enum=*/false);
            if (!result)
              return result.takeError();
            return success;
          }
      }
    }

    // Handle the artificial base class fields.
    if (supers.size() > 1) {
      auto *type_ref = supers[1].get_typeref();
      auto *objc_tr =
          llvm::dyn_cast_or_null<swift::reflection::ObjCClassTypeRef>(type_ref);
      // SwiftASTContext hides the ObjC base class for Swift classes.
      if (!m_hide_superclass &&
          (!objc_tr || objc_tr->getName() != "_TtCs12_SwiftObject")) {
        if (!visit_only || *visit_only == i) {
          // A synthetic field for the base class itself.  Only the direct
          // base class gets injected. Its parent will be a nested
          // field in the base class.
          if (!type_ref) {
            auto get_name = [&]() -> std::string { return "<base class>"; };
            auto get_info = [&]() -> llvm::Expected<ChildInfo> {
              return ChildInfo();
            };
            if (auto err =
                    visit_callback(CompilerType(), 0, get_name, get_info))
              return err;
            if (visit_only)
              return success;
          }

          CompilerType super_type = GetTypeFromTypeRef(ts, type_ref, m_flavor);
          auto get_name = [&]() -> std::string {
            auto child_name = super_type.GetTypeName().GetStringRef().str();
            // FIXME: This should be fixed in GetDisplayTypeName instead!
            if (child_name == "__C.NSObject")
              child_name = "ObjectiveC.NSObject";
            return child_name;
          };
          auto get_info = [&]() -> llvm::Expected<ChildInfo> {
            ChildInfo child;
            if (auto *rti = supers[1].get_record_type_info())
              child.byte_size = rti->getSize();
            // FIXME: This seems wrong in SwiftASTContext.
            child.byte_size = ts.GetPointerByteSize();
            child.byte_offset = 0;
            child.bitfield_bit_size = 0;
            child.bitfield_bit_offset = 0;
            child.is_base_class = true;
            child.is_deref_of_parent = false;
            child.language_flags = 0;
            return child;
          };
          if (auto err = visit_callback(super_type, 0, get_name, get_info))
            return err;
          if (visit_only)
            return success;
        }
        ++i;
      }
    }
    if (supers.empty())
      return llvm::createStringError("could not get instance type info");

    // Handle the "real" fields.
    auto *object = supers[0].get_record_type_info();
    if (!object)
      return llvm::createStringError("no record type info");
    auto &fields = object->getFields();
    if (count_only)
      return i + fields.size();
    for (unsigned j = 0; j < fields.size(); ++j)
      if (!visit_only || *visit_only == i + j) {
        auto result = visit_field_info(fields[j], {},
                                       /*hide_existentials=*/true,
                                       /*is_enum=*/false);
        if (!result)
          return result.takeError();
      }
    return success;
  }

  // Fixed array.
  if (auto *ati = llvm::dyn_cast<swift::reflection::ArrayTypeInfo>(ti)) {
    LLDB_LOG(GetLog(LLDBLog::Types), "{0}: ArrayTypeInfo()",
             m_type.GetMangledTypeName().GetCString());
    auto *el_ti = ati->getElementTypeInfo();
    if (!el_ti)
      return llvm::createStringError("array without element type info: " +
                                     m_type.GetMangledTypeName().GetString());
    // We could also get the value out of the mangled type name, but
    // this is cheaper.
    unsigned stride = el_ti->getStride();
    if (!stride)
      return llvm::createStringError("Array without element stride: " +
                                     m_type.GetMangledTypeName().GetString());
    unsigned count = ati->getSize() / stride;
    if (count_only)
      return count;

    swift::Demangle::Demangler dem;
    swift::Demangle::NodePointer global =
        dem.demangleSymbol(m_type.GetMangledTypeName().GetStringRef());
    using Kind = Node::Kind;
    auto *dem_array_type = swift_demangle::ChildAtPath(
        global, {Kind::TypeMangling, Kind::Type, Kind::BuiltinFixedArray});
    if (!dem_array_type || dem_array_type->getNumChildren() != 2)
      return llvm::createStringError("Expected fixed array, but found: " +
                                     m_type.GetMangledTypeName().GetString());
    auto flavor = SwiftLanguageRuntime::GetManglingFlavor(
        m_type.GetMangledTypeName().GetStringRef());
    CompilerType type =
        ts.RemangleAsType(dem, dem_array_type->getChild(1), flavor);

    auto visit_element = [&](unsigned idx) -> llvm::Error {
      auto get_name = [&]() -> std::string {
        std::string child_name;
        llvm::raw_string_ostream(child_name) << idx;
        return child_name;
      };
      auto get_info = [&]() -> llvm::Expected<ChildInfo> {
        ChildInfo child;
        child.byte_size = el_ti->getSize();
        child.byte_offset = el_ti->getStride() * idx;
        // FIXME:
        // if (!ignore_array_bounds &&
        //     (int64_t)child_byte_offset > (int64_t)ati->getSize())
        //   return llvm::createStringError("array index out of bounds");

        child.bitfield_bit_size = 0;
        child.bitfield_bit_offset = 0;
        child.is_base_class = false;
        child.is_deref_of_parent = false;
        child.language_flags = 0;
        return child;
      };
      return visit_callback(type, 0, get_name, get_info);
    };
    for (unsigned i = 0; i < count; ++i)
      if (!visit_only || *visit_only == i)
        if (auto err = visit_element(i))
          return err;
    return success;
  }
  if (llvm::dyn_cast_or_null<swift::reflection::BuiltinTypeInfo>(ti)) {
    // Clang enums have an artificial rawValue property. We could
    // consider handling them here, but
    // TypeSystemSwiftTypeRef::GetChildCompilerTypeAtIndex can also
    // handle them and without a Process.
    if (!TypeSystemSwiftTypeRef::IsBuiltinType(m_type) &&
        !Flags(m_type.GetTypeInfo()).AnySet(eTypeIsMetatype) &&
        !m_type.IsFunctionType()) {
      LLDB_LOG(GetLog(LLDBLog::Types),
               "{0}: unrecognized builtin type info or this is a Clang type "
               "without DWARF debug info",
               m_type.GetMangledTypeName());
      return llvm::createStringError("missing debug info for Clang type \"" +
                                     m_type.GetDisplayTypeName().GetString() +
                                     "\"");
    }
    if (count_only)
      return 0;
    if (auto err = visit_callback(
            CompilerType(), 0, []() -> std::string { return ""; },
            []() -> llvm::Expected<ChildInfo> { return ChildInfo(); }))
      return err;
    return success;
  }
  LogUnimplementedTypeKind(__FUNCTION__, m_type);
  return llvm::createStringError("not implemented");
}

llvm::Expected<uint32_t>
SwiftLanguageRuntime::GetNumFields(CompilerType type,
                                   ExecutionContext *exe_ctx) {
  if (exe_ctx)
    return GetNumChildren(type, exe_ctx->GetBestExecutionContextScope(), false,
                          false);
  return llvm::createStringError("no execution context");
}

llvm::Expected<uint32_t> SwiftLanguageRuntime::GetNumChildren(
    CompilerType type, ExecutionContextScope *exe_scope,
    bool include_superclass, bool include_clang_types) {
  SwiftRuntimeTypeVisitor visitor(*this, type, exe_scope, !include_superclass,
                                  include_clang_types);
  return visitor.CountChildren();
}

llvm::Expected<std::string> SwiftLanguageRuntime::GetEnumCaseName(
    CompilerType type, const DataExtractor &data, ExecutionContext *exe_ctx) {
  using namespace swift::reflection;
  using namespace swift::remote;
  auto ti_or_err = GetSwiftRuntimeTypeInfo(type, exe_ctx->GetFramePtr());
  if (!ti_or_err)
    return ti_or_err.takeError();
  auto *ti = &*ti_or_err;

  // FIXME: Not reported as an error. There seems to be an odd
  // compiler optimization happening with single-case payload carrying
  // enums, which report their type as the inner type.
  if (ti->getKind() != TypeInfoKind::Enum)
    return "";

  auto *eti = llvm::cast<EnumTypeInfo>(ti);
  auto buffer_holder = PushLocalBuffer((int64_t)data.GetDataStart(), data.GetByteSize());
  RemoteAddress addr = RemoteAddress((uint64_t)data.GetDataStart(), swift::reflection::RemoteAddress::DefaultAddressSpace);
  int case_index;
  if (eti->projectEnumValue(*GetMemoryReader(), addr, &case_index))
    return eti->getCases()[case_index].Name;

  // TODO: uncomment this after fixing projection for every type:
  // rdar://138424904
  LogUnimplementedTypeKind(__FUNCTION__, type);
  return llvm::createStringError("unimplemented enum kind");
}

llvm::Expected<ValueObjectSP>
SwiftLanguageRuntime::ProjectEnum(ValueObject &valobj) {
  TypeSystemSwiftTypeRefSP ts_sp;
  if (auto target_sp = valobj.GetTargetSP()) {
    auto type_system_or_err =
        target_sp->GetScratchTypeSystemForLanguage(lldb::eLanguageTypeSwift);
    if (!type_system_or_err)
      return type_system_or_err.takeError();
    auto ts_ptr = type_system_or_err->get();
    ts_sp = llvm::cast<TypeSystemSwift>(ts_ptr)->GetTypeSystemSwiftTypeRef();
  }
  if (!ts_sp)
    return llvm::createStringError("no target");
  auto &ts = *ts_sp;
  auto exe_ctx = valobj.GetExecutionContextRef().Lock(true);
  CompilerType enum_type = valobj.GetCompilerType();

  auto ti_or_err = GetSwiftRuntimeTypeInfo(
      enum_type, exe_ctx.GetBestExecutionContextScope());
  if (!ti_or_err)
    return ti_or_err.takeError();
  auto *ti = &*ti_or_err;
  auto flavor =
      SwiftLanguageRuntime::GetManglingFlavor(enum_type.GetMangledTypeName());

  auto project_indirect_enum =
      [&](uint64_t offset, std::string name) -> llvm::Expected<ValueObjectSP> {
    lldb::addr_t pointer = ::MaskMaybeBridgedPointer(
        GetProcess(), valobj.GetPointerValue().address);
    lldb::addr_t payload = pointer + offset;

    ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
    // The indirect enum field should point to a closure context.
    LLDBTypeInfoProvider tip(*this, ts);
    auto ti_or_err = reflection_ctx->GetTypeInfoFromInstance(
        payload, &tip, ts.GetDescriptorFinder());
    if (!ti_or_err)
      return ti_or_err.takeError();

    CompilerType payload_type;
    auto *ti = &*ti_or_err;
    if (auto *rti = llvm::dyn_cast<swift::reflection::RecordTypeInfo>(ti)) {
      switch (rti->getRecordKind()) {
      case swift::reflection::RecordKind::ClosureContext: {
        if (!rti->getFields().size())
          return llvm::createStringError("closure context has no fields");
        auto &field = rti->getFields()[0];
        auto *type_ref = field.TR;
        payload += field.Offset;
        payload_type = GetTypeFromTypeRef(ts, type_ref, flavor);
        break;
      }
      case swift::reflection::RecordKind::Tuple: {
        std::vector<TypeSystemSwift::TupleElement> elts;
        for (auto &field : rti->getFields())
          elts.emplace_back(ConstString(),
                            GetTypeFromTypeRef(ts, field.TR, flavor));
        payload_type = ts.CreateTupleType(elts);
        break;
      }
      default:
        return llvm::createStringError(
            "unexpected kind of indirect record enum");
      }
    } else {
      payload_type = ts.GetBuiltinRawPointerType();
    }

    return ValueObjectMemory::Create(exe_ctx.GetBestExecutionContextScope(),
                                     "$indirect." + name, payload,
                                     payload_type);
  };

  // Is this single-case indirect enum? These get lowered into their payload
  // type.
  if (ti->getKind() != swift::reflection::TypeInfoKind::Enum)
    return project_indirect_enum(0, "$single_case");

  // Prepare to project the enum to get the active case.
  MemoryReaderLocalBufferHolder holder;
  auto [addr, address_type] = valobj.GetAddressOf(false);
  if (addr == LLDB_INVALID_ADDRESS || addr == 0) {
    Value &value = valobj.GetValue();
    switch (value.GetValueType()) {
    default:
      return llvm::createStringError("unexpected address space");
    case Value::ValueType::HostAddress: {
      addr = value.GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
      if (addr == LLDB_INVALID_ADDRESS)
        return llvm::createStringError("could not get address");
      auto size = valobj.GetByteSize();
      if (!size)
        return size.takeError();
      holder = PushLocalBuffer(addr, *size);
      break;
    }
    case Value::ValueType::Scalar: {
      addr = value.GetScalar().ULongLong(LLDB_INVALID_ADDRESS);
      holder = PushLocalBuffer((uint64_t)&addr, sizeof(addr));
      break;
    }
    }
  }

  auto remote_addr = swift::reflection::RemoteAddress(addr, 0);
  int case_index;
  auto *eti = llvm::cast<swift::reflection::EnumTypeInfo>(ti);
  if (!eti->projectEnumValue(*GetMemoryReader(), remote_addr, &case_index))
    return llvm::createStringError("could not project enum case");
  const swift::reflection::FieldInfo &field_info = eti->getCases()[case_index];

  // Is there a payload?
  if (!field_info.TR)
    return ValueObjectSP();

  bool is_indirect_enum =
      !field_info.Offset && field_info.TR &&
      llvm::isa<swift::reflection::BuiltinTypeRef>(field_info.TR) &&
      llvm::isa<swift::reflection::ReferenceTypeInfo>(field_info.TI) &&
      llvm::cast<swift::reflection::ReferenceTypeInfo>(field_info.TI)
              .getReferenceKind() == swift::reflection::ReferenceKind::Strong;
  if (is_indirect_enum)
    return project_indirect_enum(field_info.Offset, field_info.Name);

  CompilerType projected_type = GetTypeFromTypeRef(ts, field_info.TR, flavor);
  if (field_info.Offset != 0) {
    assert(false);
    return llvm::createStringError("enum with unexpected offset");
  }
  return ValueObjectCast::Create(valobj, ConstString(field_info.Name),
                                 projected_type);
}

std::pair<SwiftLanguageRuntime::LookupResult, std::optional<size_t>>
SwiftLanguageRuntime::GetIndexOfChildMemberWithName(
    CompilerType type, llvm::StringRef name, ExecutionContext *exe_ctx,
    bool omit_empty_base_classes, std::vector<uint32_t> &child_indexes) {
  SwiftRuntimeTypeVisitor visitor(*this, type, exe_ctx, false, false, true);
  bool found = false;
  unsigned i = 0, last_depth = 0;
  llvm::Error error = visitor.VisitAllChildren(
      [&](CompilerType type, unsigned depth, auto get_child_name,
          auto get_child_info) -> llvm::Error {
        if (depth != last_depth) {
          i = 0;
          last_depth = depth;
        }
        if (found)
          return llvm::Error::success();
        auto info_or_err = get_child_info();
        if (!info_or_err)
          return info_or_err.takeError();
        // All enum children are index 0.
        if (info_or_err->is_enum || name == get_child_name()) {
          // The only access paths supperted are into base classes,
          // which are always at index 0.
          for (unsigned j = 0; j < depth; ++j)
            child_indexes.push_back(0);
          child_indexes.push_back(i);
          found = true;
        }
        ++i;
        return llvm::Error::success();
      });
  if (error) {
    llvm::consumeError(std::move(error));
    return {SwiftLanguageRuntime::eError, {}};
  }

  if (!found)
    return {SwiftLanguageRuntime::eNotFound, {}};
  return {SwiftLanguageRuntime::eFound, child_indexes.size()};
}

llvm::Expected<CompilerType> SwiftLanguageRuntime::GetChildCompilerTypeAtIndex(
    CompilerType type, size_t idx, bool transparent_pointers,
    bool omit_empty_base_classes, bool ignore_array_bounds,
    std::string &child_name, uint32_t &child_byte_size,
    int32_t &child_byte_offset, uint32_t &child_bitfield_bit_size,
    uint32_t &child_bitfield_bit_offset, bool &child_is_base_class,
    bool &child_is_deref_of_parent, ValueObject *valobj,
    uint64_t &language_flags) {
  CompilerType child_type;
  bool found = false;
  SwiftRuntimeTypeVisitor visitor(*this, type, valobj);
  llvm::Error error = visitor.VisitChildAtIndex(
      idx,
      [&](CompilerType type, unsigned depth, auto get_child_name,
          auto get_child_info) -> llvm::Error {
        auto info_or_err = get_child_info();
        if (!info_or_err)
          return info_or_err.takeError();
        auto child = *info_or_err;
        found = true;
        child_type = type;
        child_name = get_child_name();
        child_byte_size = child.byte_size;
        child_byte_offset = child.byte_offset;
        child_bitfield_bit_size = child.bitfield_bit_size;
        child_bitfield_bit_offset = child.bitfield_bit_offset;
        child_is_base_class = child.is_base_class;
        child_is_deref_of_parent = child.is_deref_of_parent;
        language_flags = child.language_flags;
        return llvm::Error::success();
      });
  if (error)
    return error;
  if (!found)
    return llvm::createStringError("index out of bounds");
  return child_type;
}

CompilerType SwiftLanguageRuntime::GetBaseClass(CompilerType class_ty) {
  ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
  if (!reflection_ctx)
    return {};
  auto ts_sp = class_ty.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
  if (!ts_sp)
    return {};
  auto tr_ts = ts_sp->GetTypeSystemSwiftTypeRef();
  if (!tr_ts)
    return {};
  auto type_ref_or_err =
      reflection_ctx->GetTypeRef(class_ty.GetMangledTypeName().GetStringRef(),
                                 tr_ts->GetDescriptorFinder());
  if (!type_ref_or_err) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Expressions | LLDBLog::Types),
                   type_ref_or_err.takeError(), "{0}");
    return {};
  }
  auto *super_tr = reflection_ctx->LookupSuperclass(
      *type_ref_or_err, tr_ts->GetDescriptorFinder());
  auto flavor =
      SwiftLanguageRuntime::GetManglingFlavor(class_ty.GetMangledTypeName());
  return GetTypeFromTypeRef(*tr_ts, super_tr, flavor);
}

bool SwiftLanguageRuntime::ForEachSuperClassType(
    ValueObject &instance, std::function<bool(SuperClassType)> fn) {
  ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
  if (!reflection_ctx)
    return false;
  CompilerType instance_type = instance.GetCompilerType();
  auto ts_sp = instance_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
  if (!ts_sp)
    return false;
  auto tr_ts = ts_sp->GetTypeSystemSwiftTypeRef();
  if (!tr_ts)
    return false;
  auto &ts = *tr_ts;

  ExecutionContext exe_ctx(instance.GetExecutionContextRef());
  LLDBTypeInfoProvider tip(*this, ts);
  lldb::addr_t pointer = instance.GetPointerValue().address;
  return reflection_ctx->ForEachSuperClassType(&tip, ts.GetDescriptorFinder(),
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
         node_ptr->getKind() == swift::Demangle::Node::Kind::Allocator ||
         node_ptr->getKind() == swift::Demangle::Node::Kind::ExplicitClosure;
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

llvm::Expected<CompilerType>
SwiftLanguageRuntime::BindGenericPackType(StackFrame &frame,
                                          CompilerType pack_type, bool *is_indirect) {
  // This mode is used only by GetDynamicTypeAndAddress_Pack(). It would be
  // cleaner if we could get rid of it.
  bool rewrite_indirect_packs = (is_indirect != nullptr);
  swift::Demangle::Demangler dem;
  Target &target = GetProcess().GetTarget();
  size_t ptr_size = GetProcess().GetAddressByteSize();
  ConstString func_name = frame.GetSymbolContext(eSymbolContextFunction)
                              .GetFunctionName(Mangled::ePreferMangled);
  ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
  if (!reflection_ctx)
    return llvm::createStringError("no reflection context");

  // Extract the generic signature from the function symbol.
  auto ts =
      pack_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwiftTypeRef>();
  if (!ts)
    return llvm::createStringError("no type system");
  auto signature =
      SwiftLanguageRuntime::GetGenericSignature(func_name.GetStringRef(), *ts);
  if (!signature)
    return llvm::createStringError(
        "cannot decode pack_expansion type: failed to decode generic signature "
        "from function name");

  auto expand_pack_type = [&](ConstString mangled_pack_type,
                              bool rewrite_indirect,
                              swift::Mangle::ManglingFlavor flavor)
      -> llvm::Expected<swift::Demangle::NodePointer> {
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
    if (!pack_expansion)
      return llvm::createStringError(
          "cannot decode pack_expansion type: failed to find a matching type "
          "in the function signature");

    // Extract the count.
    llvm::SmallString<16> buf;
    llvm::raw_svector_ostream os(buf);
    os << "$pack_count_" << signature->GetCountForValuePack(i);
    StringRef count_var = os.str();
    std::optional<lldb::addr_t> count =
        GetTypeMetadataForTypeNameAndFrame(count_var, frame);
    if (!count)
      return llvm::createStringError(
          "cannot decode pack_expansion type: failed to find count argument "
          "\"%s\" in frame",
          count_var.str().c_str());

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
              GetTypeMetadataForTypeNameAndFrame(mds_var, frame);
          if (!mds_ptr) {
            LLDB_LOG(GetLog(LLDBLog::Types),
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
      return llvm::createStringError("cannot decode pack_expansion type");

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
        if (!status.Success())
          return llvm::createStringError(
                    "cannot decode pack_expansion type: failed to read type "
                    "pack for type %d/%d of type pack with shape %d %d",
                    j, (unsigned)*count, depth, index);

        auto type_ref_or_err =
            reflection_ctx->ReadTypeFromMetadata(md, ts->GetDescriptorFinder());
        if (!type_ref_or_err)
          return llvm::joinErrors(
              llvm::createStringError(
                  "cannot decode pack_expansion type: failed to decode type "
                  "metadata for type %d/%d of type pack with shape %d %d",
                  j, (unsigned)*count, depth, index),
              type_ref_or_err.takeError());
        substitutions.insert({{depth, index}, &*type_ref_or_err});
      }
      if (substitutions.empty())
        return llvm::createStringError("found no substitutions");

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
      auto type_ref_or_err = reflection_ctx->GetTypeRef(
          dem, pack_element, ts->GetDescriptorFinder());
      if (!type_ref_or_err)
        return type_ref_or_err.takeError();
      auto &type_ref = *type_ref_or_err;

      // Apply the substitutions.
      auto bound_typeref_or_err = reflection_ctx->ApplySubstitutions(
          type_ref, substitutions, ts->GetDescriptorFinder());
      if (!bound_typeref_or_err)
        return bound_typeref_or_err.takeError();
      swift::Demangle::NodePointer node = bound_typeref_or_err->getDemangling(dem);
      CompilerType type = ts->RemangleAsType(dem, node, flavor);

      // Add the substituted type to the tuple.
      elements.push_back({{}, type});
    }

    // TODO: Could we get rid of this code path?
    if (rewrite_indirect) {
      // Create a tuple type with all the concrete types in the pack.
      CompilerType tuple = ts->CreateTupleType(elements);
      // TODO: Remove unnecessary mangling roundtrip.
      // Wrap the type inside a SILPackType to mark it for GetChildAtIndex.
      CompilerType sil_pack_type = ts->CreateSILPackType(tuple, rewrite_indirect);
      swift::Demangle::NodePointer global =
          dem.demangleSymbol(sil_pack_type.GetMangledTypeName().GetStringRef());
      using Kind = Node::Kind;
      auto *dem_sil_pack_type =
          swift_demangle::ChildAtPath(global, {Kind::TypeMangling, Kind::Type});
      return dem_sil_pack_type;
    }
    return CreatePackType(dem, *ts, elements);
  };

  swift::Demangle::Context dem_ctx;
  auto node = dem_ctx.demangleSymbolAsNode(
      pack_type.GetMangledTypeName().GetStringRef());

  bool indirect = false;
  auto flavor =
      SwiftLanguageRuntime::GetManglingFlavor(pack_type.GetMangledTypeName());

  // Expand all the pack types that appear in the incoming type,
  // either at the root level or as arguments of bound generic types.
  auto transformed = TypeSystemSwiftTypeRef::TryTransform(
      dem, node,
      [&](swift::Demangle::NodePointer node)
          -> llvm::Expected<swift::Demangle::NodePointer> {
        if (node->getKind() == swift::Demangle::Node::Kind::SILPackIndirect)
          indirect = true;
        if (node->getKind() != swift::Demangle::Node::Kind::SILPackIndirect &&
            node->getKind() != swift::Demangle::Node::Kind::SILPackDirect &&
            node->getKind() != swift::Demangle::Node::Kind::Pack)
          return node;

        if (node->getNumChildren() != 1)
          return node;
        node = node->getChild(0);
        CompilerType pack_type = ts->RemangleAsType(dem, node, flavor);
        ConstString mangled_pack_type = pack_type.GetMangledTypeName();
        LLDB_LOG(GetLog(LLDBLog::Types), "decoded pack_expansion type: {0}",
                 mangled_pack_type);
        return expand_pack_type(mangled_pack_type,
                                rewrite_indirect_packs && indirect, flavor);
      });

  if (!transformed)
    return transformed.takeError();

  if (is_indirect)
    *is_indirect = indirect;

  return ts->RemangleAsType(dem, *transformed, flavor);
}

bool SwiftLanguageRuntime::GetDynamicTypeAndAddress_Pack(
    ValueObject &in_value, CompilerType pack_type,
    lldb::DynamicValueType use_dynamic, TypeAndOrName &pack_type_or_name,
    Address &address, Value::ValueType &value_type) {
  Log *log(GetLog(LLDBLog::Types));
  // Return a tuple type, with one element per pack element and its
  // type has all DependentGenericParamType that appear in type packs
  // substituted.

  StackFrameSP frame = in_value.GetExecutionContextRef().GetFrameSP();
  if (!frame)
    return false;

  // This type has already been resolved?
  auto ts =
      pack_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwiftTypeRef>();
  if (!ts)
    return false;
  if (auto info = ts->IsSILPackType(pack_type))
    if (info->expanded)
      return false;

  bool indirect = false;
  llvm::Expected<CompilerType> expanded_type =
      BindGenericPackType(*frame, pack_type, &indirect);
  if (!expanded_type) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Types), expanded_type.takeError(), "{0}");
    return false;
  }
  pack_type_or_name.SetCompilerType(*expanded_type);

  auto [addr, address_type] = in_value.GetAddressOf(true);
  value_type = Value::GetValueTypeFromAddressType(address_type);
  if (indirect) {
    Status status;
    addr = GetProcess().ReadPointerFromMemory(addr, status);
    if (status.Fail()) {
      LLDB_LOG(log, "failed to dereference indirect pack: {0}",
               expanded_type->GetMangledTypeName());
      return false;
    }
  }
  address.SetRawAddress(addr);
  return true;
}

CompilerType SwiftLanguageRuntime::GetDynamicTypeAndAddress_EmbeddedClass(
    uint64_t instance_ptr, CompilerType class_type) {
  ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
  if (!reflection_ctx)
    return {};
  /// If this is an embedded Swift type, there is no metadata, and the
  /// pointer points to the type's vtable. We can still resolve the type by
  /// reading the vtable's symbol name.
  auto pointer = reflection_ctx->ReadPointer(instance_ptr);
  if (!pointer)
    return {};
  llvm::StringRef symbol_name;
  if (pointer->getSymbol().empty() || pointer->getOffset()) {
    // Find the symbol name at this address.
    Address address;
    address.SetLoadAddress(pointer->getResolvedAddress().getRawAddress(),
                           &GetProcess().GetTarget());
    Symbol *symbol = address.CalculateSymbolContextSymbol();
    if (!symbol)
      return {};
    Mangled mangled = symbol->GetMangled();
    if (!mangled)
      return {};
    symbol_name = symbol->GetMangled().GetMangledName().GetStringRef();
  } else {
    symbol_name = pointer->getSymbol();
  }
  TypeSystemSwiftTypeRefSP ts = class_type.GetTypeSystem()
                                    .dyn_cast_or_null<TypeSystemSwift>()
                                    ->GetTypeSystemSwiftTypeRef();
  // The symbol name will be something like "type metadata for Class", extract
  // "Class" from that name.
  auto dynamic_type = ts->GetTypeFromTypeMetadataNode(symbol_name);
  return dynamic_type;
}

bool SwiftLanguageRuntime::GetDynamicTypeAndAddress_Class(
    ValueObject &in_value, CompilerType class_type,
    lldb::DynamicValueType use_dynamic, TypeAndOrName &class_type_or_name,
    Address &address, Value::ValueType &value_type,
    llvm::ArrayRef<uint8_t> &local_buffer) {
  auto [instance_ptr, address_type] = in_value.GetPointerValue();
  value_type = Value::GetValueTypeFromAddressType(address_type);

  if (instance_ptr == LLDB_INVALID_ADDRESS || instance_ptr == 0)
    return false;

  // Unwrap reference types.
  Status error;
  instance_ptr = FixupAddress(instance_ptr, class_type, error);
  if (!error.Success())
    return false;
  address.SetRawAddress(instance_ptr);

  // We are going to use process information to resolve the type, so
  // the result needs to be in the target's scratch context, not a
  // long-lived per-module typesystem.
  TypeSystemSwiftTypeRefSP ts;
  if (auto target_sp = in_value.GetTargetSP()) {
    auto type_system_or_err =
        target_sp->GetScratchTypeSystemForLanguage(lldb::eLanguageTypeSwift);
    if (!type_system_or_err) {
      llvm::consumeError(type_system_or_err.takeError());
      return false;
    }
    auto ts_sp = *type_system_or_err;
    ts = llvm::cast<TypeSystemSwift>(ts_sp.get())->GetTypeSystemSwiftTypeRef();
  }
  if (!ts)
    return false;

  auto resolve_swift = [&]() {
    // Scope reflection_ctx to minimize its lock scope.
    ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
    if (!reflection_ctx)
      return false;

    CompilerType dynamic_type;
    if (SwiftLanguageRuntime::GetManglingFlavor(
            class_type.GetMangledTypeName()) ==
        swift::Mangle::ManglingFlavor::Default) {

      const swift::reflection::TypeRef *typeref = nullptr;
      {
        auto typeref_or_err = reflection_ctx->ReadTypeFromInstance(
            instance_ptr, ts->GetDescriptorFinder(), true);
        if (typeref_or_err)
          typeref = &*typeref_or_err;
        else
          LLDB_LOG_ERRORV(GetLog(LLDBLog::Types), typeref_or_err.takeError(),
                          "{0}");
      }

      if (!typeref) {
        HEALTH_LOG(
            "could not read typeref for type: {0} (instance_ptr = {1:x})",
            class_type.GetMangledTypeName(), instance_ptr);
        return false;
      }

      auto flavor = SwiftLanguageRuntime::GetManglingFlavor(
          class_type.GetMangledTypeName());

      swift::Demangle::Demangler dem;
      swift::Demangle::NodePointer node = typeref->getDemangling(dem);
      dynamic_type = ts->RemangleAsType(dem, node, flavor);
    } else {
      dynamic_type =
          GetDynamicTypeAndAddress_EmbeddedClass(instance_ptr, class_type);
      if (!dynamic_type) {
        HEALTH_LOG("could not resolve dynamic type of embedded swift class: "
                   "{0} (instance_ptr = {1:x})",
                   class_type.GetMangledTypeName(), instance_ptr);
        return false;
      }
    }
    class_type_or_name.SetCompilerType(dynamic_type);
    LLDB_LOG(GetLog(LLDBLog::Types),
             "dynamic type of instance_ptr {0:x} is {1}", instance_ptr,
             class_type.GetMangledTypeName());
    return true;
  };

  if (!resolve_swift()) {
    // When returning false here, the next compatible runtime (=
    // Objective-C) will get ask to resolve this type.
    return false;
  }

#ifndef NDEBUG
  if (ModuleList::GetGlobalModuleListProperties()
          .GetSwiftValidateTypeSystem()) {
    ConstString a = class_type_or_name.GetCompilerType().GetMangledTypeName();
    ConstString b = SwiftLanguageRuntime::GetDynamicTypeName_ClassRemoteAST(
        in_value, instance_ptr);
    if (b && a != b)
      llvm::dbgs() << "RemoteAST and runtime diverge " << a << " != " << b
                   << "\n";
  }
#endif
  return true;
}

bool SwiftLanguageRuntime::IsValidErrorValue(ValueObject &in_value) {
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
    if (auto objc_runtime =
            SwiftLanguageRuntime::GetObjCRuntime(GetProcess())) {
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

  if (SwiftLanguageRuntime::GetObjCRuntime(GetProcess())) {
    // this is a swift native error but it can be bridged to ObjC
    // so it needs to be layout compatible

    size_t ptr_size = GetProcess().GetAddressByteSize();
    size_t metadata_offset =
        ptr_size + 4 + (ptr_size == 8 ? 4 : 0);        // CFRuntimeBase
    metadata_offset += ptr_size + ptr_size + ptr_size; // CFIndex + 2*CFRef

    metadata_location += metadata_offset;
    Status error;
    lldb::addr_t metadata_ptr_value =
        GetProcess().ReadPointerFromMemory(metadata_location, error);
    if (metadata_ptr_value == 0 || metadata_ptr_value == LLDB_INVALID_ADDRESS ||
        error.Fail())
      return false;
  } else {
    // this is a swift native error and it has no way to be bridged to ObjC
    // so it adopts a more compact layout

    Status error;

    size_t ptr_size = GetProcess().GetAddressByteSize();
    size_t metadata_offset = 2 * ptr_size;
    metadata_location += metadata_offset;
    lldb::addr_t metadata_ptr_value =
        GetProcess().ReadPointerFromMemory(metadata_location, error);
    if (metadata_ptr_value == 0 || metadata_ptr_value == LLDB_INVALID_ADDRESS ||
        error.Fail())
      return false;
  }

  return true;
}

bool SwiftLanguageRuntime::GetDynamicTypeAndAddress_Existential(
    ValueObject &in_value, CompilerType existential_type,
    lldb::DynamicValueType use_dynamic, TypeAndOrName &class_type_or_name,
    Address &address) {
  Log *log(GetLog(LLDBLog::Types));
  auto tss =
      existential_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
  if (!tss) {
    if (log)
      log->Printf("Could not get type system swift");
    return false;
  }

  auto tr_or_err =
      GetTypeRef(existential_type, tss->GetTypeSystemSwiftTypeRef().get());
  if (!tr_or_err) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Types), tr_or_err.takeError(),
                   "Could not get existential typeref: {0}");
    return false;
  }
  const swift::reflection::TypeRef *existential_typeref = &*tr_or_err;

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
    existential_address = in_value.GetAddressOf().address;
  }

  if (log)
    log->Printf("existential address is 0x%" PRIx64, existential_address);

  if (!existential_address || existential_address == LLDB_INVALID_ADDRESS) {
    if (log)
      log->Printf("Existential address is invalid");
    return false;
  }

  // This scope is needed because the validation code will call PushLocalBuffer,
  // so we need to pop it before that call.
  {
    MemoryReaderLocalBufferHolder holder;
    if (use_local_buffer)
      holder = PushLocalBuffer(
          existential_address,
          llvm::expectedToOptional(in_value.GetByteSize()).value_or(0));

    auto remote_existential = swift::remote::RemoteAddress(
        existential_address, swift::remote::RemoteAddress::DefaultAddressSpace);

    ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
    if (!reflection_ctx)
      return false;
    auto tr_ts = tss->GetTypeSystemSwiftTypeRef();
    if (!tr_ts)
      return false;

    auto flavor = SwiftLanguageRuntime::GetManglingFlavor(
        existential_type.GetMangledTypeName());
    CompilerType dynamic_type;
    uint64_t dynamic_address = 0;
    if (flavor == swift::Mangle::ManglingFlavor::Default) {
      auto pair = reflection_ctx->ProjectExistentialAndUnwrapClass(
          remote_existential, *existential_typeref,
          tr_ts->GetDescriptorFinder());

      if (!pair) {
        if (log)
          log->Printf("Runtime failed to get dynamic type of existential");
        return false;
      }

      const swift::reflection::TypeRef *typeref;
      swift::remote::RemoteAddress out_address;
      std::tie(typeref, out_address) = *pair;

      auto ts = tss->GetTypeSystemSwiftTypeRef();
      if (!ts)
        return false;
      swift::Demangle::Demangler dem;
      swift::Demangle::NodePointer node = typeref->getDemangling(dem);
      dynamic_type = ts->RemangleAsType(dem, node, flavor);
      dynamic_address = out_address.getRawAddress();
    } else {
      // In the embedded Swift case, the existential container just points to
      // the instance.
      auto reflection_ctx = GetReflectionContext();
      if (!reflection_ctx)
        return false;
      auto maybe_addr_or_symbol =
          reflection_ctx->ReadPointer(existential_address);
      if (!maybe_addr_or_symbol)
        return false;

      uint64_t address = 0;
      if (maybe_addr_or_symbol->getSymbol().empty() &&
          maybe_addr_or_symbol->getOffset() == 0) {
        address = maybe_addr_or_symbol->getResolvedAddress().getRawAddress();
      } else {
        SymbolContextList sc_list;
        auto &module_list = GetProcess().GetTarget().GetImages();
        module_list.FindSymbolsWithNameAndType(
            ConstString(maybe_addr_or_symbol->getSymbol()), eSymbolTypeAny,
            sc_list);
        if (sc_list.GetSize() != 1)
          return false;

        SymbolContext sc = sc_list[0];
        Symbol *symbol = sc.symbol;
        address = symbol->GetLoadAddress(&GetProcess().GetTarget());
      }

      dynamic_type =
          GetDynamicTypeAndAddress_EmbeddedClass(address, existential_type);
      if (!dynamic_type)
        return false;
      dynamic_address =
          maybe_addr_or_symbol->getResolvedAddress().getRawAddress();
    }
    class_type_or_name.SetCompilerType(dynamic_type);
    address.SetRawAddress(dynamic_address);
  }

#ifndef NDEBUG
  if (ModuleList::GetGlobalModuleListProperties()
          .GetSwiftValidateTypeSystem()) {
    auto reference_pair = GetDynamicTypeAndAddress_ExistentialRemoteAST(
        in_value, existential_type, use_local_buffer, existential_address);

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

bool SwiftLanguageRuntime::GetDynamicTypeAndAddress_ExistentialMetatype(
    ValueObject &in_value, CompilerType meta_type,
    lldb::DynamicValueType use_dynamic, TypeAndOrName &class_type_or_name,
    Address &address, Value::ValueType &value_type) {
  // Resolve the dynamic type of the metatype.
  auto [ptr, address_type] = in_value.GetPointerValue();
  if (ptr == LLDB_INVALID_ADDRESS || ptr == 0)
    return false;

  ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
  if (!reflection_ctx)
    return false;

  auto tss = meta_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
  if (!tss)
    return false;

  auto tr_ts = tss->GetTypeSystemSwiftTypeRef();
  if (!tr_ts)
    return false;

  auto type_ref_or_err =
      reflection_ctx->ReadTypeFromMetadata(ptr, tr_ts->GetDescriptorFinder());
  if (!type_ref_or_err) {
    LLDB_LOG_ERRORV(GetLog(LLDBLog::Types), type_ref_or_err.takeError(), "{0}");
    return false;
  }

  const swift::reflection::TypeRef &type_ref = *type_ref_or_err;

  using namespace swift::Demangle;
  Demangler dem;
  NodePointer node = type_ref.getDemangling(dem);
  // Wrap the resolved type in a metatype again for the data formatter to
  // recognize.
  if (!node || node->getKind() != Node::Kind::Type)
    return false;
  NodePointer wrapped = dem.createNode(Node::Kind::Type);
  NodePointer meta = dem.createNode(Node::Kind::Metatype);
  meta->addChild(node, dem);
  wrapped->addChild(meta,dem);

  auto flavor =
      SwiftLanguageRuntime::GetManglingFlavor(meta_type.GetMangledTypeName());
  meta_type =
      tss->GetTypeSystemSwiftTypeRef()->RemangleAsType(dem, wrapped, flavor);
  class_type_or_name.SetCompilerType(meta_type);
  address.SetRawAddress(ptr);
  value_type = Value::ValueType::LoadAddress;
  return true;
}

CompilerType SwiftLanguageRuntime::GetTypeFromMetadata(TypeSystemSwift &ts,
                                                       Address address) {
  lldb::addr_t ptr = address.GetLoadAddress(&GetProcess().GetTarget());
  if (ptr == LLDB_INVALID_ADDRESS)
    return {};

  ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
  if (!reflection_ctx)
    return {};

  auto tr_ts = ts.GetTypeSystemSwiftTypeRef();
  if (!tr_ts)
    return {};

  auto type_ref_or_err =
      reflection_ctx->ReadTypeFromMetadata(ptr, tr_ts->GetDescriptorFinder());
  if (!type_ref_or_err) {
    LLDB_LOG_ERRORV(GetLog(LLDBLog::Types), type_ref_or_err.takeError(), "{0}");
    return {};
  }

  const swift::reflection::TypeRef &type_ref = *type_ref_or_err;

  using namespace swift::Demangle;
  Demangler dem;
  NodePointer node = type_ref.getDemangling(dem);
  // TODO: the mangling flavor should come from the TypeRef.
  return ts.GetTypeSystemSwiftTypeRef()->RemangleAsType(
      dem, node, ts.GetTypeSystemSwiftTypeRef()->GetManglingFlavor());
}

std::optional<lldb::addr_t>
SwiftLanguageRuntime::GetTypeMetadataForTypeNameAndFrame(StringRef mdvar_name,
                                                         StackFrame &frame) {
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

CompilerType SwiftLanguageRuntime::BindGenericTypeParameters(
    CompilerType unbound_type,
    std::function<CompilerType(unsigned, unsigned)> type_resolver) {
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

  auto tr_ts = ts->GetTypeSystemSwiftTypeRef();
  if (!tr_ts)
    return unbound_type;

  auto type_ref_or_err = reflection_ctx->GetTypeRef(
      dem, unbound_node, tr_ts->GetDescriptorFinder());
  if (!type_ref_or_err) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Expressions | LLDBLog::Types),
                   type_ref_or_err.takeError(),
                   "Couldn't get type ref of unbound type: {0}");
    return {};
  }

  auto &type_ref = *type_ref_or_err;

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

    auto type_ref_or_err = reflection_ctx->GetTypeRef(
        type.GetMangledTypeName().GetStringRef(), tr_ts->GetDescriptorFinder());
    if (!type_ref_or_err) {
      LLDB_LOG_ERROR(
          GetLog(LLDBLog::Expressions | LLDBLog::Types),
          type_ref_or_err.takeError(),
          "Couldn't get type ref when binding generic type parameters: {0}");
      failure = true;
      return;
    }

    substitutions.insert({{depth, index}, &*type_ref_or_err});
  });

  if (failure)
    return {};

  // Apply the substitutions.
  auto bound_type_ref_or_err = reflection_ctx->ApplySubstitutions(
      type_ref, substitutions, tr_ts->GetDescriptorFinder());
  if (!bound_type_ref_or_err) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Expressions | LLDBLog::Types),
                   bound_type_ref_or_err.takeError(),
                   "Couldn't apply substitutions: {0}");
    return {};
  }

  NodePointer node = bound_type_ref_or_err->getDemangling(dem);
  return ts->GetTypeSystemSwiftTypeRef()->RemangleAsType(
      dem, node,
      SwiftLanguageRuntime::GetManglingFlavor(
          unbound_type.GetMangledTypeName()));
}

llvm::Expected<CompilerType>
SwiftLanguageRuntime::BindGenericTypeParameters(StackFrame &stack_frame,
                                                TypeSystemSwiftTypeRef &ts,
                                                ConstString mangled_name) {
  using namespace swift::Demangle;

  ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
  if (!reflection_ctx)
    return llvm::createStringError("no reflection context");

  ConstString func_name = stack_frame.GetSymbolContext(eSymbolContextFunction)
                              .GetFunctionName(Mangled::ePreferMangled);
  // Extract the generic signature from the function symbol.
  auto generic_signature =
      SwiftLanguageRuntime::GetGenericSignature(func_name.GetStringRef(), ts);
  Demangler dem;

  NodePointer canonical = TypeSystemSwiftTypeRef::GetStaticSelfType(
      dem, dem.demangleSymbol(mangled_name.GetStringRef()));
  canonical = ts.DesugarNode(dem, canonical);

  // Build the list of type substitutions.
  swift::reflection::GenericArgumentMap substitutions;
  ForEachGenericParameter(canonical, [&](unsigned depth, unsigned index) {
    if (substitutions.count({depth, index}))
      return;
    // Packs will be substituted in a second pass.
    if (generic_signature && generic_signature->IsPack(depth, index))
      return;
    StreamString mdvar_name;
    mdvar_name.Printf(u8"$\u03C4_%d_%d", depth, index);

    std::optional<lldb::addr_t> metadata_location =
        GetTypeMetadataForTypeNameAndFrame(mdvar_name.GetString(), stack_frame);
    if (!metadata_location)
      return;
    auto type_ref_or_err = reflection_ctx->ReadTypeFromMetadata(
        *metadata_location, ts.GetDescriptorFinder());
    if (!type_ref_or_err) {
      LLDB_LOG_ERRORV(
          GetLog(LLDBLog::Expressions | LLDBLog::Types),
          type_ref_or_err.takeError(),
          "Couldn't get type ref when binding generic type parameters: {0}");
      return;
    }

    substitutions.insert({{depth, index}, &*type_ref_or_err});
  });
  auto flavor =
      SwiftLanguageRuntime::GetManglingFlavor(mangled_name.GetStringRef());

  // Nothing to do if there are no type parameters.
  auto get_canonical = [&]() {
    auto mangling = mangleNode(canonical, flavor);
    if (!mangling.isSuccess())
      return CompilerType();
    return ts.GetTypeFromMangledTypename(ConstString(mangling.result()));
  };
  if (substitutions.empty() &&
      !(generic_signature && generic_signature->HasPacks()))
    return get_canonical();

  // Build a TypeRef from the demangle tree.
  auto type_ref_or_err =
      reflection_ctx->GetTypeRef(dem, canonical, ts.GetDescriptorFinder());
  if (!type_ref_or_err)
    return llvm::joinErrors(
        llvm::createStringError("cannot bind generic parameters"),
        type_ref_or_err.takeError());

  // Apply the substitutions.
  auto bound_type_ref_or_err = reflection_ctx->ApplySubstitutions(
      *type_ref_or_err, substitutions, ts.GetDescriptorFinder());
  if (!bound_type_ref_or_err)
    return bound_type_ref_or_err.takeError();

  NodePointer node = bound_type_ref_or_err->getDemangling(dem);

  // Import the type into the scratch context. Subsequent conversions
  // to Swift types must be performed in the scratch context, since
  // the bound type may combine types from different
  // lldb::Modules. Contrary to the AstContext variant of this
  // function, we don't want to do this earlier, because the
  // canonicalization in GetCanonicalDemangleTree() must be performed in
  // the original context as to resolve type aliases correctly.
  auto &target = GetProcess().GetTarget();
  auto scratch_ctx = TypeSystemSwiftTypeRefForExpressions::GetForTarget(target);
  if (!scratch_ctx)
    return llvm::createStringError("No scratch context available.");

  CompilerType bound_type = scratch_ctx->RemangleAsType(dem, node, flavor);
  LLDB_LOG(GetLog(LLDBLog::Expressions | LLDBLog::Types), "Bound {0} -> {1}.",
           mangled_name, bound_type.GetMangledTypeName());

  if (generic_signature && generic_signature->HasPacks()) {
    auto bound_type_or_err = BindGenericPackType(stack_frame, bound_type);
    if (!bound_type_or_err)
      return bound_type_or_err.takeError();
    bound_type = *bound_type_or_err;
  }

  return bound_type;
}

llvm::Expected<CompilerType>
SwiftLanguageRuntime::BindGenericTypeParameters(StackFrame &stack_frame,
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

bool SwiftLanguageRuntime::GetDynamicTypeAndAddress_Value(
    ValueObject &in_value, CompilerType &bound_type,
    lldb::DynamicValueType use_dynamic, TypeAndOrName &class_type_or_name,
    Address &address, Value::ValueType &value_type,
    llvm::ArrayRef<uint8_t> &local_buffer) {
  auto static_type = in_value.GetCompilerType();
  value_type = Value::ValueType::Invalid;
  class_type_or_name.SetCompilerType(bound_type);

  ExecutionContext exe_ctx = in_value.GetExecutionContextRef().Lock(true);
  ExecutionContextScope *exe_scope = exe_ctx.GetBestExecutionContextScope();
  if (!exe_scope)
    return false;
  std::optional<uint64_t> size = llvm::expectedToOptional(
      bound_type.GetByteSize(exe_ctx.GetBestExecutionContextScope()));
  if (!size)
    return false;
  auto [val_address, address_type] = in_value.GetAddressOf(true);
  // If we couldn't find a load address, but the value object has a local
  // buffer, use that.
  if (val_address == LLDB_INVALID_ADDRESS && address_type == eAddressTypeHost) {
    // Check if the dynamic type fits in the value object's buffer.
    auto in_value_buffer = in_value.GetLocalBuffer();

    // If we can't find the local buffer we can't safely know if the
    // dynamic type fits in it.
    if (in_value_buffer.empty())
      return false;
    // If the dynamic type doesn't in the buffer we can't use it either.
    if (in_value_buffer.size() <
        llvm::expectedToOptional(bound_type.GetByteSize(exe_scope)).value_or(0))
      return false;

    value_type = Value::GetValueTypeFromAddressType(address_type);
    local_buffer = in_value_buffer;
    return true;
  }
  if (*size && (!val_address || val_address == LLDB_INVALID_ADDRESS))
    return false;

  value_type = Value::GetValueTypeFromAddressType(address_type);
  address.SetLoadAddress(val_address, in_value.GetTargetSP().get());
  return true;
}

void SwiftLanguageRuntime::DumpTyperef(CompilerType type,
                                       TypeSystemSwiftTypeRef *module_holder,
                                       Stream *s) {
  if (!s)
    return;

  auto typeref_or_err = GetTypeRef(type, module_holder);
  if (!typeref_or_err) {
    *s << llvm::toString(typeref_or_err.takeError());
    return;
  }

  std::ostringstream string_stream;
  typeref_or_err->dump(string_stream);
  s->PutCString(string_stream.str());
}

Value::ValueType SwiftLanguageRuntime::GetValueType(
    ValueObject &in_value, CompilerType dynamic_type,
    Value::ValueType static_value_type, llvm::ArrayRef<uint8_t> &local_buffer) {
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
        existential_address = in_value.GetAddressOf().address;
      }

      MemoryReaderLocalBufferHolder holder;
      if (use_local_buffer)
        holder = PushLocalBuffer(
            existential_address,
            llvm::expectedToOptional(in_value.GetByteSize()).value_or(0));

      // Read the value witness table and check if the data is inlined in
      // the existential container or not.
      auto remote_existential = swift::remote::RemoteAddress(
          existential_address,
          swift::remote::RemoteAddress::DefaultAddressSpace);
      if (ThreadSafeReflectionContext reflection_ctx = GetReflectionContext()) {
        std::optional<bool> is_inlined =
            reflection_ctx->IsValueInlinedInExistentialContainer(
                remote_existential);


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
    // If we found a host address and the dynamic type fits in there, and
    // this is not a pointer from an existential container, then this points to
    // the local buffer.
    if (static_value_type == Value::ValueType::HostAddress &&
        !local_buffer.empty())
      return static_value_type;

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
      if (static_type_flags.AllClear(eTypeIsBuiltIn))
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
  if (!objc_runtime.GetTargetRef().ResolveLoadAddress(isa_load_addr, isa))
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

bool SwiftLanguageRuntime::GetDynamicTypeAndAddress_ClangType(
    ValueObject &in_value, lldb::DynamicValueType use_dynamic,
    TypeAndOrName &class_type_or_name, Address &address,
    Value::ValueType &value_type, llvm::ArrayRef<uint8_t> &local_buffer) {
  AppleObjCRuntime *objc_runtime =
      SwiftLanguageRuntime::GetObjCRuntime(GetProcess());
  if (!objc_runtime)
    return false;

  // This is a Clang type, which means it must have been an
  // Objective-C protocol. Protocols are not represented in DWARF and
  // LLDB's ObjC runtime implementation doesn't know how to deal with
  // them either.  Use the Objective-C runtime to perform dynamic type
  // resolution first, and then map the dynamic Objective-C type back
  // into Swift.
  TypeAndOrName dyn_class_type_or_name = class_type_or_name;
  if (!objc_runtime->GetDynamicTypeAndAddress(in_value, use_dynamic,
                                              dyn_class_type_or_name, address,
                                              value_type, local_buffer))
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
    address.SetRawAddress(in_value.GetPointerValue().address);
  } else {
    swift_class.module = swift::MANGLING_MODULE_OBJC;
    swift_class.identifier = dyn_name;
  }

  std::string remangled;
  {
    auto type = in_value.GetCompilerType();
    std::optional<swift::Mangle::ManglingFlavor> mangling_flavor =
        SwiftLanguageRuntime::GetManglingFlavor(
            type.GetMangledTypeName().GetStringRef());
    if (!mangling_flavor)
      return false;

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

    auto mangling = mangleNode(global, *mangling_flavor);
    if (!mangling.isSuccess())
      return false;
    remangled = mangling.result();
  }

  // Import the remangled dynamic name into the scratch context.
  auto scratch_ctx = TypeSystemSwiftTypeRefForExpressions::GetForTarget(
      in_value.GetTargetSP());
  if (!scratch_ctx)
    return false;
  CompilerType swift_type =
      scratch_ctx->GetTypeFromMangledTypename(ConstString(remangled));

  // Roll back the ObjC dynamic type resolution.
  if (!swift_type)
    return false;
  class_type_or_name = dyn_class_type_or_name;
  class_type_or_name.SetCompilerType(swift_type);
  value_type = Value::ValueType::Scalar;
  return true;
}

static bool CouldHaveDynamicValue(ValueObject &in_value) {
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

bool SwiftLanguageRuntime::GetDynamicTypeAndAddress(
    ValueObject &in_value, lldb::DynamicValueType use_dynamic,
    TypeAndOrName &class_type_or_name, Address &address,
    Value::ValueType &value_type, llvm::ArrayRef<uint8_t> &local_buffer) {
  class_type_or_name.Clear();
  if (use_dynamic == lldb::eNoDynamicValues)
    return false;
  CompilerType val_type(in_value.GetCompilerType());
  Value::ValueType static_value_type = Value::ValueType::Invalid;

  // Try to import a Clang type into Swift.
  if (in_value.GetObjectRuntimeLanguage() == eLanguageTypeObjC) {
    if (GetDynamicTypeAndAddress_ClangType(in_value, use_dynamic,
                                           class_type_or_name, address,
                                           value_type, local_buffer))
      return true;
    return GetDynamicTypeAndAddress_Class(in_value, val_type, use_dynamic,
                                          class_type_or_name, address,
                                          static_value_type, local_buffer);
  }

  if (!CouldHaveDynamicValue(in_value))
    return false;

  Flags type_info(val_type.GetTypeInfo());
  if (!type_info.AnySet(eTypeIsSwift))
    return false;

  bool success = false;
  if (type_info.AnySet(eTypeIsPack))
    success = GetDynamicTypeAndAddress_Pack(in_value, val_type, use_dynamic,
                                            class_type_or_name, address,
                                            static_value_type);
  else if (type_info.AnySet(eTypeIsClass) ||
           type_info.AllSet(eTypeIsBuiltIn | eTypeIsPointer | eTypeHasValue))
    success = GetDynamicTypeAndAddress_Class(in_value, val_type, use_dynamic,
                                             class_type_or_name, address,
                                             static_value_type, local_buffer);
  else if (type_info.AllSet(eTypeIsMetatype | eTypeIsProtocol)) {
    success = GetDynamicTypeAndAddress_ExistentialMetatype(
        in_value, val_type, use_dynamic, class_type_or_name, address, static_value_type);
  } else if (type_info.AnySet(eTypeIsProtocol)) {
    if (type_info.AnySet(eTypeIsObjC))
      success = GetDynamicTypeAndAddress_Class(in_value, val_type, use_dynamic,
                                               class_type_or_name, address,
                                               static_value_type, local_buffer);
    else
      success = GetDynamicTypeAndAddress_Existential(
          in_value, val_type, use_dynamic, class_type_or_name, address);
  } else {
    CompilerType bound_type;
    if (type_info.AnySet(eTypeHasUnboundGeneric | eTypeHasDynamicSelf)) {
      // Perform generic type resolution.
      StackFrameSP frame = in_value.GetExecutionContextRef().GetFrameSP();
      if (!frame)
        return false;

      bound_type = llvm::expectedToOptional(
                       BindGenericTypeParameters(*frame.get(), val_type))
                       .value_or(CompilerType());
      if (!bound_type)
        return false;
    } else {
      bound_type = val_type;
    }

    Flags subst_type_info(bound_type.GetTypeInfo());
    if (subst_type_info.AnySet(eTypeIsClass)) {
      success = GetDynamicTypeAndAddress_Class(
          in_value, bound_type, use_dynamic, class_type_or_name, address,
          static_value_type, local_buffer);
    } else if (subst_type_info.AnySet(eTypeIsProtocol)) {
      success = GetDynamicTypeAndAddress_Existential(
          in_value, bound_type, use_dynamic, class_type_or_name, address);
    } else {
      success = GetDynamicTypeAndAddress_Value(
          in_value, bound_type, use_dynamic, class_type_or_name, address,
          static_value_type, local_buffer);
    }
  }

  if (success) {
    // If we haven't found a better static value type, use the value object's
    // one.
    if (static_value_type == Value::ValueType::Invalid)
      static_value_type = in_value.GetValue().GetValueType();

    value_type = GetValueType(in_value, class_type_or_name.GetCompilerType(),
                              static_value_type, local_buffer);
  }
  return success;
}

TypeAndOrName
SwiftLanguageRuntime::FixUpDynamicType(const TypeAndOrName &type_and_or_name,
                                       ValueObject &static_value) {
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
        type_flags.AllClear(eTypeIsGenericTypeParam | eTypeIsBuiltIn);
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

bool SwiftLanguageRuntime::IsTaggedPointer(lldb::addr_t addr,
                                           CompilerType type) {
  Demangler dem;
  auto *root = dem.demangleSymbol(type.GetMangledTypeName().GetStringRef());
  using Kind = Node::Kind;
  auto *unowned_node = swift_demangle::ChildAtPath(
      root, {Kind::TypeMangling, Kind::Type, Kind::Unowned});
  if (!unowned_node)
    return false;

  Target &target = GetProcess().GetTarget();
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
SwiftLanguageRuntime::FixupPointerValue(lldb::addr_t addr, CompilerType type) {
  // Check for an unowned Darwin Objective-C reference.
  if (IsTaggedPointer(addr, type)) {
    // Clear the discriminator bit to get at the pointer to Objective-C object.
    bool needs_deref = true;
    return {addr & ~1ULL, needs_deref};
  }

  // Adjust the pointer to strip away the spare bits.
  Target &target = GetProcess().GetTarget();
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

lldb::addr_t SwiftLanguageRuntime::FixupAddress(lldb::addr_t addr,
                                                CompilerType type,
                                                Status &error) {
  // Peek into the reference to see whether it needs an extra deref.
  // If yes, return the fixed-up address we just read.
  lldb::addr_t stripped_addr = LLDB_INVALID_ADDRESS;
  bool extra_deref;
  std::tie(stripped_addr, extra_deref) = FixupPointerValue(addr, type);
  if (extra_deref) {
    Target &target = GetProcess().GetTarget();
    size_t ptr_size = GetProcess().GetAddressByteSize();
    lldb::addr_t refd_addr = LLDB_INVALID_ADDRESS;
    target.ReadMemory(stripped_addr, &refd_addr, ptr_size, error, true);
    return refd_addr;
  }
  return addr;
}

bool SwiftLanguageRuntime::IsObjCInstance(ValueObject &instance) {
  bool found = false;
  ForEachSuperClassType(instance, [&](SuperClassType sc) -> bool {
    auto *tr = sc.get_typeref();
    if (!tr)
      return true;
    if (llvm::isa<swift::reflection::ObjCClassTypeRef>(tr)) {
      found = true;
      return true;
    }
    return false;
  });
  return found;
}

llvm::Expected<const swift::reflection::TypeRef &>
SwiftLanguageRuntime::GetTypeRef(CompilerType type,
                                 TypeSystemSwiftTypeRef *module_holder) {
  Log *log(GetLog(LLDBLog::Types));
  if (log && log->GetVerbose())
    LLDB_LOGF(log,
              "[SwiftLanguageRuntime::GetTypeRef] Getting typeref for "
              "type: %s\n",
              type.GetMangledTypeName().GetCString());

  // Demangle the mangled name.
  swift::Demangle::Demangler dem;
  llvm::StringRef mangled_name = type.GetMangledTypeName().GetStringRef();
  auto ts = type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
  if (!ts)
    return llvm::createStringError("not a Swift type");

  // List of commonly used types known to have been been annotated with
  // @_originallyDefinedIn to a different module.
  static llvm::StringMap<llvm::StringRef> known_types_with_redefined_modules = {
      {"$s14CoreFoundation7CGFloatVD", "$s12CoreGraphics7CGFloatVD"}};

  auto it = known_types_with_redefined_modules.find(mangled_name);
  if (it != known_types_with_redefined_modules.end())
    mangled_name = it->second;

  if (!module_holder)
    return llvm::createStringError("no module holder");

  swift::Demangle::NodePointer node =
      module_holder->GetCanonicalDemangleTree(dem, mangled_name);
  if (!node)
    return llvm::createStringError("could not demangle");

  // Build a TypeRef.
  ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
  if (!reflection_ctx)
    return llvm::createStringError("no reflection context");

  auto type_ref_or_err = reflection_ctx->GetTypeRef(
      dem, node, module_holder->GetDescriptorFinder());
  if (!type_ref_or_err)
    return llvm::joinErrors(
        llvm::createStringError("cannot get typeref for type %s",
                                type.GetMangledTypeName().GetCString()),
        type_ref_or_err.takeError());

  if (log && log->GetVerbose()) {
    std::stringstream ss;
    type_ref_or_err->dump(ss);
    LLDB_LOG(log,
             "[SwiftLanguageRuntime::GetTypeRef] Found typeref for "
             "type: {0}:\n{1}",
             type.GetMangledTypeName(), ss.str());
  }
  return type_ref_or_err;
}

llvm::Expected<TypeSystemSwiftTypeRefSP>
SwiftLanguageRuntime::GetReflectionTypeSystem(CompilerType for_type,
                                              ExecutionContext exe_ctx) {
  auto ts_sp = for_type.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
  if (!ts_sp)
    return llvm::createStringError("not a Swift type");

  // The TypeSystemSwiftTypeRefForExpressions doesn't have a SymbolFile,
  // so any DWARF lookups for Embedded Swift fail.
  //
  // FIXME: It's unclear whether this is safe to do in a non-LTO Swift program.
  if (auto *tr_ts =
          llvm::dyn_cast_or_null<TypeSystemSwiftTypeRefForExpressions>(
              ts_sp.get())) {
    if (tr_ts->GetManglingFlavor(&exe_ctx) ==
        swift::Mangle::ManglingFlavor::Embedded) {
      if (auto *frame = exe_ctx.GetFramePtr()) {
        auto &sc = frame->GetSymbolContext(eSymbolContextModule);
        if (sc.module_sp) {
          auto ts_or_err =
              sc.module_sp->GetTypeSystemForLanguage(eLanguageTypeSwift);
          if (!ts_or_err)
            return ts_or_err.takeError();
          if (auto *tr_ts =
                  llvm::dyn_cast_or_null<TypeSystemSwift>(ts_or_err->get()))
            ts_sp = tr_ts->GetTypeSystemSwiftTypeRef();
        }
      }
    }
  }
  auto tr_ts = ts_sp->GetTypeSystemSwiftTypeRef();
  if (!tr_ts)
    return llvm::createStringError("no Swift typesystem");
  return tr_ts;
}

llvm::Expected<const swift::reflection::TypeInfo &>
SwiftLanguageRuntime::GetSwiftRuntimeTypeInfo(
    CompilerType type, ExecutionContextScope *exe_scope,
    swift::reflection::TypeRef const **out_tr) {
  Log *log(GetLog(LLDBLog::Types));
  if (log && log->GetVerbose())
    LLDB_LOG(log,
             "[SwiftLanguageRuntime::GetSwiftRuntimeTypeInfo] Getting "
             "type info for type: {0}",
             type.GetMangledTypeName());
  StackFrame *frame = nullptr;
  ExecutionContext exe_ctx;
  if (exe_scope) {
    frame = exe_scope->CalculateStackFrame().get();
    if (frame)
      frame->CalculateExecutionContext(exe_ctx);
  }
  auto ts_or_err = GetReflectionTypeSystem(type, exe_ctx);
  if (!ts_or_err)
    return ts_or_err.takeError();
  auto &ts = *ts_or_err->get();

  // Resolve all type aliases.
  type = type.GetCanonicalType();
  if (!type)  {
    // FIXME: We could print a better error message if
    // GetCanonicalType() returned an Expected.
    return llvm::createStringError(
        "could not get canonical type (possibly due to unresolved typealias)");
  }

  // Resolve all generic type parameters in the type for the current
  // frame. Generic parameter binding has to happen in the scratch
  // context.
  if (frame) {
    frame->CalculateExecutionContext(exe_ctx);
    auto bound_type_or_err = BindGenericTypeParameters(*frame, type);
    if (!bound_type_or_err)
      return bound_type_or_err.takeError();
    type = *bound_type_or_err;
  }

  // BindGenericTypeParameters imports the type into the scratch
  // context, but we need to resolve (any DWARF links in) the typeref
  // in the original module.
  auto type_ref_or_err = GetTypeRef(type, &ts);
  if (!type_ref_or_err)
    return type_ref_or_err.takeError();

  if (out_tr)
    *out_tr = &*type_ref_or_err;

  ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
  if (!reflection_ctx)
    return llvm::createStringError("no reflection context");

  LLDBTypeInfoProvider provider(*this, ts);
  return reflection_ctx->GetTypeInfo(*type_ref_or_err, &provider,
                                     ts.GetDescriptorFinder());
}

bool SwiftLanguageRuntime::IsStoredInlineInBuffer(CompilerType type) {
  auto type_info_or_err = GetSwiftRuntimeTypeInfo(type, nullptr);
  if (!type_info_or_err) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Types), type_info_or_err.takeError(), "{0}");
    return true;
  }

  auto &type_info = *type_info_or_err;
  return type_info.isBitwiseTakable() && type_info.getSize() <= 24;
}

llvm::Expected<CompilerType>
SwiftLanguageRuntime::ResolveTypeAlias(CompilerType alias) {
  using namespace swift::Demangle;
  Demangler dem;

  auto tss = alias.GetTypeSystem().dyn_cast_or_null<TypeSystemSwift>();
  if (!tss)
    return llvm::createStringError("not a Swift type");
  auto tr_ts = tss->GetTypeSystemSwiftTypeRef();
  if (!tr_ts)
    return llvm::createStringError("could not get typesystem");

  auto flavor = SwiftLanguageRuntime::GetManglingFlavor(alias.GetMangledTypeName());
  // Extract the mangling of the alias type's parent type and its
  // generic substitutions if any.
  auto mangled = alias.GetMangledTypeName().GetStringRef();
  NodePointer type = swift_demangle::GetDemangledType(dem, mangled);
  if (!type || type->getKind() != Node::Kind::TypeAlias ||
      type->getNumChildren() != 2)
    return llvm::createStringError("not a type alias");

  NodePointer alias_node = type->getChild(1);
  if (!alias_node || alias_node->getKind() != Node::Kind::Identifier ||
      !alias_node->hasText())
    return llvm::createStringError("type alias has no name");
  std::string member = alias_node->getText().str();

  NodePointer parent_node = type->getChild(0);
  if (!parent_node)
    return llvm::createStringError("top-level type alias");
  NodePointer subst_node = nullptr;

  switch (parent_node->getKind()) {
  case Node::Kind::Class:
  case Node::Kind::Structure:
  case Node::Kind::Enum:
    break;
    // For the lookup, get the unbound type.
  case Node::Kind::BoundGenericClass:
    subst_node =
        swift_demangle::ChildAtPath(parent_node, {Node::Kind::TypeList});
    parent_node = swift_demangle::ChildAtPath(
        parent_node, {Node::Kind::Type, Node::Kind::Class});
    break;
  case Node::Kind::BoundGenericStructure:
    subst_node =
        swift_demangle::ChildAtPath(parent_node, {Node::Kind::TypeList});
    parent_node = swift_demangle::ChildAtPath(
        parent_node, {Node::Kind::Type, Node::Kind::Structure});
    break;
  case Node::Kind::BoundGenericEnum:
    subst_node =
        swift_demangle::ChildAtPath(parent_node, {Node::Kind::TypeList});
    parent_node = swift_demangle::ChildAtPath(
        parent_node, {Node::Kind::Type, Node::Kind::Enum});
    break;
  default:
    return llvm::createStringError("unsupported parent kind");
  }
  if (!parent_node)
    return llvm::createStringError("unsupported generic parent kind");

  NodePointer global = dem.createNode(Node::Kind::Global);
  global->addChild(parent_node, dem);
  auto mangling = swift::Demangle::mangleNode(global);
  if (!mangling.isSuccess())
    return llvm::createStringError("mangling error");
  std::string in_type =
      swift::Demangle::dropSwiftManglingPrefix(mangling.result()).str();

  // `in_type` now holds the type alias' parent type.
  // `member` is the name of the type alias.
  // Scan through the witness tables of all of the parent's conformances.
  ThreadSafeReflectionContext reflection_ctx = GetReflectionContext();
  if (!reflection_ctx)
    return llvm::createStringError("no reflection context");

  // FIXME: The current implementation that loads all conformances
  // up-front creates too much small memory traffic during the
  // LookupTypeWitness step.
  if (!ModuleList::GetGlobalModuleListProperties().GetSwiftLoadConformances())
    return llvm::createStringError("conformance loading disabled in settings");

  for (const std::string &protocol : GetConformances(in_type)) {
    auto type_ref_or_err =
        reflection_ctx->LookupTypeWitness(in_type, member, protocol);
    if (!type_ref_or_err) {
      LLDB_LOG_ERRORV(GetLog(LLDBLog::Types), type_ref_or_err.takeError(),
                      "{0}");
      continue;
    }
    const swift::reflection::TypeRef *type_ref = &*type_ref_or_err;

    // Success, we found an associated type in the conformance.  If
    // the parent type was generic, the type alias could point to a
    // type parameter. Reapply any substitutions from the parent type.
    if (subst_node) {
      swift::reflection::GenericArgumentMap substitutions;
      unsigned idx = 0;
      for (auto &child : *subst_node) {
        auto mangling = swift_demangle::GetMangledName(dem, child, flavor);
        if (!mangling.isSuccess())
          continue;
        auto type_ref_or_err = reflection_ctx->GetTypeRef(
            mangling.result(), tr_ts->GetDescriptorFinder());
        if (!type_ref_or_err) {
          LLDB_LOG_ERRORV(GetLog(LLDBLog::Types), type_ref_or_err.takeError(),
                          "{0}");
        }
        auto &type_ref = *type_ref_or_err;
        substitutions.insert({{0, idx++}, &type_ref});
      }
      auto type_ref_or_err = reflection_ctx->ApplySubstitutions(
          *type_ref, substitutions, tr_ts->GetDescriptorFinder());
      if (!type_ref_or_err) {
        LLDB_LOG_ERRORV(GetLog(LLDBLog::Types), type_ref_or_err.takeError(),
                        "{0}");
        continue;
      }
      type_ref = &*type_ref_or_err;
    }

    CompilerType resolved = GetTypeFromTypeRef(*tr_ts, type_ref, flavor);
    LLDB_LOG(GetLog(LLDBLog::Types),
             "Resolved type alias {0} = {1} using reflection metadata.",
             alias.GetMangledTypeName(), resolved.GetMangledTypeName());
    return resolved;
  }
  return llvm::createStringError("cannot resolve type alias via reflection");
}

llvm::Expected<uint64_t>
SwiftLanguageRuntime::GetBitSize(CompilerType type,
                                 ExecutionContextScope *exe_scope) {
  auto type_info_or_err = GetSwiftRuntimeTypeInfo(type, exe_scope);
  if (!type_info_or_err)
    return type_info_or_err.takeError();

  return type_info_or_err->getSize() * 8;
}

std::optional<uint64_t> SwiftLanguageRuntime::GetByteStride(CompilerType type) {
  auto type_info_or_err = GetSwiftRuntimeTypeInfo(type, nullptr);
  if (!type_info_or_err) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Types), type_info_or_err.takeError(), "{0}");
    return {};
  }

  return type_info_or_err->getStride();
}

std::optional<size_t>
SwiftLanguageRuntime::GetBitAlignment(CompilerType type,
                                      ExecutionContextScope *exe_scope) {
  auto type_info_or_err = GetSwiftRuntimeTypeInfo(type, exe_scope);
  if (!type_info_or_err) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Types), type_info_or_err.takeError(), "{0}");
    return {};
  }

  return type_info_or_err->getAlignment() * 8;
}

bool SwiftLanguageRuntime::IsAllowedRuntimeValue(ConstString name) {
  return name.GetStringRef() == "self";
}

bool SwiftLanguageRuntime::CouldHaveDynamicValue(ValueObject &in_value) {
  return ::CouldHaveDynamicValue(in_value);
}

CompilerType
SwiftLanguageRuntime::GetConcreteType(ExecutionContextScope *exe_scope,
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

  const SymbolContext &sc = frame->GetSymbolContext(eSymbolContextFunction);
  return promise_sp->FulfillTypePromise(sc);
}

} // namespace lldb_private
