//===-- ReflectionContext.cpp --------------------------------------------===//
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

#include "SwiftLanguageRuntimeImpl.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "swift/Demangling/Demangle.h"

using namespace lldb;
using namespace lldb_private;

namespace {

/// An implementation of the generic ReflectionContextInterface that
/// is templatized on target pointer width and specialized to either
/// 32-bit or 64-bit pointers, with and without ObjC interoperability.
template <typename ReflectionContext>
class TargetReflectionContext
    : public SwiftLanguageRuntimeImpl::ReflectionContextInterface {
  ReflectionContext m_reflection_ctx;
  swift::reflection::TypeConverter m_type_converter;

public:
  TargetReflectionContext(
      std::shared_ptr<swift::reflection::MemoryReader> reader,
      SwiftMetadataCache *swift_metadata_cache)
      : m_reflection_ctx(reader, swift_metadata_cache),
        m_type_converter(m_reflection_ctx.getBuilder()) {}

  llvm::Optional<uint32_t> AddImage(
      llvm::function_ref<std::pair<swift::remote::RemoteRef<void>, uint64_t>(
          swift::ReflectionSectionKind)>
          find_section,
      llvm::SmallVector<llvm::StringRef, 1> likely_module_names) override {
    return m_reflection_ctx.addImage(find_section, likely_module_names);
  }

  llvm::Optional<uint32_t>
  AddImage(swift::remote::RemoteAddress image_start,
           llvm::SmallVector<llvm::StringRef, 1> likely_module_names) override {
    return m_reflection_ctx.addImage(image_start, likely_module_names);
  }

  llvm::Optional<uint32_t> ReadELF(
      swift::remote::RemoteAddress ImageStart,
      llvm::Optional<llvm::sys::MemoryBlock> FileBuffer,
      llvm::SmallVector<llvm::StringRef, 1> likely_module_names = {}) override {
    return m_reflection_ctx.readELF(ImageStart, FileBuffer,
                                    likely_module_names);
  }

  const swift::reflection::TypeRef *
  GetTypeRefOrNull(StringRef mangled_type_name) override {
    swift::Demangle::Demangler dem;
    swift::Demangle::NodePointer node = dem.demangleSymbol(mangled_type_name);
    const swift::reflection::TypeRef *type_ref = GetTypeRefOrNull(dem, node);
    if (!type_ref)
      LLDB_LOG(GetLog(LLDBLog::Types), "Could not find typeref for type: {0}",
               mangled_type_name);
    return type_ref;
  }

  virtual const swift::reflection::TypeRef *
  GetTypeRefOrNull(swift::Demangle::Demangler &dem,
                   swift::Demangle::NodePointer node) override {
    auto type_ref_or_err =
        swift::Demangle::decodeMangledType(m_reflection_ctx.getBuilder(), node);
    if (type_ref_or_err.isError()) {
      LLDB_LOG(GetLog(LLDBLog::Types),
               "Could not find typeref: decode mangled type failed. Error: {0}",
               type_ref_or_err.getError()->copyErrorString());
      return nullptr;
    }
    return type_ref_or_err.getType();
  }

  const swift::reflection::TypeInfo *
  GetClassInstanceTypeInfo(const swift::reflection::TypeRef *type_ref,
                           swift::remote::TypeInfoProvider *provider) override {
    if (!type_ref)
      return nullptr;
    return m_type_converter.getClassInstanceTypeInfo(type_ref, 0, provider);
  }

  const swift::reflection::TypeInfo *
  GetTypeInfo(const swift::reflection::TypeRef *type_ref,
              swift::remote::TypeInfoProvider *provider) override {
    if (!type_ref)
      return nullptr;

    Log *log(GetLog(LLDBLog::Types));
    if (log && log->GetVerbose()) {
      std::stringstream ss;
      type_ref->dump(ss);
      LLDB_LOGF(log,
                "[TargetReflectionContext::getTypeInfo] Getting "
                "type info for typeref:\n%s",
                ss.str().c_str());
    }

    auto type_info = m_reflection_ctx.getTypeInfo(type_ref, provider);
    if (log && !type_info) {
      std::stringstream ss;
      type_ref->dump(ss);
      LLDB_LOGF(log,
                "[TargetReflectionContext::getTypeInfo] Could not get "
                "type info for typeref:\n%s",
                ss.str().c_str());
    }

    if (type_info && log && log->GetVerbose()) {
      std::stringstream ss;
      type_info->dump(ss);
      log->Printf("[TargetReflectionContext::getTypeInfo] Found "
                  "type info:\n%s",
                  ss.str().c_str());
    }
    return type_info;
  }

  const swift::reflection::TypeInfo *
  GetTypeInfoFromInstance(lldb::addr_t instance,
                          swift::remote::TypeInfoProvider *provider) override {
    return m_reflection_ctx.getInstanceTypeInfo(instance, provider);
  }

  swift::reflection::MemoryReader &GetReader() override {
    return m_reflection_ctx.getReader();
  }

  const swift::reflection::TypeRef *
  LookupSuperclass(const swift::reflection::TypeRef *tr) override {
    return m_reflection_ctx.getBuilder().lookupSuperclass(tr);
  }

  bool ForEachSuperClassType(
      swift::remote::TypeInfoProvider *tip, lldb::addr_t pointer,
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

  llvm::Optional<int32_t>
  ProjectEnumValue(swift::remote::RemoteAddress enum_addr,
                   const swift::reflection::TypeRef *enum_type_ref,
                   swift::remote::TypeInfoProvider *provider) override {
    int32_t case_idx;
    if (m_reflection_ctx.projectEnumValue(enum_addr, enum_type_ref, &case_idx,
                                          provider))
      return case_idx;
    return {};
  }

  llvm::Optional<std::pair<const swift::reflection::TypeRef *,
                           swift::reflection::RemoteAddress>>
  ProjectExistentialAndUnwrapClass(
      swift::reflection::RemoteAddress existential_address,
      const swift::reflection::TypeRef &existential_tr) override {
    return m_reflection_ctx.projectExistentialAndUnwrapClass(
        existential_address, existential_tr);
  }

  const swift::reflection::TypeRef *
  ReadTypeFromMetadata(lldb::addr_t metadata_address,
                       bool skip_artificial_subclasses) override {
    return m_reflection_ctx.readTypeFromMetadata(metadata_address,
                                                 skip_artificial_subclasses);
  }

  const swift::reflection::TypeRef *
  ReadTypeFromInstance(lldb::addr_t instance_address,
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

  llvm::Optional<bool> IsValueInlinedInExistentialContainer(
      swift::remote::RemoteAddress existential_address) override {
    return m_reflection_ctx.isValueInlinedInExistentialContainer(
        existential_address);
  }

  const swift::reflection::TypeRef *ApplySubstitutions(
      const swift::reflection::TypeRef *type_ref,
      swift::reflection::GenericArgumentMap substitutions) override{
    return type_ref->subst(m_reflection_ctx.getBuilder(), substitutions);
  }

  swift::remote::RemoteAbsolutePointer StripSignedPointer(
      swift::remote::RemoteAbsolutePointer pointer) override {
    return m_reflection_ctx.stripSignedPointer(pointer);
  }
};

} // namespace

namespace lldb_private {
std::unique_ptr<SwiftLanguageRuntimeImpl::ReflectionContextInterface>
SwiftLanguageRuntimeImpl::ReflectionContextInterface::CreateReflectionContext(
    uint8_t ptr_size, std::shared_ptr<swift::remote::MemoryReader> reader,
    bool ObjCInterop, SwiftMetadataCache *swift_metadata_cache) {
  using ReflectionContext32ObjCInterop =
      TargetReflectionContext<swift::reflection::ReflectionContext<
          swift::External<swift::WithObjCInterop<swift::RuntimeTarget<4>>>>>;
  using ReflectionContext32NoObjCInterop =
      TargetReflectionContext<swift::reflection::ReflectionContext<
          swift::External<swift::NoObjCInterop<swift::RuntimeTarget<4>>>>>;
  using ReflectionContext64ObjCInterop =
      TargetReflectionContext<swift::reflection::ReflectionContext<
          swift::External<swift::WithObjCInterop<swift::RuntimeTarget<8>>>>>;
  using ReflectionContext64NoObjCInterop =
      TargetReflectionContext<swift::reflection::ReflectionContext<
          swift::External<swift::NoObjCInterop<swift::RuntimeTarget<8>>>>>;
  if (ptr_size == 4) {
    if (ObjCInterop)
      return std::make_unique<ReflectionContext32ObjCInterop>(
          reader, swift_metadata_cache);
    return std::make_unique<ReflectionContext32NoObjCInterop>(
        reader, swift_metadata_cache);
  }
  if (ptr_size == 8) {
    if (ObjCInterop)
      return std::make_unique<ReflectionContext64ObjCInterop>(
          reader, swift_metadata_cache);
    return std::make_unique<ReflectionContext64NoObjCInterop>(
        reader, swift_metadata_cache);
  }
  return {};
}
} // namespace lldb_private
