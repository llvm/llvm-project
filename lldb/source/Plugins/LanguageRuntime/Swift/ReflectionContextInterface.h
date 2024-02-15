//===-- ReflectionContextInterface.h ----------------------------*- C++ -*-===//
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

#ifndef liblldb_SwiftReflectionContextInterface_h_
#define liblldb_SwiftReflectionContextInterface_h_

#include <mutex>

#include "lldb/lldb-types.h"
#include "swift/ABI/ObjectFile.h"
#include "swift/Remote/RemoteAddress.h"
#include "swift/RemoteInspection/TypeRef.h"
#include <optional>
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Memory.h"

namespace swift {
namespace Demangle {
class Demangler;
} // namespace Demangle
namespace reflection {
struct DescriptorFinder;
class RecordTypeInfo;
class TypeInfo;
} // namespace reflection
namespace remote {
class MemoryReader;
struct TypeInfoProvider;
} // namespace remote
} // namespace swift

namespace lldb_private {
struct SwiftMetadataCache;

/// Returned by \ref ForEachSuperClassType. Not every user of \p
/// ForEachSuperClassType needs all of these. By returning this
/// object we call into the runtime only when needed.
/// Using function objects to avoid instantiating ReflectionContext in this
/// header.
struct SuperClassType {
  std::function<const swift::reflection::RecordTypeInfo *()>
      get_record_type_info;
  std::function<const swift::reflection::TypeRef *()> get_typeref;
};

/// An abstract interface to swift::reflection::ReflectionContext
/// objects of varying pointer sizes.  This class encapsulates all
/// traffic to ReflectionContext and abstracts the detail that
/// ReflectionContext is a template that needs to be specialized for
/// a specific pointer width.
class ReflectionContextInterface {
public:
  /// Return a reflection context.
  static std::unique_ptr<ReflectionContextInterface> CreateReflectionContext(
      uint8_t pointer_size, std::shared_ptr<swift::remote::MemoryReader> reader,
      bool objc_interop, SwiftMetadataCache *swift_metadata_cache);

  virtual ~ReflectionContextInterface() = default;

  virtual std::optional<uint32_t> AddImage(
      llvm::function_ref<std::pair<swift::remote::RemoteRef<void>, uint64_t>(
          swift::ReflectionSectionKind)>
          find_section,
      llvm::SmallVector<llvm::StringRef, 1> likely_module_names = {}) = 0;
  virtual std::optional<uint32_t>
  AddImage(swift::remote::RemoteAddress image_start,
           llvm::SmallVector<llvm::StringRef, 1> likely_module_names = {}) = 0;
  virtual std::optional<uint32_t>
  ReadELF(swift::remote::RemoteAddress ImageStart,
          std::optional<llvm::sys::MemoryBlock> FileBuffer,
          llvm::SmallVector<llvm::StringRef, 1> likely_module_names = {}) = 0;
  virtual const swift::reflection::TypeRef *GetTypeRefOrNull(
      llvm::StringRef mangled_type_name,
      swift::reflection::DescriptorFinder *descriptor_finder) = 0;
  virtual const swift::reflection::TypeRef *GetTypeRefOrNull(
      swift::Demangle::Demangler &dem, swift::Demangle::NodePointer node,
      swift::reflection::DescriptorFinder *descriptor_finder) = 0;
  virtual const swift::reflection::TypeInfo *GetClassInstanceTypeInfo(
      const swift::reflection::TypeRef *type_ref,
      swift::remote::TypeInfoProvider *provider,
      swift::reflection::DescriptorFinder *descriptor_finder) = 0;
  virtual const swift::reflection::TypeInfo *
  GetTypeInfo(const swift::reflection::TypeRef *type_ref,
              swift::remote::TypeInfoProvider *provider,
              swift::reflection::DescriptorFinder *descriptor_finder) = 0;
  virtual const swift::reflection::TypeInfo *GetTypeInfoFromInstance(
      lldb::addr_t instance, swift::remote::TypeInfoProvider *provider,
      swift::reflection::DescriptorFinder *descriptor_finder) = 0;
  virtual swift::remote::MemoryReader &GetReader() = 0;
  virtual const swift::reflection::TypeRef *LookupSuperclass(
      const swift::reflection::TypeRef *tr,
      swift::reflection::DescriptorFinder *descriptor_finder) = 0;
  virtual bool
  ForEachSuperClassType(swift::remote::TypeInfoProvider *tip,
                        swift::reflection::DescriptorFinder *descriptor_finder,
                        lldb::addr_t pointer,
                        std::function<bool(SuperClassType)> fn) = 0;

  /// Traverses the superclass hierarchy using the typeref, as opposed to the
  /// other version of the function that uses the instance's pointer. This
  /// version is useful when reflection metadata has been stripped from the
  /// binary (for example, when debugging embedded Swift programs).
  virtual bool
  ForEachSuperClassType(swift::remote::TypeInfoProvider *tip,
                        swift::reflection::DescriptorFinder *descriptor_finder,
                        const swift::reflection::TypeRef *tr,
                        std::function<bool(SuperClassType)> fn) = 0;

  virtual std::optional<std::pair<const swift::reflection::TypeRef *,
                                  swift::remote::RemoteAddress>>
  ProjectExistentialAndUnwrapClass(
      swift::remote::RemoteAddress existential_addess,
      const swift::reflection::TypeRef &existential_tr,
      swift::reflection::DescriptorFinder *descriptor_finder) = 0;
  virtual std::optional<int32_t> ProjectEnumValue(
      swift::remote::RemoteAddress enum_addr,
      const swift::reflection::TypeRef *enum_type_ref,
      swift::remote::TypeInfoProvider *provider,
      swift::reflection::DescriptorFinder *descriptor_finder) = 0;
  virtual const swift::reflection::TypeRef *ReadTypeFromMetadata(
      lldb::addr_t metadata_address,
      swift::reflection::DescriptorFinder *descriptor_finder,
      bool skip_artificial_subclasses = false) = 0;
  virtual const swift::reflection::TypeRef *ReadTypeFromInstance(
      lldb::addr_t instance_address,
      swift::reflection::DescriptorFinder *descriptor_finder,
      bool skip_artificial_subclasses = false) = 0;
  virtual std::optional<bool> IsValueInlinedInExistentialContainer(
      swift::remote::RemoteAddress existential_address) = 0;
  virtual const swift::reflection::TypeRef *ApplySubstitutions(
      const swift::reflection::TypeRef *type_ref,
      swift::reflection::GenericArgumentMap substitutions,
      swift::reflection::DescriptorFinder *descriptor_finder) = 0;
  virtual swift::remote::RemoteAbsolutePointer
  StripSignedPointer(swift::remote::RemoteAbsolutePointer pointer) = 0;
};

/// A wrapper around TargetReflectionContext, which holds a lock to ensure
/// exclusive access.
struct ThreadSafeReflectionContext {
  ThreadSafeReflectionContext(ReflectionContextInterface *reflection_ctx,
                              std::recursive_mutex &mutex)
      : m_reflection_ctx(reflection_ctx), m_lock(mutex, std::adopt_lock) {}

  static ThreadSafeReflectionContext MakeInvalid() {
    // This exists so we can create an "empty" reflection context in the stub
    // language runtime.
    static std::recursive_mutex mutex;
    return ThreadSafeReflectionContext(nullptr, mutex);
  }

  ReflectionContextInterface *operator->() const { return m_reflection_ctx; }

  operator bool() const { return m_reflection_ctx != nullptr; }

private:
  ReflectionContextInterface *m_reflection_ctx;
  // This lock operates on a recursive mutex because the initialization
  // of ReflectionContext recursive calls itself (see
  // SwiftLanguageRuntimeImpl::SetupReflection).
  std::lock_guard<std::recursive_mutex> m_lock;
};

} // namespace lldb_private
#endif
