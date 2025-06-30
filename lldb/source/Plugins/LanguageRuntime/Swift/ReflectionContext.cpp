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

#include "swift/RemoteInspection/ReflectionContext.h"
#include "ReflectionContextInterface.h"
#include "SwiftLanguageRuntime.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "swift/Demangling/Demangle.h"
#include "swift/RemoteInspection/DescriptorFinder.h"

using namespace lldb;
using namespace lldb_private;

namespace {

/// The descriptor finder needs to be an instance variable of the
/// TypeRefBuilder, but we would still want to swap out the descriptor finder,
/// as they are tied to each type system typeref's symbol file. This class's
/// only purpose is to allow this swapping.
struct DescriptorFinderForwarder : public swift::reflection::DescriptorFinder {
  DescriptorFinderForwarder() = default;
  ~DescriptorFinderForwarder() override = default;

  std::unique_ptr<swift::reflection::BuiltinTypeDescriptorBase>
  getBuiltinTypeDescriptor(const swift::reflection::TypeRef *TR) override {
    if (!m_descriptor_finders.empty() && shouldConsultDescriptorFinder())
      return m_descriptor_finders.back()->getBuiltinTypeDescriptor(TR);
    return nullptr;
  }

  std::unique_ptr<swift::reflection::FieldDescriptorBase>
  getFieldDescriptor(const swift::reflection::TypeRef *TR) override {
    if (!m_descriptor_finders.empty() && shouldConsultDescriptorFinder())
      return m_descriptor_finders.back()->getFieldDescriptor(TR);
    return nullptr;
  }

  std::unique_ptr<swift::reflection::MultiPayloadEnumDescriptorBase>
  getMultiPayloadEnumDescriptor(const swift::reflection::TypeRef *TR) override {
    if (!m_descriptor_finders.empty() && shouldConsultDescriptorFinder())
      return m_descriptor_finders.back()->getMultiPayloadEnumDescriptor(TR);
    return nullptr;
  }

  void PushExternalDescriptorFinder(
      swift::reflection::DescriptorFinder *descriptor_finder) {
    m_descriptor_finders.push_back(descriptor_finder);
  }

  void PopExternalDescriptorFinder() {
    assert(!m_descriptor_finders.empty() && "m_descriptor_finders is empty!");
    m_descriptor_finders.pop_back();
  }

  void SetImageAdded(bool image_added) {
    m_image_added |= image_added;
  }

private:
  bool shouldConsultDescriptorFinder() {
    switch (ModuleList::GetGlobalModuleListProperties()
                .GetSwiftEnableFullDwarfDebugging()) {
    case lldb_private::AutoBool::True:
      return true;
    case lldb_private::AutoBool::False:
      return false;
    case lldb_private::AutoBool::Auto:
      // Full DWARF debugging is auto-enabled if there is no reflection metadata
      // to read from.
      return !m_image_added;
    }
  }

  llvm::SmallVector<swift::reflection::DescriptorFinder *, 1>
      m_descriptor_finders;
  bool m_image_added = false;
};

/// An implementation of the generic ReflectionContextInterface that
/// is templatized on target pointer width and specialized to either
/// 32-bit or 64-bit pointers, with and without ObjC interoperability.
template <typename ReflectionContext, bool ObjCEnabled, unsigned PointerSize>
class TargetReflectionContext : public ReflectionContextInterface {
  DescriptorFinderForwarder m_forwader;
  ReflectionContext m_reflection_ctx;
  swift::reflection::TypeConverter &m_type_converter;

public:
  TargetReflectionContext(
      std::shared_ptr<swift::reflection::MemoryReader> reader,
      SwiftMetadataCache *swift_metadata_cache)
      : m_reflection_ctx(reader, swift_metadata_cache, &m_forwader),
        m_type_converter(m_reflection_ctx.getBuilder().getTypeConverter()) {
    m_type_converter.enableErrorCache();
  }

  std::optional<uint32_t> AddImage(
      llvm::function_ref<std::pair<swift::remote::RemoteRef<void>, uint64_t>(
          swift::ReflectionSectionKind)>
          find_section,
      llvm::SmallVector<llvm::StringRef, 1> likely_module_names) override {
    auto id = m_reflection_ctx.addImage(find_section, likely_module_names);
    m_forwader.SetImageAdded(id.has_value());
    return id;
  }

  std::optional<uint32_t>
  AddImage(swift::remote::RemoteAddress image_start,
           llvm::SmallVector<llvm::StringRef, 1> likely_module_names) override {
    auto id = m_reflection_ctx.addImage(image_start, likely_module_names);
    m_forwader.SetImageAdded(id.has_value());
    return id;
  }

  std::optional<uint32_t> ReadELF(
      swift::remote::RemoteAddress ImageStart,
      std::optional<llvm::sys::MemoryBlock> FileBuffer,
      llvm::SmallVector<llvm::StringRef, 1> likely_module_names = {}) override {
    auto id = m_reflection_ctx.readELF(ImageStart, FileBuffer,
                                    likely_module_names);
    m_forwader.SetImageAdded(id.has_value());
    return id;
  }

  llvm::Expected<const swift::reflection::TypeRef &>
  GetTypeRef(StringRef mangled_type_name,
             swift::reflection::DescriptorFinder *descriptor_finder) override {
    swift::Demangle::Demangler dem;
    swift::Demangle::NodePointer node = dem.demangleSymbol(mangled_type_name);
    return GetTypeRef(dem, node, descriptor_finder);
  }

  /// Sets the descriptor finder, and on scope exit clears it out.
  auto PushDescriptorFinderAndPopOnExit(
      swift::reflection::DescriptorFinder *descriptor_finder) {
    m_forwader.PushExternalDescriptorFinder(descriptor_finder);
    return llvm::make_scope_exit(
        [&]() { m_forwader.PopExternalDescriptorFinder(); });
  }

  llvm::Expected<const swift::reflection::TypeRef &>
  GetTypeRef(swift::Demangle::Demangler &dem, swift::Demangle::NodePointer node,
             swift::reflection::DescriptorFinder *descriptor_finder) override {
    auto on_exit = PushDescriptorFinderAndPopOnExit(descriptor_finder);
    auto type_ref_or_err =
        swift::Demangle::decodeMangledType(m_reflection_ctx.getBuilder(), node);
    if (type_ref_or_err.isError())
      return llvm::createStringError(
          type_ref_or_err.getError()->copyErrorString());
    auto *tr = type_ref_or_err.getType();
    if (!tr)
      return llvm::createStringError(
          "decoder returned nullptr typeref but no error");
    return *tr;
  }

  llvm::Expected<const swift::reflection::RecordTypeInfo &>
  GetClassInstanceTypeInfo(
      const swift::reflection::TypeRef &type_ref,
      swift::remote::TypeInfoProvider *provider,
      swift::reflection::DescriptorFinder *descriptor_finder) override {
    auto on_exit = PushDescriptorFinderAndPopOnExit(descriptor_finder);
    auto start =
        m_reflection_ctx.computeUnalignedFieldStartOffset(&type_ref, provider);
    if (!start) {
      std::stringstream ss;
      type_ref.dump(ss);
      return llvm::createStringError(
          "Could not compute start field offset for typeref: " + ss.str());
    }

    auto *rti =
        m_type_converter.getClassInstanceTypeInfo(&type_ref, *start, provider);
    if (!rti)
      return llvm::createStringError(m_type_converter.takeLastError());
    return *rti;
  }

  llvm::Expected<const swift::reflection::TypeInfo &>
  GetTypeInfo(const swift::reflection::TypeRef &type_ref,
              swift::remote::TypeInfoProvider *provider,
              swift::reflection::DescriptorFinder *descriptor_finder) override {
    auto on_exit = PushDescriptorFinderAndPopOnExit(descriptor_finder);

    Log *log(GetLog(LLDBLog::Types));
    if (log && log->GetVerbose()) {
      std::stringstream ss;
      type_ref.dump(ss);
      LLDB_LOG(log,
               "[TargetReflectionContext[{0:x}]::getTypeInfo] Getting type "
               "info for typeref {1}",
               provider ? provider->getId() : 0, ss.str());
    }

    auto type_info_or_err = m_reflection_ctx.getTypeInfo(type_ref, provider);
    if (!type_info_or_err)
      return llvm::joinErrors(
          llvm::createStringError(
              "Could not find reflection metadata for type"),
          type_info_or_err.takeError());

    if (log && log->GetVerbose()) {
      std::stringstream ss;
      type_info_or_err->dump(ss);
      LLDB_LOG(log,
               "[TargetReflectionContext::getTypeInfo] Found type info {0}",
               ss.str());
    }
    return *type_info_or_err;
  }

  llvm::Expected<const swift::reflection::TypeInfo &> GetTypeInfoFromInstance(
      lldb::addr_t instance, swift::remote::TypeInfoProvider *provider,
      swift::reflection::DescriptorFinder *descriptor_finder) override {
    auto on_exit = PushDescriptorFinderAndPopOnExit(descriptor_finder);
    auto *ti = m_reflection_ctx.getInstanceTypeInfo(
        swift::remote::RemoteAddress(
            instance, swift::remote::RemoteAddress::DefaultAddressSpace),
        provider);
    if (!ti)
      return llvm::createStringError("could not get instance type info");
    return *ti;
  }

  swift::reflection::MemoryReader &GetReader() override {
    return m_reflection_ctx.getReader();
  }

  const swift::reflection::TypeRef *LookupSuperclass(
      const swift::reflection::TypeRef &tr,
      swift::reflection::DescriptorFinder *descriptor_finder) override {
    auto on_exit = PushDescriptorFinderAndPopOnExit(descriptor_finder);
    return m_reflection_ctx.getBuilder().lookupSuperclass(&tr);
  }

  bool
  ForEachSuperClassType(swift::remote::TypeInfoProvider *tip,
                        swift::reflection::DescriptorFinder *descriptor_finder,
                        const swift::reflection::TypeRef *tr,
                        std::function<bool(SuperClassType)> fn) override {
    // Guard against faulty self-referential metadata.
    unsigned limit = 256;
    while (tr && --limit) {
      if (fn({[=]() -> const swift::reflection::RecordTypeInfo * {
                auto ti_or_err = GetRecordTypeInfo(*tr, tip, descriptor_finder);
                if (!ti_or_err) {
                  LLDB_LOG_ERRORV(GetLog(LLDBLog::Types), ti_or_err.takeError(),
                                  "ForEachSuperClassType: {0}");
                  return nullptr;
                }
                return &*ti_or_err;
              },
              [=]() -> const swift::reflection::TypeRef * { return tr; }}))
        return true;

      tr = LookupSuperclass(*tr, descriptor_finder);
    }
    return false;
  }

  bool
  ForEachSuperClassType(swift::remote::TypeInfoProvider *tip,
                        swift::reflection::DescriptorFinder *descriptor_finder,
                        lldb::addr_t pointer,
                        std::function<bool(SuperClassType)> fn) override {
    auto on_exit = PushDescriptorFinderAndPopOnExit(descriptor_finder);
    // Guard against faulty self-referential metadata.
    unsigned limit = 256;
    auto md_ptr =
        m_reflection_ctx.readMetadataFromInstance(swift::remote::RemoteAddress(
            pointer, swift::remote::RemoteAddress::DefaultAddressSpace));
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

  std::optional<int32_t> ProjectEnumValue(
      swift::remote::RemoteAddress enum_addr,
      const swift::reflection::TypeRef *enum_type_ref,
      swift::remote::TypeInfoProvider *provider,
      swift::reflection::DescriptorFinder *descriptor_finder) override {
    auto on_exit = PushDescriptorFinderAndPopOnExit(descriptor_finder);
    int32_t case_idx;
    if (m_reflection_ctx.projectEnumValue(enum_addr, enum_type_ref, &case_idx,
                                          provider))
      return case_idx;
    return {};
  }

  std::optional<std::pair<const swift::reflection::TypeRef *,
                          swift::reflection::RemoteAddress>>
  ProjectExistentialAndUnwrapClass(
      swift::reflection::RemoteAddress existential_address,
      const swift::reflection::TypeRef &existential_tr,
      swift::reflection::DescriptorFinder *descriptor_finder) override {
    auto on_exit = PushDescriptorFinderAndPopOnExit(descriptor_finder);
    return m_reflection_ctx.projectExistentialAndUnwrapClass(
        existential_address, existential_tr);
  }

  llvm::Expected<const swift::reflection::TypeRef &>
  LookupTypeWitness(const std::string &MangledTypeName,
                    const std::string &Member, StringRef Protocol) override {
    if (auto *tr = m_type_converter.getBuilder().lookupTypeWitness(
            MangledTypeName, Member, Protocol))
      return *tr;
    return llvm::createStringError("could not lookup type witness");
  }
  swift::reflection::ConformanceCollectionResult GetAllConformances() override {
    swift::reflection::TypeRefBuilder &b = m_type_converter.getBuilder();
    if (ObjCEnabled)
      return b.collectAllConformances<swift::WithObjCInterop, PointerSize>();
    return b.collectAllConformances<swift::NoObjCInterop, PointerSize>();
  }

  llvm::Expected<const swift::reflection::TypeRef &>
  ReadTypeFromMetadata(lldb::addr_t metadata_address,
                       swift::reflection::DescriptorFinder *descriptor_finder,
                       bool skip_artificial_subclasses) override {
    auto on_exit = PushDescriptorFinderAndPopOnExit(descriptor_finder);
    if (auto *tr = m_reflection_ctx.readTypeFromMetadata(
            swift::remote::RemoteAddress(
                metadata_address,
                swift::remote::RemoteAddress::DefaultAddressSpace),
            skip_artificial_subclasses))
      return *tr;
    return llvm::createStringError("could not read type from metadata");
  }

  llvm::Expected<const swift::reflection::TypeRef &>
  ReadTypeFromInstance(lldb::addr_t instance_address,
                       swift::reflection::DescriptorFinder *descriptor_finder,
                       bool skip_artificial_subclasses) override {
    auto on_exit = PushDescriptorFinderAndPopOnExit(descriptor_finder);
    auto metadata_address =
        m_reflection_ctx.readMetadataFromInstance(swift::remote::RemoteAddress(
            instance_address,
            swift::remote::RemoteAddress::DefaultAddressSpace));
    if (!metadata_address)
      return llvm::createStringError(
          llvm::formatv("could not read heap metadata for object at {0:x}",
                        instance_address));

    if (auto *tr = m_reflection_ctx.readTypeFromMetadata(
            *metadata_address, skip_artificial_subclasses))
      return *tr;
    return llvm::createStringError("could not read type from metadata");
  }

  std::optional<swift::remote::RemoteAbsolutePointer>
  ReadPointer(lldb::addr_t instance_address) override {
    auto ptr = m_reflection_ctx.readPointer(swift::remote::RemoteAddress(
        instance_address, swift::remote::RemoteAddress::DefaultAddressSpace));
    return ptr;
  }

  std::optional<bool> IsValueInlinedInExistentialContainer(
      swift::remote::RemoteAddress existential_address) override {
    return m_reflection_ctx.isValueInlinedInExistentialContainer(
        existential_address);
  }

  llvm::Expected<const swift::reflection::TypeRef &> ApplySubstitutions(
      const swift::reflection::TypeRef &type_ref,
      swift::reflection::GenericArgumentMap substitutions,
      swift::reflection::DescriptorFinder *descriptor_finder) override {
    auto on_exit = PushDescriptorFinderAndPopOnExit(descriptor_finder);
    if (auto *tr = type_ref.subst(m_reflection_ctx.getBuilder(), substitutions))
      return *tr;
    return llvm::createStringError("failed to apply substitutions");
  }

  swift::remote::RemoteAbsolutePointer
  StripSignedPointer(swift::remote::RemoteAbsolutePointer pointer) override {
    return m_reflection_ctx.stripSignedPointer(pointer);
  }

  llvm::Expected<AsyncTaskInfo>
  asyncTaskInfo(lldb::addr_t AsyncTaskPtr, unsigned ChildTaskLimit,
                unsigned AsyncBacktraceLimit) override {
    auto [error, task_info] = m_reflection_ctx.asyncTaskInfo(
        swift::remote::RemoteAddress(
            AsyncTaskPtr, swift::remote::RemoteAddress::DefaultAddressSpace),
        ChildTaskLimit, AsyncBacktraceLimit);
    if (error)
      return llvm::createStringError(*error);

    AsyncTaskInfo result;
    result.isChildTask = task_info.IsChildTask;
    result.isFuture = task_info.IsFuture;
    result.isGroupChildTask = task_info.IsGroupChildTask;
    result.isAsyncLetTask = task_info.IsAsyncLetTask;
    result.isCancelled = task_info.IsCancelled;
    result.isStatusRecordLocked = task_info.IsStatusRecordLocked;
    result.isEscalated = task_info.IsEscalated;
    result.hasIsRunning = task_info.HasIsRunning;
    result.isRunning = task_info.IsRunning;
    result.isEnqueued = task_info.IsEnqueued;
    result.isComplete = task_info.IsComplete;
    result.isSuspended = task_info.IsSuspended;
    result.id = task_info.Id;
    result.kind = task_info.Kind;
    result.enqueuePriority = task_info.EnqueuePriority;
    result.resumeAsyncContext = task_info.ResumeAsyncContext;
    result.runJob = task_info.RunJob;
    for (auto child : task_info.ChildTasks)
      result.childTasks.push_back(child);
    return result;
  }

private:
  /// Return a description of the layout of the record (classes, structs and
  /// tuples) type given its typeref.
  llvm::Expected<const swift::reflection::RecordTypeInfo &>
  GetRecordTypeInfo(const swift::reflection::TypeRef &type_ref,
                    swift::remote::TypeInfoProvider *tip,
                    swift::reflection::DescriptorFinder *descriptor_finder) {
    auto type_info_or_err = GetTypeInfo(type_ref, tip, descriptor_finder);
    if (!type_info_or_err)
      return type_info_or_err.takeError();
    auto *type_info = &*type_info_or_err;
    if (auto record_type_info =
            llvm::dyn_cast_or_null<swift::reflection::RecordTypeInfo>(
                type_info))
      return *record_type_info;
    if (llvm::isa_and_nonnull<swift::reflection::ReferenceTypeInfo>(type_info))
      return GetClassInstanceTypeInfo(type_ref, tip, descriptor_finder);
    std::stringstream ss;
    type_ref.dump(ss);
    return llvm::createStringError(
        "Could not get record type info for typeref: " + ss.str());
  }
};
} // namespace

namespace lldb_private {
std::unique_ptr<ReflectionContextInterface>
ReflectionContextInterface::CreateReflectionContext(
    uint8_t ptr_size, std::shared_ptr<swift::remote::MemoryReader> reader,
    bool ObjCInterop, SwiftMetadataCache *swift_metadata_cache) {
  using ReflectionContext32ObjCInterop = TargetReflectionContext<
      swift::reflection::ReflectionContext<
          swift::External<swift::WithObjCInterop<swift::RuntimeTarget<4>>>>,
      true, 4>;
  using ReflectionContext32NoObjCInterop = TargetReflectionContext<
      swift::reflection::ReflectionContext<
          swift::External<swift::NoObjCInterop<swift::RuntimeTarget<4>>>>,
      false, 4>;
  using ReflectionContext64ObjCInterop = TargetReflectionContext<
      swift::reflection::ReflectionContext<
          swift::External<swift::WithObjCInterop<swift::RuntimeTarget<8>>>>,
      true, 8>;
  using ReflectionContext64NoObjCInterop = TargetReflectionContext<
      swift::reflection::ReflectionContext<
          swift::External<swift::NoObjCInterop<swift::RuntimeTarget<8>>>>,
      false, 8>;
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
