//===-- SwiftLanguageRuntime.h ----------------------------------*- C++ -*-===//
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

#ifndef liblldb_SwiftLanguageRuntimeImpl_h_
#define liblldb_SwiftLanguageRuntimeImpl_h_

#include "LLDBMemoryReader.h"
#include "SwiftLanguageRuntime.h"
#include "SwiftMetadataCache.h"
#include "llvm/Support/Memory.h"

namespace swift {
namespace reflection {
class TypeRef;
}
} // namespace swift

namespace lldb_private {
class Process;
class LLDBTypeInfoProvider;

/// A full LLDB language runtime backed by the Swift runtime library
/// in the process.
class SwiftLanguageRuntimeImpl {
  Process &m_process;

public:
  SwiftLanguageRuntimeImpl(Process &process);
  SwiftLanguageRuntimeImpl(const SwiftLanguageRuntimeImpl &) = delete;
  const SwiftLanguageRuntimeImpl &
  operator=(const SwiftLanguageRuntimeImpl &) = delete;

  static lldb::BreakpointPreconditionSP
  GetBreakpointExceptionPrecondition(lldb::LanguageType language,
                                     bool throw_bp);

  class SwiftExceptionPrecondition : public BreakpointPrecondition {
  public:
    SwiftExceptionPrecondition();

    virtual ~SwiftExceptionPrecondition() {}

    bool EvaluatePrecondition(StoppointCallbackContext &context) override;
    void GetDescription(Stream &stream, lldb::DescriptionLevel level) override;
    Status ConfigurePrecondition(Args &args) override;

  protected:
    void AddTypeName(const char *type_name);
    void AddEnumSpec(const char *enum_name, const char *element_name);

  private:
    std::unordered_set<std::string> m_type_names;
    std::unordered_map<std::string, std::vector<std::string>> m_enum_spec;
  };

  void ModulesDidLoad(const ModuleList &module_list);

  bool GetObjectDescription(Stream &str, ValueObject &object);

  SwiftExceptionPrecondition *GetExceptionPrecondition();

  /// This call should return true if it could set the name and/or the type.
  bool GetDynamicTypeAndAddress(ValueObject &in_value,
                                lldb::DynamicValueType use_dynamic,
                                TypeAndOrName &class_type_or_name,
                                Address &address, Value::ValueType &value_type);

  TypeAndOrName FixUpDynamicType(const TypeAndOrName &type_and_or_name,
                                 ValueObject &static_value);
  bool IsTaggedPointer(lldb::addr_t addr, CompilerType type);
  std::pair<lldb::addr_t, bool> FixupPointerValue(lldb::addr_t addr,
                                                  CompilerType type);
  /// This allows a language runtime to adjust references depending on the type.
  lldb::addr_t FixupAddress(lldb::addr_t addr, CompilerType type,
                            Status &error);

  /// Ask Remote Mirrors for the type info about a Swift type.
  /// This will return a nullptr if the lookup fails.
  const swift::reflection::TypeInfo *
  GetSwiftRuntimeTypeInfo(CompilerType type, ExecutionContextScope *exe_scope,
                          swift::reflection::TypeRef const **out_tr = nullptr);

  llvm::Optional<const swift::reflection::TypeInfo *>
  lookupClangTypeInfo(CompilerType clang_type);

  const swift::reflection::TypeInfo *emplaceClangTypeInfo(
      CompilerType clang_type, llvm::Optional<uint64_t> byte_size,
      llvm::Optional<size_t> bit_align,
      llvm::ArrayRef<swift::reflection::FieldInfo> fields);

  bool IsStoredInlineInBuffer(CompilerType type);

  /// Ask Remote Mirrors for the size of a Swift type.
  llvm::Optional<uint64_t> GetBitSize(CompilerType type,
                                      ExecutionContextScope *exe_scope);

  /// Ask Remote mirrors for the stride of a Swift type.
  llvm::Optional<uint64_t> GetByteStride(CompilerType type);

  /// Ask Remote mirrors for the alignment of a Swift type.
  llvm::Optional<size_t> GetBitAlignment(CompilerType type,
                                         ExecutionContextScope *exe_scope);

  CompilerType GetTypeFromMetadata(TypeSystemSwift &tss, Address address);

  llvm::Optional<uint64_t>
  GetMemberVariableOffsetRemoteAST(CompilerType instance_type,
                                   ValueObject *instance,
                                   llvm::StringRef member_name);
  llvm::Optional<uint64_t> GetMemberVariableOffsetRemoteMirrors(
      CompilerType instance_type, ValueObject *instance,
      llvm::StringRef member_name, Status *error);
  llvm::Optional<uint64_t> GetMemberVariableOffset(CompilerType instance_type,
                                                   ValueObject *instance,
                                                   llvm::StringRef member_name,
                                                   Status *error);

  llvm::Optional<unsigned> GetNumChildren(CompilerType type,
                                          ExecutionContextScope *exe_scope);

  llvm::Optional<unsigned> GetNumFields(CompilerType type,
                                        ExecutionContext *exe_ctx);

  llvm::Optional<std::string> GetEnumCaseName(CompilerType type,
                                              const DataExtractor &data,
                                              ExecutionContext *exe_ctx);

  std::pair<bool, llvm::Optional<size_t>> GetIndexOfChildMemberWithName(
      CompilerType type, llvm::StringRef name, ExecutionContext *exe_ctx,
      bool omit_empty_base_classes, std::vector<uint32_t> &child_indexes);

  CompilerType GetChildCompilerTypeAtIndex(
      CompilerType type, size_t idx, bool transparent_pointers,
      bool omit_empty_base_classes, bool ignore_array_bounds,
      std::string &child_name, uint32_t &child_byte_size,
      int32_t &child_byte_offset, uint32_t &child_bitfield_bit_size,
      uint32_t &child_bitfield_bit_offset, bool &child_is_base_class,
      bool &child_is_deref_of_parent, ValueObject *valobj,
      uint64_t &language_flags);

  CompilerType BindGenericTypeParameters(
      CompilerType unbound_type,
      std::function<CompilerType(unsigned, unsigned)> type_resolver);

  /// Like \p BindGenericTypeParameters but for TypeSystemSwiftTypeRef.
  CompilerType BindGenericTypeParameters(StackFrame &stack_frame,
                                         TypeSystemSwiftTypeRef &ts,
                                         ConstString mangled_name);

  /// Like \p BindGenericTypeParameters but for RemoteAST.
  CompilerType BindGenericTypeParametersRemoteAST(StackFrame &stack_frame,
                                                  CompilerType base_type);

  /// \see SwiftLanguageRuntime::BindGenericTypeParameters().
  CompilerType BindGenericTypeParameters(StackFrame &stack_frame,
                                         CompilerType base_type);

  CompilerType GetConcreteType(ExecutionContextScope *exe_scope,
                               ConstString abstract_type_name);

  /// Retrieve the remote AST context for the given Swift AST context.
  swift::remoteAST::RemoteASTContext &
  GetRemoteASTContext(SwiftASTContext &swift_ast_ctx);

  /// Release the RemoteASTContext associated with the given swift::ASTContext.
  /// Note that a RemoteASTContext must be destroyed before its associated
  /// swift::ASTContext is destroyed.
  void ReleaseAssociatedRemoteASTContext(swift::ASTContext *ctx);

  void AddToLibraryNegativeCache(llvm::StringRef library_name);
  bool IsInLibraryNegativeCache(llvm::StringRef library_name);
  void WillStartExecutingUserExpression(bool runs_in_playground_or_repl);
  void DidFinishExecutingUserExpression(bool runs_in_playground_or_repl);
  bool IsValidErrorValue(ValueObject &in_value);

  ConstString GetErrorBackstopName();
  ConstString GetStandardLibraryName();
  ConstString GetStandardLibraryBaseName();

  lldb::SyntheticChildrenSP
  GetBridgedSyntheticChildProvider(ValueObject &valobj);

  bool IsABIStable();

  void DumpTyperef(CompilerType type, TypeSystemSwiftTypeRef *module_holder,
                   Stream *s);
  /// Returned by \ref ForEachSuperClassType. Not every user of \p
  /// ForEachSuperClassType needs all of these. By returning this
  /// object we call into the runtime only when needed.
  /// Using function objects to avoid instantiating ReflectionContext in this header.
  struct SuperClassType {
    std::function<const swift::reflection::RecordTypeInfo *()> get_record_type_info;
    std::function<const swift::reflection::TypeRef *()> get_typeref;
  };

  Process &GetProcess() const;

  /// An abstract interface to swift::reflection::ReflectionContext
  /// objects of varying pointer sizes.  This class encapsulates all
  /// traffic to ReflectionContext and abstracts the detail that
  /// ReflectionContext is a template that needs to be specialized for
  /// a specific pointer width.
  class ReflectionContextInterface {
  public:
    /// Return a reflection context.
    static std::unique_ptr<ReflectionContextInterface>
    CreateReflectionContext(uint8_t pointer_size,
        std::shared_ptr<swift::remote::MemoryReader> reader, bool objc_interop,
        SwiftMetadataCache *swift_metadata_cache);

    virtual ~ReflectionContextInterface();

    virtual llvm::Optional<uint32_t> AddImage(
        llvm::function_ref<std::pair<swift::remote::RemoteRef<void>, uint64_t>(
            swift::ReflectionSectionKind)>
            find_section,
        llvm::SmallVector<llvm::StringRef, 1> likely_module_names = {}) = 0;
    virtual llvm::Optional<uint32_t> AddImage(
        swift::remote::RemoteAddress image_start,
        llvm::SmallVector<llvm::StringRef, 1> likely_module_names = {}) = 0;
    virtual llvm::Optional<uint32_t>
    ReadELF(swift::remote::RemoteAddress ImageStart,
            llvm::Optional<llvm::sys::MemoryBlock> FileBuffer,
            llvm::SmallVector<llvm::StringRef, 1> likely_module_names = {}) = 0;
    virtual const swift::reflection::TypeRef *
    GetTypeRefOrNull(StringRef mangled_type_name) = 0;
    virtual const swift::reflection::TypeRef *
    GetTypeRefOrNull(swift::Demangle::Demangler &dem,
                     swift::Demangle::NodePointer node) = 0;
    virtual const swift::reflection::TypeInfo *
    GetClassInstanceTypeInfo(const swift::reflection::TypeRef *type_ref,
                             swift::remote::TypeInfoProvider *provider) = 0;
    virtual const swift::reflection::TypeInfo *
    GetTypeInfo(const swift::reflection::TypeRef *type_ref,
                swift::remote::TypeInfoProvider *provider) = 0;
    virtual const swift::reflection::TypeInfo *
    GetTypeInfoFromInstance(lldb::addr_t instance,
                            swift::remote::TypeInfoProvider *provider) = 0;
    virtual swift::remote::MemoryReader &GetReader() = 0;
    virtual const swift::reflection::TypeRef *
    LookupSuperclass(const swift::reflection::TypeRef *tr) = 0;
    virtual bool
    ForEachSuperClassType(swift::remote::TypeInfoProvider *tip,
                          lldb::addr_t pointer,
                          std::function<bool(SuperClassType)> fn) = 0;
    virtual llvm::Optional<std::pair<const swift::reflection::TypeRef *,
                                     swift::remote::RemoteAddress>>
    ProjectExistentialAndUnwrapClass(
        swift::remote::RemoteAddress existential_addess,
        const swift::reflection::TypeRef &existential_tr) = 0;
    virtual llvm::Optional<int32_t>
    ProjectEnumValue(swift::remote::RemoteAddress enum_addr,
                     const swift::reflection::TypeRef *enum_type_ref,
                     swift::remote::TypeInfoProvider *provider) = 0;
    virtual const swift::reflection::TypeRef *
    ReadTypeFromMetadata(lldb::addr_t metadata_address,
                         bool skip_artificial_subclasses = false) = 0;
    virtual const swift::reflection::TypeRef *
    ReadTypeFromInstance(lldb::addr_t instance_address,
                         bool skip_artificial_subclasses = false) = 0;
    virtual llvm::Optional<bool> IsValueInlinedInExistentialContainer(
        swift::remote::RemoteAddress existential_address) = 0;
    virtual const swift::reflection::TypeRef *
    ApplySubstitutions(const swift::reflection::TypeRef *type_ref,
                       swift::reflection::GenericArgumentMap substitutions) = 0;
    virtual swift::remote::RemoteAbsolutePointer
    StripSignedPointer(swift::remote::RemoteAbsolutePointer pointer) = 0;
  };

  /// A wrapper around TargetReflectionContext, which holds a lock to ensure
  /// exclusive access.
  struct ThreadSafeReflectionContext {
    ThreadSafeReflectionContext(ReflectionContextInterface *reflection_ctx,
                             std::recursive_mutex &mutex)
        : m_reflection_ctx(reflection_ctx), m_lock(mutex, std::adopt_lock) {}

    ReflectionContextInterface *operator->() const {
      return m_reflection_ctx;
    }

    operator bool() const {
      return m_reflection_ctx != nullptr;
    }

  private:
    ReflectionContextInterface *m_reflection_ctx;
    // This lock operates on a recursive mutex because the initialization
    // of ReflectionContext recursive calls itself (see
    // SwiftLanguageRuntimeImpl::SetupReflection).
    std::lock_guard<std::recursive_mutex> m_lock;
  };

protected:
  static void
  ForEachGenericParameter(swift::Demangle::NodePointer node,
                          std::function<void(unsigned, unsigned)> callback) {
    SwiftLanguageRuntime::ForEachGenericParameter(node, callback);
  }

  /// Use the reflection context to build a TypeRef object.
  const swift::reflection::TypeRef *
  GetTypeRef(CompilerType type, TypeSystemSwiftTypeRef *module_holder);

  /// If \p instance points to a Swift object, retrieve its
  /// RecordTypeInfo and pass it to the callback \p fn. Repeat the
  /// process with all superclasses. If \p fn returns \p true, early
  /// exit and return \p true. Otherwise return \p false.
  bool ForEachSuperClassType(ValueObject &instance,
                             std::function<bool(SuperClassType)> fn);

  // Classes that inherit from SwiftLanguageRuntime can see and modify these
  Value::ValueType GetValueType(ValueObject &in_value,
                                CompilerType dynamic_type,
                                Value::ValueType static_value_type,
                                bool is_indirect_enum_case);
  bool GetDynamicTypeAndAddress_Pack(ValueObject &in_value,
                                     CompilerType pack_type,
                                     lldb::DynamicValueType use_dynamic,
                                     TypeAndOrName &class_type_or_name,
                                     Address &address,
                                     Value::ValueType &value_type);

  bool GetDynamicTypeAndAddress_Class(ValueObject &in_value,
                                      CompilerType class_type,
                                      lldb::DynamicValueType use_dynamic,
                                      TypeAndOrName &class_type_or_name,
                                      Address &address,
                                      Value::ValueType &value_type);
#ifndef NDEBUG
  ConstString GetDynamicTypeName_ClassRemoteAST(ValueObject &in_value,
                                                lldb::addr_t instance_ptr);
#endif
  bool GetDynamicTypeAndAddress_Protocol(ValueObject &in_value,
                                         CompilerType protocol_type,
                                         lldb::DynamicValueType use_dynamic,
                                         TypeAndOrName &class_type_or_name,
                                         Address &address);
#ifndef NDEBUG
  llvm::Optional<std::pair<CompilerType, Address>>
  GetDynamicTypeAndAddress_ProtocolRemoteAST(ValueObject &in_value,
                                             CompilerType protocol_type,
                                             bool use_local_buffer,
                                             lldb::addr_t existential_address);
#endif

  bool GetDynamicTypeAndAddress_ExistentialMetatype(
      ValueObject &in_value, CompilerType meta_type,
      lldb::DynamicValueType use_dynamic, TypeAndOrName &class_type_or_name,
      Address &address);

  bool GetDynamicTypeAndAddress_Value(ValueObject &in_value,
                                      CompilerType &bound_type,
                                      lldb::DynamicValueType use_dynamic,
                                      TypeAndOrName &class_type_or_name,
                                      Address &address,
                                      Value::ValueType &value_type);

  bool GetDynamicTypeAndAddress_IndirectEnumCase(
      ValueObject &in_value, lldb::DynamicValueType use_dynamic,
      TypeAndOrName &class_type_or_name, Address &address,
      Value::ValueType &value_type);

  bool GetDynamicTypeAndAddress_ClangType(ValueObject &in_value,
                                          lldb::DynamicValueType use_dynamic,
                                          TypeAndOrName &class_type_or_name,
                                          Address &address,
                                          Value::ValueType &value_type);

  /// A proxy object to support lazy binding of Archetypes.
  class MetadataPromise {
    friend class SwiftLanguageRuntimeImpl;

    MetadataPromise(ValueObject &, SwiftLanguageRuntimeImpl &, lldb::addr_t);

    lldb::ValueObjectSP m_for_object_sp;
    SwiftLanguageRuntimeImpl &m_swift_runtime;
    lldb::addr_t m_metadata_location;
    llvm::Optional<swift::MetadataKind> m_metadata_kind;
    llvm::Optional<CompilerType> m_compiler_type;

  public:
    CompilerType FulfillTypePromise(Status *error = nullptr);
  };
  typedef std::shared_ptr<MetadataPromise> MetadataPromiseSP;

  MetadataPromiseSP GetMetadataPromise(lldb::addr_t addr,
                                       ValueObject &for_object);
  MetadataPromiseSP
  GetPromiseForTypeNameAndFrame(const char *type_name, StackFrame *frame);

  llvm::Optional<lldb::addr_t>
  GetTypeMetadataForTypeNameAndFrame(llvm::StringRef mdvar_name,
                                     StackFrame &frame);

  const CompilerType &GetBoxMetadataType();

  std::shared_ptr<LLDBMemoryReader> GetMemoryReader();

  void PushLocalBuffer(uint64_t local_buffer, uint64_t local_buffer_size);

  void PopLocalBuffer();

  // These are the helper functions for GetObjectDescription for various
  // types of swift objects.
  std::string GetObjectDescriptionExpr_Result(ValueObject &object);
  std::string GetObjectDescriptionExpr_Ref(ValueObject &object);
  std::string GetObjectDescriptionExpr_Copy(ValueObject &object, 
      lldb::addr_t &copy_location);
  bool RunObjectDescriptionExpr(ValueObject &object, std::string &expr_string, 
                                Stream &result);
  /// We have to load swift dependent libraries by hand, but if they
  /// are missing, we shouldn't keep trying.
  llvm::StringSet<> m_library_negative_cache;
  std::mutex m_negative_cache_mutex;

  std::shared_ptr<LLDBMemoryReader> m_memory_reader_sp;

  llvm::DenseMap<std::pair<swift::ASTContext *, lldb::addr_t>,
                 MetadataPromiseSP>
      m_promises_map;

  llvm::DenseMap<swift::ASTContext *,
                 std::unique_ptr<swift::remoteAST::RemoteASTContext>>
      m_remote_ast_contexts;

  /// Uses ConstStrings as keys to avoid storing the strings twice.
  llvm::DenseMap<const char *, lldb::SyntheticChildrenSP>
      m_bridged_synthetics_map;

  /// Cached member variable offsets.
  using MemberID = std::pair<const swift::TypeBase *, const char *>;
  llvm::DenseMap<MemberID, uint64_t> m_member_offsets;

  CompilerType m_box_metadata_type;

private:
  /// Don't call these directly.
  /// \{
  /// There is a global variable \p _swift_classIsSwiftMask that is
  /// used to communicate with the Swift language runtime. It needs to
  /// be initialized by us, but could in theory also be written to by
  /// the runtime.
  void SetupABIBit();
  void SetupExclusivity();
  void SetupReflection();
  void SetupSwiftError();
  /// \}

  /// Whether \p SetupReflection() has been run.
  bool m_initialized_reflection_ctx = false;

  /// Lazily initialize and return \p m_dynamic_exclusivity_flag_addr.
  llvm::Optional<lldb::addr_t> GetDynamicExclusivityFlagAddr();

  /// Lazily initialize the reflection context. Returns a
  /// ThreadSafeReflectionContext with a \p nullptr on failure.
  ThreadSafeReflectionContext GetReflectionContext();

  // Add the modules in m_modules_to_add to the Reflection Context. The
  // ModulesDidLoad() callback appends to m_modules_to_add.
  void ProcessModulesToAdd();

  /// Lazily initialize and return \p m_SwiftNativeNSErrorISA.
  llvm::Optional<lldb::addr_t> GetSwiftNativeNSErrorISA();

  SwiftMetadataCache *GetSwiftMetadataCache();

  /// These members are used to track and toggle the state of the "dynamic
  /// exclusivity enforcement flag" in the swift runtime. This flag is set to
  /// true when an LLDB expression starts running, and reset to its original
  /// state after that expression (and any other concurrently running
  /// expressions) terminates.
  /// \{
  std::mutex m_active_user_expr_mutex;
  uint32_t m_active_user_expr_count = 0;

  bool m_original_dynamic_exclusivity_flag_state = false;
  llvm::Optional<lldb::addr_t> m_dynamic_exclusivity_flag_addr;
  /// \}

  /// Reflection context.
  /// \{
  std::unique_ptr<ReflectionContextInterface> m_reflection_ctx;

  /// Mutex guarding accesses to the reflection context.
  std::recursive_mutex m_reflection_ctx_mutex;

  SwiftMetadataCache m_swift_metadata_cache;

  /// Record modules added through ModulesDidLoad, which are to be
  /// added to the reflection context once it's being initialized.
  ModuleList m_modules_to_add;

  /// Add the image to the reflection context.
  /// \return true on success.
  bool AddModuleToReflectionContext(const lldb::ModuleSP &module_sp);
  /// \}

  /// Add the contents of the object file to the reflection context.
  /// \return true on success.
  bool AddJitObjectFileToReflectionContext(
      ObjectFile &obj_file, llvm::Triple::ObjectFormatType obj_format_type,
      llvm::SmallVector<llvm::StringRef, 1> likely_module_names);

  /// Add the reflections sections to the reflection context by extracting
  /// the directly from the object file.
  /// \return the info id of the newly registered reflection info on success, or
  /// llvm::None otherwise.
  llvm::Optional<uint32_t> AddObjectFileToReflectionContext(
      lldb::ModuleSP module,
      llvm::SmallVector<llvm::StringRef, 1> likely_module_names);

  /// Cache for the debug-info-originating type infos.
  /// \{
  llvm::DenseMap<lldb::opaque_compiler_type_t,
                 llvm::Optional<swift::reflection::TypeInfo>>
      m_clang_type_info;
  llvm::DenseMap<lldb::opaque_compiler_type_t,
                 llvm::Optional<swift::reflection::RecordTypeInfo>>
      m_clang_record_type_info;
  std::recursive_mutex m_clang_type_info_mutex;
  /// \}

  /// Swift native NSError isa.
  llvm::Optional<lldb::addr_t> m_SwiftNativeNSErrorISA;

#ifndef NDEBUG
  /// Assert helper to determine if the scratch SwiftASTContext is locked.
  static bool IsScratchContextLocked(Target &target);
  /// Assert helper to determine if the scratch SwiftASTContext is locked.
  static bool IsScratchContextLocked(lldb::TargetSP target);
#endif
};

} // namespace lldb_private
#endif
