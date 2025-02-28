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

#ifndef liblldb_SwiftLanguageRuntime_h_
#define liblldb_SwiftLanguageRuntime_h_

#include "Plugins/LanguageRuntime/ObjC/AppleObjCRuntime/AppleObjCRuntimeV2.h"
#include "Plugins/LanguageRuntime/Swift/SwiftMetadataCache.h"
#include "Plugins/TypeSystem/Swift/SwiftASTContext.h"
#include "lldb/Breakpoint/BreakpointPrecondition.h"
#include "lldb/Core/PluginInterface.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/lldb-private.h"
#include "swift/Demangling/ManglingFlavor.h"

#include <optional>
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Casting.h"

#include <mutex>
#include <tuple>
#include <vector>

namespace swift {
namespace remote {
class MemoryReader;
class RemoteAddress;
} // namespace remote

template <typename Runtime> struct External;
template <unsigned PointerSize> struct RuntimeTarget;

namespace reflection {
template <typename T> class ReflectionContext;
class TypeInfo;
struct FieldInfo;
class TypeRef;
class RecordTypeInfo;
} // namespace reflection

namespace remoteAST {
class RemoteASTContext;
}
enum class MetadataKind : uint32_t;
class TypeBase;
} // namespace swift

namespace lldb_private {
template <typename T>
struct LockGuarded;

class SwiftLanguageRuntimeStub;
class SwiftLanguageRuntimeImpl;
class ReflectionContextInterface;
class LLDBMemoryReader;
struct SuperClassType;

using ThreadSafeReflectionContext = LockGuarded<ReflectionContextInterface>;

class SwiftLanguageRuntime : public LanguageRuntime {
protected:
  SwiftLanguageRuntime(Process &process);

public:
  ThreadSafeReflectionContext GetReflectionContext();
  static char ID;

  bool isA(const void *ClassID) const override {
    return ClassID == &ID || LanguageRuntime::isA(ClassID);
  }

  /// Static Functions.
  /// \{
  static void Initialize();
  static void Terminate();

  static lldb_private::LanguageRuntime *
  CreateInstance(Process *process, lldb::LanguageType language);

  static llvm::StringRef GetPluginNameStatic() { return "swift"; }

  static bool classof(const LanguageRuntime *runtime) {
    return runtime->isA(&ID);
  }

  static SwiftLanguageRuntime *Get(Process *process) {
    return process ? llvm::cast_or_null<SwiftLanguageRuntime>(
                         process->GetLanguageRuntime(lldb::eLanguageTypeSwift))
                   : nullptr;
  }

  static SwiftLanguageRuntime *Get(lldb::ProcessSP process_sp) {
    return SwiftLanguageRuntime::Get(process_sp.get());
  }

  /// Returns the Module containing the Swift Concurrency runtime, if it exists.
  static lldb::ModuleSP FindConcurrencyModule(Process &process);

  /// Returns the version of the swift concurrency runtime debug layout.
  /// If no Concurrency module is found, or if errors occur, nullopt is
  /// returned.
  /// Returns 0 for versions of the module prior to the introduction
  /// of versioning.
  static std::optional<uint32_t> FindConcurrencyDebugVersion(Process &process);
  /// \}

  /// PluginInterface protocol.
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  /// It is impossible to create a SwiftLanguageRuntime without a process.
  Process &GetProcess() const { return *m_process; }

  llvm::Error
  GetObjectDescription(Stream &str, Value &value,
                       ExecutionContextScope *exe_scope) override {
    // This is only interesting to do with a ValueObject for Swift.
    return llvm::createStringError(
        "Swift values do not have an object description");
  }

  lldb::LanguageType GetLanguageType() const override {
    return lldb::eLanguageTypeSwift;
  }

  unsigned GetGeneration() const { return m_generation; }
  void SymbolsDidLoad(const ModuleList &module_list) override {
    ++m_generation;
  }
  void ModulesDidLoad(const ModuleList &module_list) override;

  bool IsSymbolARuntimeThunk(const Symbol &symbol) override;

  /// Mangling support.
  /// \{
  /// Use these passthrough functions rather than calling into Swift directly,
  /// since some day we may want to support more than one swift variant.
  static bool IsSwiftMangledName(llvm::StringRef name);

  static swift::Mangle::ManglingFlavor
  GetManglingFlavor(llvm::StringRef mangledName) {
    if (mangledName.starts_with("$e") || mangledName.starts_with("_$e"))
      return swift::Mangle::ManglingFlavor::Embedded;
    return swift::Mangle::ManglingFlavor::Default;
  }
  enum class FuncletComparisonResult {
    NotBothFunclets,
    DifferentAsyncFunctions,
    SameAsyncFunction
  };

  /// Compares name1 and name2 to decide whether they are both async funclets.
  /// If either is not an async funclet, returns NotBothFunclets.
  /// If they are both funclets but of different async functions, returns
  /// DifferentAsyncFunctions.
  /// Otherwise, returns SameAsyncFunction.
  static FuncletComparisonResult
  AreFuncletsOfSameAsyncFunction(llvm::StringRef name1, llvm::StringRef name2);

  /// Return true if name is a Swift async function symbol.
  static bool IsSwiftAsyncFunctionSymbol(llvm::StringRef name);

  /// Return true if name is a Swift async function, await resume partial
  /// function, or suspend resume partial function symbol.
  static bool IsAnySwiftAsyncFunctionSymbol(llvm::StringRef name);

  /// Return true if node is a Swift async function, await resume partial
  /// function, or suspend resume partial function symbol.
  static bool IsAnySwiftAsyncFunctionSymbol(swift::Demangle::NodePointer node);

  /// Return the async context address using the target's specific register.
  static lldb::addr_t GetAsyncContext(RegisterContext *regctx);

  static bool
  IsSwiftAsyncAwaitResumePartialFunctionSymbol(llvm::StringRef name);

  enum DemangleMode { eSimplified, eTypeName, eDisplayTypeName };
  static std::string
  DemangleSymbolAsString(llvm::StringRef symbol, DemangleMode mode,
                         const SymbolContext *sc = nullptr,
                         const ExecutionContext *exe_ctx = nullptr);

  /// Demangle a symbol to a swift::Demangle node tree.
  ///
  /// This is a central point of access, for purposes such as logging.
  static swift::Demangle::NodePointer
  DemangleSymbolAsNode(llvm::StringRef symbol, swift::Demangle::Context &ctx);

  void DumpTyperef(CompilerType type, TypeSystemSwiftTypeRef *module_holder,
                   Stream *s);
  class MethodName {
  public:
    enum Type {
      eTypeInvalid,
      eTypeUnknownMethod,
      eTypeClassMethod,
      eTypeInstanceMethod,
      eTypeOperator,
      eTypeConstructor,
      eTypeDestructor,
      eTypeAllocator,
      eTypeDeallocator
    };

    MethodName() = default;
    MethodName(ConstString s, bool do_parse = false) : m_full(s) {
      if (do_parse)
        Parse();
    }

    void Clear();

    bool IsValid() const {
      if (m_parse_error)
        return false;
      if (m_type == eTypeInvalid)
        return false;
      return (bool)m_full;
    }

    Type GetType() const { return m_type; }
    ConstString GetFullName() const { return m_full; }
    llvm::StringRef GetBasename();

    static bool ExtractFunctionBasenameFromMangled(ConstString mangled,
                                                   ConstString &basename,
                                                   bool &is_method);

  protected:
    void Parse();

    ConstString m_full; ///< Full name:    "foo.bar.baz : <A : AProtocol>
                        ///< (foo.bar.metatype)(x : Swift.Int64) -> A"
    llvm::StringRef m_basename;      ///< Basename:     "baz"
    llvm::StringRef m_context;       ///< Decl context: "foo.bar"
    llvm::StringRef m_metatype_ref;  ///< Meta type:    "(foo.bar.metatype)"
    llvm::StringRef m_template_args; ///< Generic args: "<A: AProtocol>
    llvm::StringRef m_arguments;     ///< Arguments:    "(x : Swift.Int64)"
    llvm::StringRef m_qualifiers;    ///< Qualifiers:   "const"
    llvm::StringRef m_return_type;   ///< Return type:  "A"
    Type m_type = eTypeInvalid;
    bool m_parsed = false;
    bool m_parse_error = false;
  };
  /// \}

  bool GetDynamicTypeAndAddress(ValueObject &in_value,
                                lldb::DynamicValueType use_dynamic,
                                TypeAndOrName &class_type_or_name,
                                Address &address, Value::ValueType &value_type,
                                llvm::ArrayRef<uint8_t> &local_buffer) override;

  CompilerType BindGenericTypeParameters(
      CompilerType unbound_type,
      std::function<CompilerType(unsigned, unsigned)> finder);

  /// Extract the value object which contains the Swift type's "contents".
  /// Returns None if this is not a C++ wrapping a Swift type, returns
  /// the a pair containing the extracted value object and a boolean indicating
  /// whether the corresponding Swift type should be a pointer (for example, if
  /// the Swift type is a value type but the storage is behind a C pointer.
  static std::optional<std::pair<lldb::ValueObjectSP, bool>>
  ExtractSwiftValueObjectFromCxxWrapper(ValueObject &valobj);

  TypeAndOrName FixUpDynamicType(const TypeAndOrName &type_and_or_name,
                                 ValueObject &static_value) override;
  lldb::BreakpointResolverSP CreateExceptionResolver(const lldb::BreakpointSP &bkpt,
                                                     bool catch_bp,
                                                     bool throw_bp) override;
  bool CouldHaveDynamicValue(ValueObject &in_value) override;
  llvm::Error GetObjectDescription(Stream &str, ValueObject &object) override;
  CompilerType GetConcreteType(ExecutionContextScope *exe_scope,
                               ConstString abstract_type_name) override;

  CompilerType GetTypeFromMetadata(TypeSystemSwift &tss, Address address);
  /// Build the artificial type metadata variable name for \p swift_type.
  static bool GetAbstractTypeName(StreamString &name, swift::Type swift_type);

  /// A pair of depth and index.
  using ArchetypePath = std::pair<uint64_t, uint64_t>;
  /// Populate a map with the names of all archetypes in a function's generic
  /// context.
  static void GetGenericParameterNamesForFunction(
      const SymbolContext &sc, const ExecutionContext *exe_ctx,
      swift::Mangle::ManglingFlavor flavor,
      llvm::DenseMap<ArchetypePath, llvm::StringRef> &dict);

  /// Invoke callback for each DependentGenericParamType.
  static void
  ForEachGenericParameter(swift::Demangle::NodePointer node,
                          std::function<void(unsigned, unsigned)> callback);

  /// One element for each value pack / pack expansion in the signature.
  struct GenericSignature {
    /// Represents a single generic parameter.
    struct GenericParam {
      unsigned depth;
      unsigned index;
      /// A vector of |generic_params| bits, indicating which other
      /// generic_params share the same shape.
      llvm::BitVector same_shape;
      bool is_pack = false;
      GenericParam(unsigned d, unsigned i, unsigned nparams)
          : depth(d), index(i), same_shape(nparams) {}
    };

    struct PackExpansion {
      llvm::BitVector generic_params;
      ConstString mangled_type;
      unsigned shape;
      PackExpansion(unsigned nparams, unsigned shape)
          : generic_params(nparams), shape(shape) {}
    };

    llvm::SmallVector<GenericParam, 4> generic_params;
    llvm::SmallVector<PackExpansion> pack_expansions;

    llvm::SmallVector<unsigned, 4> count_for_value_pack;
    llvm::SmallVector<unsigned, 4> count_for_type_pack;
    unsigned dependent_generic_param_count = 0;
    unsigned num_counts = 0;

    unsigned GetNumValuePacks() { return count_for_value_pack.size(); }
    unsigned GetNumTypePacks() { return count_for_type_pack.size(); }
    unsigned GetCountForValuePack(unsigned i) {
      return count_for_value_pack[i];
    }
    unsigned GetCountForTypePack(unsigned i) { return count_for_type_pack[i]; }
  };
  /// Extract the generic signature out of a mangled Swift function name.
  static std::optional<GenericSignature>
  GetGenericSignature(llvm::StringRef function_name,
                      TypeSystemSwiftTypeRef &ts);

  /// Using the generic type parameters of \p stack_frame return a
  /// version of \p base_type that replaces all generic type
  /// parameters with bound generic types. If a generic type parameter
  /// cannot be resolved, the input type is returned.
  CompilerType BindGenericTypeParameters(StackFrame &stack_frame,
                                         CompilerType base_type);

  bool IsStoredInlineInBuffer(CompilerType type) override;

  /// Check if this type alias is listed in any witness tables and resolve it.
  llvm::Expected<CompilerType> ResolveTypeAlias(CompilerType alias);

  /// Retrieve the offset of the named member variable within an instance
  /// of the given type.
  ///
  /// \param instance_type
  std::optional<uint64_t> GetMemberVariableOffset(CompilerType instance_type,
                                                   ValueObject *instance,
                                                   llvm::StringRef member_name,
                                                   Status *error = nullptr);

  /// Ask Remote Mirrors about the children of a composite type.
  llvm::Expected<uint32_t> GetNumChildren(CompilerType type,
                                          ExecutionContextScope *exe_scope);

  /// Determine the enum case name for the \p data value of the enum \p type.
  /// This is performed using Swift reflection.
  llvm::Expected<std::string> GetEnumCaseName(CompilerType type,
                                              const DataExtractor &data,
                                              ExecutionContext *exe_ctx);

  enum LookupResult {
    /// Failed due to missing reflection meatadata or unimplemented
    /// functionality. Should retry with SwiftASTContext.
    eError = 0,
    /// Success.
    eFound,
    /// Found complete type info, lookup unsuccessful.
    /// Do not waste time retrying.
    eNotFound
  };

  /// Behaves like the CompilerType::GetIndexOfChildMemberWithName()
  /// except for the more nuanced return value.
  ///
  /// \returns {false, {}} on error.
  //
  /// \returns {true, {}} if the member exists, but it is an enum case
  ///                     without payload. Enum cases without payload
  ///                     don't have an index.
  ///
  /// \returns {true, {num_idexes}} on success.
  std::pair<LookupResult, std::optional<size_t>>
  GetIndexOfChildMemberWithName(CompilerType type, llvm::StringRef name,
                                ExecutionContext *exe_ctx,
                                bool omit_empty_base_classes,
                                std::vector<uint32_t> &child_indexes);

  /// Ask Remote Mirrors about a child of a composite type.
  llvm::Expected<CompilerType> GetChildCompilerTypeAtIndex(
      CompilerType type, size_t idx, bool transparent_pointers,
      bool omit_empty_base_classes, bool ignore_array_bounds,
      std::string &child_name, uint32_t &child_byte_size,
      int32_t &child_byte_offset, uint32_t &child_bitfield_bit_size,
      uint32_t &child_bitfield_bit_offset, bool &child_is_base_class,
      bool &child_is_deref_of_parent, ValueObject *valobj,
      uint64_t &language_flags);

  /// Ask Remote Mirrors about the fields of a composite type.
  std::optional<unsigned> GetNumFields(CompilerType type,
                                        ExecutionContext *exe_ctx);

  /// Ask Remote Mirrors for the size of a Swift type.
  std::optional<uint64_t> GetBitSize(CompilerType type,
                                      ExecutionContextScope *exe_scope);

  /// Ask Remote mirrors for the stride of a Swift type.
  std::optional<uint64_t> GetByteStride(CompilerType type);

  /// Ask Remote mirrors for the alignment of a Swift type.
  std::optional<size_t> GetBitAlignment(CompilerType type,
                                         ExecutionContextScope *exe_scope);

  /// Release the RemoteASTContext associated with the given swift::ASTContext.
  /// Note that a RemoteASTContext must be destroyed before its associated
  /// swift::ASTContext is destroyed.
  void ReleaseAssociatedRemoteASTContext(swift::ASTContext *ctx);

  void AddToLibraryNegativeCache(llvm::StringRef library_name);
  bool IsInLibraryNegativeCache(llvm::StringRef library_name);

  // Swift uses a few known-unused bits in ObjC pointers
  // to record useful-for-bridging information
  // This API's task is to return such pointer+info aggregates
  // back to a pure pointer
  lldb::addr_t MaskMaybeBridgedPointer(lldb::addr_t, lldb::addr_t * = nullptr);

  /// Swift uses a few known-unused bits in weak,unowned,unmanaged
  /// references to record useful runtime information.  This API's
  /// task is to strip those bits if necessary and return a pure
  /// pointer (or a tagged pointer).
  lldb::addr_t MaybeMaskNonTrivialReferencePointer(
      lldb::addr_t, TypeSystemSwift::NonTriviallyManagedReferenceKind kind);
  /// \return true if this is a Swift tagged pointer (as opposed to an
  /// Objective-C tagged pointer).
  bool IsTaggedPointer(lldb::addr_t addr, CompilerType type);
  std::pair<lldb::addr_t, bool> FixupPointerValue(lldb::addr_t addr,
                                                  CompilerType type) override;
  lldb::addr_t FixupAddress(lldb::addr_t addr, CompilerType type,
                            Status &error) override;

  lldb::ThreadPlanSP GetStepThroughTrampolinePlan(Thread &thread,
                                                  bool stop_others) override;

  StructuredData::ObjectSP GetLanguageSpecificData(SymbolContext sc) override;

  /// If you are at the initial instruction of the frame passed in,
  /// then this will examine the call arguments, and if any of them is
  /// a function pointer, this will push the address of the function
  /// into addresses.  If debug_only is true, then it will only push
  /// function pointers that are in user code.
  void FindFunctionPointersInCall(StackFrame &frame,
                                  std::vector<Address> &addresses,
                                  bool debug_only = true,
                                  bool resolve_thunks = true) override;

  /// Error value handling.
  /// \{
  static lldb::ValueObjectSP CalculateErrorValue(lldb::StackFrameSP frame_sp,
                                                 ConstString name);

  lldb::ValueObjectSP CalculateErrorValueObjectFromValue(Value &value,
                                                         ConstString name,
                                                         bool persistent);

  std::optional<Value>
  GetErrorReturnLocationAfterReturn(lldb::StackFrameSP frame_sp);

  std::optional<Value>
  GetErrorReturnLocationBeforeReturn(lldb::StackFrameSP frame_sp,
                                     bool &need_to_check_after_return);

  static void RegisterGlobalError(Target &target, ConstString name,
                                  lldb::addr_t addr);

  // Provide a quick and yet somewhat reasonable guess as to whether
  // this ValueObject represents something that validly conforms
  // to the magic ErrorType protocol.
  bool IsValidErrorValue(ValueObject &in_value);
  /// \}

  static const char *GetErrorBackstopName();
  ConstString GetStandardLibraryName();
  static const char *GetStandardLibraryBaseName();
  static const char *GetConcurrencyLibraryBaseName();

  static bool IsSwiftClassName(const char *name);
  /// Determines wether \c variable is the "self" object.
  static bool IsSelf(Variable &variable);
  bool IsAllowedRuntimeValue(ConstString name) override;

  lldb::SyntheticChildrenSP
  GetBridgedSyntheticChildProvider(ValueObject &valobj);

  /// Expression Callbacks.
  /// \{
  void WillStartExecutingUserExpression(bool);
  void DidFinishExecutingUserExpression(bool);
  /// \}

  bool IsABIStable();

  SwiftLanguageRuntime(const SwiftLanguageRuntime &) = delete;
  const SwiftLanguageRuntime &operator=(const SwiftLanguageRuntime &) = delete;

  static AppleObjCRuntimeV2 *GetObjCRuntime(lldb_private::Process &process);

protected:
  friend class LLDBTypeInfoProvider;
  /// Enter an anonymous Clang type with a name key into a side table.
  void RegisterAnonymousClangType(const char *key, CompilerType clang_type);
  /// Look up an anonymous Clang type with a name key into a side table.
  CompilerType LookupAnonymousClangType(const char *key);

  std::optional<const swift::reflection::TypeInfo *>
  lookupClangTypeInfo(CompilerType clang_type);

  const swift::reflection::TypeInfo *
  emplaceClangTypeInfo(CompilerType clang_type,
                       std::optional<uint64_t> byte_size,
                       std::optional<size_t> bit_align,
                       llvm::ArrayRef<swift::reflection::FieldInfo> fields);

  /// Use the reflection context to build a TypeRef object.
  llvm::Expected<const swift::reflection::TypeRef &>
  GetTypeRef(CompilerType type, TypeSystemSwiftTypeRef *module_holder);

  /// Ask Remote Mirrors for the type info about a Swift type.
  /// This will return a nullptr if the lookup fails.
  llvm::Expected<const swift::reflection::TypeInfo &>
  GetSwiftRuntimeTypeInfo(CompilerType type, ExecutionContextScope *exe_scope,
                          swift::reflection::TypeRef const **out_tr = nullptr);

  std::optional<uint64_t>
  GetMemberVariableOffsetRemoteAST(CompilerType instance_type,
                                   ValueObject *instance,
                                   llvm::StringRef member_name);
  std::optional<uint64_t> GetMemberVariableOffsetRemoteMirrors(
      CompilerType instance_type, ValueObject *instance,
      llvm::StringRef member_name, Status *error);

  /// If \p instance points to a Swift object, retrieve its
  /// RecordTypeInfo and pass it to the callback \p fn. Repeat the
  /// process with all superclasses. If \p fn returns \p true, early
  /// exit and return \p true. Otherwise return \p false.
  bool ForEachSuperClassType(ValueObject &instance,
                             std::function<bool(SuperClassType)> fn);

  /// Retrieve the remote AST context for the given Swift AST context.
  swift::remoteAST::RemoteASTContext &
  GetRemoteASTContext(SwiftASTContext &swift_ast_ctx);

  /// Like \p BindGenericTypeParameters but for TypeSystemSwiftTypeRef.
  CompilerType BindGenericTypeParameters(StackFrame &stack_frame,
                                         TypeSystemSwiftTypeRef &ts,
                                         ConstString mangled_name);

  /// Like \p BindGenericTypeParameters but for RemoteAST.
  CompilerType BindGenericTypeParametersRemoteAST(StackFrame &stack_frame,
                                                  CompilerType base_type);

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
                                      Value::ValueType &value_type,
                                      llvm::ArrayRef<uint8_t> &local_buffer);
#ifndef NDEBUG
  ConstString GetDynamicTypeName_ClassRemoteAST(ValueObject &in_value,
                                                lldb::addr_t instance_ptr);
#endif
  bool GetDynamicTypeAndAddress_Existential(ValueObject &in_value,
                                            CompilerType protocol_type,
                                            lldb::DynamicValueType use_dynamic,
                                            TypeAndOrName &class_type_or_name,
                                            Address &address);
#ifndef NDEBUG
  std::optional<std::pair<CompilerType, Address>>
  GetDynamicTypeAndAddress_ExistentialRemoteAST(
      ValueObject &in_value, CompilerType protocol_type, bool use_local_buffer,
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
                                      Value::ValueType &value_type,
                                      llvm::ArrayRef<uint8_t> &local_buffer);

  bool GetDynamicTypeAndAddress_IndirectEnumCase(
      ValueObject &in_value, lldb::DynamicValueType use_dynamic,
      TypeAndOrName &class_type_or_name, Address &address,
      Value::ValueType &value_type, llvm::ArrayRef<uint8_t> &local_buffer);

  bool GetDynamicTypeAndAddress_ClangType(
      ValueObject &in_value, lldb::DynamicValueType use_dynamic,
      TypeAndOrName &class_type_or_name, Address &address,
      Value::ValueType &value_type, llvm::ArrayRef<uint8_t> &local_buffer);

  /// Dynamic type resolution tends to want to generate scalar data -
  /// but there are caveats Per original comment here "Our address is
  /// the location of the dynamic type stored in memory.  It isn't a
  /// load address, because we aren't pointing to the LOCATION that
  /// stores the pointer to us, we're pointing to us..."  See inlined
  /// comments for exceptions to this general rule.
  Value::ValueType GetValueType(ValueObject &in_value,
                                CompilerType dynamic_type,
                                Value::ValueType static_value_type,
                                bool is_indirect_enum_case,
                                llvm::ArrayRef<uint8_t> &local_buffer);

  lldb::UnwindPlanSP
  GetRuntimeUnwindPlan(lldb::ProcessSP process_sp,
                       lldb_private::RegisterContext *regctx,
                       bool &behaves_like_zeroth_frame) override;

  bool GetTargetOfPartialApply(SymbolContext &curr_sc, ConstString &apply_name,
                               SymbolContext &sc);
  AppleObjCRuntimeV2 *GetObjCRuntime();

  /// Creates an UnwindPlan for following the AsyncContext chain up the stack,
  /// from a current AsyncContext frame.
  lldb::UnwindPlanSP
  GetFollowAsyncContextUnwindPlan(lldb::ProcessSP process_sp,
                                  RegisterContext *regctx, ArchSpec &arch,
                                  bool &behaves_like_zeroth_frame);

  /// Given the async register of a funclet, extract its continuation pointer,
  /// compute the prologue size of the continuation function, and return the
  /// address of the first non-prologue instruction.
  std::optional<lldb::addr_t>
  TrySkipVirtualParentProlog(lldb::addr_t async_reg_val, Process &process,
                             unsigned num_indirections = 0);

  const CompilerType &GetBoxMetadataType();

  /// A proxy object to support lazy binding of Archetypes.
  class MetadataPromise {
    friend class SwiftLanguageRuntime;

    MetadataPromise(ValueObject &, SwiftLanguageRuntime &, lldb::addr_t);

    lldb::ValueObjectSP m_for_object_sp;
    SwiftLanguageRuntime &m_swift_runtime;
    lldb::addr_t m_metadata_location;
    std::optional<swift::MetadataKind> m_metadata_kind;
    std::optional<CompilerType> m_compiler_type;

  public:
    CompilerType FulfillTypePromise(const SymbolContext &sc,
                                    Status *error = nullptr);
  };
  typedef std::shared_ptr<MetadataPromise> MetadataPromiseSP;

  MetadataPromiseSP GetMetadataPromise(const SymbolContext &sc,
                                       lldb::addr_t addr,
                                       ValueObject &for_object);
  MetadataPromiseSP GetPromiseForTypeNameAndFrame(const char *type_name,
                                                  StackFrame *frame);

  std::optional<lldb::addr_t>
  GetTypeMetadataForTypeNameAndFrame(llvm::StringRef mdvar_name,
                                     StackFrame &frame);

  std::shared_ptr<LLDBMemoryReader> GetMemoryReader();

  void PushLocalBuffer(uint64_t local_buffer, uint64_t local_buffer_size);

  void PopLocalBuffer();

  // These are the helper functions for GetObjectDescription for various
  // types of swift objects.
  std::string GetObjectDescriptionExpr_Result(ValueObject &object);
  std::string GetObjectDescriptionExpr_Ref(ValueObject &object);
  std::string GetObjectDescriptionExpr_Copy(ValueObject &object,
                                            lldb::addr_t &copy_location);
  llvm::Error RunObjectDescriptionExpr(ValueObject &object,
                                       std::string &expr_string,
                                       Stream &result);

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

  llvm::StringMap<std::vector<std::string>> m_conformances;

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
  std::optional<lldb::addr_t> GetDynamicExclusivityFlagAddr();

  // Add the modules in m_modules_to_add to the Reflection Context. The
  // ModulesDidLoad() callback appends to m_modules_to_add.
  void ProcessModulesToAdd();

  /// Lazily initialize and return \p m_SwiftNativeNSErrorISA.
  std::optional<lldb::addr_t> GetSwiftNativeNSErrorISA();

  SwiftMetadataCache *GetSwiftMetadataCache();

  /// Find all conformances for a nominal type in the reflection metadata.
  std::vector<std::string> GetConformances(llvm::StringRef mangled_name);

  /// These members are used to track and toggle the state of the "dynamic
  /// exclusivity enforcement flag" in the swift runtime. This flag is set to
  /// true when an LLDB expression starts running, and reset to its original
  /// state after that expression (and any other concurrently running
  /// expressions) terminates.
  /// \{
  std::mutex m_active_user_expr_mutex;
  uint32_t m_active_user_expr_count = 0;

  bool m_original_dynamic_exclusivity_flag_state = false;
  std::optional<lldb::addr_t> m_dynamic_exclusivity_flag_addr;
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

  /// Increased every time SymbolsDidLoad is called.
  unsigned m_generation = 0;
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
  /// std::nullopt otherwise.
  std::optional<uint32_t> AddObjectFileToReflectionContext(
      lldb::ModuleSP module,
      llvm::SmallVector<llvm::StringRef, 1> likely_module_names);

  /// Cache for the debug-info-originating type infos.
  /// \{
  llvm::DenseMap<lldb::opaque_compiler_type_t,
                 std::optional<swift::reflection::TypeInfo>>
      m_clang_type_info;
  llvm::DenseMap<lldb::opaque_compiler_type_t,
                 std::optional<swift::reflection::RecordTypeInfo>>
      m_clang_record_type_info;
  llvm::DenseMap<const char *, CompilerType> m_anonymous_clang_types;
  unsigned m_num_anonymous_clang_types = 0;
  std::recursive_mutex m_clang_type_info_mutex;
  /// \}

  /// Swift native NSError isa.
  std::optional<lldb::addr_t> m_SwiftNativeNSErrorISA;
};

/// The target specific register numbers used for async unwinding.
///
/// For UnwindPlans, these use eh_frame / dwarf register numbering.
struct AsyncUnwindRegisterNumbers {
  uint32_t async_ctx_regnum;
  uint32_t pc_regnum;

  /// All register numbers in this struct are given in the eRegisterKindDWARF
  /// domain.
  lldb::RegisterKind GetRegisterKind() const { return lldb::eRegisterKindDWARF; }
};

std::optional<AsyncUnwindRegisterNumbers>
GetAsyncUnwindRegisterNumbers(llvm::Triple::ArchType triple);

/// Inspects thread local storage to find the address of the currently executing
/// task.
llvm::Expected<lldb::addr_t> GetTaskAddrFromThreadLocalStorage(Thread &thread);
} // namespace lldb_private

#endif // liblldb_SwiftLanguageRuntime_h_
