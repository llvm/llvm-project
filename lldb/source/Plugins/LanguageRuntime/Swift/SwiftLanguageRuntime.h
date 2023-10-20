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
#include "Plugins/TypeSystem/Swift/SwiftASTContext.h"
#include "lldb/Breakpoint/BreakpointPrecondition.h"
#include "lldb/Core/PluginInterface.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/lldb-private.h"

#include "llvm/ADT/Optional.h"
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
} // namespace reflection

namespace remoteAST {
class RemoteASTContext;
}
enum class MetadataKind : uint32_t;
class TypeBase;
} // namespace swift

namespace lldb_private {

class SwiftLanguageRuntimeStub;
class SwiftLanguageRuntimeImpl;

class SwiftLanguageRuntime : public LanguageRuntime {
protected:
  SwiftLanguageRuntime(Process *process);
  /// The private implementation object, either a stub or a full
  /// runtime.
  ///
  /// TODO: Instead of using these pImpl objects, it would be more
  ///   elegant to have CreateInstance return the right object,
  ///   unfortunately Process wants to cache the returned language
  ///   runtimes and doesn't call CreateInstance() ever again.
  std::unique_ptr<SwiftLanguageRuntimeStub> m_stub;
  std::unique_ptr<SwiftLanguageRuntimeImpl> m_impl;

public:
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
  /// \}

  /// PluginInterface protocol.
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  bool GetObjectDescription(Stream &str, Value &value,
                            ExecutionContextScope *exe_scope) override {
    // This is only interesting to do with a ValueObject for Swift.
    return false;
   }

  lldb::LanguageType GetLanguageType() const override {
    return lldb::eLanguageTypeSwift;
  }

  void ModulesDidLoad(const ModuleList &module_list) override;

  bool IsSymbolARuntimeThunk(const Symbol &symbol) override;

  /// Mangling support.
  /// \{
  /// Use these passthrough functions rather than calling into Swift directly,
  /// since some day we may want to support more than one swift variant.
  static bool IsSwiftMangledName(llvm::StringRef name);

  /// Return true if name is a Swift async function symbol.
  static bool IsSwiftAsyncFunctionSymbol(llvm::StringRef name);

  /// Return true if name is a Swift async function, await resume partial
  /// function, or suspend resume partial function symbol.
  static bool IsAnySwiftAsyncFunctionSymbol(llvm::StringRef name);
  
  /// Return the async context address using the target's specific register.
  static lldb::addr_t GetAsyncContext(RegisterContext *regctx);

  static bool
  IsSwiftAsyncAwaitResumePartialFunctionSymbol(llvm::StringRef name);

  enum DemangleMode { eSimplified, eTypeName, eDisplayTypeName };
  static std::string DemangleSymbolAsString(llvm::StringRef symbol,
                                            DemangleMode mode,
                                            const SymbolContext *sc = nullptr);

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
                                Address &address,
                                Value::ValueType &value_type) override;

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
  bool GetObjectDescription(Stream &str, ValueObject &object) override;
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
      const SymbolContext &sc,
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
  static llvm::Optional<GenericSignature>
  GetGenericSignature(llvm::StringRef function_name,
                      TypeSystemSwiftTypeRef &ts);

  /// Using the generic type parameters of \p stack_frame return a
  /// version of \p base_type that replaces all generic type
  /// parameters with bound generic types. If a generic type parameter
  /// cannot be resolved, the input type is returned.
  CompilerType BindGenericTypeParameters(StackFrame &stack_frame,
                                         CompilerType base_type);

  bool IsStoredInlineInBuffer(CompilerType type) override;

  /// Retrieve the offset of the named member variable within an instance
  /// of the given type.
  ///
  /// \param instance_type
  llvm::Optional<uint64_t> GetMemberVariableOffset(CompilerType instance_type,
                                                   ValueObject *instance,
                                                   llvm::StringRef member_name,
                                                   Status *error = nullptr);

  /// Ask Remote Mirrors about the children of a composite type.
  llvm::Optional<unsigned> GetNumChildren(CompilerType type,
                                          ExecutionContextScope *exe_scope);

  /// Determine the enum case name for the \p data value of the enum \p type.
  /// This is performed using Swift reflection.
  llvm::Optional<std::string> GetEnumCaseName(CompilerType type,
                                              const DataExtractor &data,
                                              ExecutionContext *exe_ctx);

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
  std::pair<bool, llvm::Optional<size_t>> GetIndexOfChildMemberWithName(
      CompilerType type, llvm::StringRef name, ExecutionContext *exe_ctx,
      bool omit_empty_base_classes, std::vector<uint32_t> &child_indexes);

  /// Ask Remote Mirrors about a child of a composite type.
  CompilerType GetChildCompilerTypeAtIndex(
      CompilerType type, size_t idx, bool transparent_pointers,
      bool omit_empty_base_classes, bool ignore_array_bounds,
      std::string &child_name, uint32_t &child_byte_size,
      int32_t &child_byte_offset, uint32_t &child_bitfield_bit_size,
      uint32_t &child_bitfield_bit_offset, bool &child_is_base_class,
      bool &child_is_deref_of_parent, ValueObject *valobj,
      uint64_t &language_flags);

  /// Ask Remote Mirrors about the fields of a composite type.
  llvm::Optional<unsigned> GetNumFields(CompilerType type,
                                        ExecutionContext *exe_ctx);

  /// Ask Remote Mirrors for the size of a Swift type.
  llvm::Optional<uint64_t> GetBitSize(CompilerType type,
                                      ExecutionContextScope *exe_scope);

  /// Ask Remote mirrors for the stride of a Swift type.
  llvm::Optional<uint64_t> GetByteStride(CompilerType type);

  /// Ask Remote mirrors for the alignment of a Swift type.
  llvm::Optional<size_t> GetBitAlignment(CompilerType type,
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

  StructuredDataImpl *GetLanguageSpecificData(StackFrame &frame) override;

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

  llvm::Optional<Value>
  GetErrorReturnLocationAfterReturn(lldb::StackFrameSP frame_sp);

  llvm::Optional<Value>
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
  lldb::UnwindPlanSP
  GetRuntimeUnwindPlan(lldb::ProcessSP process_sp,
                       lldb_private::RegisterContext *regctx,
                       bool &behaves_like_zeroth_frame) override;

  bool GetTargetOfPartialApply(SymbolContext &curr_sc, ConstString &apply_name,
                               SymbolContext &sc);
  AppleObjCRuntimeV2 *GetObjCRuntime();
};

} // namespace lldb_private

#endif // liblldb_SwiftLanguageRuntime_h_
