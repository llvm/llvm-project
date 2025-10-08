//===-- TypeSystemSwiftTypeRef.h --------------------------------*- C++ -*-===//
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

#ifndef liblldb_TypeSystemSwiftTypeRef_h_
#define liblldb_TypeSystemSwiftTypeRef_h_

#include "Plugins/TypeSystem/Swift/TypeSystemSwift.h"
#include "lldb/Core/SwiftForward.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Utility/ThreadSafeDenseMap.h"
#include "lldb/Utility/ThreadSafeStringMap.h"
#include "lldb/lldb-types.h"
#include "swift/Demangling/ManglingFlavor.h"

// FIXME: needed only for the DenseMap.
#include "clang/APINotes/APINotesManager.h"
#include "clang/Basic/Module.h"

#include "llvm/ADT/StringRef.h"

namespace swift {
class DWARFImporterDelegate;
namespace Demangle {
class Node;
using NodePointer = Node *;
class Demangler;
template <typename T>
class ManglingErrorOr;
} // namespace Demangle
namespace reflection {
struct DescriptorFinder;
class TypeInfo;
} // namespace reflection
namespace remote {
struct TypeInfoProvider;
} // namespace remote
} // namespace swift

namespace lldb_private {
class ClangExternalASTSourceCallbacks;
class ClangNameImporter;
class SwiftASTContext;
class SwiftASTContextForExpressions;
class SwiftDWARFImporterForClangTypes;
class SwiftPersistentExpressionState;

/// A Swift TypeSystem that does not own a swift::ASTContext.
class TypeSystemSwiftTypeRef : public TypeSystemSwift {
  /// LLVM RTTI support.
  static char ID;

public:
  /// LLVM RTTI support.
  /// \{
  bool isA(const void *ClassID) const override {
    return ClassID == &ID || TypeSystemSwift::isA(ClassID);
  }
  static bool classof(const TypeSystem *ts) { return ts->isA(&ID); }
  /// \}

  /// Provided only for unit tests.
  /// \{
  friend struct TestTypeSystemSwiftTypeRef;
  TypeSystemSwiftTypeRef();
  /// \}
  ~TypeSystemSwiftTypeRef();
  TypeSystemSwiftTypeRef(Module &module);
  /// Get the corresponding SwiftASTContext, and create one if necessary.
  SwiftASTContextSP GetSwiftASTContext(const SymbolContext &sc) const override;
  /// Convenience helpers.
  SymbolContext GetSymbolContext(ExecutionContextScope *exe_scope) const;
  SymbolContext GetSymbolContext(const ExecutionContext *exe_ctx) const;
  /// Return SwiftASTContext, iff one has already been created.
  virtual SwiftASTContextSP
  GetSwiftASTContextOrNull(const SymbolContext &sc) const;
  TypeSystemSwiftTypeRefSP GetTypeSystemSwiftTypeRef() override {
    return std::static_pointer_cast<TypeSystemSwiftTypeRef>(shared_from_this());
  }
  std::shared_ptr<const TypeSystemSwiftTypeRef>
  GetTypeSystemSwiftTypeRef() const override {
    return std::static_pointer_cast<const TypeSystemSwiftTypeRef>(
        shared_from_this());
  }
  SwiftDWARFImporterForClangTypes &GetSwiftDWARFImporterForClangTypes();
  ClangNameImporter *GetNameImporter() const;
  llvm::Triple GetTriple() const;
  void SetTriple(const SymbolContext &sc, const llvm::Triple triple) override;
  void ClearModuleDependentCaches() override;
  lldb::TargetWP GetTargetWP() const override { return {}; }

  /// Return a SwiftASTContext type for type.
  CompilerType ReconstructType(CompilerType type,
                               const ExecutionContext *exe_ctx);
  CompilerType
  GetTypeFromMangledTypename(ConstString mangled_typename) override;

  CompilerType GetGenericArgumentType(lldb::opaque_compiler_type_t type,
                                      size_t idx) override;

  /// Returns the list of DependentGenericParamTypes (depth, index pairs) that a
  /// type has, if any.
  static llvm::SmallVector<std::pair<int, int>, 1>
  GetDependentGenericParamListForType(llvm::StringRef type);

  // PluginInterface functions
  llvm::StringRef GetPluginName() override { return "TypeSystemSwiftTypeRef"; }

  bool SupportsLanguage(lldb::LanguageType language) override;
  Status IsCompatible() override;

  plugin::dwarf::DWARFASTParser *GetDWARFParser() override;
  // CompilerDecl functions
  ConstString DeclGetName(void *opaque_decl) override {
    return ConstString("");
  }

  // CompilerDeclContext functions
  std::vector<CompilerDecl>
  DeclContextFindDeclByName(void *opaque_decl_ctx, ConstString name,
                            const bool ignore_imported_decls) override {
    return {};
  }

  bool DeclContextIsContainedInLookup(void *opaque_decl_ctx,
                                      void *other_opaque_decl_ctx) override {
    if (opaque_decl_ctx == other_opaque_decl_ctx)
      return true;
    return false;
  }

  CompilerType GetParentType(lldb::opaque_compiler_type_t type);
  std::vector<std::vector<CompilerType>>
  /// Extract the substitutions from a bound generic type.
  GetSubstitutions(lldb::opaque_compiler_type_t type);
  /// Apply substitutions to a bound generic type that is mapped out of context.
  CompilerType ApplySubstitutions(lldb::opaque_compiler_type_t type,
                                  std::vector<std::vector<CompilerType>> subs);
  /// Apply substitutions to a bound generic type that is mapped out of context.
  CompilerType MapOutOfContext(lldb::opaque_compiler_type_t type);

  Module *GetModule() const { return m_module; }

  /// Return a key for the SwiftASTContext map. If there is debug info it's the
  /// name of the owning Swift module for a function.
  static const char *DeriveKeyFor(const SymbolContext &sc);
  /// Return the name of the owning Swift module for a function.
  static ConstString GetSwiftModuleFor(const SymbolContext &sc);

  // Tests
#ifndef NDEBUG
  bool Verify(lldb::opaque_compiler_type_t type) override;
#endif
  bool IsArrayType(lldb::opaque_compiler_type_t type,
                   CompilerType *element_type, uint64_t *size,
                   bool *is_incomplete) override;
  bool IsAggregateType(lldb::opaque_compiler_type_t type) override;
  bool IsDefined(lldb::opaque_compiler_type_t type) override;
  bool IsFunctionType(lldb::opaque_compiler_type_t type) override;
  size_t
  GetNumberOfFunctionArguments(lldb::opaque_compiler_type_t type) override;
  CompilerType GetFunctionArgumentAtIndex(lldb::opaque_compiler_type_t type,
                                          const size_t index) override;
  bool IsFunctionPointerType(lldb::opaque_compiler_type_t type) override;
  bool IsPossibleDynamicType(lldb::opaque_compiler_type_t type,
                             CompilerType *target_type, // Can pass NULL
                             bool check_cplusplus, bool check_objc) override;
  bool IsPointerType(lldb::opaque_compiler_type_t type,
                     CompilerType *pointee_type) override;
  bool IsVoidType(lldb::opaque_compiler_type_t type) override;
  // AST related queries
  uint32_t GetPointerByteSize() override;
  // Accessors
  ConstString GetTypeName(lldb::opaque_compiler_type_t type,
                          bool BaseOnly) override;
  ConstString GetDisplayTypeName(lldb::opaque_compiler_type_t type,
                                 const SymbolContext *sc) override;
  ConstString GetMangledTypeName(lldb::opaque_compiler_type_t type) override;
  uint32_t GetTypeInfo(lldb::opaque_compiler_type_t type,
                       CompilerType *pointee_or_element_clang_type) override;
  lldb::TypeClass GetTypeClass(lldb::opaque_compiler_type_t type) override;

  // Creating related types
  CompilerType GetArrayElementType(lldb::opaque_compiler_type_t type,
                                   ExecutionContextScope *exe_scope) override;
  CompilerType GetCanonicalType(lldb::opaque_compiler_type_t type) override;
  int GetFunctionArgumentCount(lldb::opaque_compiler_type_t type) override;
  CompilerType GetFunctionArgumentTypeAtIndex(lldb::opaque_compiler_type_t type,
                                              size_t idx) override;
  CompilerType
  GetFunctionReturnType(lldb::opaque_compiler_type_t type) override;
  size_t GetNumMemberFunctions(lldb::opaque_compiler_type_t type) override;
  TypeMemberFunctionImpl
  GetMemberFunctionAtIndex(lldb::opaque_compiler_type_t type,
                           size_t idx) override;
  CompilerType GetPointeeType(lldb::opaque_compiler_type_t type) override;
  CompilerType GetPointerType(lldb::opaque_compiler_type_t type) override;

  /// Get a function type that returns nothing and take no parameters.
  CompilerType GetVoidFunctionType();

  // Exploring the type
  llvm::Expected<uint64_t>
  GetBitSize(lldb::opaque_compiler_type_t type,
             ExecutionContextScope *exe_scope) override;
  std::optional<uint64_t>
  GetByteStride(lldb::opaque_compiler_type_t type,
                ExecutionContextScope *exe_scope) override;
  lldb::Encoding GetEncoding(lldb::opaque_compiler_type_t type,
                             uint64_t &count) override;
  llvm::Expected<uint32_t>
  GetNumChildren(lldb::opaque_compiler_type_t type,
                 bool omit_empty_base_classes,
                 const ExecutionContext *exe_ctx) override;
  uint32_t GetNumFields(lldb::opaque_compiler_type_t type,
                        ExecutionContext *exe_ctx = nullptr) override;
  CompilerType GetFieldAtIndex(lldb::opaque_compiler_type_t type, size_t idx,
                               std::string &name, uint64_t *bit_offset_ptr,
                               uint32_t *bitfield_bit_size_ptr,
                               bool *is_bitfield_ptr) override;
  llvm::Expected<CompilerType> GetChildCompilerTypeAtIndex(
      lldb::opaque_compiler_type_t type, ExecutionContext *exe_ctx, size_t idx,
      bool transparent_pointers, bool omit_empty_base_classes,
      bool ignore_array_bounds, std::string &child_name,
      uint32_t &child_byte_size, int32_t &child_byte_offset,
      uint32_t &child_bitfield_bit_size, uint32_t &child_bitfield_bit_offset,
      bool &child_is_base_class, bool &child_is_deref_of_parent,
      ValueObject *valobj, uint64_t &language_flags) override;
  size_t
  GetIndexOfChildMemberWithName(lldb::opaque_compiler_type_t type,
                                llvm::StringRef name, ExecutionContext *exe_ctx,
                                bool omit_empty_base_classes,
                                std::vector<uint32_t> &child_indexes) override;
  size_t GetNumTemplateArguments(lldb::opaque_compiler_type_t type,
                                 bool expand_pack) override;
  CompilerType GetTypeForFormatters(lldb::opaque_compiler_type_t type) override;
  LazyBool ShouldPrintAsOneLiner(lldb::opaque_compiler_type_t type,
                                 ValueObject *valobj) override;
  bool IsMeaninglessWithoutDynamicResolution(
      lldb::opaque_compiler_type_t type) override;

  // Dumping types
#ifndef NDEBUG
  /// Convenience LLVM-style dump method for use in the debugger only.
  LLVM_DUMP_METHOD virtual void
  dump(lldb::opaque_compiler_type_t type) const override;
#endif

  bool DumpTypeValue(lldb::opaque_compiler_type_t type, Stream &s,
                     lldb::Format format, const DataExtractor &data,
                     lldb::offset_t data_offset, size_t data_byte_size,
                     uint32_t bitfield_bit_size, uint32_t bitfield_bit_offset,
                     ExecutionContextScope *exe_scope,
                     bool is_base_class) override;

  void DumpTypeDescription(
      lldb::opaque_compiler_type_t type,
      lldb::DescriptionLevel level = lldb::eDescriptionLevelFull,
      ExecutionContextScope *exe_scope = nullptr) override;
  void DumpTypeDescription(
      lldb::opaque_compiler_type_t type, Stream &s,
      lldb::DescriptionLevel level = lldb::eDescriptionLevelFull,
      ExecutionContextScope *exe_scope = nullptr) override;
  void DumpTypeDescription(
      lldb::opaque_compiler_type_t type, bool print_help_if_available,
      bool print_extensions_if_available,
      lldb::DescriptionLevel level = lldb::eDescriptionLevelFull,
      ExecutionContextScope *exe_scope = nullptr) override;
  void DumpTypeDescription(
      lldb::opaque_compiler_type_t type, Stream *s,
      bool print_help_if_available, bool print_extensions_if_available,
      lldb::DescriptionLevel level = lldb::eDescriptionLevelFull,
      ExecutionContextScope *exe_scope = nullptr) override;

  bool IsPointerOrReferenceType(lldb::opaque_compiler_type_t type,
                                CompilerType *pointee_type) override;
  std::optional<size_t>
  GetTypeBitAlign(lldb::opaque_compiler_type_t type,
                  ExecutionContextScope *exe_scope) override;
  CompilerType GetBuiltinTypeForEncodingAndBitSize(lldb::Encoding encoding,
                                                   size_t bit_size) override {
    return CompilerType();
  }
  bool IsTypedefType(lldb::opaque_compiler_type_t type) override;
  CompilerType GetTypedefedType(lldb::opaque_compiler_type_t type) override;
  CompilerType
  GetFullyUnqualifiedType(lldb::opaque_compiler_type_t type) override;
  uint32_t GetNumDirectBaseClasses(lldb::opaque_compiler_type_t type) override;
  CompilerType GetDirectBaseClassAtIndex(lldb::opaque_compiler_type_t type,
                                         size_t idx,
                                         uint32_t *bit_offset_ptr) override;
  bool IsReferenceType(lldb::opaque_compiler_type_t type,
                       CompilerType *pointee_type, bool *is_rvalue) override;

  // Swift-specific methods.
  lldb::TypeSP GetCachedType(ConstString mangled);
  lldb::TypeSP GetCachedType(lldb::opaque_compiler_type_t type);
  void SetCachedType(ConstString mangled, const lldb::TypeSP &type_sp);
  bool IsImportedType(lldb::opaque_compiler_type_t type,
                      CompilerType *original_type) override;
  /// Determine whether this is a builtin SIMD type.
  static bool IsSIMDType(CompilerType type);
  static bool IsOptionalType(lldb::opaque_compiler_type_t type);
  static CompilerType GetOptionalType(CompilerType type);
  /// Like \p IsImportedType(), but even returns Clang types that are also Swift
  /// builtins (int <-> Swift.Int) as Clang types.
  CompilerType GetAsClangTypeOrNull(lldb::opaque_compiler_type_t type,
                                    bool *is_imported = nullptr);
  bool IsErrorType(lldb::opaque_compiler_type_t type) override;
  CompilerType GetErrorType() override;
  CompilerType GetWeakReferent(lldb::opaque_compiler_type_t type) override;
  CompilerType GetReferentType(lldb::opaque_compiler_type_t type) override;
  CompilerType GetInstanceType(lldb::opaque_compiler_type_t type,
                               ExecutionContextScope *exe_scope) override;
  CompilerType GetStaticSelfType(lldb::opaque_compiler_type_t type) override;
  static swift::Demangle::NodePointer
  GetStaticSelfType(swift::Demangle::Demangler &dem,
                    swift::Demangle::NodePointer node);

  /// Wrap type inside a SILPackType.
  CompilerType CreateSILPackType(CompilerType type, bool indirect);
  struct PackTypeInfo {
    unsigned count = 0;
    bool indirect = false;
    bool expanded = false;
  };
  std::optional<PackTypeInfo> IsSILPackType(CompilerType type);
  CompilerType GetSILPackElementAtIndex(CompilerType type, unsigned i);
  CompilerType
  CreateTupleType(const std::vector<TupleElement> &elements) override;
  bool IsTupleType(lldb::opaque_compiler_type_t type) override;
  std::optional<NonTriviallyManagedReferenceKind>
  GetNonTriviallyManagedReferenceKind(
      lldb::opaque_compiler_type_t type) override;

  /// Return the nth tuple element's type and name, if it has one.
  std::optional<TupleElement>
  GetTupleElement(lldb::opaque_compiler_type_t type, size_t idx);

  /// Returns true if the compiler type is a Builtin (belongs to the "Builtin
  /// module").
  static bool IsBuiltinType(CompilerType type);

  /// Creates a GenericTypeParamType with the desired depth and index.
  CompilerType
  CreateGenericTypeParamType(unsigned int depth, unsigned int index,
                             swift::Mangle::ManglingFlavor flavor) override;

  /// Create a __C imported struct type.
  CompilerType CreateClangStructType(llvm::StringRef name);

  /// Builds a bound generic struct demangle tree with the name, module name,
  /// and the struct's elements.
  static swift::Demangle::NodePointer CreateBoundGenericStruct(
      llvm::StringRef name, llvm::StringRef module_name,
      llvm::ArrayRef<swift::Demangle::NodePointer> type_list_elements,
      swift::Demangle::Demangler &dem);

  /// Get the Swift raw pointer type.
  CompilerType GetRawPointerType();
  /// Determine whether \p type is a protocol.
  bool IsExistentialType(lldb::opaque_compiler_type_t type);
  bool IsBoundGenericAliasType(lldb::opaque_compiler_type_t type);
  bool ContainsBoundGenericType(lldb::opaque_compiler_type_t type);

  /// Recursively transform the demangle tree starting a \p node by
  /// doing a post-order traversal and replacing each node with
  /// fn(node).
  /// The NodePointer passed to \p fn is guaranteed to be non-null.
  static swift::Demangle::NodePointer Transform(
      swift::Demangle::Demangler &dem, swift::Demangle::NodePointer node,
      std::function<swift::Demangle::NodePointer(swift::Demangle::NodePointer)>
          visitor);

  /// Recursively transform the demangle tree starting a \p node by
  /// doing a post-order traversal and replacing each node with
  /// fn(node).
  /// The NodePointer passed to \p fn is guaranteed to be non-null.
  static llvm::Expected<swift::Demangle::NodePointer>
  TryTransform(swift::Demangle::Demangler &dem,
               swift::Demangle::NodePointer node,
               std::function<llvm::Expected<swift::Demangle::NodePointer>(
                   swift::Demangle::NodePointer)>
                   visitor);

  /// A left-to-right preorder traversal. Don't visit children if
  /// visitor returns false.
  /// The NodePointer passed to \p fn is guaranteed to be non-null.
  static void
  PreOrderTraversal(swift::Demangle::NodePointer node,
                    std::function<bool(swift::Demangle::NodePointer)>);

  /// Canonicalize Array, Dictionary and Optional to their sugared form.
  static swift::Demangle::NodePointer
  CanonicalizeSugar(swift::Demangle::Demangler &dem,
                    swift::Demangle::NodePointer node);

  /// Recursively desugars sugared types (arrays, dictionaries, optionals, etc.)
  /// in a demangle tree.
  swift::Demangle::NodePointer DesugarNode(swift::Demangle::Demangler &dem,
                                           swift::Demangle::NodePointer node);

  /// Finds the nominal type node (struct, class, enum) that contains the
  /// module and identifier nodes for that type. If \p node is not a valid
  /// type node, returns a nullptr.
  static swift::Demangle::NodePointer
  FindTypeWithModuleAndIdentifierNode(swift::Demangle::NodePointer node);

  /// Types with the @_originallyDefinedIn attribute are serialized with with
  /// the original module name in reflection metadata. At the same time the type
  /// is serialized with the swiftmodule name in debug info, but with a parent
  /// module with the original module name. This function adjusts \type to look
  /// up the type in reflection metadata if necessary.
  std::string
  AdjustTypeForOriginallyDefinedInModule(llvm::StringRef mangled_typename);

  /// Return the canonicalized Demangle tree for a Swift mangled type name.
  /// It resolves all type aliases and removes sugar.
  swift::Demangle::NodePointer
  GetCanonicalDemangleTree(swift::Demangle::Demangler &dem,
                           llvm::StringRef mangled_name);
  /// Return the base name of the topmost nominal type.
  static llvm::StringRef GetBaseName(swift::Demangle::NodePointer node);
  static std::string GetBaseName(lldb::opaque_compiler_type_t type);

  /// Given a mangled name that mangles a "type metadata for Type", return a
  /// CompilerType with that Type.
  CompilerType GetTypeFromTypeMetadataNode(llvm::StringRef mangled_name);

  /// Use API notes to determine the swiftified name of \p clang_decl.
  std::string GetSwiftName(const clang::Decl *clang_decl,
                           TypeSystemClang &clang_typesystem) override;

  /// Wrap \p node as \p Global(TypeMangling(node)), remangle the type
  /// and create a CompilerType from it.
  CompilerType RemangleAsType(swift::Demangle::Demangler &dem,
                              swift::Demangle::NodePointer node,
                              swift::Mangle::ManglingFlavor flavor);

  /// Search the debug info for a non-nested Clang type with the specified name
  /// and cache the result. Users should prefer the version that takes in the
  /// decl_context.
  lldb::TypeSP LookupClangType(llvm::StringRef name_ref, SymbolContext sc = {});

  /// Search the debug info for a Clang type with the specified name and decl
  /// context.
  virtual lldb::TypeSP
  LookupClangType(llvm::StringRef name_ref,
                  llvm::ArrayRef<CompilerContext> decl_context,
                  bool ignore_modules, SymbolContext sc = {});

  /// Attempts to convert a Clang type into a Swift type.
  /// For example, int is converted to Int32.
  CompilerType ConvertClangTypeToSwiftType(CompilerType clang_type) override;

  /// Gets the descriptor finder belonging to this instance's
  /// module.
  swift::reflection::DescriptorFinder *GetDescriptorFinder();

  /// Lookup a type in the debug info.
  lldb::TypeSP FindTypeInModule(lldb::opaque_compiler_type_t type);

  /// Returns the mangling flavor associated with the ASTContext corresponding
  /// with this TypeSystem.
  swift::Mangle::ManglingFlavor
  GetManglingFlavor(const ExecutionContext *exe_ctx = nullptr);

protected:
  /// Determine whether the fallback is enabled via setting.
  bool UseSwiftASTContextFallback(const char *func_name,
                                  lldb::opaque_compiler_type_t type);
  /// Print a warning that a fallback was necessary.
  void DiagnoseSwiftASTContextFallback(const char *func_name,
                                       lldb::opaque_compiler_type_t type);

  /// Helper that creates an AST type from \p type.
  ///
  /// FIXME: This API is dangerous, it would be better to return a
  /// CompilerType so the caller isn't responsible for matching the
  /// exact same SwiftASTContext.
  void *ReconstructType(lldb::opaque_compiler_type_t type,
                        const ExecutionContext *exe_ctx = nullptr);
  void *ReconstructType(lldb::opaque_compiler_type_t type,
                        ExecutionContextScope *exe_scope);
  /// Cast \p opaque_type as a mangled name.
  static const char *AsMangledName(lldb::opaque_compiler_type_t type);

  /// Demangle the mangled name of the canonical type of \p type and
  /// drill into the Global(TypeMangling(Type())).
  ///
  /// \return the child of Type or a nullptr.
  swift::Demangle::NodePointer
  DemangleCanonicalType(swift::Demangle::Demangler &dem,
                        lldb::opaque_compiler_type_t type);

  /// Demangle the mangled name of \p type after canonicalizing its
  /// outermost type node and drill into the
  /// Global(TypeMangling(Type())).
  ///
  /// \return the child of Type or a nullptr.
  swift::Demangle::NodePointer
  DemangleCanonicalOutermostType(swift::Demangle::Demangler &dem,
                                 lldb::opaque_compiler_type_t type);

  /// Desugar to this node and if it is a type alias resolve it by
  /// looking up its type in the debug info.
  swift::Demangle::NodePointer
  Canonicalize(swift::Demangle::Demangler &dem,
               swift::Demangle::NodePointer node,
               swift::Mangle::ManglingFlavor flavor);

  /// Iteratively desugar and resolve all type aliases in \p node by
  /// looking up their types in the debug info.
  swift::Demangle::NodePointer
  GetCanonicalNode(swift::Demangle::Demangler &dem,
                   swift::Demangle::NodePointer node,
                   swift::Mangle::ManglingFlavor flavor);

  /// If \p node is a Struct/Class/Typedef in the __C module, return a
  /// Swiftified node by looking up the name in the corresponding APINotes and
  /// optionally putting it into the correctly named module.
  swift::Demangle::NodePointer
  GetSwiftified(swift::Demangle::Demangler &dem,
                swift::Demangle::NodePointer node,
                swift::Mangle::ManglingFlavor flavor, bool resolve_objc_module);

  /// Replace all "__C" module names with their actual Clang module
  /// names.  This is the recursion step of \p
  /// GetDemangleTreeForPrinting(). Don't call it directly.
  swift::Demangle::NodePointer GetNodeForPrintingImpl(
      swift::Demangle::Demangler &dem, swift::Demangle::NodePointer node,
      swift::Mangle::ManglingFlavor flavor, bool resolve_objc_module);

  /// Return the demangle tree representation with all "__C" module
  /// names with their actual Clang module names.
  swift::Demangle::NodePointer
  GetDemangleTreeForPrinting(swift::Demangle::Demangler &dem,
                             const char *mangled_name,
                             bool resolve_objc_module);

  /// Return an APINotes manager for the module with module id \id.
  /// APINotes are used to get at the SDK swiftification annotations.
  clang::api_notes::APINotesManager *
  GetAPINotesManager(ClangExternalASTSourceCallbacks *source, unsigned id);

  CompilerType
  LookupClangForwardType(llvm::StringRef name,
                         llvm::ArrayRef<CompilerContext> decl_context,
                         bool ignore_modules);

  /// Resolve a type alias node and return a demangle tree for the
  /// resolved type. If the type alias resolves to a Clang type, return
  /// a Clang CompilerType.
  ///
  /// \param prefer_clang_types if this is true, type aliases in the
  ///                           __C module are resolved as Clang types.
  ///
  std::pair<swift::Demangle::NodePointer, CompilerType>
  ResolveTypeAlias(swift::Demangle::Demangler &dem,
                   swift::Demangle::NodePointer node,
                   swift::Mangle::ManglingFlavor flavor,
                   bool prefer_clang_types = false);

  uint32_t CollectTypeInfo(swift::Demangle::Demangler &dem,
                           swift::Demangle::NodePointer node,
                           swift::Mangle::ManglingFlavor flavor,
                           bool &unresolved_typealias);

  swift::Demangle::NodePointer
  GetClangTypeNode(CompilerType clang_type, swift::Demangle::Demangler &dem);

  swift::Demangle::NodePointer
  GetClangTypeTypeNode(swift::Demangle::Demangler &dem,
                       CompilerType clang_type);

  /// Determine if this type contains a type from a module that looks
  /// like it was JIT-compiled by LLDB.
  bool IsExpressionEvaluatorDefined(lldb::opaque_compiler_type_t type);

#ifndef NDEBUG
  /// Check whether the type being dealt with is tricky to validate due to
  /// discrepancies between TypeSystemSwiftTypeRef and SwiftASTContext.
  bool ShouldSkipValidation(lldb::opaque_compiler_type_t type);
#endif

  /// Perform an action on all subling SwiftASTContexts.
  void NotifyAllTypeSystems(std::function<void(lldb::TypeSystemSP)> fn);

  struct TypeSystemAndCount {
    lldb::TypeSystemSP typesystem;
    /// Count how often this typesystem was initialized.
    unsigned char retry_count = 0;
  };

  std::once_flag m_fallback_warning;
  mutable std::mutex m_swift_ast_context_lock;
  /// The "precise" SwiftASTContexts managed by this scratch context. There
  /// exists one per Swift module. The keys in this map are module names.
  mutable llvm::DenseMap<const char *, TypeSystemAndCount>
      m_swift_ast_context_map;
  /// A list of types that turn SwiftASTContext into a fatal error
  /// state after type reconstruction (presumably due to additional
  /// module imports). The key is a pair of SymbolContext string and
  /// mangled type name.
  mutable llvm::DenseSet<std::pair<const char *, const char *>>
      m_dangerous_types;

  mutable std::unique_ptr<SwiftDWARFImporterForClangTypes>
      m_dwarf_importer_for_clang_types_up;
  mutable std::unique_ptr<ClangNameImporter> m_name_importer_up;
  std::unique_ptr<plugin::dwarf::DWARFASTParser> m_dwarf_ast_parser_up;

  /// The APINotesManager responsible for each Clang module.
  llvm::DenseMap<clang::Module *,
                 std::unique_ptr<clang::api_notes::APINotesManager>>
      m_apinotes_manager;

  /// All lldb::Type pointers produced by DWARFASTParser Swift go here.
  ThreadSafeDenseMap<const char *, lldb::TypeSP> m_swift_type_map;
  /// An LRU cache for \ref GetManglingFlavor().
  std::pair<CompileUnit *, swift::Mangle::ManglingFlavor> m_lru_is_embedded = {
      nullptr, swift::Mangle::ManglingFlavor::Default};
};

/// This one owns a SwiftASTContextForExpressions.
class TypeSystemSwiftTypeRefForExpressions : public TypeSystemSwiftTypeRef {
  // LLVM RTTI support
  static char ID;

public:
  /// LLVM RTTI support.
  /// \{
  bool isA(const void *ClassID) const override {
    return ClassID == &ID || TypeSystemSwiftTypeRef::isA(ClassID);
  }
  static bool classof(const TypeSystem *ts) { return ts->isA(&ID); }
  /// \}

  TypeSystemSwiftTypeRefForExpressions(lldb::LanguageType language,
                                       Target &target, bool repl,
                                       bool playground,
                                       const char *extra_options);

  static TypeSystemSwiftTypeRefForExpressionsSP GetForTarget(Target &target);
  static TypeSystemSwiftTypeRefForExpressionsSP
  GetForTarget(lldb::TargetSP target);

  SwiftASTContextSP GetSwiftASTContext(const SymbolContext &sc) const override;
  SwiftASTContextSP
  GetSwiftASTContextOrNull(const SymbolContext &sc) const override;
  /// This API needs to be called for a REPL or Playground before the first call
  /// to GetSwiftASTContext is being made.
  void SetCompilerOptions(bool repl, bool playground,
                         const char *compiler_options) {
    m_repl = repl;
    m_playground = playground;
    m_compiler_options = compiler_options;
  }
  lldb::TargetWP GetTargetWP() const override { return m_target_wp; }

  void ModulesDidLoad(ModuleList &module_list);

  /// Forwards to SwiftASTContext.
  UserExpression *GetUserExpression(llvm::StringRef expr,
                                    llvm::StringRef prefix,
                                    SourceLanguage language,
                                    Expression::ResultType desired_type,
                                    const EvaluateExpressionOptions &options,
                                    ValueObject *ctx_obj) override;

  /// Forwards to SwiftASTContext.
  PersistentExpressionState *GetPersistentExpressionState() override;
  Status PerformCompileUnitImports(const SymbolContext &sc);
  /// Returns how often ModulesDidLoad was called.
  unsigned GetGeneration() const { return m_generation; }
  /// Performs a target-wide search.
  /// \param exe_ctx is a hint for where to look first.
  lldb::TypeSP LookupClangType(llvm::StringRef name_ref,
                               llvm::ArrayRef<CompilerContext> decl_context,
                               bool ignore_modules,
                               SymbolContext sc = {}) override;

  friend class SwiftASTContextForExpressions;
protected:
  lldb::TargetWP m_target_wp;
  unsigned m_generation = 0;
  bool m_repl = false;
  bool m_playground = false;
  const char *m_compiler_options = nullptr;

  /// This exists to implement the PerformCompileUnitImports
  /// mechanism.
  ///
  /// FIXME: The mechanism's implementation is unreliable since it
  ///        depends on where the scratch context is first
  ///        initialized. It should be replaced by something more
  ///        deterministic.
  /// Perform all the implicit imports for the current frame.
  mutable std::unique_ptr<SymbolContext> m_initial_symbol_context_up;
  std::unique_ptr<SwiftPersistentExpressionState> m_persistent_state_up;
  /// Map ConstString Clang type identifiers and the concatenation of the
  /// compiler context used to find them to Clang types.
  ThreadSafeStringMap<lldb::TypeSP> m_clang_type_cache;
};

} // namespace lldb_private
#endif
