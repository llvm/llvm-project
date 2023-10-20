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
#include "lldb/Utility/ThreadSafeDenseMap.h"

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
} // namespace Demangle
} // namespace swift

namespace lldb_private {
class ClangExternalASTSourceCallbacks;
class ClangNameImporter;
class SwiftASTContext;
class SwiftASTContextForExpressions;
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
  TypeSystemSwiftTypeRef();
  ~TypeSystemSwiftTypeRef();
  TypeSystemSwiftTypeRef(Module &module);
  /// Get the corresponding SwiftASTContext, and create one if necessary.
  SwiftASTContext *GetSwiftASTContext() const override;
  /// Return SwiftASTContext, iff one has already been created.
  SwiftASTContext *GetSwiftASTContextOrNull() const;
  TypeSystemSwiftTypeRef &GetTypeSystemSwiftTypeRef() override { return *this; }
  const TypeSystemSwiftTypeRef &GetTypeSystemSwiftTypeRef() const override {
    return *this;
  }
  swift::DWARFImporterDelegate &GetDWARFImporterDelegate();
  ClangNameImporter *GetNameImporter() const;
  llvm::Triple GetTriple() const;
  void SetTriple(const llvm::Triple triple) override;
  void ClearModuleDependentCaches() override;
  lldb::TargetWP GetTargetWP() const override { return {}; }

  CompilerType ReconstructType(CompilerType type);
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

  void DiagnoseWarnings(Process &process, Module &module) const override;
  DWARFASTParser *GetDWARFParser() override;
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

  Module *GetModule() const { return m_module; }

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
  llvm::Optional<uint64_t>
  GetBitSize(lldb::opaque_compiler_type_t type,
             ExecutionContextScope *exe_scope) override;
  llvm::Optional<uint64_t>
  GetByteStride(lldb::opaque_compiler_type_t type,
                ExecutionContextScope *exe_scope) override;
  lldb::Encoding GetEncoding(lldb::opaque_compiler_type_t type,
                             uint64_t &count) override;
  uint32_t GetNumChildren(lldb::opaque_compiler_type_t type,
                          bool omit_empty_base_classes,
                          const ExecutionContext *exe_ctx) override;
  uint32_t GetNumFields(lldb::opaque_compiler_type_t type,
                        ExecutionContext *exe_ctx = nullptr) override;
  CompilerType GetFieldAtIndex(lldb::opaque_compiler_type_t type, size_t idx,
                               std::string &name, uint64_t *bit_offset_ptr,
                               uint32_t *bitfield_bit_size_ptr,
                               bool *is_bitfield_ptr) override;
  CompilerType GetChildCompilerTypeAtIndex(
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
  llvm::Optional<size_t>
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
  /// Like \p IsImportedType(), but even returns Clang types that are also Swift
  /// builtins (int <-> Swift.Int) as Clang types.
  CompilerType GetAsClangTypeOrNull(lldb::opaque_compiler_type_t type,
                                    bool *is_imported = nullptr);
  bool IsErrorType(lldb::opaque_compiler_type_t type) override;
  CompilerType GetErrorType() override;
  CompilerType GetReferentType(lldb::opaque_compiler_type_t type) override;
  CompilerType GetInstanceType(lldb::opaque_compiler_type_t type) override;
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
  llvm::Optional<PackTypeInfo> IsSILPackType(CompilerType type);
  CompilerType GetSILPackElementAtIndex(CompilerType type, unsigned i);
  CompilerType
  CreateTupleType(const std::vector<TupleElement> &elements) override;
  bool IsTupleType(lldb::opaque_compiler_type_t type) override;
  llvm::Optional<NonTriviallyManagedReferenceKind>
  GetNonTriviallyManagedReferenceKind(
      lldb::opaque_compiler_type_t type) override;

  /// Return the nth tuple element's type and name, if it has one.
  llvm::Optional<TupleElement>
  GetTupleElement(lldb::opaque_compiler_type_t type, size_t idx);

  /// Creates a GenericTypeParamType with the desired depth and index.
  CompilerType CreateGenericTypeParamType(unsigned int depth,
                                    unsigned int index) override;

  /// Get the Swift raw pointer type.
  CompilerType GetRawPointerType();
  /// Determine whether \p type is a protocol.
  bool IsExistentialType(lldb::opaque_compiler_type_t type);

  /// Recursively transform the demangle tree starting a \p node by
  /// doing a post-order traversal and replacing each node with
  /// fn(node).
  /// The NodePointer passed to \p fn is guaranteed to be non-null.
  static swift::Demangle::NodePointer Transform(
      swift::Demangle::Demangler &dem, swift::Demangle::NodePointer node,
      std::function<swift::Demangle::NodePointer(swift::Demangle::NodePointer)>
          visitor);

  /// A left-to-right preorder traversal. Don't visit children if
  /// visitor returns false.
  static void
  PreOrderTraversal(swift::Demangle::NodePointer node,
                    std::function<bool(swift::Demangle::NodePointer)>);

  /// Canonicalize Array, Dictionary and Optional to their sugared form.
  static swift::Demangle::NodePointer
  CanonicalizeSugar(swift::Demangle::Demangler &dem,
                    swift::Demangle::NodePointer node);

  /// Return the canonicalized Demangle tree for a Swift mangled type name.
  swift::Demangle::NodePointer
  GetCanonicalDemangleTree(swift::Demangle::Demangler &dem,
                           llvm::StringRef mangled_name);
  /// Return the base name of the topmost nominal type.
  static llvm::StringRef GetBaseName(swift::Demangle::NodePointer node);

  /// Return whether the type is known to be specially handled by the compiler.
  static bool IsKnownSpecialImportedType(llvm::StringRef name);

  /// Use API notes to determine the swiftified name of \p clang_decl.
  std::string GetSwiftName(const clang::Decl *clang_decl,
                           TypeSystemClang &clang_typesystem) override;

  CompilerType GetBuiltinRawPointerType() override;

  /// Wrap \p node as \p Global(TypeMangling(node)), remangle the type
  /// and create a CompilerType from it.
  CompilerType RemangleAsType(swift::Demangle::Demangler &dem,
                              swift::Demangle::NodePointer node);

  /// Search the debug info for a non-nested Clang type with the specified name
  /// and cache the result. Users should prefer the version that takes in the
  /// decl_context.
  lldb::TypeSP LookupClangType(llvm::StringRef name_ref);

  /// Search the debug info for a Clang type with the specified name and decl
  /// context, and cache the result.
  lldb::TypeSP LookupClangType(llvm::StringRef name_ref,
                               llvm::ArrayRef<CompilerContext> decl_context);

  /// Attempts to convert a Clang type into a Swift type.
  /// For example, int is converted to Int32.
  CompilerType ConvertClangTypeToSwiftType(CompilerType clang_type) override;

protected:
  /// Helper that creates an AST type from \p type.
  void *ReconstructType(lldb::opaque_compiler_type_t type);
  /// Cast \p opaque_type as a mangled name.
  static const char *AsMangledName(lldb::opaque_compiler_type_t type);

  /// Lookup a type in the debug info.
  lldb::TypeSP FindTypeInModule(lldb::opaque_compiler_type_t type);

  /// Demangle the mangled name of the canonical type of \p type and
  /// drill into the Global(TypeMangling(Type())).
  ///
  /// \return the child of Type or a nullptr.
  swift::Demangle::NodePointer
  DemangleCanonicalType(swift::Demangle::Demangler &dem,
                        lldb::opaque_compiler_type_t type);

  /// If \p node is a Struct/Class/Typedef in the __C module, return a
  /// Swiftified node by looking up the name in the corresponding APINotes and
  /// optionally putting it into the correctly named module.
  swift::Demangle::NodePointer GetSwiftified(swift::Demangle::Demangler &dem,
                                             swift::Demangle::NodePointer node,
                                             bool resolve_objc_module);

  /// Replace all "__C" module names with their actual Clang module
  /// names.  This is the recursion step of \p
  /// GetDemangleTreeForPrinting(). Don't call it directly.
  swift::Demangle::NodePointer
  GetNodeForPrintingImpl(swift::Demangle::Demangler &dem,
                         swift::Demangle::NodePointer node,
                         bool resolve_objc_module);

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

  CompilerType LookupClangForwardType(llvm::StringRef name, 
                  llvm::ArrayRef<CompilerContext> decl_context);

  std::pair<swift::Demangle::NodePointer, CompilerType>
  ResolveTypeAlias(swift::Demangle::Demangler &dem,
                   swift::Demangle::NodePointer node,
                   bool prefer_clang_types = false);

  swift::Demangle::NodePointer
  GetCanonicalNode(swift::Demangle::Demangler &dem,
                   swift::Demangle::NodePointer node);

  uint32_t CollectTypeInfo(swift::Demangle::Demangler &dem,
                           swift::Demangle::NodePointer node,
                           bool &unresolved_typealias);

  swift::Demangle::NodePointer
  GetClangTypeNode(CompilerType clang_type, swift::Demangle::Demangler &dem);

  swift::Demangle::NodePointer
  GetClangTypeTypeNode(swift::Demangle::Demangler &dem,
                       CompilerType clang_type);
#ifndef NDEBUG
  /// Check whether the type being dealt with is tricky to validate due to
  /// discrepancies between TypeSystemSwiftTypeRef and SwiftASTContext.
  bool ShouldSkipValidation(lldb::opaque_compiler_type_t type);
#endif

  /// The sibling SwiftASTContext.
  mutable bool m_swift_ast_context_initialized = false;
  mutable lldb::TypeSystemSP m_swift_ast_context_sp;
  mutable SwiftASTContext *m_swift_ast_context = nullptr;
  mutable std::unique_ptr<swift::DWARFImporterDelegate>
      m_dwarf_importer_delegate_up;
  mutable std::unique_ptr<ClangNameImporter> m_name_importer_up;
  std::unique_ptr<DWARFASTParser> m_dwarf_ast_parser_up;

  /// The APINotesManager responsible for each Clang module.
  llvm::DenseMap<clang::Module *,
                 std::unique_ptr<clang::api_notes::APINotesManager>>
      m_apinotes_manager;

  /// All lldb::Type pointers produced by DWARFASTParser Swift go here.
  ThreadSafeDenseMap<const char *, lldb::TypeSP> m_swift_type_map;
  /// Map ConstString Clang type identifiers to Clang types.
  ThreadSafeDenseMap<const char *, lldb::TypeSP> m_clang_type_cache;
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
                                       Target &target,
                                       const char *extra_options);

  /// For per-module fallback contexts.
  TypeSystemSwiftTypeRefForExpressions(lldb::LanguageType language,
                                       Target &target, Module &module);

  SwiftASTContext *GetSwiftASTContext() const override;
  lldb::TargetWP GetTargetWP() const override { return m_target_wp; }

  /// Forwards to SwiftASTContext.
  UserExpression *GetUserExpression(llvm::StringRef expr,
                                    llvm::StringRef prefix,
                                    lldb::LanguageType language,
                                    Expression::ResultType desired_type,
                                    const EvaluateExpressionOptions &options,
                                    ValueObject *ctx_obj) override;

  /// Forwards to SwiftASTContext.
  PersistentExpressionState *GetPersistentExpressionState() override;
  Status PerformCompileUnitImports(SymbolContext &sc);

  friend class SwiftASTContextForExpressions;
protected:
  lldb::TargetWP m_target_wp;

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
};

swift::DWARFImporterDelegate *
CreateSwiftDWARFImporterDelegate(TypeSystemSwiftTypeRef &ts);
  
} // namespace lldb_private
#endif
