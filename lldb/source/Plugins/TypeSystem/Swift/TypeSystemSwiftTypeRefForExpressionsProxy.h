//===-- TypeSystemSwiftTypeRefForExpressionsProxy.h ----------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2026 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_TypeSystemSwiftTypeRefForExpressionsProxy_h_
#define liblldb_TypeSystemSwiftTypeRefForExpressionsProxy_h_

#include "Plugins/ExpressionParser/Swift/SwiftPersistentExpressionState.h"
#include "Plugins/TypeSystem/Swift/TypeSystemSwiftTypeRef.h"

namespace lldb_private {

/// A variant of TypeSystemSwiftTypeRefForExpressions that forwards
/// all calls to another TypeSystemSwiftTypeRefForExpressions.
class TypeSystemSwiftTypeRefForExpressionsProxy
    : public TypeSystemSwiftTypeRefForExpressions {
public:
  /// Constructing the base class with repl and playground both false skips
  /// eagerly creating a SwiftASTContext that would never be used: this
  /// object never accesses its own base-class state, only forwards to
  /// \p real.
  TypeSystemSwiftTypeRefForExpressionsProxy(
      std::shared_ptr<TypeSystemSwiftTypeRefForExpressions> real)
      : TypeSystemSwiftTypeRefForExpressions(lldb::eLanguageTypeSwift,
                                             *real->GetTargetWP().lock(),
                                             /*repl=*/false,
                                             /*playground=*/false),
        m_real(std::move(real)) {}

  SwiftASTContextSP GetSwiftASTContext(const SymbolContext &sc) const override {
    return m_real->GetSwiftASTContext(sc);
  }
  SwiftASTContextSP
  GetSwiftASTContextOrNull(const SymbolContext &sc) const override {
    return m_real->GetSwiftASTContextOrNull(sc);
  }
  void SetTriple(const SymbolContext &sc, const llvm::Triple triple) override {
    m_real->SetTriple(sc, triple);
  }
  void ClearModuleDependentCaches() override {
    m_real->ClearModuleDependentCaches();
  }
  lldb::TargetWP GetTargetWP() const override { return m_real->GetTargetWP(); }
  CompilerType
  GetTypeFromMangledTypename(ConstString mangled_typename) override {
    return m_real->GetTypeFromMangledTypename(mangled_typename);
  }
  CompilerType GetGenericArgumentType(lldb::opaque_compiler_type_t type,
                                      size_t idx) override {
    return m_real->GetGenericArgumentType(type, idx);
  }
  llvm::StringRef GetPluginName() override { return m_real->GetPluginName(); }
  bool SupportsLanguage(lldb::LanguageType language) override {
    return m_real->SupportsLanguage(language);
  }
  Status IsCompatible() override { return m_real->IsCompatible(); }
  plugin::dwarf::DWARFASTParser *GetDWARFParser() override {
    return m_real->GetDWARFParser();
  }
  npdb::PdbAstBuilder *GetNativePDBParser() override {
    return m_real->GetNativePDBParser();
  }
  ConstString DeclGetName(void *opaque_decl) override {
    return m_real->DeclGetName(opaque_decl);
  }
  std::vector<CompilerDecl>
  DeclContextFindDeclByName(void *opaque_decl_ctx, ConstString name,
                            const bool ignore_imported_decls) override {
    return m_real->DeclContextFindDeclByName(opaque_decl_ctx, name,
                                             ignore_imported_decls);
  }
  bool DeclContextIsContainedInLookup(void *opaque_decl_ctx,
                                      void *other_opaque_decl_ctx) override {
    return m_real->DeclContextIsContainedInLookup(opaque_decl_ctx,
                                                  other_opaque_decl_ctx);
  }
  bool IsAggregateType(lldb::opaque_compiler_type_t type) override {
    return m_real->IsAggregateType(type);
  }
  bool IsDefined(lldb::opaque_compiler_type_t type) override {
    return m_real->IsDefined(type);
  }
  bool IsFunctionType(lldb::opaque_compiler_type_t type) override {
    return m_real->IsFunctionType(type);
  }
  size_t
  GetNumberOfFunctionArguments(lldb::opaque_compiler_type_t type) override {
    return m_real->GetNumberOfFunctionArguments(type);
  }
  CompilerType GetFunctionArgumentAtIndex(lldb::opaque_compiler_type_t type,
                                          const size_t index) override {
    return m_real->GetFunctionArgumentAtIndex(type, index);
  }
  bool IsFunctionPointerType(lldb::opaque_compiler_type_t type) override {
    return m_real->IsFunctionPointerType(type);
  }
  bool IsPossibleDynamicType(lldb::opaque_compiler_type_t type,
                             CompilerType *target_type, bool check_cplusplus,
                             bool check_objc) override {
    return m_real->IsPossibleDynamicType(type, target_type, check_cplusplus,
                                         check_objc);
  }
  bool IsPointerType(lldb::opaque_compiler_type_t type,
                     CompilerType *pointee_type) override {
    return m_real->IsPointerType(type, pointee_type);
  }
  bool IsVoidType(lldb::opaque_compiler_type_t type) override {
    return m_real->IsVoidType(type);
  }
  uint32_t GetPointerByteSize() override {
    return m_real->GetPointerByteSize();
  }
  ConstString GetTypeName(lldb::opaque_compiler_type_t type,
                          bool BaseOnly) override {
    return m_real->GetTypeName(type, BaseOnly);
  }
  ConstString GetDisplayTypeName(lldb::opaque_compiler_type_t type,
                                 const SymbolContext *sc) override {
    return m_real->GetDisplayTypeName(type, sc);
  }
  ConstString GetMangledTypeName(lldb::opaque_compiler_type_t type) override {
    return m_real->GetMangledTypeName(type);
  }
  uint32_t GetTypeInfo(lldb::opaque_compiler_type_t type,
                       CompilerType *pointee_or_element_clang_type) override {
    return m_real->GetTypeInfo(type, pointee_or_element_clang_type);
  }
  lldb::TypeClass GetTypeClass(lldb::opaque_compiler_type_t type) override {
    return m_real->GetTypeClass(type);
  }
  CompilerType GetArrayElementType(lldb::opaque_compiler_type_t type,
                                   ExecutionContextScope *exe_scope) override {
    return m_real->GetArrayElementType(type, exe_scope);
  }
  CompilerType GetCanonicalType(lldb::opaque_compiler_type_t type) override {
    return m_real->GetCanonicalType(type);
  }
  int GetFunctionArgumentCount(lldb::opaque_compiler_type_t type) override {
    return m_real->GetFunctionArgumentCount(type);
  }
  CompilerType GetFunctionArgumentTypeAtIndex(lldb::opaque_compiler_type_t type,
                                              size_t idx) override {
    return m_real->GetFunctionArgumentTypeAtIndex(type, idx);
  }
  CompilerType
  GetFunctionReturnType(lldb::opaque_compiler_type_t type) override {
    return m_real->GetFunctionReturnType(type);
  }
  size_t GetNumMemberFunctions(lldb::opaque_compiler_type_t type) override {
    return m_real->GetNumMemberFunctions(type);
  }
  TypeMemberFunctionImpl
  GetMemberFunctionAtIndex(lldb::opaque_compiler_type_t type,
                           size_t idx) override {
    return m_real->GetMemberFunctionAtIndex(type, idx);
  }
  CompilerType GetPointeeType(lldb::opaque_compiler_type_t type) override {
    return m_real->GetPointeeType(type);
  }
  CompilerType GetPointerType(lldb::opaque_compiler_type_t type) override {
    return m_real->GetPointerType(type);
  }
  CompilerType GetBasicTypeFromAST(lldb::BasicType basic_type) override {
    return m_real->GetBasicTypeFromAST(basic_type);
  }
  llvm::Expected<uint64_t>
  GetBitSize(lldb::opaque_compiler_type_t type,
             ExecutionContextScope *exe_scope) override {
    return m_real->GetBitSize(type, exe_scope);
  }
  std::optional<uint64_t>
  GetByteStride(lldb::opaque_compiler_type_t type,
                ExecutionContextScope *exe_scope) override {
    return m_real->GetByteStride(type, exe_scope);
  }
  lldb::Encoding GetEncoding(lldb::opaque_compiler_type_t type) override {
    return m_real->GetEncoding(type);
  }
  llvm::Expected<uint32_t>
  GetNumChildren(lldb::opaque_compiler_type_t type,
                 bool omit_empty_base_classes,
                 const ExecutionContext *exe_ctx) override {
    return m_real->GetNumChildren(type, omit_empty_base_classes, exe_ctx);
  }
  uint32_t GetNumFields(lldb::opaque_compiler_type_t type,
                        ExecutionContext *exe_ctx = nullptr) override {
    return m_real->GetNumFields(type, exe_ctx);
  }
  CompilerType GetFieldAtIndex(lldb::opaque_compiler_type_t type, size_t idx,
                               std::string &name, uint64_t *bit_offset_ptr,
                               uint32_t *bitfield_bit_size_ptr,
                               bool *is_bitfield_ptr) override {
    return m_real->GetFieldAtIndex(type, idx, name, bit_offset_ptr,
                                   bitfield_bit_size_ptr, is_bitfield_ptr);
  }
  llvm::Expected<CompilerType> GetChildCompilerTypeAtIndex(
      lldb::opaque_compiler_type_t type, ExecutionContext *exe_ctx, size_t idx,
      bool transparent_pointers, bool omit_empty_base_classes,
      bool ignore_array_bounds, std::string &child_name,
      uint32_t &child_byte_size, int32_t &child_byte_offset,
      uint32_t &child_bitfield_bit_size, uint32_t &child_bitfield_bit_offset,
      bool &child_is_base_class, bool &child_is_deref_of_parent,
      ValueObject *valobj, uint64_t &language_flags) override {
    return m_real->GetChildCompilerTypeAtIndex(
        type, exe_ctx, idx, transparent_pointers, omit_empty_base_classes,
        ignore_array_bounds, child_name, child_byte_size, child_byte_offset,
        child_bitfield_bit_size, child_bitfield_bit_offset, child_is_base_class,
        child_is_deref_of_parent, valobj, language_flags);
  }
  size_t
  GetIndexOfChildMemberWithName(lldb::opaque_compiler_type_t type,
                                llvm::StringRef name, ExecutionContext *exe_ctx,
                                bool omit_empty_base_classes,
                                std::vector<uint32_t> &child_indexes) override {
    return m_real->GetIndexOfChildMemberWithName(
        type, name, exe_ctx, omit_empty_base_classes, child_indexes);
  }
  size_t GetNumTemplateArguments(lldb::opaque_compiler_type_t type,
                                 bool expand_pack) override {
    return m_real->GetNumTemplateArguments(type, expand_pack);
  }
  lldb::TemplateArgumentKind
  GetTemplateArgumentKind(lldb::opaque_compiler_type_t type, size_t idx,
                          bool expand_pack) override {
    return m_real->GetTemplateArgumentKind(type, idx, expand_pack);
  }
  CompilerType GetTypeTemplateArgument(lldb::opaque_compiler_type_t type,
                                       size_t idx, bool expand_pack) override {
    return m_real->GetTypeTemplateArgument(type, idx, expand_pack);
  }
  CompilerType
  GetTypeForFormatters(lldb::opaque_compiler_type_t type) override {
    return m_real->GetTypeForFormatters(type);
  }
  LazyBool ShouldPrintAsOneLiner(lldb::opaque_compiler_type_t type,
                                 ValueObject *valobj) override {
    return m_real->ShouldPrintAsOneLiner(type, valobj);
  }
  bool IsMeaninglessWithoutDynamicResolution(
      lldb::opaque_compiler_type_t type) override {
    return m_real->IsMeaninglessWithoutDynamicResolution(type);
  }
  void DumpTypeDescription(
      lldb::opaque_compiler_type_t type,
      lldb::DescriptionLevel level = lldb::eDescriptionLevelFull,
      ExecutionContextScope *exe_scope = nullptr) override {
    m_real->DumpTypeDescription(type, level, exe_scope);
  }
  void DumpTypeDescription(
      lldb::opaque_compiler_type_t type, Stream &s,
      lldb::DescriptionLevel level = lldb::eDescriptionLevelFull,
      ExecutionContextScope *exe_scope = nullptr) override {
    m_real->DumpTypeDescription(type, s, level, exe_scope);
  }
  void DumpTypeDescription(
      lldb::opaque_compiler_type_t type, bool print_help_if_available,
      bool print_extensions_if_available,
      lldb::DescriptionLevel level = lldb::eDescriptionLevelFull,
      ExecutionContextScope *exe_scope = nullptr) override {
    m_real->DumpTypeDescription(type, print_help_if_available,
                                print_extensions_if_available, level,
                                exe_scope);
  }
  void DumpTypeDescription(
      lldb::opaque_compiler_type_t type, Stream *s,
      bool print_help_if_available, bool print_extensions_if_available,
      lldb::DescriptionLevel level = lldb::eDescriptionLevelFull,
      ExecutionContextScope *exe_scope = nullptr) override {
    m_real->DumpTypeDescription(type, s, print_help_if_available,
                                print_extensions_if_available, level,
                                exe_scope);
  }
  bool IsPointerOrReferenceType(lldb::opaque_compiler_type_t type,
                                CompilerType *pointee_type) override {
    return m_real->IsPointerOrReferenceType(type, pointee_type);
  }
  std::optional<size_t>
  GetTypeBitAlign(lldb::opaque_compiler_type_t type,
                  ExecutionContextScope *exe_scope) override {
    return m_real->GetTypeBitAlign(type, exe_scope);
  }
  CompilerType GetBuiltinTypeForEncodingAndBitSize(lldb::Encoding encoding,
                                                   size_t bit_size) override {
    return m_real->GetBuiltinTypeForEncodingAndBitSize(encoding, bit_size);
  }
  bool IsTypedefType(lldb::opaque_compiler_type_t type) override {
    return m_real->IsTypedefType(type);
  }
  CompilerType GetTypedefedType(lldb::opaque_compiler_type_t type) override {
    return m_real->GetTypedefedType(type);
  }
  CompilerType
  GetFullyUnqualifiedType(lldb::opaque_compiler_type_t type) override {
    return m_real->GetFullyUnqualifiedType(type);
  }
  uint32_t GetNumDirectBaseClasses(lldb::opaque_compiler_type_t type) override {
    return m_real->GetNumDirectBaseClasses(type);
  }
  CompilerType GetDirectBaseClassAtIndex(lldb::opaque_compiler_type_t type,
                                         size_t idx,
                                         uint32_t *bit_offset_ptr) override {
    return m_real->GetDirectBaseClassAtIndex(type, idx, bit_offset_ptr);
  }
  bool IsReferenceType(lldb::opaque_compiler_type_t type,
                       CompilerType *pointee_type, bool *is_rvalue) override {
    return m_real->IsReferenceType(type, pointee_type, is_rvalue);
  }
  bool IsImportedType(lldb::opaque_compiler_type_t type,
                      CompilerType *original_type) override {
    return m_real->IsImportedType(type, original_type);
  }
  bool IsErrorType(lldb::opaque_compiler_type_t type,
                   const ExecutionContext *exe_ctx) override {
    return m_real->IsErrorType(type, exe_ctx);
  }
  CompilerType GetErrorType(swift::Mangle::ManglingFlavor flavor) override {
    return m_real->GetErrorType(flavor);
  }
  CompilerType GetWeakReferent(lldb::opaque_compiler_type_t type) override {
    return m_real->GetWeakReferent(type);
  }
  CompilerType GetReferentType(lldb::opaque_compiler_type_t type) override {
    return m_real->GetReferentType(type);
  }
  CompilerType GetInstanceType(lldb::opaque_compiler_type_t type,
                               ExecutionContextScope *exe_scope) override {
    return m_real->GetInstanceType(type, exe_scope);
  }
  CompilerType GetStaticSelfType(lldb::opaque_compiler_type_t type) override {
    return m_real->GetStaticSelfType(type);
  }
  CompilerType CreateTupleType(const std::vector<TupleElement> &elements,
                               swift::Mangle::ManglingFlavor flavor) override {
    return m_real->CreateTupleType(elements, flavor);
  }
  bool IsTupleType(lldb::opaque_compiler_type_t type) override {
    return m_real->IsTupleType(type);
  }
  std::optional<NonTriviallyManagedReferenceKind>
  GetNonTriviallyManagedReferenceKind(
      lldb::opaque_compiler_type_t type) override {
    return m_real->GetNonTriviallyManagedReferenceKind(type);
  }
  CompilerType
  CreateGenericTypeParamType(unsigned int depth, unsigned int index,
                             swift::Mangle::ManglingFlavor flavor) override {
    return m_real->CreateGenericTypeParamType(depth, index, flavor);
  }
  std::string GetSwiftName(const clang::Decl *clang_decl,
                           TypeSystemClang &clang_typesystem) override {
    return m_real->GetSwiftName(clang_decl, clang_typesystem);
  }
  CompilerType
  ConvertClangTypeToSwiftType(CompilerType clang_type,
                              swift::Mangle::ManglingFlavor flavor) override {
    return m_real->ConvertClangTypeToSwiftType(clang_type, flavor);
  }
  UserExpression *GetUserExpression(llvm::StringRef expr,
                                    llvm::StringRef prefix,
                                    SourceLanguage language,
                                    Expression::ResultType desired_type,
                                    const EvaluateExpressionOptions &options,
                                    ValueObject *ctx_obj) override {
    return m_real->GetUserExpression(expr, prefix, language, desired_type,
                                     options, ctx_obj);
  }
  PersistentExpressionState *GetPersistentExpressionState() override {
    return m_real->GetPersistentExpressionState();
  }
  ExecutionContextRef
  GetExecutionContextForType(lldb::opaque_compiler_type_t type) override {
    return m_real->GetExecutionContextForType(type);
  }

private:
  std::shared_ptr<TypeSystemSwiftTypeRefForExpressions> m_real;
};

} // namespace lldb_private

#endif // liblldb_TypeSystemSwiftTypeRefForExpressionsProxy_h_
