//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_NATIVEPDB_PDBASTBUILDERCLANG_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_NATIVEPDB_PDBASTBUILDERCLANG_H

#include "PdbAstBuilder.h"

#include "Plugins/ExpressionParser/Clang/ClangASTImporter.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/DebugInfo/CodeView/CVRecord.h"
#include "llvm/Support/Threading.h"

namespace clang {
class TagDecl;
class DeclContext;
class Decl;
class QualType;
class FunctionDecl;
class NamespaceDecl;
class BlockDecl;
class VarDecl;
} // namespace clang

namespace llvm {
namespace codeview {
class PointerRecord;
class ModifierRecord;
class ArrayRecord;
class TagRecord;
class EnumRecord;
enum class CallingConvention : uint8_t;
} // namespace codeview
} // namespace llvm

namespace lldb_private {
namespace npdb {

struct DeclStatus {
  DeclStatus() = default;
  DeclStatus(lldb::user_id_t uid, bool resolved)
      : uid(uid), resolved(resolved) {}
  lldb::user_id_t uid = 0;
  bool resolved = false;
};

class PdbAstBuilderClang : public PdbAstBuilder {
public:
  PdbAstBuilderClang(TypeSystemClang &clang);

  CompilerDecl GetOrCreateDeclForUid(PdbSymUid uid) override;
  CompilerDeclContext GetOrCreateDeclContextForUid(PdbSymUid uid) override;
  CompilerDeclContext GetParentDeclContext(PdbSymUid uid) override;

  void EnsureFunction(PdbCompilandSymId func_id) override;
  void EnsureInlinedFunction(PdbCompilandSymId inlinesite_id) override;
  void EnsureBlock(PdbCompilandSymId block_id) override;
  void EnsureVariable(PdbCompilandSymId scope_id,
                      PdbCompilandSymId var_id) override;
  void EnsureVariable(PdbGlobalSymId var_id) override;

  CompilerType GetOrCreateType(PdbTypeSymId type) override;
  CompilerType GetOrCreateTypedefType(PdbGlobalSymId id) override;
  bool CompleteType(CompilerType ct) override;

  void ParseDeclsForContext(CompilerDeclContext context) override;

  CompilerDeclContext FindNamespaceDecl(CompilerDeclContext parent_ctx,
                                        llvm::StringRef name) override;

  void Dump(Stream &stream, llvm::StringRef filter, bool show_color) override;

  // Clang-specific
  clang::QualType GetBasicType(lldb::BasicType type);
  clang::QualType GetOrCreateClangType(PdbTypeSymId type);
  clang::DeclContext *GetOrCreateClangDeclContextForUid(PdbSymUid uid);

  CompilerDecl ToCompilerDecl(clang::Decl *decl);
  CompilerType ToCompilerType(clang::QualType qt);
  CompilerDeclContext ToCompilerDeclContext(clang::DeclContext *context);
  clang::QualType FromCompilerType(CompilerType ct);
  clang::Decl *FromCompilerDecl(CompilerDecl decl);
  clang::DeclContext *FromCompilerDeclContext(CompilerDeclContext context);

  bool CompleteTagDecl(clang::TagDecl &tag);

  TypeSystemClang &clang() { return m_clang; }
  ClangASTImporter &GetClangASTImporter() { return m_importer; }

private:
  CompilerDeclContext GetTranslationUnitDecl();
  clang::DeclContext *GetParentClangDeclContext(PdbSymUid uid);

  clang::Decl *TryGetDecl(PdbSymUid uid) const;

  clang::FunctionDecl *GetOrCreateFunctionDecl(PdbCompilandSymId func_id);
  clang::FunctionDecl *
  GetOrCreateInlinedFunctionDecl(PdbCompilandSymId inlinesite_id);
  clang::BlockDecl *GetOrCreateBlockDecl(PdbCompilandSymId block_id);
  clang::VarDecl *GetOrCreateVariableDecl(PdbCompilandSymId scope_id,
                                          PdbCompilandSymId var_id);
  clang::VarDecl *GetOrCreateVariableDecl(PdbGlobalSymId var_id);

  using TypeIndex = llvm::codeview::TypeIndex;

  clang::QualType
  CreatePointerType(const llvm::codeview::PointerRecord &pointer);
  clang::QualType
  CreateModifierType(const llvm::codeview::ModifierRecord &modifier);
  clang::QualType CreateArrayType(const llvm::codeview::ArrayRecord &array);
  clang::QualType CreateRecordType(PdbTypeSymId id,
                                   const llvm::codeview::TagRecord &record);
  clang::QualType CreateEnumType(PdbTypeSymId id,
                                 const llvm::codeview::EnumRecord &record);
  clang::QualType
  CreateFunctionType(TypeIndex args_type_idx, TypeIndex return_type_idx,
                     llvm::codeview::CallingConvention calling_convention);
  clang::QualType CreateType(PdbTypeSymId type);

  void CreateFunctionParameters(PdbCompilandSymId func_id,
                                clang::FunctionDecl &function_decl,
                                uint32_t param_count);
  clang::Decl *GetOrCreateSymbolForId(PdbCompilandSymId id);
  clang::VarDecl *CreateVariableDecl(PdbSymUid uid,
                                     llvm::codeview::CVSymbol sym,
                                     clang::DeclContext &scope);
  clang::NamespaceDecl *GetOrCreateNamespaceDecl(const char *name,
                                                 clang::DeclContext &context);
  clang::FunctionDecl *CreateFunctionDeclFromId(PdbTypeSymId func_tid,
                                                PdbCompilandSymId func_sid);
  clang::FunctionDecl *
  CreateFunctionDecl(PdbCompilandSymId func_id, llvm::StringRef func_name,
                     TypeIndex func_ti, CompilerType func_ct,
                     uint32_t param_count, clang::StorageClass func_storage,
                     bool is_inline, clang::DeclContext *parent);
  void ParseNamespace(clang::DeclContext &parent);
  void ParseAllTypes();
  void ParseAllFunctionsAndNonLocalVars();
  void ParseDeclsForSimpleContext(clang::DeclContext &context);
  void ParseBlockChildren(PdbCompilandSymId block_id);

  std::pair<clang::DeclContext *, std::string>
  CreateDeclInfoForType(const llvm::codeview::TagRecord &record, TypeIndex ti);
  std::pair<clang::DeclContext *, std::string>
  CreateDeclInfoForUndecoratedName(llvm::StringRef uname);
  clang::QualType CreateSimpleType(TypeIndex ti);

  TypeSystemClang &m_clang;

  ClangASTImporter m_importer;
  llvm::once_flag m_parse_functions_and_non_local_vars;
  llvm::once_flag m_parse_all_types;
  llvm::DenseMap<clang::Decl *, DeclStatus> m_decl_to_status;
  llvm::DenseMap<lldb::user_id_t, clang::Decl *> m_uid_to_decl;
  llvm::DenseMap<lldb::user_id_t, clang::QualType> m_uid_to_type;

  // From class/struct's opaque_compiler_type_t to a set containing the pairs of
  // method's name and CompilerType.
  llvm::DenseMap<lldb::opaque_compiler_type_t,
                 llvm::SmallSet<std::pair<llvm::StringRef, CompilerType>, 8>>
      m_cxx_record_map;

  using NamespaceSet = llvm::DenseSet<clang::NamespaceDecl *>;

  // These namespaces are fully parsed
  NamespaceSet m_parsed_namespaces;

  // We know about these namespaces, but they might not be completely parsed yet
  NamespaceSet m_known_namespaces;
  llvm::DenseMap<clang::DeclContext *, NamespaceSet> m_parent_to_namespaces;
};

} // namespace npdb
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_NATIVEPDB_PDBASTBUILDERCLANG_H
