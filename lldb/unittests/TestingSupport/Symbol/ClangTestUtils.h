//===- ClangTestUtils.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UNITTESTS_TESTINGSUPPORT_SYMBOL_CLANGTESTUTILS_H
#define LLDB_UNITTESTS_TESTINGSUPPORT_SYMBOL_CLANGTESTUTILS_H

#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ClangUtil.h"

namespace lldb_private {
namespace clang_utils {
inline clang::DeclarationName getDeclarationName(ClangASTContext &ast,
                                                 llvm::StringRef name) {
  clang::IdentifierInfo &II = ast.getASTContext().Idents.get(name);
  return ast.getASTContext().DeclarationNames.getIdentifier(&II);
}

inline std::unique_ptr<ClangASTContext> createAST() {
  return std::make_unique<ClangASTContext>(HostInfo::GetTargetTriple());
}

inline CompilerType createRecord(ClangASTContext &ast, llvm::StringRef name) {
  return ast.CreateRecordType(ast.getASTContext().getTranslationUnitDecl(),
                              lldb::AccessType::eAccessPublic, name, 0,
                              lldb::LanguageType::eLanguageTypeC);
}

/// Create a record with the given name and a field with the given type
/// and name.
inline CompilerType createRecordWithField(ClangASTContext &ast,
                                          llvm::StringRef record_name,
                                          CompilerType field_type,
                                          llvm::StringRef field_name) {
  CompilerType t = createRecord(ast, record_name);

  ClangASTContext::StartTagDeclarationDefinition(t);
  ast.AddFieldToRecordType(t, field_name, field_type,
                           lldb::AccessType::eAccessPublic, 7);
  ClangASTContext::CompleteTagDeclarationDefinition(t);

  return t;
}

/// Constructs a ClangASTContext that contains a single RecordDecl that contains
/// a single FieldDecl. Utility class as this setup is a common starting point
/// for unit test that exercise the ASTImporter.
struct SourceASTWithRecord {
  std::unique_ptr<ClangASTContext> ast;
  CompilerType record_type;
  clang::RecordDecl *record_decl = nullptr;
  clang::FieldDecl *field_decl = nullptr;
  SourceASTWithRecord() {
    ast = createAST();
    record_type = createRecordWithField(
        *ast, "Source", ast->GetBasicType(lldb::BasicType::eBasicTypeChar),
        "a_field");
    record_decl =
        llvm::cast<clang::RecordDecl>(ClangUtil::GetAsTagDecl(record_type));
    field_decl = *record_decl->fields().begin();
    assert(field_decl);
  }
};

} // namespace clang_utils
} // namespace lldb_private

#endif
