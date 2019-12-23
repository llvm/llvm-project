//===-- ClangExpressionDeclMapTest.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ExpressionParser/Clang/ClangExpressionDeclMap.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/Symbol/ClangTestUtils.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ClangUtil.h"
#include "lldb/lldb-defines.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb;

namespace {
struct FakeClangExpressionDeclMap : public ClangExpressionDeclMap {
  FakeClangExpressionDeclMap(const ClangASTImporterSP &importer)
      : ClangExpressionDeclMap(false, nullptr, lldb::TargetSP(), importer,
                               nullptr) {
    m_scratch_context = clang_utils::createAST();
  }
  std::unique_ptr<ClangASTContext> m_scratch_context;
  /// Adds a persistent decl that can be found by the ClangExpressionDeclMap
  /// via GetPersistentDecl.
  void AddPersistentDeclForTest(clang::NamedDecl *d) {
    // The declaration needs to have '$' prefix in its name like every
    // persistent declaration and must be inside the scratch AST context.
    assert(d);
    assert(d->getName().startswith("$"));
    assert(&d->getASTContext() == &m_scratch_context->getASTContext());
    m_persistent_decls[d->getName()] = d;
  }

protected:
  // ClangExpressionDeclMap hooks.

  clang::NamedDecl *GetPersistentDecl(ConstString name) override {
    // ClangExpressionDeclMap wants to know if there is a persistent decl
    // with the given name. Check the
    return m_persistent_decls.lookup(name.GetStringRef());
  }

private:
  /// The persistent decls in this test with their names as keys.
  llvm::DenseMap<llvm::StringRef, clang::NamedDecl *> m_persistent_decls;
};
} // namespace

namespace {
struct ClangExpressionDeclMapTest : public testing::Test {
  SubsystemRAII<FileSystem, HostInfo> subsystems;

  /// The ClangASTImporter used during the test.
  ClangASTImporterSP importer;
  /// The ExpressionDeclMap for the current test case.
  std::unique_ptr<FakeClangExpressionDeclMap> decl_map;

  /// The target AST that lookup results should be imported to.
  std::unique_ptr<ClangASTContext> target_ast;

  void SetUp() override {
    importer = std::make_shared<ClangASTImporter>();
    decl_map = std::make_unique<FakeClangExpressionDeclMap>(importer);
    target_ast = clang_utils::createAST();
    decl_map->InstallASTContext(*target_ast);
  }

  void TearDown() override {
    importer.reset();
    decl_map.reset();
    target_ast.reset();
  }
};
} // namespace

TEST_F(ClangExpressionDeclMapTest, TestUnknownIdentifierLookup) {
  // Tests looking up an identifier that can't be found anywhere.

  // Setup a NameSearchContext for 'foo'.
  llvm::SmallVector<clang::NamedDecl *, 16> decls;
  clang::DeclarationName name =
      clang_utils::getDeclarationName(*target_ast, "foo");
  const clang::DeclContext *dc = target_ast->GetTranslationUnitDecl();
  NameSearchContext search(*decl_map, decls, name, dc);

  decl_map->FindExternalVisibleDecls(search);

  // This shouldn't exist so we should get no lookups.
  EXPECT_EQ(0U, decls.size());
}

TEST_F(ClangExpressionDeclMapTest, TestPersistentDeclLookup) {
  // Tests looking up a persistent decl from the scratch AST context.

  // Create a '$persistent_class' record and add it as a persistent variable
  // to the scratch AST context.
  llvm::StringRef decl_name = "$persistent_class";
  CompilerType persistent_type =
      clang_utils::createRecord(*decl_map->m_scratch_context, decl_name);
  decl_map->AddPersistentDeclForTest(ClangUtil::GetAsTagDecl(persistent_type));

  // Setup a NameSearchContext for $persistent_class;
  llvm::SmallVector<clang::NamedDecl *, 16> decls;
  clang::DeclarationName name =
      clang_utils::getDeclarationName(*target_ast, decl_name);
  const clang::DeclContext *dc = target_ast->GetTranslationUnitDecl();
  NameSearchContext search(*decl_map, decls, name, dc);

  // Search and check that we found $persistent_class.
  decl_map->FindExternalVisibleDecls(search);
  EXPECT_EQ(1U, decls.size());
  EXPECT_EQ(decl_name, decls.front()->getQualifiedNameAsString());
  auto *record = llvm::cast<clang::RecordDecl>(decls.front());
  // The class was minimally imported from the scratch AST context.
  EXPECT_TRUE(record->hasExternalLexicalStorage());
}
