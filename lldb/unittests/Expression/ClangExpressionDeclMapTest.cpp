//===-- ClangExpressionDeclMapTest.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ExpressionParser/Clang/ClangExpressionDeclMap.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/lldb-defines.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb;

namespace {
struct ClangExpressionDeclMapTest : public testing::Test {
  static void SetUpTestCase() {
    FileSystem::Initialize();
    HostInfo::Initialize();
  }
  static void TearDownTestCase() {
    HostInfo::Terminate();
    FileSystem::Terminate();
  }

  std::unique_ptr<ClangASTContext> createAST() {
    return std::make_unique<ClangASTContext>(HostInfo::GetTargetTriple());
  }

  clang::DeclarationName getDeclarationName(ClangASTContext &ast,
                                            llvm::StringRef name) {
    clang::IdentifierInfo &II = ast.getIdentifierTable()->get(name);
    return ast.getASTContext()->DeclarationNames.getIdentifier(&II);
  }
};
} // namespace

TEST_F(ClangExpressionDeclMapTest, TestIdentifierLookupInEmptyTU) {
  ClangASTImporterSP importer = std::make_shared<ClangASTImporter>();
  ClangExpressionDeclMap map(false, nullptr, lldb::TargetSP(), importer,
                             nullptr);

  std::unique_ptr<ClangASTContext> ast = createAST();
  map.InstallASTContext(*ast, *ast->getFileManager());

  llvm::SmallVector<clang::NamedDecl *, 16> decls;
  clang::DeclarationName name = getDeclarationName(*ast, "does_no_exist");
  const clang::DeclContext *dc = ast->GetTranslationUnitDecl();

  NameSearchContext search(map, decls, name, dc);
  map.FindExternalVisibleDecls(search);

  EXPECT_EQ(0U, decls.size());
}
