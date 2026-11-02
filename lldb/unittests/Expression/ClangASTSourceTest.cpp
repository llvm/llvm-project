//===-- ClangASTSourceTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ExpressionParser/Clang/ClangASTSource.h"
#include "Plugins/ExpressionParser/Clang/ASTUtils.h"
#include "Plugins/ExpressionParser/Clang/ClangASTImporter.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/Symbol/ClangTestUtils.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb;

namespace {
/// Records the qualified lookup notifications it receives.
struct QualifiedLookupRecorder : public clang::ExternalSemaSource {
  void StartedQualifiedLookup(const clang::DeclContext *DC) override {
    started.push_back(DC);
  }
  void FinishedQualifiedLookup(const clang::DeclContext *DC) override {
    finished.push_back(DC);
  }
  std::vector<const clang::DeclContext *> started;
  std::vector<const clang::DeclContext *> finished;
};

struct ClangASTSourceTest : public testing::Test {
  SubsystemRAII<FileSystem, HostInfo> subsystems;

  std::shared_ptr<ClangASTImporter> importer;
  std::unique_ptr<clang_utils::TypeSystemClangHolder> holder;
  TypeSystemClang *ast = nullptr;
  std::unique_ptr<ClangASTSource> source;

  clang::DeclContext *tu = nullptr;
  clang::DeclContext *ns_a = nullptr;
  clang::DeclContext *ns_b = nullptr;

  void SetUp() override {
    importer = std::make_shared<ClangASTImporter>();
    holder = std::make_unique<clang_utils::TypeSystemClangHolder>("ast");
    ast = holder->GetAST();
    source = std::make_unique<ClangASTSource>(lldb::TargetSP(), importer);
    source->InstallASTContext(*ast);

    tu = ast->getASTContext().getTranslationUnitDecl();
    ns_a = ast->GetUniqueNamespaceDeclaration("A", tu, OptionalClangModuleID());
    ns_b = ast->GetUniqueNamespaceDeclaration("B", tu, OptionalClangModuleID());
  }

  void TearDown() override {
    source.reset();
    holder.reset();
    importer.reset();
  }
};
} // namespace

TEST_F(ClangASTSourceTest, NoQualifiedLookupByDefault) {
  EXPECT_FALSE(source->IsInQualifiedLookup(tu));
  EXPECT_FALSE(source->IsInQualifiedLookup(ns_a));
}

TEST_F(ClangASTSourceTest, StartAndFinishQualifiedLookup) {
  source->StartedQualifiedLookup(tu);
  EXPECT_TRUE(source->IsInQualifiedLookup(tu));
  EXPECT_FALSE(source->IsInQualifiedLookup(ns_a));

  source->FinishedQualifiedLookup(tu);
  EXPECT_FALSE(source->IsInQualifiedLookup(tu));
}

TEST_F(ClangASTSourceTest, NestedQualifiedLookupsInDifferentContexts) {
  source->StartedQualifiedLookup(tu);
  source->StartedQualifiedLookup(ns_a);
  EXPECT_TRUE(source->IsInQualifiedLookup(tu));
  EXPECT_TRUE(source->IsInQualifiedLookup(ns_a));
  EXPECT_FALSE(source->IsInQualifiedLookup(ns_b));

  // Finishing the inner lookup must keep the outer one active.
  source->FinishedQualifiedLookup(ns_a);
  EXPECT_TRUE(source->IsInQualifiedLookup(tu));
  EXPECT_FALSE(source->IsInQualifiedLookup(ns_a));

  source->FinishedQualifiedLookup(tu);
  EXPECT_FALSE(source->IsInQualifiedLookup(tu));
}

TEST_F(ClangASTSourceTest, NestedQualifiedLookupsInSameContext) {
  // Sema can re-enter a qualified lookup into the same DeclContext. Finishing
  // the inner lookup must not end the outer one.
  source->StartedQualifiedLookup(ns_a);
  source->StartedQualifiedLookup(ns_a);
  EXPECT_TRUE(source->IsInQualifiedLookup(ns_a));

  source->FinishedQualifiedLookup(ns_a);
  EXPECT_TRUE(source->IsInQualifiedLookup(ns_a));

  source->FinishedQualifiedLookup(ns_a);
  EXPECT_FALSE(source->IsInQualifiedLookup(ns_a));
}

TEST_F(ClangASTSourceTest, ProxyForwardsQualifiedLookup) {
  llvm::IntrusiveRefCntPtr<clang::ExternalASTSource> proxy =
      source->CreateProxy();
  // The proxy has to be an ExternalSemaSource, otherwise Sema can't notify it.
  auto *sema_proxy = llvm::dyn_cast<clang::ExternalSemaSource>(proxy.get());
  ASSERT_NE(sema_proxy, nullptr);

  sema_proxy->StartedQualifiedLookup(ns_a);
  EXPECT_TRUE(source->IsInQualifiedLookup(ns_a));
  sema_proxy->FinishedQualifiedLookup(ns_a);
  EXPECT_FALSE(source->IsInQualifiedLookup(ns_a));
}

TEST_F(ClangASTSourceTest, WrapperForwardsQualifiedLookupToProxy) {
  // This mirrors how ClangExpressionParser hands the source to Sema.
  auto wrapper = llvm::makeIntrusiveRefCnt<ExternalASTSourceWrapper>(
      source->CreateProxy());

  wrapper->StartedQualifiedLookup(tu);
  EXPECT_TRUE(source->IsInQualifiedLookup(tu));
  wrapper->FinishedQualifiedLookup(tu);
  EXPECT_FALSE(source->IsInQualifiedLookup(tu));
}

TEST_F(ClangASTSourceTest, WrapperIgnoresNonSemaSource) {
  // Wrapping a plain ExternalASTSource must not crash when Sema reports
  // qualified lookups.
  auto plain = llvm::makeIntrusiveRefCnt<clang::ExternalASTSource>();
  auto wrapper = llvm::makeIntrusiveRefCnt<ExternalASTSourceWrapper>(plain);
  wrapper->StartedQualifiedLookup(tu);
  wrapper->FinishedQualifiedLookup(tu);
}

TEST_F(ClangASTSourceTest, SemaSourceWithPrioritiesForwardsToAllSources) {
  auto high = llvm::makeIntrusiveRefCnt<QualifiedLookupRecorder>();
  auto low = llvm::makeIntrusiveRefCnt<QualifiedLookupRecorder>();
  auto multiplexer =
      llvm::makeIntrusiveRefCnt<SemaSourceWithPriorities>(high, low);

  multiplexer->StartedQualifiedLookup(ns_a);
  multiplexer->FinishedQualifiedLookup(ns_a);

  std::vector<const clang::DeclContext *> expected = {ns_a};
  EXPECT_EQ(high->started, expected);
  EXPECT_EQ(high->finished, expected);
  EXPECT_EQ(low->started, expected);
  EXPECT_EQ(low->finished, expected);
}
