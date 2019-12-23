//===-- TestClangASTImporter.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/Symbol/ClangTestUtils.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ClangASTImporter.h"
#include "lldb/Symbol/ClangASTMetadata.h"
#include "lldb/Symbol/ClangUtil.h"
#include "lldb/Symbol/Declaration.h"
#include "clang/AST/DeclCXX.h"

using namespace clang;
using namespace lldb;
using namespace lldb_private;

class TestClangASTImporter : public testing::Test {
public:
  SubsystemRAII<FileSystem, HostInfo> subsystems;
};

TEST_F(TestClangASTImporter, CanImportInvalidType) {
  ClangASTImporter importer;
  EXPECT_FALSE(importer.CanImport(CompilerType()));
}

TEST_F(TestClangASTImporter, ImportInvalidType) {
  ClangASTImporter importer;
  EXPECT_FALSE(importer.Import(CompilerType()));
}

TEST_F(TestClangASTImporter, CopyDeclTagDecl) {
  // Tests that the ClangASTImporter::CopyDecl can copy TagDecls.
  clang_utils::SourceASTWithRecord source;

  std::unique_ptr<ClangASTContext> target_ast = clang_utils::createAST();

  ClangASTImporter importer;
  clang::Decl *imported =
      importer.CopyDecl(&target_ast->getASTContext(), source.record_decl);
  ASSERT_NE(nullptr, imported);

  // Check that we got the correct decl by just comparing their qualified name.
  clang::TagDecl *imported_tag_decl = llvm::cast<clang::TagDecl>(imported);
  EXPECT_EQ(source.record_decl->getQualifiedNameAsString(),
            imported_tag_decl->getQualifiedNameAsString());
  // We did a minimal import of the tag decl.
  EXPECT_TRUE(imported_tag_decl->hasExternalLexicalStorage());

  // Check that origin was set for the imported declaration.
  ClangASTImporter::DeclOrigin origin = importer.GetDeclOrigin(imported);
  EXPECT_TRUE(origin.Valid());
  EXPECT_EQ(origin.ctx, &source.ast->getASTContext());
  EXPECT_EQ(origin.decl, source.record_decl);
}

TEST_F(TestClangASTImporter, CopyTypeTagDecl) {
  // Tests that the ClangASTImporter::CopyType can copy TagDecls types.
  clang_utils::SourceASTWithRecord source;

  std::unique_ptr<ClangASTContext> target_ast = clang_utils::createAST();

  ClangASTImporter importer;
  CompilerType imported = importer.CopyType(*target_ast, source.record_type);
  ASSERT_TRUE(imported.IsValid());

  // Check that we got the correct decl by just comparing their qualified name.
  clang::TagDecl *imported_tag_decl = ClangUtil::GetAsTagDecl(imported);
  EXPECT_EQ(source.record_decl->getQualifiedNameAsString(),
            imported_tag_decl->getQualifiedNameAsString());
  // We did a minimal import of the tag decl.
  EXPECT_TRUE(imported_tag_decl->hasExternalLexicalStorage());

  // Check that origin was set for the imported declaration.
  ClangASTImporter::DeclOrigin origin =
      importer.GetDeclOrigin(imported_tag_decl);
  EXPECT_TRUE(origin.Valid());
  EXPECT_EQ(origin.ctx, &source.ast->getASTContext());
  EXPECT_EQ(origin.decl, source.record_decl);
}

TEST_F(TestClangASTImporter, DeportDeclTagDecl) {
  // Tests that the ClangASTImporter::DeportDecl completely copies TagDecls.
  clang_utils::SourceASTWithRecord source;

  std::unique_ptr<ClangASTContext> target_ast = clang_utils::createAST();

  ClangASTImporter importer;
  clang::Decl *imported =
      importer.DeportDecl(&target_ast->getASTContext(), source.record_decl);
  ASSERT_NE(nullptr, imported);

  // Check that we got the correct decl by just comparing their qualified name.
  clang::TagDecl *imported_tag_decl = llvm::cast<clang::TagDecl>(imported);
  EXPECT_EQ(source.record_decl->getQualifiedNameAsString(),
            imported_tag_decl->getQualifiedNameAsString());
  // The record should be completed as we deported it.
  EXPECT_FALSE(imported_tag_decl->hasExternalLexicalStorage());

  // Deporting doesn't update the origin map.
  EXPECT_FALSE(importer.GetDeclOrigin(imported_tag_decl).Valid());
}

TEST_F(TestClangASTImporter, DeportTypeTagDecl) {
  // Tests that the ClangASTImporter::CopyType can deport TagDecl types.
  clang_utils::SourceASTWithRecord source;

  std::unique_ptr<ClangASTContext> target_ast = clang_utils::createAST();

  ClangASTImporter importer;
  CompilerType imported = importer.DeportType(*target_ast, source.record_type);
  ASSERT_TRUE(imported.IsValid());

  // Check that we got the correct decl by just comparing their qualified name.
  clang::TagDecl *imported_tag_decl = ClangUtil::GetAsTagDecl(imported);
  EXPECT_EQ(source.record_decl->getQualifiedNameAsString(),
            imported_tag_decl->getQualifiedNameAsString());
  // The record should be completed as we deported it.
  EXPECT_FALSE(imported_tag_decl->hasExternalLexicalStorage());

  // Deporting doesn't update the origin map.
  EXPECT_FALSE(importer.GetDeclOrigin(imported_tag_decl).Valid());
}

TEST_F(TestClangASTImporter, MetadataPropagation) {
  // Tests that AST metadata is propagated when copying declarations.

  clang_utils::SourceASTWithRecord source;

  const lldb::user_id_t metadata = 123456;
  source.ast->SetMetadataAsUserID(source.record_decl, metadata);

  std::unique_ptr<ClangASTContext> target_ast = clang_utils::createAST();

  ClangASTImporter importer;
  clang::Decl *imported =
      importer.CopyDecl(&target_ast->getASTContext(), source.record_decl);
  ASSERT_NE(nullptr, imported);

  // Check that we got the same Metadata.
  ASSERT_NE(nullptr, importer.GetDeclMetadata(imported));
  EXPECT_EQ(metadata, importer.GetDeclMetadata(imported)->GetUserID());
}

TEST_F(TestClangASTImporter, MetadataPropagationIndirectImport) {
  // Tests that AST metadata is propagated when copying declarations when
  // importing one declaration into a temporary context and then to the
  // actual destination context.

  clang_utils::SourceASTWithRecord source;

  const lldb::user_id_t metadata = 123456;
  source.ast->SetMetadataAsUserID(source.record_decl, metadata);

  std::unique_ptr<ClangASTContext> temporary_ast = clang_utils::createAST();

  ClangASTImporter importer;
  clang::Decl *temporary_imported =
      importer.CopyDecl(&temporary_ast->getASTContext(), source.record_decl);
  ASSERT_NE(nullptr, temporary_imported);

  std::unique_ptr<ClangASTContext> target_ast = clang_utils::createAST();
  clang::Decl *imported =
      importer.CopyDecl(&target_ast->getASTContext(), temporary_imported);
  ASSERT_NE(nullptr, imported);

  // Check that we got the same Metadata.
  ASSERT_NE(nullptr, importer.GetDeclMetadata(imported));
  EXPECT_EQ(metadata, importer.GetDeclMetadata(imported)->GetUserID());
}

TEST_F(TestClangASTImporter, MetadataPropagationAfterCopying) {
  // Tests that AST metadata is propagated when copying declarations even
  // when the metadata was set after the declaration has already been copied.

  clang_utils::SourceASTWithRecord source;
  const lldb::user_id_t metadata = 123456;

  std::unique_ptr<ClangASTContext> target_ast = clang_utils::createAST();

  ClangASTImporter importer;
  clang::Decl *imported =
      importer.CopyDecl(&target_ast->getASTContext(), source.record_decl);
  ASSERT_NE(nullptr, imported);

  // The TagDecl has been imported. Now set the metadata of the source and
  // make sure the imported one will directly see it.
  source.ast->SetMetadataAsUserID(source.record_decl, metadata);

  // Check that we got the same Metadata.
  ASSERT_NE(nullptr, importer.GetDeclMetadata(imported));
  EXPECT_EQ(metadata, importer.GetDeclMetadata(imported)->GetUserID());
}

TEST_F(TestClangASTImporter, RecordLayout) {
  // Test that it is possible to register RecordDecl layouts and then later
  // correctly retrieve them.

  clang_utils::SourceASTWithRecord source;

  ClangASTImporter importer;
  ClangASTImporter::LayoutInfo layout_info;
  layout_info.bit_size = 15;
  layout_info.alignment = 2;
  layout_info.field_offsets[source.field_decl] = 1;
  importer.SetRecordLayout(source.record_decl, layout_info);

  uint64_t bit_size;
  uint64_t alignment;
  llvm::DenseMap<const clang::FieldDecl *, uint64_t> field_offsets;
  llvm::DenseMap<const clang::CXXRecordDecl *, clang::CharUnits> base_offsets;
  llvm::DenseMap<const clang::CXXRecordDecl *, clang::CharUnits> vbase_offsets;
  importer.LayoutRecordType(source.record_decl, bit_size, alignment,
                            field_offsets, base_offsets, vbase_offsets);

  EXPECT_EQ(15U, bit_size);
  EXPECT_EQ(2U, alignment);
  EXPECT_EQ(1U, field_offsets.size());
  EXPECT_EQ(1U, field_offsets[source.field_decl]);
  EXPECT_EQ(0U, base_offsets.size());
  EXPECT_EQ(0U, vbase_offsets.size());
}
