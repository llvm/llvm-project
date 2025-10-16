//===-- TestClangASTImporter.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "Plugins/ExpressionParser/Clang/ClangASTImporter.h"
#include "Plugins/ExpressionParser/Clang/ClangASTMetadata.h"
#include "Plugins/ExpressionParser/Clang/ClangUtil.h"
#include "Plugins/SymbolFile/DWARF/DWARFASTParserClang.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/Symbol/ClangTestUtils.h"
#include "lldb/Core/Declaration.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
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

  auto holder =
      std::make_unique<clang_utils::TypeSystemClangHolder>("target ast");
  auto *target_ast = holder->GetAST();

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

  auto holder =
      std::make_unique<clang_utils::TypeSystemClangHolder>("target ast");
  auto *target_ast = holder->GetAST();

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

TEST_F(TestClangASTImporter, CompleteFwdDeclWithOtherOrigin) {
  // Create an AST with a full type that is defined.
  clang_utils::SourceASTWithRecord source_with_definition;

  // Create an AST with a type thst is only a forward declaration with the
  // same name as the one in the other source.
  auto holder =
      std::make_unique<clang_utils::TypeSystemClangHolder>("ast");
  auto *fwd_decl_source = holder->GetAST();
  CompilerType fwd_decl_type = clang_utils::createRecord(
      *fwd_decl_source, source_with_definition.record_decl->getName());

  // Create a target and import the forward decl.
  auto target_holder =
      std::make_unique<clang_utils::TypeSystemClangHolder>("target ast");
  auto *target = target_holder->GetAST();
  ClangASTImporter importer;
  CompilerType imported = importer.CopyType(*target, fwd_decl_type);
  ASSERT_TRUE(imported.IsValid());
  EXPECT_FALSE(imported.IsDefined());

  // Now complete the forward decl with the definition from the other source.
  // This should define the decl and give it the fields of the other origin.
  clang::TagDecl *imported_tag_decl = ClangUtil::GetAsTagDecl(imported);
  importer.CompleteTagDeclWithOrigin(imported_tag_decl,
                                     source_with_definition.record_decl);
  ASSERT_TRUE(imported.IsValid());
  EXPECT_TRUE(imported.IsDefined());
  EXPECT_EQ(1U, imported.GetNumFields());
}

TEST_F(TestClangASTImporter, DeportDeclTagDecl) {
  // Tests that the ClangASTImporter::DeportDecl completely copies TagDecls.
  clang_utils::SourceASTWithRecord source;

  auto holder =
      std::make_unique<clang_utils::TypeSystemClangHolder>("target ast");
  auto *target_ast = holder->GetAST();

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

  auto holder =
      std::make_unique<clang_utils::TypeSystemClangHolder>("target ast");
  auto *target_ast = holder->GetAST();

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

  auto holder =
      std::make_unique<clang_utils::TypeSystemClangHolder>("target ast");
  auto *target_ast = holder->GetAST();

  ClangASTImporter importer;
  clang::Decl *imported =
      importer.CopyDecl(&target_ast->getASTContext(), source.record_decl);
  ASSERT_NE(nullptr, imported);

  // Check that we got the same Metadata.
  ASSERT_NE(std::nullopt, importer.GetDeclMetadata(imported));
  EXPECT_EQ(metadata, importer.GetDeclMetadata(imported)->GetUserID());
}

TEST_F(TestClangASTImporter, MetadataPropagationIndirectImport) {
  // Tests that AST metadata is propagated when copying declarations when
  // importing one declaration into a temporary context and then to the
  // actual destination context.

  clang_utils::SourceASTWithRecord source;

  const lldb::user_id_t metadata = 123456;
  source.ast->SetMetadataAsUserID(source.record_decl, metadata);

  auto tmp_holder =
      std::make_unique<clang_utils::TypeSystemClangHolder>("tmp ast");
  auto *temporary_ast = tmp_holder->GetAST();

  ClangASTImporter importer;
  clang::Decl *temporary_imported =
      importer.CopyDecl(&temporary_ast->getASTContext(), source.record_decl);
  ASSERT_NE(nullptr, temporary_imported);

  auto holder =
      std::make_unique<clang_utils::TypeSystemClangHolder>("target ast");
  auto *target_ast = holder->GetAST();
  clang::Decl *imported =
      importer.CopyDecl(&target_ast->getASTContext(), temporary_imported);
  ASSERT_NE(nullptr, imported);

  // Check that we got the same Metadata.
  ASSERT_NE(std::nullopt, importer.GetDeclMetadata(imported));
  EXPECT_EQ(metadata, importer.GetDeclMetadata(imported)->GetUserID());
}

TEST_F(TestClangASTImporter, MetadataPropagationAfterCopying) {
  // Tests that AST metadata is propagated when copying declarations even
  // when the metadata was set after the declaration has already been copied.

  clang_utils::SourceASTWithRecord source;
  const lldb::user_id_t metadata = 123456;

  auto holder =
      std::make_unique<clang_utils::TypeSystemClangHolder>("target ast");
  auto *target_ast = holder->GetAST();

  ClangASTImporter importer;
  clang::Decl *imported =
      importer.CopyDecl(&target_ast->getASTContext(), source.record_decl);
  ASSERT_NE(nullptr, imported);

  // The TagDecl has been imported. Now set the metadata of the source and
  // make sure the imported one will directly see it.
  source.ast->SetMetadataAsUserID(source.record_decl, metadata);

  // Check that we got the same Metadata.
  ASSERT_NE(std::nullopt, importer.GetDeclMetadata(imported));
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

TEST_F(TestClangASTImporter, RecordLayoutFromOrigin) {
  // Tests that we can retrieve the layout of a record that has
  // an origin with an already existing LayoutInfo. We expect
  // the layout to be retrieved from the ClangASTImporter of the
  // origin decl.

  clang_utils::SourceASTWithRecord source;

  auto *dwarf_parser =
      static_cast<DWARFASTParserClang *>(source.ast->GetDWARFParser());
  auto &importer = dwarf_parser->GetClangASTImporter();

  // Set the layout for the origin decl in the origin ClangASTImporter.
  ClangASTImporter::LayoutInfo layout_info;
  layout_info.bit_size = 32;
  layout_info.alignment = 16;
  layout_info.field_offsets[source.field_decl] = 1;
  importer.SetRecordLayout(source.record_decl, layout_info);

  auto holder =
      std::make_unique<clang_utils::TypeSystemClangHolder>("target ast");
  auto *target_ast = holder->GetAST();

  // Import the decl into a new TypeSystemClang.
  CompilerType imported = importer.CopyType(*target_ast, source.record_type);
  ASSERT_TRUE(imported.IsValid());

  auto *imported_decl = cast<CXXRecordDecl>(ClangUtil::GetAsTagDecl(imported));
  ClangASTImporter::DeclOrigin origin = importer.GetDeclOrigin(imported_decl);
  ASSERT_TRUE(origin.Valid());
  ASSERT_EQ(origin.decl, source.record_decl);

  uint64_t bit_size;
  uint64_t alignment;
  llvm::DenseMap<const clang::FieldDecl *, uint64_t> field_offsets;
  llvm::DenseMap<const clang::CXXRecordDecl *, clang::CharUnits> base_offsets;
  llvm::DenseMap<const clang::CXXRecordDecl *, clang::CharUnits> vbase_offsets;

  // Make sure we correctly read out the layout (despite not Having
  // called SetRecordLayout on the new TypeSystem's ClangASTImporter).
  auto success =
      importer.LayoutRecordType(imported_decl, bit_size, alignment,
                                field_offsets, base_offsets, vbase_offsets);
  EXPECT_TRUE(success);

  EXPECT_EQ(32U, bit_size);
  EXPECT_EQ(16U, alignment);
  EXPECT_EQ(1U, field_offsets.size());
  EXPECT_EQ(1U, field_offsets[*imported_decl->field_begin()]);
  EXPECT_EQ(0U, base_offsets.size());
  EXPECT_EQ(0U, vbase_offsets.size());
}
