//===-- TestClangRedeclarations.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ExpressionParser/Clang/ClangASTMetadata.h"
#include "Plugins/ExpressionParser/Clang/ClangUtil.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/Symbol/ClangTestUtils.h"
#include "lldb/Core/Declaration.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Type.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace lldb;
using namespace lldb_private;

struct TestClangRedeclarations : public testing::Test {
  SubsystemRAII<FileSystem, HostInfo> subsystems;

  void SetUp() override {
    m_holder =
        std::make_unique<clang_utils::TypeSystemClangHolder>("test ASTContext");
    m_ast = m_holder->GetAST();
  }

  void TearDown() override {
    m_ast = nullptr;
    m_holder.reset();
  }

  TypeSystemClang *m_ast = nullptr;
  std::unique_ptr<clang_utils::TypeSystemClangHolder> m_holder;
};

TEST_F(TestClangRedeclarations, RedeclareCppClass) {
  // Test redeclaring C++ classes.

  OptionalClangModuleID module_id(1);
  CompilerType class_type = m_ast->CreateRecordType(
      m_ast->GetTranslationUnitDecl(), module_id, lldb::eAccessNone, "A",
      llvm::to_underlying(TagTypeKind::Class), lldb::eLanguageTypeC_plus_plus);
  auto *record = llvm::cast<CXXRecordDecl>(ClangUtil::GetAsTagDecl(class_type));

  m_ast->CreateRedeclaration(class_type);
  m_ast->StartTagDeclarationDefinition(class_type);
  // For C++ classes, the definition is already available (but it is currently
  // being defined). Make sure the definition is last in the redecl chain.
  CXXRecordDecl *def = record->getDefinition();
  ASSERT_TRUE(def);
  ASSERT_TRUE(def->isBeingDefined());
  ASSERT_NE(def, record);
  EXPECT_EQ(def->getPreviousDecl(), record);

  // Add a method.
  std::vector<CompilerType> args;
  CompilerType func_type =
      m_ast->CreateFunctionType(m_ast->GetBasicType(lldb::eBasicTypeInt),
                                args.data(), args.size(), /*is_variadic=*/false,
                                /*type_quals=*/0, clang::CallingConv::CC_C);
  const bool is_virtual = false;
  const bool is_static = false;
  const bool is_inline = false;
  const bool is_explicit = false;
  const bool is_attr_used = false;
  const bool is_artificial = false;
  clang::CXXMethodDecl *method = m_ast->AddMethodToCXXRecordType(
      class_type.GetOpaqueQualType(), "A", nullptr, func_type,
      lldb::eAccessPublic, is_virtual, is_static, is_inline, is_explicit,
      is_attr_used, is_artificial);
  // Check that the method was created and is in the definition.
  ASSERT_NE(method, nullptr);
  EXPECT_EQ(method->getParent(), def);

  // Add an ivar and check that it was added to the definition.
  FieldDecl *member_var = m_ast->AddFieldToRecordType(
      class_type, "f", m_ast->GetBasicType(lldb::eBasicTypeInt),
      lldb::eAccessPublic,
      /*bitfield_bit_size=*/0);
  ASSERT_TRUE(member_var);
  EXPECT_EQ(member_var->getParent(), def);

  // Complete the class and check that the last decl is the definition.
  m_ast->CompleteTagDeclarationDefinition(class_type);
  EXPECT_FALSE(record->isThisDeclarationADefinition());
  EXPECT_TRUE(def->isThisDeclarationADefinition());
  // Check that the module is identical.
  EXPECT_EQ(def->getOwningModuleID(), module_id.GetValue());

  // Make sure forward decl and definition have the same type.
  EXPECT_EQ(def->getTypeForDecl(), record->getTypeForDecl());
}

TEST_F(TestClangRedeclarations, RedeclareCppTemplateClass) {
  // Test redeclaring C++ template classes.

  OptionalClangModuleID module_id(1);
  auto args = std::make_unique<TypeSystemClang::TemplateParameterInfos>();
  args->InsertArg("T", TemplateArgument(m_ast->getASTContext().IntTy));

  ClassTemplateDecl *template_decl = m_ast->CreateClassTemplateDecl(
      m_ast->GetTranslationUnitDecl(), module_id, lldb::eAccessNone, "A",
      llvm::to_underlying(TagTypeKind::Struct), *args);
  ClassTemplateSpecializationDecl *fwd_decl =
      m_ast->CreateClassTemplateSpecializationDecl(
          m_ast->GetTranslationUnitDecl(), module_id, template_decl,
          llvm::to_underlying(TagTypeKind::Struct), *args);
  CompilerType spec_type =
      m_ast->CreateClassTemplateSpecializationType(fwd_decl);

  // Delete the TemplateParameterInfos to make sure TypeSystemClang doesn't
  // rely on the caller to keep them around.
  args.reset();

  m_ast->CreateRedeclaration(spec_type);
  m_ast->StartTagDeclarationDefinition(spec_type);
  // For C++ classes, the definition is already available (but it is currently
  // being defined). Make sure the definition is last in the redecl chain.
  CXXRecordDecl *def = fwd_decl->getDefinition();
  ASSERT_TRUE(def);
  ASSERT_TRUE(def->isBeingDefined());
  ASSERT_NE(def, fwd_decl);
  EXPECT_EQ(def->getPreviousDecl(), fwd_decl);

  // Add an ivar and check that it was added to the definition.
  FieldDecl *member_var = m_ast->AddFieldToRecordType(
      spec_type, "f", m_ast->GetBasicType(lldb::eBasicTypeInt),
      lldb::eAccessPublic,
      /*bitfield_bit_size=*/0);
  ASSERT_TRUE(member_var);
  EXPECT_EQ(member_var->getParent(), def);

  // Complete the class and check that the last decl is the definition.
  m_ast->CompleteTagDeclarationDefinition(spec_type);
  EXPECT_FALSE(fwd_decl->isThisDeclarationADefinition());
  EXPECT_TRUE(def->isThisDeclarationADefinition());

  // Check that the module is identical.
  EXPECT_EQ(def->getOwningModuleID(), module_id.GetValue());

  // Make sure forward decl and definition have the same type.
  EXPECT_EQ(def->getTypeForDecl(), fwd_decl->getTypeForDecl());
}

TEST_F(TestClangRedeclarations, RedeclareObjCClass) {
  // Test redeclaring Objective-C interfaces.

  OptionalClangModuleID module_id(1);
  CompilerType objc_class =
      m_ast->CreateObjCClass("A", m_ast->GetTranslationUnitDecl(), module_id,
                             /*isForwardDecl=*/false,
                             /*isInternal=*/false);
  ObjCInterfaceDecl *interface = m_ast->GetAsObjCInterfaceDecl(objc_class);
  m_ast->CreateRedeclaration(objc_class);
  m_ast->StartTagDeclarationDefinition(objc_class);
  ObjCInterfaceDecl *def = interface->getDefinition();
  ASSERT_TRUE(def);
  ASSERT_NE(def, interface);
  EXPECT_EQ(def->getPreviousDecl(), interface);

  // Add a method.
  std::vector<CompilerType> args;
  CompilerType func_type =
      m_ast->CreateFunctionType(m_ast->GetBasicType(lldb::eBasicTypeInt),
                                args.data(), args.size(), /*is_variadic=*/false,
                                /*type_quals=*/0, clang::CallingConv::CC_C);
  const bool variadic = false;
  const bool artificial = false;
  const bool objc_direct = false;
  clang::ObjCMethodDecl *method = TypeSystemClang::AddMethodToObjCObjectType(
      objc_class, "-[A foo]", func_type, artificial, variadic, objc_direct);
  // Check that the method was created and is in the definition.
  ASSERT_NE(method, nullptr);
  EXPECT_EQ(*def->meth_begin(), method);

  // Add an ivar and check that it was added to the definition.
  FieldDecl *ivar = m_ast->AddFieldToRecordType(
      objc_class, "f", m_ast->GetBasicType(lldb::eBasicTypeInt),
      lldb::eAccessPublic,
      /*bitfield_bit_size=*/0);
  ASSERT_TRUE(ivar);
  EXPECT_EQ(*def->ivar_begin(), ivar);

  m_ast->CompleteTagDeclarationDefinition(objc_class);
  // The forward declaration should be unchanged.
  EXPECT_FALSE(interface->isThisDeclarationADefinition());
  EXPECT_TRUE(def->isThisDeclarationADefinition());
  // Check that the module is identical.
  EXPECT_EQ(def->getOwningModuleID(), module_id.GetValue());

  // Make sure forward decl and definition have the same type.
  EXPECT_EQ(def->getTypeForDecl(), interface->getTypeForDecl());
}

TEST_F(TestClangRedeclarations, RedeclareEnum) {
  // Test redeclaring enums.

  OptionalClangModuleID module_id(1);
  Declaration decl;
  CompilerType enum_type = m_ast->CreateEnumerationType(
      "A", m_ast->GetTranslationUnitDecl(), module_id, decl,
      m_ast->GetBasicType(lldb::eBasicTypeInt), /*is_scoped=*/true);

  EnumDecl *fwd_decl = m_ast->GetAsEnumDecl(enum_type);
  m_ast->CreateRedeclaration(enum_type);
  m_ast->StartTagDeclarationDefinition(enum_type);
  m_ast->AddEnumerationValueToEnumerationType(
      enum_type, decl, "case1", /*enum_value=*/1, /*enum_value_bit_size=*/32);
  m_ast->CompleteTagDeclarationDefinition(enum_type);

  // There should now be a definition at the end of the redeclaration chain.
  EnumDecl *def = fwd_decl->getDefinition();
  ASSERT_TRUE(def);
  ASSERT_NE(def, fwd_decl);
  EXPECT_EQ(def->getPreviousDecl(), fwd_decl);
  // The forward declaration should be unchanged.
  EXPECT_FALSE(fwd_decl->isThisDeclarationADefinition());
  EXPECT_TRUE(def->isThisDeclarationADefinition());
  // Check that the module is identical.
  EXPECT_EQ(def->getOwningModuleID(), module_id.GetValue());

  // Make sure forward decl and definition have the same type.
  EXPECT_EQ(def->getTypeForDecl(), fwd_decl->getTypeForDecl());

  // Check that ForEachEnumerator uses the definition.
  bool seen_value = false;
  m_ast->ForEachEnumerator(enum_type.GetOpaqueQualType(),
                           [&seen_value](const CompilerType &, ConstString name,
                                         const llvm::APSInt &) {
                             EXPECT_EQ(name, "case1");
                             seen_value = true;
                             return true;
                           });
  EXPECT_TRUE(seen_value);
}

TEST_F(TestClangRedeclarations, NestedDecls) {
  // Tests that nested declarations pick the right redeclaration as their
  // DeclContext.

  // Create a class.
  CompilerType context_class = m_ast->CreateRecordType(
      m_ast->GetTranslationUnitDecl(), OptionalClangModuleID(),
      lldb::eAccessNone, "A", llvm::to_underlying(TagTypeKind::Class),
      lldb::eLanguageTypeC_plus_plus);
  auto *fwd_decl =
      llvm::cast<CXXRecordDecl>(ClangUtil::GetAsTagDecl(context_class));

  // Give it a redeclaration that defines it.
  m_ast->CreateRedeclaration(context_class);
  m_ast->StartTagDeclarationDefinition(context_class);
  m_ast->CompleteTagDeclarationDefinition(context_class);

  // Check that there is one forward declaration and a definition now.
  CXXRecordDecl *def = fwd_decl->getDefinition();
  ASSERT_TRUE(def);
  EXPECT_FALSE(fwd_decl->isThisDeclarationADefinition());
  EXPECT_TRUE(def->isThisDeclarationADefinition());

  // Create a nested class and make sure it picks the definition as its
  // DeclContext.
  CompilerType nested_class = m_ast->CreateRecordType(
      fwd_decl, OptionalClangModuleID(), lldb::eAccessPublic, "A",
      llvm::to_underlying(TagTypeKind::Class), lldb::eLanguageTypeC_plus_plus);
  EXPECT_EQ(ClangUtil::GetAsTagDecl(nested_class)->getDeclContext(), def);

  CompilerType int_type = m_ast->GetBasicType(lldb::eBasicTypeInt);

  // Create a typedef and make sure it picks the definition as its DeclContext.
  CompilerType nested_typedef = int_type.CreateTypedef(
      "t", CompilerDeclContext(m_ast, static_cast<DeclContext *>(fwd_decl)),
      /*payload=*/0);
  const TypedefType *typedef_type =
      ClangUtil::GetQualType(nested_typedef)->getAs<TypedefType>();
  ASSERT_TRUE(typedef_type);
  TypedefNameDecl *nested_typedef_decl = typedef_type->getDecl();
  ASSERT_TRUE(nested_typedef_decl);
  EXPECT_EQ(nested_typedef_decl->getDeclContext(), def);

  TypeSystemClang::TemplateParameterInfos args;
  args.InsertArg("T", TemplateArgument(m_ast->getASTContext().IntTy));

  // Create a class template and specialization and check that their DeclContext
  // is the definition.
  ClassTemplateDecl *template_decl = m_ast->CreateClassTemplateDecl(
      fwd_decl, OptionalClangModuleID(), lldb::eAccessPublic, "A",
      llvm::to_underlying(TagTypeKind::Struct), args);
  EXPECT_EQ(template_decl->getDeclContext(), def);
  ClassTemplateSpecializationDecl *template_spec_decl =
      m_ast->CreateClassTemplateSpecializationDecl(
          fwd_decl, OptionalClangModuleID(), template_decl,
          llvm::to_underlying(TagTypeKind::Struct), args);
  EXPECT_EQ(template_spec_decl->getDeclContext(), def);
}

TEST_F(TestClangRedeclarations, MetadataRedeclaration) {
  // Tests that metadata is shared between redeclarations.

  // Create a class with the test metadata.
  CompilerType class_with_metadata = m_ast->CreateRecordType(
      m_ast->GetTranslationUnitDecl(), OptionalClangModuleID(),
      lldb::eAccessPublic, "A", llvm::to_underlying(TagTypeKind::Class),
      lldb::eLanguageTypeC_plus_plus);
  auto *record =
      llvm::cast<CXXRecordDecl>(ClangUtil::GetAsTagDecl(class_with_metadata));
  ClangASTMetadata metadata;
  metadata.SetUserID(1234);
  m_ast->SetMetadata(record, metadata);
  ASSERT_EQ(m_ast->GetMetadata(record)->GetUserID(), 1234U);

  // Redeclare and define the redeclaration.
  m_ast->CreateRedeclaration(class_with_metadata);
  m_ast->StartTagDeclarationDefinition(class_with_metadata);
  m_ast->CompleteTagDeclarationDefinition(class_with_metadata);
  CXXRecordDecl *def = record->getDefinition();
  ASSERT_TRUE(def);
  ASSERT_NE(def, record);
  EXPECT_EQ(def->getPreviousDecl(), record);

  // Check that the redeclaration has the right metadata;
  ASSERT_TRUE(m_ast->GetMetadata(def));
  EXPECT_EQ(m_ast->GetMetadata(def)->GetUserID(), 1234U);
}
