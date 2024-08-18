//===-- TestTypeSystemClang.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ExpressionParser/Clang/ClangUtil.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/Symbol/ClangTestUtils.h"
#include "lldb/Core/Declaration.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ExprCXX.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace lldb;
using namespace lldb_private;

class TestTypeSystemClang : public testing::Test {
public:
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

protected:
  
  TypeSystemClang *m_ast = nullptr;
  std::unique_ptr<clang_utils::TypeSystemClangHolder> m_holder;

  QualType GetBasicQualType(BasicType type) const {
    return ClangUtil::GetQualType(m_ast->GetBasicTypeFromAST(type));
  }

  QualType GetBasicQualType(const char *name) const {
    return ClangUtil::GetQualType(
        m_ast->GetBuiltinTypeByName(ConstString(name)));
  }
};

TEST_F(TestTypeSystemClang, TestGetBasicTypeFromEnum) {
  clang::ASTContext &context = m_ast->getASTContext();

  EXPECT_TRUE(
      context.hasSameType(GetBasicQualType(eBasicTypeBool), context.BoolTy));
  EXPECT_TRUE(
      context.hasSameType(GetBasicQualType(eBasicTypeChar), context.CharTy));
  EXPECT_TRUE(context.hasSameType(GetBasicQualType(eBasicTypeChar8),
                                  context.Char8Ty));
  EXPECT_TRUE(context.hasSameType(GetBasicQualType(eBasicTypeChar16),
                                  context.Char16Ty));
  EXPECT_TRUE(context.hasSameType(GetBasicQualType(eBasicTypeChar32),
                                  context.Char32Ty));
  EXPECT_TRUE(context.hasSameType(GetBasicQualType(eBasicTypeDouble),
                                  context.DoubleTy));
  EXPECT_TRUE(context.hasSameType(GetBasicQualType(eBasicTypeDoubleComplex),
                                  context.getComplexType(context.DoubleTy)));
  EXPECT_TRUE(
      context.hasSameType(GetBasicQualType(eBasicTypeFloat), context.FloatTy));
  EXPECT_TRUE(context.hasSameType(GetBasicQualType(eBasicTypeFloatComplex),
                                  context.getComplexType(context.FloatTy)));
  EXPECT_TRUE(
      context.hasSameType(GetBasicQualType(eBasicTypeHalf), context.HalfTy));
  EXPECT_TRUE(
      context.hasSameType(GetBasicQualType(eBasicTypeInt), context.IntTy));
  EXPECT_TRUE(context.hasSameType(GetBasicQualType(eBasicTypeInt128),
                                  context.Int128Ty));
  EXPECT_TRUE(
      context.hasSameType(GetBasicQualType(eBasicTypeLong), context.LongTy));
  EXPECT_TRUE(context.hasSameType(GetBasicQualType(eBasicTypeLongDouble),
                                  context.LongDoubleTy));
  EXPECT_TRUE(
      context.hasSameType(GetBasicQualType(eBasicTypeLongDoubleComplex),
                          context.getComplexType(context.LongDoubleTy)));
  EXPECT_TRUE(context.hasSameType(GetBasicQualType(eBasicTypeLongLong),
                                  context.LongLongTy));
  EXPECT_TRUE(context.hasSameType(GetBasicQualType(eBasicTypeNullPtr),
                                  context.NullPtrTy));
  EXPECT_TRUE(context.hasSameType(GetBasicQualType(eBasicTypeObjCClass),
                                  context.getObjCClassType()));
  EXPECT_TRUE(context.hasSameType(GetBasicQualType(eBasicTypeObjCID),
                                  context.getObjCIdType()));
  EXPECT_TRUE(context.hasSameType(GetBasicQualType(eBasicTypeObjCSel),
                                  context.getObjCSelType()));
  EXPECT_TRUE(
      context.hasSameType(GetBasicQualType(eBasicTypeShort), context.ShortTy));
  EXPECT_TRUE(context.hasSameType(GetBasicQualType(eBasicTypeSignedChar),
                                  context.SignedCharTy));
  EXPECT_TRUE(context.hasSameType(GetBasicQualType(eBasicTypeUnsignedChar),
                                  context.UnsignedCharTy));
  EXPECT_TRUE(context.hasSameType(GetBasicQualType(eBasicTypeUnsignedInt),
                                  context.UnsignedIntTy));
  EXPECT_TRUE(context.hasSameType(GetBasicQualType(eBasicTypeUnsignedInt128),
                                  context.UnsignedInt128Ty));
  EXPECT_TRUE(context.hasSameType(GetBasicQualType(eBasicTypeUnsignedLong),
                                  context.UnsignedLongTy));
  EXPECT_TRUE(context.hasSameType(GetBasicQualType(eBasicTypeUnsignedLongLong),
                                  context.UnsignedLongLongTy));
  EXPECT_TRUE(context.hasSameType(GetBasicQualType(eBasicTypeUnsignedShort),
                                  context.UnsignedShortTy));
  EXPECT_TRUE(
      context.hasSameType(GetBasicQualType(eBasicTypeVoid), context.VoidTy));
  EXPECT_TRUE(
      context.hasSameType(GetBasicQualType(eBasicTypeWChar), context.WCharTy));
}

TEST_F(TestTypeSystemClang, TestGetBasicTypeFromName) {
  EXPECT_EQ(GetBasicQualType(eBasicTypeChar), GetBasicQualType("char"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeSignedChar),
            GetBasicQualType("signed char"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeUnsignedChar),
            GetBasicQualType("unsigned char"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeWChar), GetBasicQualType("wchar_t"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeSignedWChar),
            GetBasicQualType("signed wchar_t"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeUnsignedWChar),
            GetBasicQualType("unsigned wchar_t"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeShort), GetBasicQualType("short"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeShort), GetBasicQualType("short int"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeUnsignedShort),
            GetBasicQualType("unsigned short"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeUnsignedShort),
            GetBasicQualType("unsigned short int"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeInt), GetBasicQualType("int"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeInt), GetBasicQualType("signed int"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeUnsignedInt),
            GetBasicQualType("unsigned int"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeUnsignedInt),
            GetBasicQualType("unsigned"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeLong), GetBasicQualType("long"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeLong), GetBasicQualType("long int"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeUnsignedLong),
            GetBasicQualType("unsigned long"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeUnsignedLong),
            GetBasicQualType("unsigned long int"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeLongLong),
            GetBasicQualType("long long"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeLongLong),
            GetBasicQualType("long long int"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeUnsignedLongLong),
            GetBasicQualType("unsigned long long"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeUnsignedLongLong),
            GetBasicQualType("unsigned long long int"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeInt128), GetBasicQualType("__int128_t"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeUnsignedInt128),
            GetBasicQualType("__uint128_t"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeVoid), GetBasicQualType("void"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeBool), GetBasicQualType("bool"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeFloat), GetBasicQualType("float"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeDouble), GetBasicQualType("double"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeLongDouble),
            GetBasicQualType("long double"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeObjCID), GetBasicQualType("id"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeObjCSel), GetBasicQualType("SEL"));
  EXPECT_EQ(GetBasicQualType(eBasicTypeNullPtr), GetBasicQualType("nullptr"));
}

void VerifyEncodingAndBitSize(TypeSystemClang &clang_context,
                              lldb::Encoding encoding, unsigned int bit_size) {
  clang::ASTContext &context = clang_context.getASTContext();

  CompilerType type =
      clang_context.GetBuiltinTypeForEncodingAndBitSize(encoding, bit_size);
  EXPECT_TRUE(type.IsValid());

  QualType qtype = ClangUtil::GetQualType(type);
  EXPECT_FALSE(qtype.isNull());
  if (qtype.isNull())
    return;

  uint64_t actual_size = context.getTypeSize(qtype);
  EXPECT_EQ(bit_size, actual_size);

  const clang::Type *type_ptr = qtype.getTypePtr();
  EXPECT_NE(nullptr, type_ptr);
  if (!type_ptr)
    return;

  EXPECT_TRUE(type_ptr->isBuiltinType());
  switch (encoding) {
  case eEncodingSint:
    EXPECT_TRUE(type_ptr->isSignedIntegerType());
    break;
  case eEncodingUint:
    EXPECT_TRUE(type_ptr->isUnsignedIntegerType());
    break;
  case eEncodingIEEE754:
    EXPECT_TRUE(type_ptr->isFloatingType());
    break;
  default:
    FAIL() << "Unexpected encoding";
    break;
  }
}

TEST_F(TestTypeSystemClang, TestBuiltinTypeForEncodingAndBitSize) {
  // Make sure we can get types of every possible size in every possible
  // encoding.
  // We can't make any guarantee about which specific type we get, because the
  // standard
  // isn't that specific.  We only need to make sure the compiler hands us some
  // type that
  // is both a builtin type and matches the requested bit size.
  VerifyEncodingAndBitSize(*m_ast, eEncodingSint, 8);
  VerifyEncodingAndBitSize(*m_ast, eEncodingSint, 16);
  VerifyEncodingAndBitSize(*m_ast, eEncodingSint, 32);
  VerifyEncodingAndBitSize(*m_ast, eEncodingSint, 64);
  VerifyEncodingAndBitSize(*m_ast, eEncodingSint, 128);

  VerifyEncodingAndBitSize(*m_ast, eEncodingUint, 8);
  VerifyEncodingAndBitSize(*m_ast, eEncodingUint, 16);
  VerifyEncodingAndBitSize(*m_ast, eEncodingUint, 32);
  VerifyEncodingAndBitSize(*m_ast, eEncodingUint, 64);
  VerifyEncodingAndBitSize(*m_ast, eEncodingUint, 128);

  VerifyEncodingAndBitSize(*m_ast, eEncodingIEEE754, 32);
  VerifyEncodingAndBitSize(*m_ast, eEncodingIEEE754, 64);
}

TEST_F(TestTypeSystemClang, TestDisplayName) {
  TypeSystemClang ast("some name", llvm::Triple());
  EXPECT_EQ("some name", ast.getDisplayName());
}

TEST_F(TestTypeSystemClang, TestDisplayNameEmpty) {
  TypeSystemClang ast("", llvm::Triple());
  EXPECT_EQ("", ast.getDisplayName());
}

TEST_F(TestTypeSystemClang, TestGetEnumIntegerTypeInvalid) {
  EXPECT_FALSE(m_ast->GetEnumerationIntegerType(CompilerType()).IsValid());
}

TEST_F(TestTypeSystemClang, TestGetEnumIntegerTypeUnexpectedType) {
  CompilerType int_type = m_ast->GetBasicType(lldb::eBasicTypeInt);
  CompilerType t = m_ast->GetEnumerationIntegerType(int_type);
  EXPECT_FALSE(t.IsValid());
}

TEST_F(TestTypeSystemClang, TestGetEnumIntegerTypeBasicTypes) {
  // All possible underlying integer types of enums.
  const std::vector<lldb::BasicType> types_to_test = {
      eBasicTypeInt,          eBasicTypeUnsignedInt, eBasicTypeLong,
      eBasicTypeUnsignedLong, eBasicTypeLongLong,    eBasicTypeUnsignedLongLong,
  };

  for (bool scoped : {true, false}) {
    SCOPED_TRACE("scoped: " + std::to_string(scoped));
    for (lldb::BasicType basic_type : types_to_test) {
      SCOPED_TRACE(std::to_string(basic_type));

      auto holder =
          std::make_unique<clang_utils::TypeSystemClangHolder>("enum_ast");
      auto &ast = *holder->GetAST();

      CompilerType basic_compiler_type = ast.GetBasicType(basic_type);
      EXPECT_TRUE(basic_compiler_type.IsValid());

      CompilerType enum_type = ast.CreateEnumerationType(
          "my_enum", ast.GetTranslationUnitDecl(), OptionalClangModuleID(),
          Declaration(), basic_compiler_type, scoped);

      CompilerType t = ast.GetEnumerationIntegerType(enum_type);
      // Check that the type we put in at the start is found again.
      EXPECT_EQ(basic_compiler_type.GetTypeName(), t.GetTypeName());
    }
  }
}

TEST_F(TestTypeSystemClang, TestOwningModule) {
  auto holder =
      std::make_unique<clang_utils::TypeSystemClangHolder>("module_ast");
  auto &ast = *holder->GetAST();
  CompilerType basic_compiler_type = ast.GetBasicType(BasicType::eBasicTypeInt);
  CompilerType enum_type = ast.CreateEnumerationType(
      "my_enum", ast.GetTranslationUnitDecl(), OptionalClangModuleID(100),
      Declaration(), basic_compiler_type, false);
  auto *ed = TypeSystemClang::GetAsEnumDecl(enum_type);
  EXPECT_FALSE(!ed);
  EXPECT_EQ(ed->getOwningModuleID(), 100u);

  CompilerType record_type = ast.CreateRecordType(
      nullptr, OptionalClangModuleID(200), lldb::eAccessPublic, "FooRecord",
      llvm::to_underlying(clang::TagTypeKind::Struct),
      lldb::eLanguageTypeC_plus_plus, std::nullopt);
  auto *rd = TypeSystemClang::GetAsRecordDecl(record_type);
  EXPECT_FALSE(!rd);
  EXPECT_EQ(rd->getOwningModuleID(), 200u);

  CompilerType class_type =
      ast.CreateObjCClass("objc_class", ast.GetTranslationUnitDecl(),
                          OptionalClangModuleID(300), false);
  auto *cd = TypeSystemClang::GetAsObjCInterfaceDecl(class_type);
  EXPECT_FALSE(!cd);
  EXPECT_EQ(cd->getOwningModuleID(), 300u);
}

TEST_F(TestTypeSystemClang, TestIsClangType) {
  clang::ASTContext &context = m_ast->getASTContext();
  lldb::opaque_compiler_type_t bool_ctype =
      TypeSystemClang::GetOpaqueCompilerType(&context, lldb::eBasicTypeBool);
  CompilerType bool_type(m_ast->weak_from_this(), bool_ctype);
  CompilerType record_type = m_ast->CreateRecordType(
      nullptr, OptionalClangModuleID(100), lldb::eAccessPublic, "FooRecord",
      llvm::to_underlying(clang::TagTypeKind::Struct),
      lldb::eLanguageTypeC_plus_plus, std::nullopt);
  // Clang builtin type and record type should pass
  EXPECT_TRUE(ClangUtil::IsClangType(bool_type));
  EXPECT_TRUE(ClangUtil::IsClangType(record_type));

  // Default constructed type should fail
  EXPECT_FALSE(ClangUtil::IsClangType(CompilerType()));
}

TEST_F(TestTypeSystemClang, TestRemoveFastQualifiers) {
  CompilerType record_type = m_ast->CreateRecordType(
      nullptr, OptionalClangModuleID(), lldb::eAccessPublic, "FooRecord",
      llvm::to_underlying(clang::TagTypeKind::Struct),
      lldb::eLanguageTypeC_plus_plus, std::nullopt);
  QualType qt;

  qt = ClangUtil::GetQualType(record_type);
  EXPECT_EQ(0u, qt.getLocalFastQualifiers());
  record_type = record_type.AddConstModifier();
  record_type = record_type.AddVolatileModifier();
  record_type = record_type.AddRestrictModifier();
  qt = ClangUtil::GetQualType(record_type);
  EXPECT_NE(0u, qt.getLocalFastQualifiers());
  record_type = ClangUtil::RemoveFastQualifiers(record_type);
  qt = ClangUtil::GetQualType(record_type);
  EXPECT_EQ(0u, qt.getLocalFastQualifiers());
}

TEST_F(TestTypeSystemClang, TestConvertAccessTypeToAccessSpecifier) {
  EXPECT_EQ(AS_none,
            TypeSystemClang::ConvertAccessTypeToAccessSpecifier(eAccessNone));
  EXPECT_EQ(AS_none, TypeSystemClang::ConvertAccessTypeToAccessSpecifier(
                         eAccessPackage));
  EXPECT_EQ(AS_public,
            TypeSystemClang::ConvertAccessTypeToAccessSpecifier(eAccessPublic));
  EXPECT_EQ(AS_private, TypeSystemClang::ConvertAccessTypeToAccessSpecifier(
                            eAccessPrivate));
  EXPECT_EQ(AS_protected, TypeSystemClang::ConvertAccessTypeToAccessSpecifier(
                              eAccessProtected));
}

TEST_F(TestTypeSystemClang, TestUnifyAccessSpecifiers) {
  // Unifying two of the same type should return the same type
  EXPECT_EQ(AS_public,
            TypeSystemClang::UnifyAccessSpecifiers(AS_public, AS_public));
  EXPECT_EQ(AS_private,
            TypeSystemClang::UnifyAccessSpecifiers(AS_private, AS_private));
  EXPECT_EQ(AS_protected,
            TypeSystemClang::UnifyAccessSpecifiers(AS_protected, AS_protected));

  // Otherwise the result should be the strictest of the two.
  EXPECT_EQ(AS_private,
            TypeSystemClang::UnifyAccessSpecifiers(AS_private, AS_public));
  EXPECT_EQ(AS_private,
            TypeSystemClang::UnifyAccessSpecifiers(AS_private, AS_protected));
  EXPECT_EQ(AS_private,
            TypeSystemClang::UnifyAccessSpecifiers(AS_public, AS_private));
  EXPECT_EQ(AS_private,
            TypeSystemClang::UnifyAccessSpecifiers(AS_protected, AS_private));
  EXPECT_EQ(AS_protected,
            TypeSystemClang::UnifyAccessSpecifiers(AS_protected, AS_public));
  EXPECT_EQ(AS_protected,
            TypeSystemClang::UnifyAccessSpecifiers(AS_public, AS_protected));

  // None is stricter than everything (by convention)
  EXPECT_EQ(AS_none,
            TypeSystemClang::UnifyAccessSpecifiers(AS_none, AS_public));
  EXPECT_EQ(AS_none,
            TypeSystemClang::UnifyAccessSpecifiers(AS_none, AS_protected));
  EXPECT_EQ(AS_none,
            TypeSystemClang::UnifyAccessSpecifiers(AS_none, AS_private));
  EXPECT_EQ(AS_none,
            TypeSystemClang::UnifyAccessSpecifiers(AS_public, AS_none));
  EXPECT_EQ(AS_none,
            TypeSystemClang::UnifyAccessSpecifiers(AS_protected, AS_none));
  EXPECT_EQ(AS_none,
            TypeSystemClang::UnifyAccessSpecifiers(AS_private, AS_none));
}

TEST_F(TestTypeSystemClang, TestRecordHasFields) {
  CompilerType int_type = m_ast->GetBasicType(eBasicTypeInt);

  // Test that a record with no fields returns false
  CompilerType empty_base = m_ast->CreateRecordType(
      nullptr, OptionalClangModuleID(), lldb::eAccessPublic, "EmptyBase",
      llvm::to_underlying(clang::TagTypeKind::Struct),
      lldb::eLanguageTypeC_plus_plus, std::nullopt);
  TypeSystemClang::StartTagDeclarationDefinition(empty_base);
  TypeSystemClang::CompleteTagDeclarationDefinition(empty_base);

  RecordDecl *empty_base_decl = TypeSystemClang::GetAsRecordDecl(empty_base);
  EXPECT_NE(nullptr, empty_base_decl);
  EXPECT_FALSE(m_ast->RecordHasFields(empty_base_decl));

  // Test that a record with direct fields returns true
  CompilerType non_empty_base = m_ast->CreateRecordType(
      nullptr, OptionalClangModuleID(), lldb::eAccessPublic, "NonEmptyBase",
      llvm::to_underlying(clang::TagTypeKind::Struct),
      lldb::eLanguageTypeC_plus_plus, std::nullopt);
  TypeSystemClang::StartTagDeclarationDefinition(non_empty_base);
  FieldDecl *non_empty_base_field_decl = m_ast->AddFieldToRecordType(
      non_empty_base, "MyField", int_type, eAccessPublic, 0);
  TypeSystemClang::CompleteTagDeclarationDefinition(non_empty_base);
  RecordDecl *non_empty_base_decl =
      TypeSystemClang::GetAsRecordDecl(non_empty_base);
  EXPECT_NE(nullptr, non_empty_base_decl);
  EXPECT_NE(nullptr, non_empty_base_field_decl);
  EXPECT_TRUE(m_ast->RecordHasFields(non_empty_base_decl));

  std::vector<std::unique_ptr<clang::CXXBaseSpecifier>> bases;

  // Test that a record with no direct fields, but fields in a base returns true
  CompilerType empty_derived = m_ast->CreateRecordType(
      nullptr, OptionalClangModuleID(), lldb::eAccessPublic, "EmptyDerived",
      llvm::to_underlying(clang::TagTypeKind::Struct),
      lldb::eLanguageTypeC_plus_plus, std::nullopt);
  TypeSystemClang::StartTagDeclarationDefinition(empty_derived);
  std::unique_ptr<clang::CXXBaseSpecifier> non_empty_base_spec =
      m_ast->CreateBaseClassSpecifier(non_empty_base.GetOpaqueQualType(),
                                      lldb::eAccessPublic, false, false);
  bases.push_back(std::move(non_empty_base_spec));
  bool result = m_ast->TransferBaseClasses(empty_derived.GetOpaqueQualType(),
                                           std::move(bases));
  TypeSystemClang::CompleteTagDeclarationDefinition(empty_derived);
  EXPECT_TRUE(result);
  CXXRecordDecl *empty_derived_non_empty_base_cxx_decl =
      m_ast->GetAsCXXRecordDecl(empty_derived.GetOpaqueQualType());
  RecordDecl *empty_derived_non_empty_base_decl =
      TypeSystemClang::GetAsRecordDecl(empty_derived);
  EXPECT_EQ(1u, m_ast->GetNumBaseClasses(
                    empty_derived_non_empty_base_cxx_decl, false));
  EXPECT_TRUE(m_ast->RecordHasFields(empty_derived_non_empty_base_decl));

  // Test that a record with no direct fields, but fields in a virtual base
  // returns true
  CompilerType empty_derived2 = m_ast->CreateRecordType(
      nullptr, OptionalClangModuleID(), lldb::eAccessPublic, "EmptyDerived2",
      llvm::to_underlying(clang::TagTypeKind::Struct),
      lldb::eLanguageTypeC_plus_plus, std::nullopt);
  TypeSystemClang::StartTagDeclarationDefinition(empty_derived2);
  std::unique_ptr<CXXBaseSpecifier> non_empty_vbase_spec =
      m_ast->CreateBaseClassSpecifier(non_empty_base.GetOpaqueQualType(),
                                      lldb::eAccessPublic, true, false);
  bases.push_back(std::move(non_empty_vbase_spec));
  result = m_ast->TransferBaseClasses(empty_derived2.GetOpaqueQualType(),
                                      std::move(bases));
  TypeSystemClang::CompleteTagDeclarationDefinition(empty_derived2);
  EXPECT_TRUE(result);
  CXXRecordDecl *empty_derived_non_empty_vbase_cxx_decl =
      m_ast->GetAsCXXRecordDecl(empty_derived2.GetOpaqueQualType());
  RecordDecl *empty_derived_non_empty_vbase_decl =
      TypeSystemClang::GetAsRecordDecl(empty_derived2);
  EXPECT_EQ(1u, m_ast->GetNumBaseClasses(
                    empty_derived_non_empty_vbase_cxx_decl, false));
  EXPECT_TRUE(
      m_ast->RecordHasFields(empty_derived_non_empty_vbase_decl));
}

TEST_F(TestTypeSystemClang, TemplateArguments) {
  TypeSystemClang::TemplateParameterInfos infos;
  infos.InsertArg("T", TemplateArgument(m_ast->getASTContext().IntTy));

  llvm::APSInt arg(llvm::APInt(8, 47));
  infos.InsertArg("I", TemplateArgument(m_ast->getASTContext(), arg,
                                        m_ast->getASTContext().IntTy));

  // template<typename T, int I> struct foo;
  ClassTemplateDecl *decl = m_ast->CreateClassTemplateDecl(
      m_ast->GetTranslationUnitDecl(), OptionalClangModuleID(), eAccessPublic,
      "foo", llvm::to_underlying(clang::TagTypeKind::Struct), infos);
  ASSERT_NE(decl, nullptr);

  // foo<int, 47>
  ClassTemplateSpecializationDecl *spec_decl =
      m_ast->CreateClassTemplateSpecializationDecl(
          m_ast->GetTranslationUnitDecl(), OptionalClangModuleID(), decl,
          llvm::to_underlying(clang::TagTypeKind::Struct), infos);
  ASSERT_NE(spec_decl, nullptr);
  CompilerType type = m_ast->CreateClassTemplateSpecializationType(spec_decl);
  ASSERT_TRUE(type);
  m_ast->StartTagDeclarationDefinition(type);
  m_ast->CompleteTagDeclarationDefinition(type);

  // typedef foo<int, 47> foo_def;
  CompilerType typedef_type = type.CreateTypedef(
      "foo_def", m_ast->CreateDeclContext(m_ast->GetTranslationUnitDecl()), 0);

  CompilerType auto_type(
      m_ast->weak_from_this(),
      m_ast->getASTContext()
          .getAutoType(ClangUtil::GetCanonicalQualType(typedef_type),
                       clang::AutoTypeKeyword::Auto, false)
          .getAsOpaquePtr());

  CompilerType int_type(m_ast->weak_from_this(),
                        m_ast->getASTContext().IntTy.getAsOpaquePtr());
  for (CompilerType t : {type, typedef_type, auto_type}) {
    SCOPED_TRACE(t.GetTypeName().AsCString());

    const bool expand_pack = false;
    EXPECT_EQ(
        m_ast->GetTemplateArgumentKind(t.GetOpaqueQualType(), 0, expand_pack),
        eTemplateArgumentKindType);
    EXPECT_EQ(
        m_ast->GetTypeTemplateArgument(t.GetOpaqueQualType(), 0, expand_pack),
        int_type);
    EXPECT_EQ(std::nullopt, m_ast->GetIntegralTemplateArgument(
                                t.GetOpaqueQualType(), 0, expand_pack));

    EXPECT_EQ(
        m_ast->GetTemplateArgumentKind(t.GetOpaqueQualType(), 1, expand_pack),
        eTemplateArgumentKindIntegral);
    EXPECT_EQ(
        m_ast->GetTypeTemplateArgument(t.GetOpaqueQualType(), 1, expand_pack),
        CompilerType());
    auto result = m_ast->GetIntegralTemplateArgument(t.GetOpaqueQualType(), 1,
                                                     expand_pack);
    ASSERT_NE(std::nullopt, result);
    EXPECT_EQ(arg, result->value);
    EXPECT_EQ(int_type, result->type);
  }
}

class TestCreateClassTemplateDecl : public TestTypeSystemClang {
protected:
  /// The class templates created so far by the Expect* functions below.
  llvm::DenseSet<ClassTemplateDecl *> m_created_templates;

  /// Utility function for creating a class template.
  ClassTemplateDecl *
  CreateClassTemplate(const TypeSystemClang::TemplateParameterInfos &infos) {
    ClassTemplateDecl *decl = m_ast->CreateClassTemplateDecl(
        m_ast->GetTranslationUnitDecl(), OptionalClangModuleID(), eAccessPublic,
        "foo", llvm::to_underlying(clang::TagTypeKind::Struct), infos);
    return decl;
  }

  /// Creates a new class template with the given template parameters.
  /// Asserts that a new ClassTemplateDecl is created.
  /// \param description The gtest scope string that should describe the input.
  /// \param infos The template parameters that the class template should have.
  /// \returns The created ClassTemplateDecl.
  ClassTemplateDecl *
  ExpectNewTemplate(std::string description,
                    const TypeSystemClang::TemplateParameterInfos &infos) {
    SCOPED_TRACE(description);
    ClassTemplateDecl *first_template = CreateClassTemplate(infos);
    // A new template should have been created.
    EXPECT_FALSE(m_created_templates.contains(first_template))
        << "Didn't create new class template but reused this existing decl:\n"
        << ClangUtil::DumpDecl(first_template);
    m_created_templates.insert(first_template);

    // Creating a new template with the same arguments should always return
    // the template created above.
    ClassTemplateDecl *second_template = CreateClassTemplate(infos);
    EXPECT_EQ(first_template, second_template)
        << "Second attempt to create class template didn't reuse first decl:\n"
        << ClangUtil::DumpDecl(first_template) << "\nInstead created/reused:\n"
        << ClangUtil::DumpDecl(second_template);
    return first_template;
  }

  /// Tries to create a new class template but asserts that an existing class
  /// template in the current AST is reused (in contract so a new class
  /// template being created).
  /// \param description The gtest scope string that should describe the input.
  /// \param infos The template parameters that the class template should have.
  void
  ExpectReusedTemplate(std::string description,
                       const TypeSystemClang::TemplateParameterInfos &infos,
                       ClassTemplateDecl *expected) {
    SCOPED_TRACE(description);
    ClassTemplateDecl *td = CreateClassTemplate(infos);
    EXPECT_EQ(td, expected)
        << "Created/reused class template is:\n"
        << ClangUtil::DumpDecl(td) << "\nExpected to reuse:\n"
        << ClangUtil::DumpDecl(expected);
  }
};

TEST_F(TestCreateClassTemplateDecl, FindExistingTemplates) {
  // This tests the logic in TypeSystemClang::CreateClassTemplateDecl that
  // decides whether an existing ClassTemplateDecl in the AST can be reused.
  // The behaviour should follow the C++ rules for redeclaring templates
  // (e.g., parameter names can be changed/omitted.)

  // Test an empty template parameter list: <>
  ExpectNewTemplate("<>", {{}, {}});

  clang::TemplateArgument intArg(m_ast->getASTContext().IntTy);
  clang::TemplateArgument int47Arg(m_ast->getASTContext(),
                                   llvm::APSInt(llvm::APInt(32, 47)),
                                   m_ast->getASTContext().IntTy);
  clang::TemplateArgument floatArg(m_ast->getASTContext().FloatTy);
  clang::TemplateArgument char47Arg(m_ast->getASTContext(),
                                    llvm::APSInt(llvm::APInt(8, 47)),
                                    m_ast->getASTContext().SignedCharTy);

  clang::TemplateArgument char123Arg(m_ast->getASTContext(),
                                     llvm::APSInt(llvm::APInt(8, 123)),
                                     m_ast->getASTContext().SignedCharTy);

  // Test that <typename T> with T = int creates a new template.
  ClassTemplateDecl *single_type_arg =
      ExpectNewTemplate("<typename T>", {{"T"}, {intArg}});

  // Test that changing the parameter name doesn't create a new class template.
  ExpectReusedTemplate("<typename A> (A = int)", {{"A"}, {intArg}},
                       single_type_arg);

  // Test that changing the used type doesn't create a new class template.
  ExpectReusedTemplate("<typename A> (A = float)", {{"A"}, {floatArg}},
                       single_type_arg);

  // Test that <typename A, signed char I> creates a new template with A = int
  // and I = 47;
  ClassTemplateDecl *type_and_char_value =
      ExpectNewTemplate("<typename A, signed char I> (I = 47)",
                        {{"A", "I"}, {floatArg, char47Arg}});

  // Change the value of the I parameter to 123. The previously created
  // class template should still be reused.
  ExpectReusedTemplate("<typename A, signed char I> (I = 123)",
                       {{"A", "I"}, {floatArg, char123Arg}},
                       type_and_char_value);

  // Change the type of the I parameter to int so we have <typename A, int I>.
  // The class template from above can't be reused.
  ExpectNewTemplate("<typename A, int I> (I = 123)",
                    {{"A", "I"}, {floatArg, int47Arg}});

  // Test a second type parameter will also cause a new template to be created.
  // We now have <typename A, int I, typename B>.
  ClassTemplateDecl *type_and_char_value_and_type =
      ExpectNewTemplate("<typename A, int I, typename B>",
                        {{"A", "I", "B"}, {floatArg, int47Arg, intArg}});

  // Remove all the names from the parameters which shouldn't influence the
  // way the templates get merged.
  ExpectReusedTemplate("<typename, int, typename>",
                       {{"", "", ""}, {floatArg, int47Arg, intArg}},
                       type_and_char_value_and_type);
}

TEST_F(TestCreateClassTemplateDecl, FindExistingTemplatesWithParameterPack) {
  // The same as FindExistingTemplates but for templates with parameter packs.
  TypeSystemClang::TemplateParameterInfos infos;
  clang::TemplateArgument intArg(m_ast->getASTContext().IntTy);
  clang::TemplateArgument int1Arg(m_ast->getASTContext(),
                                  llvm::APSInt(llvm::APInt(32, 1)),
                                  m_ast->getASTContext().IntTy);
  clang::TemplateArgument int123Arg(m_ast->getASTContext(),
                                    llvm::APSInt(llvm::APInt(32, 123)),
                                    m_ast->getASTContext().IntTy);
  clang::TemplateArgument longArg(m_ast->getASTContext().LongTy);
  clang::TemplateArgument long1Arg(m_ast->getASTContext(),
                                   llvm::APSInt(llvm::APInt(64, 1)),
                                   m_ast->getASTContext().LongTy);

  infos.SetParameterPack(
      std::make_unique<TypeSystemClang::TemplateParameterInfos>(
          llvm::SmallVector<const char *>{"", ""},
          llvm::SmallVector<TemplateArgument>{intArg, intArg}));

  ClassTemplateDecl *type_pack =
      ExpectNewTemplate("<typename ...> (int, int)", infos);

  // Special case: An instantiation for a parameter pack with no values fits
  // to whatever class template we find. There isn't enough information to
  // do an actual comparison here.
  infos.SetParameterPack(
      std::make_unique<TypeSystemClang::TemplateParameterInfos>());
  ExpectReusedTemplate("<...> (no values in pack)", infos, type_pack);

  // Change the type content of pack type values.
  infos.SetParameterPack(
      std::make_unique<TypeSystemClang::TemplateParameterInfos>(
          llvm::SmallVector<const char *>{"", ""},
          llvm::SmallVector<TemplateArgument>{intArg, longArg}));
  ExpectReusedTemplate("<typename ...> (int, long)", infos, type_pack);

  // Change the number of pack values.
  infos.SetParameterPack(
      std::make_unique<TypeSystemClang::TemplateParameterInfos>(
          llvm::SmallVector<const char *>{""},
          llvm::SmallVector<TemplateArgument>{intArg}));
  ExpectReusedTemplate("<typename ...> (int)", infos, type_pack);

  // The names of the pack values shouldn't matter.
  infos.SetParameterPack(
      std::make_unique<TypeSystemClang::TemplateParameterInfos>(
          llvm::SmallVector<const char *>{"A"},
          llvm::SmallVector<TemplateArgument>{intArg}));
  ExpectReusedTemplate("<typename ...> (int)", infos, type_pack);

  // Changing the kind of template argument will create a new template.
  infos.SetParameterPack(
      std::make_unique<TypeSystemClang::TemplateParameterInfos>(
          llvm::SmallVector<const char *>{"A"},
          llvm::SmallVector<TemplateArgument>{int1Arg}));
  ClassTemplateDecl *int_pack = ExpectNewTemplate("<int ...> (int = 1)", infos);

  // Changing the value of integral parameters will not create a new template.
  infos.SetParameterPack(
      std::make_unique<TypeSystemClang::TemplateParameterInfos>(
          llvm::SmallVector<const char *>{"A"},
          llvm::SmallVector<TemplateArgument>{int123Arg}));
  ExpectReusedTemplate("<int ...> (int = 123)", infos, int_pack);

  // Changing the integral type will create a new template.
  infos.SetParameterPack(
      std::make_unique<TypeSystemClang::TemplateParameterInfos>(
          llvm::SmallVector<const char *>{"A"},
          llvm::SmallVector<TemplateArgument>{long1Arg}));
  ExpectNewTemplate("<long ...> (long = 1)", infos);

  // Prependinding a non-pack parameter will create a new template.
  infos.InsertArg("T", intArg);
  ExpectNewTemplate("<typename T, long...> (T = int, long = 1)", infos);
}

TEST_F(TestTypeSystemClang, OnlyPackName) {
  TypeSystemClang::TemplateParameterInfos infos;
  infos.SetPackName("A");
  EXPECT_FALSE(infos.IsValid());
}

static QualType makeConstInt(clang::ASTContext &ctxt) {
  QualType result(ctxt.IntTy);
  result.addConst();
  return result;
}

TEST_F(TestTypeSystemClang, TestGetTypeClassDeclType) {
  clang::ASTContext &ctxt = m_ast->getASTContext();
  auto *nullptr_expr = new (ctxt) CXXNullPtrLiteralExpr(ctxt.NullPtrTy, SourceLocation());
  QualType t = ctxt.getDecltypeType(nullptr_expr, makeConstInt(ctxt));
  EXPECT_EQ(lldb::eTypeClassBuiltin, m_ast->GetTypeClass(t.getAsOpaquePtr()));
}

TEST_F(TestTypeSystemClang, TestGetTypeClassTypeOf) {
  clang::ASTContext &ctxt = m_ast->getASTContext();
  QualType t = ctxt.getTypeOfType(makeConstInt(ctxt), TypeOfKind::Qualified);
  EXPECT_EQ(lldb::eTypeClassBuiltin, m_ast->GetTypeClass(t.getAsOpaquePtr()));
}

TEST_F(TestTypeSystemClang, TestGetTypeClassTypeOfExpr) {
  clang::ASTContext &ctxt = m_ast->getASTContext();
  auto *nullptr_expr = new (ctxt) CXXNullPtrLiteralExpr(ctxt.NullPtrTy, SourceLocation());
  QualType t = ctxt.getTypeOfExprType(nullptr_expr, TypeOfKind::Qualified);
  EXPECT_EQ(lldb::eTypeClassBuiltin, m_ast->GetTypeClass(t.getAsOpaquePtr()));
}

TEST_F(TestTypeSystemClang, TestGetTypeClassNested) {
  clang::ASTContext &ctxt = m_ast->getASTContext();
  QualType t_base =
      ctxt.getTypeOfType(makeConstInt(ctxt), TypeOfKind::Qualified);
  QualType t = ctxt.getTypeOfType(t_base, TypeOfKind::Qualified);
  EXPECT_EQ(lldb::eTypeClassBuiltin, m_ast->GetTypeClass(t.getAsOpaquePtr()));
}

TEST_F(TestTypeSystemClang, TestFunctionTemplateConstruction) {
  // Tests creating a function template.

  CompilerType int_type = m_ast->GetBasicType(lldb::eBasicTypeInt);
  clang::TranslationUnitDecl *TU = m_ast->GetTranslationUnitDecl();

  // Prepare the declarations/types we need for the template.
  CompilerType clang_type =
      m_ast->CreateFunctionType(int_type, nullptr, 0U, false, 0U);
  FunctionDecl *func = m_ast->CreateFunctionDeclaration(
      TU, OptionalClangModuleID(), "foo", clang_type, StorageClass::SC_None,
      false);
  TypeSystemClang::TemplateParameterInfos empty_params;

  // Create the actual function template.
  clang::FunctionTemplateDecl *func_template =
      m_ast->CreateFunctionTemplateDecl(TU, OptionalClangModuleID(), func,
                                        empty_params);

  EXPECT_EQ(TU, func_template->getDeclContext());
  EXPECT_EQ("foo", func_template->getName());
  EXPECT_EQ(clang::AccessSpecifier::AS_none, func_template->getAccess());
}

TEST_F(TestTypeSystemClang, TestFunctionTemplateInRecordConstruction) {
  // Tests creating a function template inside a record.

  CompilerType int_type = m_ast->GetBasicType(lldb::eBasicTypeInt);
  clang::TranslationUnitDecl *TU = m_ast->GetTranslationUnitDecl();

  // Create a record we can put the function template int.
  CompilerType record_type =
      clang_utils::createRecordWithField(*m_ast, "record", int_type, "field");
  clang::TagDecl *record = ClangUtil::GetAsTagDecl(record_type);

  // Prepare the declarations/types we need for the template.
  CompilerType clang_type =
      m_ast->CreateFunctionType(int_type, nullptr, 0U, false, 0U);
  // We create the FunctionDecl for the template in the TU DeclContext because:
  // 1. FunctionDecls can't be in a Record (only CXXMethodDecls can).
  // 2. It is mirroring the behavior of DWARFASTParserClang::ParseSubroutine.
  FunctionDecl *func = m_ast->CreateFunctionDeclaration(
      TU, OptionalClangModuleID(), "foo", clang_type, StorageClass::SC_None,
      false);
  TypeSystemClang::TemplateParameterInfos empty_params;

  // Create the actual function template.
  clang::FunctionTemplateDecl *func_template =
      m_ast->CreateFunctionTemplateDecl(record, OptionalClangModuleID(), func,
                                        empty_params);

  EXPECT_EQ(record, func_template->getDeclContext());
  EXPECT_EQ("foo", func_template->getName());
  EXPECT_EQ(clang::AccessSpecifier::AS_public, func_template->getAccess());
}

TEST_F(TestTypeSystemClang, TestDeletingImplicitCopyCstrDueToMoveCStr) {
  // We need to simulate this behavior in our AST that we construct as we don't
  // have a Sema instance that can do this for us:
  // C++11 [class.copy]p7, p18:
  //  If the class definition declares a move constructor or move assignment
  //  operator, an implicitly declared copy constructor or copy assignment
  //  operator is defined as deleted.

  // Create a record and start defining it.
  llvm::StringRef class_name = "S";
  CompilerType t = clang_utils::createRecord(*m_ast, class_name);
  m_ast->StartTagDeclarationDefinition(t);

  // Create a move constructor that will delete the implicit copy constructor.
  CompilerType return_type = m_ast->GetBasicType(lldb::eBasicTypeVoid);
  CompilerType param_type = t.GetRValueReferenceType();
  CompilerType function_type =
      m_ast->CreateFunctionType(return_type, &param_type, /*num_params*/ 1,
                                /*variadic=*/false, /*quals*/ 0U);
  bool is_virtual = false;
  bool is_static = false;
  bool is_inline = false;
  bool is_explicit = true;
  bool is_attr_used = false;
  bool is_artificial = false;
  m_ast->AddMethodToCXXRecordType(
      t.GetOpaqueQualType(), class_name, nullptr, function_type,
      lldb::AccessType::eAccessPublic, is_virtual, is_static, is_inline,
      is_explicit, is_attr_used, is_artificial);

  // Complete the definition and check the created record.
  m_ast->CompleteTagDeclarationDefinition(t);
  auto *record = llvm::cast<CXXRecordDecl>(ClangUtil::GetAsTagDecl(t));
  // We can't call defaultedCopyConstructorIsDeleted() as this requires that
  // the Decl passes through Sema which will actually compute this field.
  // Instead we check that there is no copy constructor declared by the user
  // which only leaves a non-deleted defaulted copy constructor as an option
  // that our record will have no simple copy constructor.
  EXPECT_FALSE(record->hasUserDeclaredCopyConstructor());
  EXPECT_FALSE(record->hasSimpleCopyConstructor());
}

TEST_F(TestTypeSystemClang, TestNotDeletingUserCopyCstrDueToMoveCStr) {
  // Tests that we don't delete the a user-defined copy constructor when
  // a move constructor is provided.
  // See also the TestDeletingImplicitCopyCstrDueToMoveCStr test.
  llvm::StringRef class_name = "S";
  CompilerType t = clang_utils::createRecord(*m_ast, class_name);
  m_ast->StartTagDeclarationDefinition(t);

  CompilerType return_type = m_ast->GetBasicType(lldb::eBasicTypeVoid);
  bool is_virtual = false;
  bool is_static = false;
  bool is_inline = false;
  bool is_explicit = true;
  bool is_attr_used = false;
  bool is_artificial = false;
  // Create a move constructor.
  {
    CompilerType param_type = t.GetRValueReferenceType();
    CompilerType function_type =
        m_ast->CreateFunctionType(return_type, &param_type, /*num_params*/ 1,
                                  /*variadic=*/false, /*quals*/ 0U);
    m_ast->AddMethodToCXXRecordType(
        t.GetOpaqueQualType(), class_name, nullptr, function_type,
        lldb::AccessType::eAccessPublic, is_virtual, is_static, is_inline,
        is_explicit, is_attr_used, is_artificial);
  }
  // Create a copy constructor.
  {
    CompilerType param_type = t.GetLValueReferenceType().AddConstModifier();
    CompilerType function_type =
        m_ast->CreateFunctionType(return_type, &param_type, /*num_params*/ 1,
                                  /*variadic=*/false, /*quals*/ 0U);
    m_ast->AddMethodToCXXRecordType(
        t.GetOpaqueQualType(), class_name, nullptr, function_type,
        lldb::AccessType::eAccessPublic, is_virtual, is_static, is_inline,
        is_explicit, is_attr_used, is_artificial);
  }

  // Complete the definition and check the created record.
  m_ast->CompleteTagDeclarationDefinition(t);
  auto *record = llvm::cast<CXXRecordDecl>(ClangUtil::GetAsTagDecl(t));
  EXPECT_TRUE(record->hasUserDeclaredCopyConstructor());
}

TEST_F(TestTypeSystemClang, AddMethodToObjCObjectType) {
  // Create an interface decl and mark it as having external storage.
  CompilerType c = m_ast->CreateObjCClass("A", m_ast->GetTranslationUnitDecl(),
                                          OptionalClangModuleID(),
                                          /*IsInternal*/ false);
  ObjCInterfaceDecl *interface = m_ast->GetAsObjCInterfaceDecl(c);
  m_ast->SetHasExternalStorage(c.GetOpaqueQualType(), true);
  EXPECT_TRUE(interface->hasExternalLexicalStorage());

  // Add a method to the interface.
  std::vector<CompilerType> args;
  CompilerType func_type =
      m_ast->CreateFunctionType(m_ast->GetBasicType(lldb::eBasicTypeInt),
                                args.data(), args.size(), /*variadic*/ false,
                                /*quals*/ 0, clang::CallingConv::CC_C);
  bool variadic = false;
  bool artificial = false;
  bool objc_direct = false;
  clang::ObjCMethodDecl *method = TypeSystemClang::AddMethodToObjCObjectType(
      c, "-[A foo]", func_type, artificial, variadic, objc_direct);
  ASSERT_NE(method, nullptr);

  // The interface decl should still have external lexical storage.
  EXPECT_TRUE(interface->hasExternalLexicalStorage());

  // Test some properties of the created ObjCMethodDecl.
  EXPECT_FALSE(method->isVariadic());
  EXPECT_TRUE(method->isImplicit());
  EXPECT_FALSE(method->isDirectMethod());
  EXPECT_EQ(method->getDeclName().getObjCSelector().getAsString(), "foo");
}

TEST_F(TestTypeSystemClang, GetFullyUnqualifiedType) {
  CompilerType bool_ = m_ast->GetBasicType(eBasicTypeBool);
  CompilerType cv_bool = bool_.AddConstModifier().AddVolatileModifier();

  // const volatile bool -> bool
  EXPECT_EQ(bool_, cv_bool.GetFullyUnqualifiedType());

  // const volatile bool[47] -> bool[47]
  EXPECT_EQ(bool_.GetArrayType(47),
            cv_bool.GetArrayType(47).GetFullyUnqualifiedType());

  // const volatile bool[47][42] -> bool[47][42]
  EXPECT_EQ(
      bool_.GetArrayType(42).GetArrayType(47),
      cv_bool.GetArrayType(42).GetArrayType(47).GetFullyUnqualifiedType());

  // const volatile bool * -> bool *
  EXPECT_EQ(bool_.GetPointerType(),
            cv_bool.GetPointerType().GetFullyUnqualifiedType());

  // const volatile bool *[47] -> bool *[47]
  EXPECT_EQ(
      bool_.GetPointerType().GetArrayType(47),
      cv_bool.GetPointerType().GetArrayType(47).GetFullyUnqualifiedType());
}

TEST(TestScratchTypeSystemClang, InferSubASTFromLangOpts) {
  LangOptions lang_opts;
  EXPECT_EQ(
      ScratchTypeSystemClang::DefaultAST,
      ScratchTypeSystemClang::InferIsolatedASTKindFromLangOpts(lang_opts));

  lang_opts.Modules = true;
  EXPECT_EQ(
      ScratchTypeSystemClang::IsolatedASTKind::CppModules,
      ScratchTypeSystemClang::InferIsolatedASTKindFromLangOpts(lang_opts));
}

TEST_F(TestTypeSystemClang, GetDeclContextByNameWhenMissingSymbolFile) {
  // Test that a type system without a symbol file is handled gracefully.
  std::vector<CompilerDecl> decls =
      m_ast->DeclContextFindDeclByName(nullptr, ConstString("SomeName"), true);

  EXPECT_TRUE(decls.empty());
}
