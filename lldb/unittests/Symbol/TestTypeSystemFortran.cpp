//===-- TestTypeSystemFortran.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/TypeSystem/Fortran/FortranTypes.h"
#include "Plugins/TypeSystem/Fortran/TypeSystemFortran.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Core/Declaration.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::plugin::fortran;

class TypeSystemFortranHolder {
  std::shared_ptr<TypeSystemFortran> m_ast;

public:
  TypeSystemFortranHolder() : m_ast(std::make_shared<TypeSystemFortran>()) {}
  TypeSystemFortran *GetAST() const { return m_ast.get(); }
};

class TestTypeSystemFortran : public testing::Test {
public:
  SubsystemRAII<FileSystem, HostInfo> subsystems;

  void SetUp() override {
    m_holder = std::make_unique<TypeSystemFortranHolder>();
    m_ast = m_holder->GetAST();
  }

  void TearDown() override {
    m_ast = nullptr;
    m_holder.reset();
  }

protected:
  TypeSystemFortran *m_ast = nullptr;
  std::unique_ptr<TypeSystemFortranHolder> m_holder;
};

TEST_F(TestTypeSystemFortran, TestBaseTypes) {
  CompilerType logical_type = m_ast->CreateType(llvm::dwarf::DW_ATE_boolean, 32,
                                                ConstString("Logical"));
  EXPECT_TRUE(logical_type.IsValid());
  auto bitsize_or_err = logical_type.GetBitSize(nullptr);
  ASSERT_THAT_EXPECTED(bitsize_or_err, llvm::Succeeded());
  EXPECT_EQ(*bitsize_or_err, 32U);
  EXPECT_EQ(m_ast->GetBasicTypeEnumeration(logical_type.GetOpaqueQualType()),
            eBasicTypeBool);

  CompilerType int8_type =
      m_ast->CreateType(llvm::dwarf::DW_ATE_signed, 8, ConstString());
  EXPECT_TRUE(int8_type.IsValid());
  bitsize_or_err = int8_type.GetBitSize(nullptr);
  ASSERT_THAT_EXPECTED(bitsize_or_err, llvm::Succeeded());
  EXPECT_EQ(*bitsize_or_err, 8U);
  EXPECT_EQ(m_ast->GetBasicTypeEnumeration(int8_type.GetOpaqueQualType()),
            eBasicTypeSignedChar);

  CompilerType int16_type =
      m_ast->CreateType(llvm::dwarf::DW_ATE_signed, 16, ConstString());
  EXPECT_TRUE(int16_type.IsValid());
  bitsize_or_err = int16_type.GetBitSize(nullptr);
  ASSERT_THAT_EXPECTED(bitsize_or_err, llvm::Succeeded());
  EXPECT_EQ(*bitsize_or_err, 16U);
  EXPECT_EQ(m_ast->GetBasicTypeEnumeration(int16_type.GetOpaqueQualType()),
            eBasicTypeShort);

  CompilerType int32_type =
      m_ast->CreateType(llvm::dwarf::DW_ATE_signed, 32, ConstString());
  EXPECT_TRUE(int32_type.IsValid());
  bitsize_or_err = int32_type.GetBitSize(nullptr);
  ASSERT_THAT_EXPECTED(bitsize_or_err, llvm::Succeeded());
  EXPECT_EQ(*bitsize_or_err, 32U);
  EXPECT_EQ(m_ast->GetBasicTypeEnumeration(int32_type.GetOpaqueQualType()),
            eBasicTypeInt);

  CompilerType int64_type =
      m_ast->CreateType(llvm::dwarf::DW_ATE_signed, 64, ConstString());
  EXPECT_TRUE(int64_type.IsValid());
  bitsize_or_err = int64_type.GetBitSize(nullptr);
  ASSERT_THAT_EXPECTED(bitsize_or_err, llvm::Succeeded());
  EXPECT_EQ(*bitsize_or_err, 64U);
  EXPECT_EQ(m_ast->GetBasicTypeEnumeration(int64_type.GetOpaqueQualType()),
            eBasicTypeLongLong);

  CompilerType int128_type =
      m_ast->CreateType(llvm::dwarf::DW_ATE_signed, 128, ConstString());
  EXPECT_TRUE(int128_type.IsValid());
  bitsize_or_err = int128_type.GetBitSize(nullptr);
  ASSERT_THAT_EXPECTED(bitsize_or_err, llvm::Succeeded());
  EXPECT_EQ(*bitsize_or_err, 128U);
  EXPECT_EQ(m_ast->GetBasicTypeEnumeration(int128_type.GetOpaqueQualType()),
            eBasicTypeInt128);

  CompilerType real16_type =
      m_ast->CreateType(llvm::dwarf::DW_ATE_float, 16, ConstString());
  EXPECT_TRUE(real16_type.IsValid());
  bitsize_or_err = real16_type.GetBitSize(nullptr);
  ASSERT_THAT_EXPECTED(bitsize_or_err, llvm::Succeeded());
  EXPECT_EQ(*bitsize_or_err, 16U);
  EXPECT_EQ(m_ast->GetBasicTypeEnumeration(real16_type.GetOpaqueQualType()),
            eBasicTypeHalf);

  CompilerType real32_type =
      m_ast->CreateType(llvm::dwarf::DW_ATE_float, 32, ConstString());
  EXPECT_TRUE(real32_type.IsValid());
  bitsize_or_err = real32_type.GetBitSize(nullptr);
  ASSERT_THAT_EXPECTED(bitsize_or_err, llvm::Succeeded());
  EXPECT_EQ(*bitsize_or_err, 32U);
  EXPECT_EQ(m_ast->GetBasicTypeEnumeration(real32_type.GetOpaqueQualType()),
            eBasicTypeFloat);

  CompilerType real64_type =
      m_ast->CreateType(llvm::dwarf::DW_ATE_float, 64, ConstString());
  EXPECT_TRUE(real64_type.IsValid());
  bitsize_or_err = real64_type.GetBitSize(nullptr);
  ASSERT_THAT_EXPECTED(bitsize_or_err, llvm::Succeeded());
  EXPECT_EQ(*bitsize_or_err, 64U);
  EXPECT_EQ(m_ast->GetBasicTypeEnumeration(real64_type.GetOpaqueQualType()),
            eBasicTypeDouble);

  CompilerType real128_type =
      m_ast->CreateType(llvm::dwarf::DW_ATE_float, 128, ConstString());
  EXPECT_TRUE(real128_type.IsValid());
  bitsize_or_err = real128_type.GetBitSize(nullptr);
  ASSERT_THAT_EXPECTED(bitsize_or_err, llvm::Succeeded());
  EXPECT_EQ(*bitsize_or_err, 128U);
  EXPECT_EQ(m_ast->GetBasicTypeEnumeration(real128_type.GetOpaqueQualType()),
            eBasicTypeFloat128);

  CompilerType complex64_type =
      m_ast->CreateType(llvm::dwarf::DW_ATE_complex_float, 64, ConstString());
  EXPECT_TRUE(complex64_type.IsValid());
  bitsize_or_err = complex64_type.GetBitSize(nullptr);
  ASSERT_THAT_EXPECTED(bitsize_or_err, llvm::Succeeded());
  EXPECT_EQ(*bitsize_or_err, 64U);
  EXPECT_EQ(m_ast->GetBasicTypeEnumeration(complex64_type.GetOpaqueQualType()),
            eBasicTypeFloatComplex);

  CompilerType complex128_type =
      m_ast->CreateType(llvm::dwarf::DW_ATE_complex_float, 128, ConstString());
  EXPECT_TRUE(complex128_type.IsValid());
  bitsize_or_err = complex128_type.GetBitSize(nullptr);
  ASSERT_THAT_EXPECTED(bitsize_or_err, llvm::Succeeded());
  EXPECT_EQ(*bitsize_or_err, 128U);
  EXPECT_EQ(m_ast->GetBasicTypeEnumeration(complex128_type.GetOpaqueQualType()),
            eBasicTypeDoubleComplex);

  CompilerType complex256_type =
      m_ast->CreateType(llvm::dwarf::DW_ATE_complex_float, 256, ConstString());
  EXPECT_TRUE(complex256_type.IsValid());
  bitsize_or_err = complex256_type.GetBitSize(nullptr);
  ASSERT_THAT_EXPECTED(bitsize_or_err, llvm::Succeeded());
  EXPECT_EQ(*bitsize_or_err, 256U);
  EXPECT_EQ(m_ast->GetBasicTypeEnumeration(complex256_type.GetOpaqueQualType()),
            eBasicTypeLongDoubleComplex);

  CompilerType invalid_int =
      m_ast->CreateType(llvm::dwarf::DW_ATE_signed, 42, ConstString());
  EXPECT_EQ(m_ast->GetBasicTypeEnumeration(invalid_int.GetOpaqueQualType()),
            eBasicTypeInvalid);
}

TEST_F(TestTypeSystemFortran, TestEncodingAndFormat) {
  CompilerType logical_type =
      m_ast->CreateType(llvm::dwarf::DW_ATE_boolean, 32, ConstString());
  CompilerType int_type =
      m_ast->CreateType(llvm::dwarf::DW_ATE_signed, 32, ConstString());
  CompilerType real_type =
      m_ast->CreateType(llvm::dwarf::DW_ATE_float, 32, ConstString());
  CompilerType complex_type =
      m_ast->CreateType(llvm::dwarf::DW_ATE_complex_float, 64, ConstString());

  EXPECT_EQ(logical_type.GetEncoding(), eEncodingUint);
  EXPECT_EQ(int_type.GetEncoding(), eEncodingSint);
  EXPECT_EQ(real_type.GetEncoding(), eEncodingIEEE754);
  EXPECT_EQ(complex_type.GetEncoding(), eEncodingIEEE754);

  EXPECT_EQ(logical_type.GetFormat(), eFormatBoolean);
  EXPECT_EQ(int_type.GetFormat(), eFormatDecimal);
  EXPECT_EQ(real_type.GetFormat(), eFormatFloat);
  EXPECT_EQ(complex_type.GetFormat(), eFormatComplex);
}

TEST_F(TestTypeSystemFortran, TestTypeClassifications) {
  CompilerType logical_type =
      m_ast->CreateType(llvm::dwarf::DW_ATE_boolean, 32, ConstString());
  CompilerType int_type =
      m_ast->CreateType(llvm::dwarf::DW_ATE_signed, 32, ConstString());
  CompilerType real_type =
      m_ast->CreateType(llvm::dwarf::DW_ATE_float, 32, ConstString());
  CompilerType complex_type =
      m_ast->CreateType(llvm::dwarf::DW_ATE_complex_float, 64, ConstString());

  bool is_signed = false;

  EXPECT_TRUE(int_type.IsIntegerType(is_signed));
  EXPECT_TRUE(is_signed);
  EXPECT_FALSE(logical_type.IsIntegerType(is_signed));
  EXPECT_FALSE(real_type.IsIntegerType(is_signed));
  EXPECT_FALSE(complex_type.IsIntegerType(is_signed));

  EXPECT_TRUE(real_type.IsFloatingPointType());
  EXPECT_FALSE(int_type.IsFloatingPointType());
  EXPECT_FALSE(logical_type.IsFloatingPointType());
  EXPECT_FALSE(complex_type.IsFloatingPointType());
}

TEST_F(TestTypeSystemFortran, TestGetTypeInfo) {
  CompilerType int_type =
      m_ast->CreateType(llvm::dwarf::DW_ATE_signed, 32, ConstString());
  CompilerType real_type =
      m_ast->CreateType(llvm::dwarf::DW_ATE_float, 32, ConstString());
  CompilerType complex_type =
      m_ast->CreateType(llvm::dwarf::DW_ATE_complex_float, 64, ConstString());

  uint32_t int_flags = int_type.GetTypeInfo();
  EXPECT_TRUE(int_flags & eTypeIsBuiltIn);
  EXPECT_TRUE(int_flags & eTypeHasValue);
  EXPECT_TRUE(int_flags & eTypeIsScalar);
  EXPECT_TRUE(int_flags & eTypeIsInteger);
  EXPECT_TRUE(int_flags & eTypeIsSigned);

  uint32_t real_flags = real_type.GetTypeInfo();
  EXPECT_TRUE(real_flags & eTypeIsScalar);
  EXPECT_TRUE(real_flags & eTypeIsFloat);

  uint32_t complex_flags = complex_type.GetTypeInfo();
  EXPECT_TRUE(complex_flags & eTypeIsComplex);
  EXPECT_FALSE(complex_flags & eTypeIsScalar);
}

TEST_F(TestTypeSystemFortran, TestTypeNameGeneration) {
  CompilerType logical32 =
      m_ast->CreateType(llvm::dwarf::DW_ATE_boolean, 32, ConstString());
  CompilerType int32 =
      m_ast->CreateType(llvm::dwarf::DW_ATE_signed, 32, ConstString());
  CompilerType real32 =
      m_ast->CreateType(llvm::dwarf::DW_ATE_float, 32, ConstString());
  CompilerType complex64 =
      m_ast->CreateType(llvm::dwarf::DW_ATE_complex_float, 64, ConstString());

  EXPECT_STREQ(logical32.GetTypeName().GetCString(), "LOGICAL");
  EXPECT_STREQ(int32.GetTypeName().GetCString(), "INTEGER");
  EXPECT_STREQ(real32.GetTypeName().GetCString(), "REAL");
  EXPECT_STREQ(complex64.GetTypeName().GetCString(), "COMPLEX");
}

TEST_F(TestTypeSystemFortran, TestFortranFunction) {
  CompilerType int_param =
      m_ast->CreateType(llvm::dwarf::DW_ATE_signed, 32, ConstString("INTEGER"));
  CompilerType real_param =
      m_ast->CreateType(llvm::dwarf::DW_ATE_float, 64, ConstString("REAL(8)"));

  llvm::SmallVector<CompilerType, 2> params = {int_param, real_param};

  CompilerType func_type =
      m_ast->GetOrCreateFortranFunction(ConstString("my_subroutine"), params);
  EXPECT_TRUE(func_type.IsValid());

  auto *fortran_func =
      static_cast<FortranFunction *>(func_type.GetOpaqueQualType());
  EXPECT_EQ(fortran_func->GetKind(), FortranType::KIND_FUNCTION);
  EXPECT_EQ(fortran_func->GetNumberOfParameters(), 2U);
  EXPECT_EQ(fortran_func->GetName().GetStringRef(), "my_subroutine");
}

TEST_F(TestTypeSystemFortran, TestFoldingSetDeduplication) {
  CompilerType int1 =
      m_ast->CreateType(llvm::dwarf::DW_ATE_signed, 32, ConstString("INTEGER"));

  CompilerType int2 =
      m_ast->CreateType(llvm::dwarf::DW_ATE_signed, 32, ConstString("INTEGER"));

  EXPECT_EQ(int1.GetOpaqueQualType(), int2.GetOpaqueQualType());
}

TEST_F(TestTypeSystemFortran, TestGetBasicTypeFromAST) {
  CompilerType int_type = m_ast->GetBasicTypeFromAST(eBasicTypeInt);
  EXPECT_TRUE(int_type.IsValid());
  EXPECT_STREQ(int_type.GetTypeName().GetCString(), "INTEGER");

  auto bitsize_or_err = int_type.GetBitSize(nullptr);
  ASSERT_THAT_EXPECTED(bitsize_or_err, llvm::Succeeded());
  EXPECT_EQ(*bitsize_or_err, 32U);

  CompilerType complex_type =
      m_ast->GetBasicTypeFromAST(eBasicTypeDoubleComplex);
  EXPECT_TRUE(complex_type.IsValid());
  EXPECT_STREQ(complex_type.GetTypeName().GetCString(), "COMPLEX(KIND=8)");
}