//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Language/CPlusPlus/LibStdcpp.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/Symbol/ClangTestUtils.h"
#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/ValueObject/ValueObject.h"

#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

namespace {

/// A minimal ValueObject mock that returns null from GetChildAtIndex for a
/// specified index. This simulates the crash scenario where GetChildAtIndex()
/// returns null due to incomplete debug info or type system errors.
class ValueObjectWithNullChild : public ValueObject {
public:
  static lldb::ValueObjectSP Create(CompilerType type, ConstString name,
                                    size_t null_child_idx) {
    auto manager = ValueObjectManager::Create();
    auto *obj =
        new ValueObjectWithNullChild(*manager, type, name, null_child_idx);
    return obj->GetSP();
  }

  ~ValueObjectWithNullChild() override = default;

  llvm::Expected<uint64_t> GetByteSize() override { return 4; }

  lldb::ValueType GetValueType() const override {
    return lldb::eValueTypeConstResult;
  }

  llvm::Expected<uint32_t> CalculateNumChildren(uint32_t max) override {
    auto num_or_err = m_type.GetNumChildren(true, nullptr);
    if (!num_or_err)
      return num_or_err.takeError();
    return *num_or_err;
  }

  ConstString GetTypeName() override { return m_type.GetTypeName(); }

  ConstString GetDisplayTypeName() override { return GetTypeName(); }

  bool IsInScope() override { return true; }

protected:
  bool UpdateValue() override {
    m_error.Clear();
    return true;
  }

  CompilerType GetCompilerTypeImpl() override { return m_type; }

  /// This is the key method - return null for the specified child index
  /// to simulate the crash scenario.
  ValueObject *CreateChildAtIndex(size_t idx) override {
    if (idx == m_null_child_idx) {
      return nullptr; // Return null to trigger the crash scenario!
    }
    return ValueObject::CreateChildAtIndex(idx);
  }

private:
  ValueObjectWithNullChild(ValueObjectManager &manager, CompilerType type,
                           ConstString name, size_t null_child_idx)
      : ValueObject(nullptr, manager), m_type(type),
        m_null_child_idx(null_child_idx) {
    SetName(name);
  }

  CompilerType m_type;
  size_t m_null_child_idx;
};

class LibStdcppTupleTest : public ::testing::Test {
public:
  SubsystemRAII<FileSystem, HostInfo> m_subsystems;

  void SetUp() override {
    m_holder = std::make_unique<clang_utils::TypeSystemClangHolder>("test");
    m_type_system = m_holder->GetAST();
  }

  /// Create a struct type with a child named "std::_Tuple_impl<0, int>"
  /// to trigger the tuple synthetic frontend's child iteration.
  CompilerType CreateTypeWithTupleImplChild() {
    // Create outer type
    CompilerType outer_type = m_type_system->CreateRecordType(
        m_type_system->getASTContext().getTranslationUnitDecl(),
        OptionalClangModuleID(), lldb::AccessType::eAccessPublic,
        "std::tuple<int>", 0, lldb::LanguageType::eLanguageTypeC_plus_plus);

    // Create inner _Tuple_impl type
    CompilerType inner_type = m_type_system->CreateRecordType(
        m_type_system->getASTContext().getTranslationUnitDecl(),
        OptionalClangModuleID(), lldb::AccessType::eAccessPublic,
        "std::_Tuple_impl<0, int>", 0,
        lldb::LanguageType::eLanguageTypeC_plus_plus);

    TypeSystemClang::StartTagDeclarationDefinition(inner_type);
    TypeSystemClang::CompleteTagDeclarationDefinition(inner_type);

    // Add the inner type as a field of the outer type
    TypeSystemClang::StartTagDeclarationDefinition(outer_type);
    m_type_system->AddFieldToRecordType(outer_type, "std::_Tuple_impl<0, int>",
                                        inner_type,
                                        lldb::AccessType::eAccessPublic, 0);
    TypeSystemClang::CompleteTagDeclarationDefinition(outer_type);

    return outer_type;
  }

  TypeSystemClang *m_type_system;

private:
  std::unique_ptr<clang_utils::TypeSystemClangHolder> m_holder;
};

} // anonymous namespace

TEST_F(LibStdcppTupleTest, CreatorHandlesNullValueObject) {
  auto *frontend = formatters::LibStdcppTupleSyntheticFrontEndCreator(
      nullptr, lldb::ValueObjectSP());
  EXPECT_EQ(frontend, nullptr);
}

/// This test verifies the null child handling fix.
/// It creates a ValueObject that returns null from GetChildAtIndex(0),
/// simulating the crash scenario from incomplete debug info.
/// WITHOUT the fix (null check), this test will crash with SIGSEGV.
/// WITH the fix, this test passes.
TEST_F(LibStdcppTupleTest, UpdateHandlesNullChild) {
  CompilerType type = CreateTypeWithTupleImplChild();

  // Create a ValueObject that returns null for child at index 0
  auto valobj_sp = ValueObjectWithNullChild::Create(
      type, ConstString("test_tuple"), 0 /* null_child_idx */);
  ASSERT_TRUE(valobj_sp);

  // Verify our mock returns null for child 0
  ASSERT_FALSE(valobj_sp->GetChildAtIndex(0));

  // Create the frontend - this calls Update() which iterates through children.
  // WITHOUT the null check fix, this crashes with SIGSEGV when trying to call
  // GetName() on a null child_sp.
  // WITH the fix, this succeeds because null children are skipped.
  auto *frontend =
      formatters::LibStdcppTupleSyntheticFrontEndCreator(nullptr, valobj_sp);
  ASSERT_NE(frontend, nullptr);

  // If we get here, the null check worked.
  auto num_children = frontend->CalculateNumChildren();
  ASSERT_TRUE(static_cast<bool>(num_children));

  delete frontend;
}
