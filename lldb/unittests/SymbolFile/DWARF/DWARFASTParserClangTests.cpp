//===-- DWARFASTParserClangTests.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/SymbolFile/DWARF/DWARFASTParserClang.h"
#include "Plugins/SymbolFile/DWARF/DWARFCompileUnit.h"
#include "Plugins/SymbolFile/DWARF/DWARFDIE.h"
#include "TestingSupport/Symbol/ClangTestUtils.h"
#include "TestingSupport/Symbol/YAMLModuleTester.h"
#include "lldb/Core/Debugger.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::dwarf;
using namespace lldb_private::plugin::dwarf;

namespace {
static std::once_flag debugger_initialize_flag;

class DWARFASTParserClangTests : public testing::Test {
  void SetUp() override {
    std::call_once(debugger_initialize_flag,
                   []() { Debugger::Initialize(nullptr); });
  }
};

class DWARFASTParserClangStub : public DWARFASTParserClang {
public:
  using DWARFASTParserClang::DWARFASTParserClang;
  using DWARFASTParserClang::LinkDeclContextToDIE;

  std::vector<const clang::DeclContext *> GetDeclContextToDIEMapKeys() {
    std::vector<const clang::DeclContext *> keys;
    for (const auto &it : m_decl_ctx_to_die)
      keys.push_back(it.first);
    return keys;
  }
};
} // namespace

// If your implementation needs to dereference the dummy pointers we are
// defining here, causing this test to fail, feel free to delete it.
TEST_F(DWARFASTParserClangTests,
       EnsureAllDIEsInDeclContextHaveBeenParsedParsesOnlyMatchingEntries) {

  /// Auxiliary debug info.
  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_386
DWARF:
  debug_abbrev:
    - Table:
        - Code:            0x00000001
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
        - Code:            0x00000002
          Tag:             DW_TAG_base_type
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_encoding
              Form:            DW_FORM_data1
            - Attribute:       DW_AT_byte_size
              Form:            DW_FORM_data1
  debug_info:
    - Version:         4
      AddrSize:        8
      Entries:
        - AbbrCode:        0x00000001
          Values:
            - Value:           0x000000000000000C
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x0000000000000007 # DW_ATE_unsigned
            - Value:           0x0000000000000004
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x0000000000000007 # DW_ATE_unsigned
            - Value:           0x0000000000000008
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x0000000000000005 # DW_ATE_signed
            - Value:           0x0000000000000008
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x0000000000000008 # DW_ATE_unsigned_char
            - Value:           0x0000000000000001
        - AbbrCode:        0x00000000
)";

  YAMLModuleTester t(yamldata);
  ASSERT_TRUE((bool)t.GetDwarfUnit());

  auto holder = std::make_unique<clang_utils::TypeSystemClangHolder>("ast");
  auto &ast_ctx = *holder->GetAST();

  DWARFASTParserClangStub ast_parser(ast_ctx);

  DWARFUnit *unit = t.GetDwarfUnit();
  const DWARFDebugInfoEntry *die_first = unit->DIE().GetDIE();
  const DWARFDebugInfoEntry *die_child0 = die_first->GetFirstChild();
  const DWARFDebugInfoEntry *die_child1 = die_child0->GetSibling();
  const DWARFDebugInfoEntry *die_child2 = die_child1->GetSibling();
  const DWARFDebugInfoEntry *die_child3 = die_child2->GetSibling();
  std::vector<DWARFDIE> dies = {
      DWARFDIE(unit, die_child0), DWARFDIE(unit, die_child1),
      DWARFDIE(unit, die_child2), DWARFDIE(unit, die_child3)};
  std::vector<clang::DeclContext *> decl_ctxs = {
      (clang::DeclContext *)1LL, (clang::DeclContext *)2LL,
      (clang::DeclContext *)2LL, (clang::DeclContext *)3LL};
  for (int i = 0; i < 4; ++i)
    ast_parser.LinkDeclContextToDIE(decl_ctxs[i], dies[i]);
  ast_parser.EnsureAllDIEsInDeclContextHaveBeenParsed(
      CompilerDeclContext(nullptr, decl_ctxs[1]));

  EXPECT_THAT(ast_parser.GetDeclContextToDIEMapKeys(),
              testing::UnorderedElementsAre(decl_ctxs[0], decl_ctxs[3]));
}

TEST_F(DWARFASTParserClangTests, TestCallingConventionParsing) {
  // Tests parsing DW_AT_calling_convention values.

  // The DWARF below just declares a list of function types with
  // DW_AT_calling_convention on them.
  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS32
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_386
DWARF:
  debug_str:
    - func1
    - func2
    - func3
    - func4
    - func5
    - func6
    - func7
    - func8
    - func9
  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x1
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
        - Code:            0x2
          Tag:             DW_TAG_subprogram
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_low_pc
              Form:            DW_FORM_addr
            - Attribute:       DW_AT_high_pc
              Form:            DW_FORM_data4
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_calling_convention
              Form:            DW_FORM_data1
            - Attribute:       DW_AT_external
              Form:            DW_FORM_flag_present
  debug_info:
    - Version:         4
      AddrSize:        4
      Entries:
        - AbbrCode:        0x1
          Values:
            - Value:           0xC
        - AbbrCode:        0x2
          Values:
            - Value:           0x0
            - Value:           0x5
            - Value:           0x00
            - Value:           0xCB
            - Value:           0x1
        - AbbrCode:        0x2
          Values:
            - Value:           0x10
            - Value:           0x5
            - Value:           0x06
            - Value:           0xB3
            - Value:           0x1
        - AbbrCode:        0x2
          Values:
            - Value:           0x20
            - Value:           0x5
            - Value:           0x0C
            - Value:           0xB1
            - Value:           0x1
        - AbbrCode:        0x2
          Values:
            - Value:           0x30
            - Value:           0x5
            - Value:           0x12
            - Value:           0xC0
            - Value:           0x1
        - AbbrCode:        0x2
          Values:
            - Value:           0x40
            - Value:           0x5
            - Value:           0x18
            - Value:           0xB2
            - Value:           0x1
        - AbbrCode:        0x2
          Values:
            - Value:           0x50
            - Value:           0x5
            - Value:           0x1E
            - Value:           0xC1
            - Value:           0x1
        - AbbrCode:        0x2
          Values:
            - Value:           0x60
            - Value:           0x5
            - Value:           0x24
            - Value:           0xC2
            - Value:           0x1
        - AbbrCode:        0x2
          Values:
            - Value:           0x70
            - Value:           0x5
            - Value:           0x2a
            - Value:           0xEE
            - Value:           0x1
        - AbbrCode:        0x2
          Values:
            - Value:           0x80
            - Value:           0x5
            - Value:           0x30
            - Value:           0x01
            - Value:           0x1
        - AbbrCode:        0x0
...
)";
  YAMLModuleTester t(yamldata);

  DWARFUnit *unit = t.GetDwarfUnit();
  ASSERT_NE(unit, nullptr);
  const DWARFDebugInfoEntry *cu_entry = unit->DIE().GetDIE();
  ASSERT_EQ(cu_entry->Tag(), DW_TAG_compile_unit);
  DWARFDIE cu_die(unit, cu_entry);

  auto holder = std::make_unique<clang_utils::TypeSystemClangHolder>("ast");
  auto &ast_ctx = *holder->GetAST();
  DWARFASTParserClangStub ast_parser(ast_ctx);

  std::vector<std::string> found_function_types;
  // The DWARF above is just a list of functions. Parse all of them to
  // extract the function types and their calling convention values.
  for (DWARFDIE func : cu_die.children()) {
    ASSERT_EQ(func.Tag(), DW_TAG_subprogram);
    SymbolContext sc;
    bool new_type = false;
    lldb::TypeSP type = ast_parser.ParseTypeFromDWARF(sc, func, &new_type);
    found_function_types.push_back(
        type->GetForwardCompilerType().GetTypeName().AsCString());
  }

  // Compare the parsed function types against the expected list of types.
  const std::vector<std::string> expected_function_types = {
      "void () __attribute__((regcall))",
      "void () __attribute__((fastcall))",
      "void () __attribute__((stdcall))",
      "void () __attribute__((vectorcall))",
      "void () __attribute__((pascal))",
      "void () __attribute__((ms_abi))",
      "void () __attribute__((sysv_abi))",
      "void ()", // invalid calling convention.
      "void ()", // DW_CC_normal -> no attribute
  };
  ASSERT_EQ(found_function_types, expected_function_types);
}

TEST_F(DWARFASTParserClangTests, TestPtrAuthParsing) {
  // Tests parsing values with type DW_TAG_LLVM_ptrauth_type corresponding to
  // explicitly signed raw function pointers

  // This is Dwarf for the following C code:
  // ```
  // void (*__ptrauth(0, 0, 42) a)();
  // ```

  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_AARCH64
DWARF:
  debug_str:
    - a
  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x01
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
        - Code:            0x02
          Tag:             DW_TAG_variable
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_type
              Form:            DW_FORM_ref4
            - Attribute:       DW_AT_external
              Form:            DW_FORM_flag_present
        - Code:            0x03
          Tag:             DW_TAG_LLVM_ptrauth_type
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_type
              Form:            DW_FORM_ref4
            - Attribute:       DW_AT_LLVM_ptrauth_key
              Form:            DW_FORM_data1
            - Attribute:       DW_AT_LLVM_ptrauth_extra_discriminator
              Form:            DW_FORM_data2
        - Code:            0x04
          Tag:             DW_TAG_pointer_type
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_type
              Form:            DW_FORM_ref4
        - Code:            0x05
          Tag:             DW_TAG_subroutine_type
          Children:        DW_CHILDREN_yes
        - Code:            0x06
          Tag:             DW_TAG_unspecified_parameters
          Children:        DW_CHILDREN_no

  debug_info:
    - Version:         5
      UnitType:        DW_UT_compile
      AddrSize:        8
      Entries:
# 0x0c: DW_TAG_compile_unit
#         DW_AT_language [DW_FORM_data2]    (DW_LANG_C99)
        - AbbrCode:        0x01
          Values:
            - Value:           0x0c

# 0x0f:   DW_TAG_variable
#           DW_AT_name [DW_FORM_strp]       (\"a\")
#           DW_AT_type [DW_FORM_ref4]       (0x00000018 \"void (*__ptrauth(0, 0, 0x02a)\")
#           DW_AT_external [DW_FORM_flag_present]   (true)
        - AbbrCode:        0x02
          Values:
            - Value:           0x00
            - Value:           0x18

# 0x18:   DW_TAG_LLVM_ptrauth_type
#           DW_AT_type [DW_FORM_ref4]       (0x00000020 \"void (*)(...)\")
#           DW_AT_LLVM_ptrauth_key [DW_FORM_data1]  (0x00)
#           DW_AT_LLVM_ptrauth_extra_discriminator [DW_FORM_data2]  (0x002a)
        - AbbrCode:        0x03
          Values:
            - Value:           0x20
            - Value:           0x00
            - Value:           0x2a

# 0x20:   DW_TAG_pointer_type
#           DW_AT_type [DW_AT_type [DW_FORM_ref4]       (0x00000025 \"void (...)\")
        - AbbrCode:        0x04
          Values:
            - Value:           0x25

# 0x25:   DW_TAG_subroutine_type
        - AbbrCode:        0x05

# 0x26:     DW_TAG_unspecified_parameters
        - AbbrCode:        0x06

        - AbbrCode:        0x00 # end of child tags of 0x25
        - AbbrCode:        0x00 # end of child tags of 0x0c
...
)";
  YAMLModuleTester t(yamldata);

  DWARFUnit *unit = t.GetDwarfUnit();
  ASSERT_NE(unit, nullptr);
  const DWARFDebugInfoEntry *cu_entry = unit->DIE().GetDIE();
  ASSERT_EQ(cu_entry->Tag(), DW_TAG_compile_unit);
  DWARFDIE cu_die(unit, cu_entry);

  auto holder = std::make_unique<clang_utils::TypeSystemClangHolder>("ast");
  auto &ast_ctx = *holder->GetAST();
  DWARFASTParserClangStub ast_parser(ast_ctx);

  DWARFDIE ptrauth_variable = cu_die.GetFirstChild();
  ASSERT_EQ(ptrauth_variable.Tag(), DW_TAG_variable);
  DWARFDIE ptrauth_type =
      ptrauth_variable.GetAttributeValueAsReferenceDIE(DW_AT_type);
  ASSERT_EQ(ptrauth_type.Tag(), DW_TAG_LLVM_ptrauth_type);

  SymbolContext sc;
  bool new_type = false;
  lldb::TypeSP type_sp =
      ast_parser.ParseTypeFromDWARF(sc, ptrauth_type, &new_type);
  CompilerType compiler_type = type_sp->GetForwardCompilerType();
  ASSERT_EQ(compiler_type.GetPtrAuthKey(), 0U);
  ASSERT_EQ(compiler_type.GetPtrAuthAddressDiversity(), false);
  ASSERT_EQ(compiler_type.GetPtrAuthDiscriminator(), 42U);
}

struct ExtractIntFromFormValueTest : public testing::Test {
  SubsystemRAII<FileSystem, HostInfo> subsystems;
  clang_utils::TypeSystemClangHolder holder;
  TypeSystemClang &ts;

  DWARFASTParserClang parser;
  ExtractIntFromFormValueTest()
      : holder("dummy ASTContext"), ts(*holder.GetAST()), parser(ts) {}

  /// Takes the given integer value, stores it in a DWARFFormValue and then
  /// tries to extract the value back via
  /// DWARFASTParserClang::ExtractIntFromFormValue.
  /// Returns the string representation of the extracted value or the error
  /// that was returned from ExtractIntFromFormValue.
  llvm::Expected<std::string> Extract(clang::QualType qt, uint64_t value) {
    DWARFFormValue form_value;
    form_value.SetUnsigned(value);
    llvm::Expected<llvm::APInt> result =
        parser.ExtractIntFromFormValue(ts.GetType(qt), form_value);
    if (!result)
      return result.takeError();
    llvm::SmallString<16> result_str;
    result->toStringUnsigned(result_str);
    return std::string(result_str.str());
  }

  /// Same as ExtractIntFromFormValueTest::Extract but takes a signed integer
  /// and treats the result as a signed integer.
  llvm::Expected<std::string> ExtractS(clang::QualType qt, int64_t value) {
    DWARFFormValue form_value;
    form_value.SetSigned(value);
    llvm::Expected<llvm::APInt> result =
        parser.ExtractIntFromFormValue(ts.GetType(qt), form_value);
    if (!result)
      return result.takeError();
    llvm::SmallString<16> result_str;
    result->toStringSigned(result_str);
    return std::string(result_str.str());
  }
};

TEST_F(ExtractIntFromFormValueTest, TestBool) {
  using namespace llvm;
  clang::ASTContext &ast = ts.getASTContext();

  EXPECT_THAT_EXPECTED(Extract(ast.BoolTy, 0), HasValue("0"));
  EXPECT_THAT_EXPECTED(Extract(ast.BoolTy, 1), HasValue("1"));
  EXPECT_THAT_EXPECTED(Extract(ast.BoolTy, 2), Failed());
  EXPECT_THAT_EXPECTED(Extract(ast.BoolTy, 3), Failed());
}

TEST_F(ExtractIntFromFormValueTest, TestInt) {
  using namespace llvm;

  clang::ASTContext &ast = ts.getASTContext();

  // Find the min/max values for 'int' on the current host target.
  constexpr int64_t int_max = std::numeric_limits<int>::max();
  constexpr int64_t int_min = std::numeric_limits<int>::min();

  // Check that the bit width of int matches the int width in our type system.
  ASSERT_EQ(sizeof(int) * 8, ast.getIntWidth(ast.IntTy));

  // Check values around int_min.
  EXPECT_THAT_EXPECTED(ExtractS(ast.IntTy, int_min - 2), llvm::Failed());
  EXPECT_THAT_EXPECTED(ExtractS(ast.IntTy, int_min - 1), llvm::Failed());
  EXPECT_THAT_EXPECTED(ExtractS(ast.IntTy, int_min),
                       HasValue(std::to_string(int_min)));
  EXPECT_THAT_EXPECTED(ExtractS(ast.IntTy, int_min + 1),
                       HasValue(std::to_string(int_min + 1)));
  EXPECT_THAT_EXPECTED(ExtractS(ast.IntTy, int_min + 2),
                       HasValue(std::to_string(int_min + 2)));

  // Check values around 0.
  EXPECT_THAT_EXPECTED(ExtractS(ast.IntTy, -128), HasValue("-128"));
  EXPECT_THAT_EXPECTED(ExtractS(ast.IntTy, -10), HasValue("-10"));
  EXPECT_THAT_EXPECTED(ExtractS(ast.IntTy, -1), HasValue("-1"));
  EXPECT_THAT_EXPECTED(ExtractS(ast.IntTy, 0), HasValue("0"));
  EXPECT_THAT_EXPECTED(ExtractS(ast.IntTy, 1), HasValue("1"));
  EXPECT_THAT_EXPECTED(ExtractS(ast.IntTy, 10), HasValue("10"));
  EXPECT_THAT_EXPECTED(ExtractS(ast.IntTy, 128), HasValue("128"));

  // Check values around int_max.
  EXPECT_THAT_EXPECTED(ExtractS(ast.IntTy, int_max - 2),
                       HasValue(std::to_string(int_max - 2)));
  EXPECT_THAT_EXPECTED(ExtractS(ast.IntTy, int_max - 1),
                       HasValue(std::to_string(int_max - 1)));
  EXPECT_THAT_EXPECTED(ExtractS(ast.IntTy, int_max),
                       HasValue(std::to_string(int_max)));
  EXPECT_THAT_EXPECTED(ExtractS(ast.IntTy, int_max + 1), llvm::Failed());
  EXPECT_THAT_EXPECTED(ExtractS(ast.IntTy, int_max + 5), llvm::Failed());

  // Check some values not near an edge case.
  EXPECT_THAT_EXPECTED(ExtractS(ast.IntTy, int_max / 2),
                       HasValue(std::to_string(int_max / 2)));
  EXPECT_THAT_EXPECTED(ExtractS(ast.IntTy, int_min / 2),
                       HasValue(std::to_string(int_min / 2)));
}

TEST_F(ExtractIntFromFormValueTest, TestUnsignedInt) {
  using namespace llvm;

  clang::ASTContext &ast = ts.getASTContext();
  constexpr uint64_t uint_max = std::numeric_limits<uint32_t>::max();

  // Check values around 0.
  EXPECT_THAT_EXPECTED(Extract(ast.UnsignedIntTy, 0), HasValue("0"));
  EXPECT_THAT_EXPECTED(Extract(ast.UnsignedIntTy, 1), HasValue("1"));
  EXPECT_THAT_EXPECTED(Extract(ast.UnsignedIntTy, 1234), HasValue("1234"));

  // Check some values not near an edge case.
  EXPECT_THAT_EXPECTED(Extract(ast.UnsignedIntTy, uint_max / 2),
                       HasValue(std::to_string(uint_max / 2)));

  // Check values around uint_max.
  EXPECT_THAT_EXPECTED(Extract(ast.UnsignedIntTy, uint_max - 2),
                       HasValue(std::to_string(uint_max - 2)));
  EXPECT_THAT_EXPECTED(Extract(ast.UnsignedIntTy, uint_max - 1),
                       HasValue(std::to_string(uint_max - 1)));
  EXPECT_THAT_EXPECTED(Extract(ast.UnsignedIntTy, uint_max),
                       HasValue(std::to_string(uint_max)));
  EXPECT_THAT_EXPECTED(Extract(ast.UnsignedIntTy, uint_max + 1),
                       llvm::Failed());
  EXPECT_THAT_EXPECTED(Extract(ast.UnsignedIntTy, uint_max + 2),
                       llvm::Failed());
}

TEST_F(DWARFASTParserClangTests, TestDefaultTemplateParamParsing) {
  // Tests parsing DW_AT_default_value for template parameters.
  auto BufferOrError = llvm::MemoryBuffer::getFile(
      GetInputFilePath("DW_AT_default_value-test.yaml"), /*IsText=*/true);
  ASSERT_TRUE(BufferOrError);
  YAMLModuleTester t(BufferOrError.get()->getBuffer());

  DWARFUnit *unit = t.GetDwarfUnit();
  ASSERT_NE(unit, nullptr);
  const DWARFDebugInfoEntry *cu_entry = unit->DIE().GetDIE();
  ASSERT_EQ(cu_entry->Tag(), DW_TAG_compile_unit);
  DWARFDIE cu_die(unit, cu_entry);

  auto holder = std::make_unique<clang_utils::TypeSystemClangHolder>("ast");
  auto &ast_ctx = *holder->GetAST();
  DWARFASTParserClangStub ast_parser(ast_ctx);

  llvm::SmallVector<lldb::TypeSP, 2> types;
  for (DWARFDIE die : cu_die.children()) {
    if (die.Tag() == DW_TAG_class_type) {
      SymbolContext sc;
      bool new_type = false;
      types.push_back(ast_parser.ParseTypeFromDWARF(sc, die, &new_type));
    }
  }

  ASSERT_EQ(types.size(), 3U);

  auto check_decl = [](auto const *decl) {
    clang::ClassTemplateSpecializationDecl const *ctsd =
        llvm::dyn_cast_or_null<clang::ClassTemplateSpecializationDecl>(decl);
    ASSERT_NE(ctsd, nullptr);

    auto const &args = ctsd->getTemplateArgs();
    ASSERT_GT(args.size(), 0U);

    for (auto const &arg : args.asArray()) {
      EXPECT_TRUE(arg.getIsDefaulted());
    }
  };

  for (auto const &type_sp : types) {
    ASSERT_NE(type_sp, nullptr);
    auto const *decl = ClangUtil::GetAsTagDecl(type_sp->GetFullCompilerType());
    if (decl->getName() == "bar" || decl->getName() == "baz") {
      check_decl(decl);
    }
  }
}

TEST_F(DWARFASTParserClangTests, TestUniqueDWARFASTTypeMap_CppInsertMapFind) {
  // This tests the behaviour of UniqueDWARFASTTypeMap under
  // following scenario:
  // 1. DWARFASTParserClang parses a forward declaration and
  // inserts it into the UniqueDWARFASTTypeMap.
  // 2. We then MapDeclDIEToDefDIE which updates the map
  // entry with the line number/file information of the definition.
  // 3. Parse the definition DIE, which should return the previously
  // parsed type from the UniqueDWARFASTTypeMap.

  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_AARCH64
DWARF:
  debug_str:
    - Foo

  debug_line:      
    - Version:         4
      MinInstLength:   1
      MaxOpsPerInst:   1
      DefaultIsStmt:   1
      LineBase:        0
      LineRange:       0
      Files:           
        - Name:            main.cpp
          DirIdx:          0
          ModTime:         0
          Length:          0

  debug_abbrev:
    - ID:              0
      Table:
        - Code:            0x01
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
            - Attribute:       DW_AT_stmt_list
              Form:            DW_FORM_sec_offset
        - Code:            0x02
          Tag:             DW_TAG_structure_type
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_declaration
              Form:            DW_FORM_flag_present
        - Code:            0x03
          Tag:             DW_TAG_structure_type
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_name
              Form:            DW_FORM_strp
            - Attribute:       DW_AT_decl_file
              Form:            DW_FORM_data1
            - Attribute:       DW_AT_decl_line
              Form:            DW_FORM_data1

  debug_info:
    - Version:         5
      UnitType:        DW_UT_compile
      AddrSize:        8
      Entries:
# 0x0c: DW_TAG_compile_unit
#         DW_AT_language [DW_FORM_data2]    (DW_LANG_C_plus_plus)
#         DW_AT_stmt_list [DW_FORM_sec_offset]
        - AbbrCode:        0x01
          Values:
            - Value:           0x04
            - Value:           0x0000000000000000

# 0x0d:   DW_TAG_structure_type
#           DW_AT_name [DW_FORM_strp]       (\"Foo\")
#           DW_AT_declaration [DW_FORM_flag_present] (true)
        - AbbrCode:        0x02
          Values:
            - Value:           0x00

# 0x0f:   DW_TAG_structure_type
#           DW_AT_name [DW_FORM_strp]       (\"Foo\")
#           DW_AT_decl_file [DW_FORM_data1] (main.cpp)
#           DW_AT_decl_line [DW_FORM_data1] (3)
        - AbbrCode:        0x03
          Values:
            - Value:           0x00
            - Value:           0x01
            - Value:           0x03

        - AbbrCode:        0x00 # end of child tags of 0x0c
...
)";
  YAMLModuleTester t(yamldata);

  DWARFUnit *unit = t.GetDwarfUnit();
  ASSERT_NE(unit, nullptr);
  const DWARFDebugInfoEntry *cu_entry = unit->DIE().GetDIE();
  ASSERT_EQ(cu_entry->Tag(), DW_TAG_compile_unit);
  ASSERT_EQ(unit->GetDWARFLanguageType(), DW_LANG_C_plus_plus);
  DWARFDIE cu_die(unit, cu_entry);

  auto holder = std::make_unique<clang_utils::TypeSystemClangHolder>("ast");
  auto &ast_ctx = *holder->GetAST();
  DWARFASTParserClangStub ast_parser(ast_ctx);

  DWARFDIE decl_die;
  DWARFDIE def_die;
  for (auto const &die : cu_die.children()) {
    if (die.Tag() != DW_TAG_structure_type)
      continue;

    if (die.GetAttributeValueAsOptionalUnsigned(llvm::dwarf::DW_AT_declaration))
      decl_die = die;
    else
      def_die = die;
  }

  ASSERT_TRUE(decl_die.IsValid());
  ASSERT_TRUE(def_die.IsValid());
  ASSERT_NE(decl_die, def_die);

  ParsedDWARFTypeAttributes attrs(def_die);
  ASSERT_TRUE(attrs.decl.IsValid());

  SymbolContext sc;
  bool new_type = false;
  lldb::TypeSP type_sp = ast_parser.ParseTypeFromDWARF(sc, decl_die, &new_type);
  ASSERT_NE(type_sp, nullptr);

  ast_parser.MapDeclDIEToDefDIE(decl_die, def_die);

  lldb::TypeSP reparsed_type_sp =
      ast_parser.ParseTypeFromDWARF(sc, def_die, &new_type);
  ASSERT_NE(reparsed_type_sp, nullptr);

  ASSERT_EQ(type_sp, reparsed_type_sp);
}
