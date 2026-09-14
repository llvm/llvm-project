//===-- DWARFASTParserFortranTests.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/SymbolFile/DWARF/DWARFASTParserFortran.h"
#include "Plugins/SymbolFile/DWARF/DWARFCompileUnit.h"
#include "Plugins/SymbolFile/DWARF/DWARFDIE.h"
#include "TestingSupport/Symbol/YAMLModuleTester.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Debugger.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::plugin::dwarf;
using namespace llvm::dwarf;

namespace {

class TypeSystemFortranHolder {
  std::shared_ptr<TypeSystemFortran> m_ast;

public:
  TypeSystemFortranHolder() : m_ast(std::make_shared<TypeSystemFortran>()) {}
  TypeSystemFortran *GetAST() const { return m_ast.get(); }
};

class DWARFASTParserFortranTests : public testing::Test {
  void SetUp() override {
    std::call_once(TestUtilities::g_debugger_initialize_flag, []() {
      Debugger::Initialize(nullptr);
      TypeSystemFortran::Initialize();
    });
  }
};

/// Helper structure for DWARFASTParserFortran tests that want to parse DWARF
/// generated using yaml2obj. On construction parses the supplied YAML data
/// into a DWARF module and thereafter vends a DWARFASTParserFortran and
/// TypeSystemFortran that are guaranteed to live for the duration of this
/// object.
class DWARFASTParserFortranYAMLTester {
public:
  DWARFASTParserFortranYAMLTester(llvm::StringRef yaml_data)
      : m_module_tester(yaml_data) {}

  DWARFDIE GetCUDIE() {
    DWARFUnit *unit = m_module_tester.GetDwarfUnit();
    assert(unit);

    const DWARFDebugInfoEntry *cu_entry = unit->DIE().GetDIE();
    assert(cu_entry->Tag() == DW_TAG_compile_unit);

    return DWARFDIE(unit, cu_entry);
  }

  DWARFASTParserFortran &GetParser() {
    auto *parser = GetTypeSystem().GetDWARFParser();

    assert(llvm::isa_and_nonnull<DWARFASTParserFortran>(parser));

    return *llvm::cast<DWARFASTParserFortran>(parser);
  }

  TypeSystemFortran &GetTypeSystem() {
    ModuleSP module_sp = m_module_tester.GetModule();
    assert(module_sp);

    SymbolFile *symfile = module_sp->GetSymbolFile();
    assert(symfile);

    TypeSystemSP ts_sp = llvm::cantFail(symfile->GetTypeSystemForLanguage(
        lldb::LanguageType::eLanguageTypeFortran90));

    assert(llvm::isa_and_nonnull<TypeSystemFortran>(ts_sp.get()));

    return llvm::cast<TypeSystemFortran>(*ts_sp);
  }

private:
  YAMLModuleTester m_module_tester;
};
} // namespace

TEST_F(DWARFASTParserFortranTests, EnsureBaseTypeParsingWorks) {
  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_X86_64
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
            - Attribute:       DW_AT_name
              Form:            DW_FORM_string
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
            - Value:           0x000000000000000E # DW_LANG_Fortran95
        # integer(kind=4)
        - AbbrCode:        0x00000002
          Values:
            - CStr:            'integer(kind=4)'
            - Value:           0x0000000000000005 # DW_ATE_signed
            - Value:           0x0000000000000004
        # integer(kind=8)
        - AbbrCode:        0x00000002
          Values:
            - CStr:            'integer(kind=8)'
            - Value:           0x0000000000000005 # DW_ATE_signed
            - Value:           0x0000000000000008
        # real(kind=4)
        - AbbrCode:        0x00000002
          Values:
            - CStr:            'real(kind=4)'
            - Value:           0x0000000000000004 # DW_ATE_float
            - Value:           0x0000000000000004
        # complex(kind=4)
        - AbbrCode:        0x00000002
          Values:
            - CStr:            'complex(kind=4)'
            - Value:           0x0000000000000003 # DW_ATE_complex_float
            - Value:           0x0000000000000008
        # logical(kind=4)
        - AbbrCode:        0x00000002
          Values:
            - CStr:            'logical(kind=4)'
            - Value:           0x0000000000000002 # DW_ATE_boolean
            - Value:           0x0000000000000004
)";

  DWARFASTParserFortranYAMLTester tester(yamldata);
  DWARFDIE cu_die = tester.GetCUDIE();

  struct ExpectedTypeInfo {
    const char *name;
    uint64_t byte_size;
    lldb::BasicType basic_type;
  };

  const std::vector<ExpectedTypeInfo> expected_types = {
      {"INTEGER", 4, lldb::eBasicTypeInt},
      {"INTEGER(KIND=8)", 8, lldb::eBasicTypeLongLong},
      {"REAL", 4, lldb::eBasicTypeFloat},
      {"COMPLEX", 8, lldb::eBasicTypeFloatComplex},
      {"LOGICAL", 4, lldb::eBasicTypeBool},
  };

  size_t type_idx = 0;
  for (DWARFDIE child_die : cu_die.children()) {
    ASSERT_EQ(child_die.Tag(), DW_TAG_base_type);
    ASSERT_LT(type_idx, expected_types.size());

    const auto &expected = expected_types[type_idx++];
    SymbolContext sc;

    bool is_new_type = false;
    lldb::TypeSP type_sp =
        tester.GetParser().ParseTypeFromDWARF(sc, child_die, &is_new_type);

    ASSERT_NE(type_sp, nullptr);
    EXPECT_TRUE(is_new_type);

    EXPECT_EQ(type_sp->GetForwardCompilerType().GetTypeName().GetString(),
              expected.name);

    llvm::Expected<uint64_t> size_or_err = type_sp->GetByteSize(nullptr);
    ASSERT_FALSE(!size_or_err);
    EXPECT_EQ(*size_or_err, expected.byte_size);

    EXPECT_EQ(type_sp->GetForwardCompilerType().GetBasicTypeEnumeration(),
              expected.basic_type);

    bool is_new_type_cached = true;
    lldb::TypeSP cached_type_sp = tester.GetParser().ParseTypeFromDWARF(
        sc, child_die, &is_new_type_cached);

    ASSERT_NE(cached_type_sp, nullptr);
    EXPECT_FALSE(is_new_type_cached);
    EXPECT_EQ(type_sp, cached_type_sp);
  }

  // Ensure all expected DIEs were iterated
  EXPECT_EQ(type_idx, expected_types.size());
}
