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
#include "TestingSupport/Symbol/YAMLModuleTester.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

namespace {
class DWARFASTParserClangTests : public testing::Test {};

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
  const char *yamldata =
      "debug_abbrev:\n"
      "  - Code:            0x00000001\n"
      "    Tag:             DW_TAG_compile_unit\n"
      "    Children:        DW_CHILDREN_yes\n"
      "    Attributes:\n"
      "      - Attribute:       DW_AT_language\n"
      "        Form:            DW_FORM_data2\n"
      "  - Code:            0x00000002\n"
      "    Tag:             DW_TAG_base_type\n"
      "    Children:        DW_CHILDREN_no\n"
      "    Attributes:\n"
      "      - Attribute:       DW_AT_encoding\n"
      "        Form:            DW_FORM_data1\n"
      "      - Attribute:       DW_AT_byte_size\n"
      "        Form:            DW_FORM_data1\n"
      "debug_info:\n"
      "  - Length:          0\n"
      "    Version:         4\n"
      "    AbbrOffset:      0\n"
      "    AddrSize:        8\n"
      "    Entries:\n"
      "      - AbbrCode:        0x00000001\n"
      "        Values:\n"
      "          - Value:           0x000000000000000C\n"
      // 0x0000000e:
      "      - AbbrCode:        0x00000002\n"
      "        Values:\n"
      "          - Value:           0x0000000000000007\n" // DW_ATE_unsigned
      "          - Value:           0x0000000000000004\n"
      // 0x00000011:
      "      - AbbrCode:        0x00000002\n"
      "        Values:\n"
      "          - Value:           0x0000000000000007\n" // DW_ATE_unsigned
      "          - Value:           0x0000000000000008\n"
      // 0x00000014:
      "      - AbbrCode:        0x00000002\n"
      "        Values:\n"
      "          - Value:           0x0000000000000005\n" // DW_ATE_signed
      "          - Value:           0x0000000000000008\n"
      // 0x00000017:
      "      - AbbrCode:        0x00000002\n"
      "        Values:\n"
      "          - Value:           0x0000000000000008\n" // DW_ATE_unsigned_char
      "          - Value:           0x0000000000000001\n"
      ""
      "      - AbbrCode:        0x00000000\n"
      "        Values:          []\n";

  YAMLModuleTester t(yamldata, "i386-unknown-linux");
  ASSERT_TRUE((bool)t.GetDwarfUnit());

  TypeSystemClang ast_ctx("dummy ASTContext", HostInfoBase::GetTargetTriple());
  DWARFASTParserClangStub ast_parser(ast_ctx);

  DWARFUnit *unit = t.GetDwarfUnit().get();
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

