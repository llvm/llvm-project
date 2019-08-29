//===-- DWARFASTParserClangTests.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "Plugins/SymbolFile/DWARF/DWARFASTParserClang.h"
#include "Plugins/SymbolFile/DWARF/DWARFDIE.h"

using namespace lldb;
using namespace lldb_private;

namespace {
class DWARFASTParserClangStub : public DWARFASTParserClang {
public:
  using DWARFASTParserClang::DWARFASTParserClang;
  using DWARFASTParserClang::LinkDeclContextToDIE;
};
} // namespace

// If your implementation needs to dereference the dummy pointers we are
// defining here, causing this test to fail, feel free to delete it.
TEST(DWARFASTParserClangTests,
     TestGetDIEForDeclContextReturnsOnlyMatchingEntries) {
  ClangASTContext ast_ctx;
  DWARFASTParserClangStub ast_parser(ast_ctx);

  DWARFUnit *unit = nullptr;
  DWARFDIE die1(unit, (DWARFDebugInfoEntry *)1LL);
  DWARFDIE die2(unit, (DWARFDebugInfoEntry *)2LL);
  DWARFDIE die3(unit, (DWARFDebugInfoEntry *)3LL);
  DWARFDIE die4(unit, (DWARFDebugInfoEntry *)4LL);
  ast_parser.LinkDeclContextToDIE((clang::DeclContext *)1LL, die1);
  ast_parser.LinkDeclContextToDIE((clang::DeclContext *)2LL, die2);
  ast_parser.LinkDeclContextToDIE((clang::DeclContext *)2LL, die3);
  ast_parser.LinkDeclContextToDIE((clang::DeclContext *)3LL, die4);

  auto die_list = ast_parser.GetDIEForDeclContext(
      CompilerDeclContext(nullptr, (clang::DeclContext *)2LL));
  ASSERT_EQ(2u, die_list.size());
  ASSERT_EQ(die2, die_list[0]);
  ASSERT_EQ(die3, die_list[1]);
}
