//===-- DWARFASTParserFortran.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFASTParserFortran.h"

using namespace lldb;
using namespace lldb_private;

DWARFASTParserFortran::DWARFASTParserFortran(
    lldb_private::TypeSystemFortran m_ast)
    : lldb_private::plugin::dwarf::DWARFASTParser(Kind::DWARFASTParserFortran),
      m_ast(m_ast) {}

DWARFASTParserFortran::~DWARFASTParserFortran() {}

lldb::TypeSP DWARFASTParserFortran::ParseTypeFromDWARF(
    const lldb_private::SymbolContext &sc,
    const lldb_private::plugin::dwarf::DWARFDIE &die, bool *type_is_new_ptr) {
  // TODO
}

lldb_private::Function *DWARFASTParserFortran::ParseFunctionFromDWARF(
    lldb_private::CompileUnit &comp_unit,
    const lldb_private::plugin::dwarf::DWARFDIE &die,
    lldb_private::AddressRanges ranges) {
  // TODO
}

bool DWARFASTParserFortran::CompleteTypeFromDWARF(
    const lldb_private::plugin::dwarf::DWARFDIE &die, lldb_private::Type *type,
    const lldb_private::CompilerType &compiler_type) {
  // TODO
  return false;
}
