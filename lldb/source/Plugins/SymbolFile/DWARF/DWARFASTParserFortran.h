//===-- DWARFASTParserFortran.h -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFASTPARSERFORTRAN_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFASTPARSERFORTRAN_H

#include "DWARFASTParser.h"
#include "Plugins/SymbolFile/DWARF/DWARFDIE.h"
#include "Plugins/TypeSystem/Fortran/TypeSystemFortran.h"

namespace lldb_private {
class CompileUnit;
class ExecutionContext;
class TypeSystemFortran;
} // namespace lldb_private

class DWARFASTParserFortran
    : public lldb_private::plugin::dwarf::DWARFASTParser {
public:
  DWARFASTParserFortran(lldb_private::TypeSystemFortran &ast);

  ~DWARFASTParserFortran() override;

  lldb::TypeSP
  ParseTypeFromDWARF(const lldb_private::SymbolContext &sc,
                     const lldb_private::plugin::dwarf::DWARFDIE &die,
                     bool *type_is_new_ptr) override;

  lldb_private::Function *
  ParseFunctionFromDWARF(lldb_private::CompileUnit &comp_unit,
                         const lldb_private::plugin::dwarf::DWARFDIE &die,
                         lldb_private::AddressRanges ranges) override;

  bool CompleteTypeFromDWARF(
      const lldb_private::plugin::dwarf::DWARFDIE &die,
      lldb_private::Type *type,
      const lldb_private::CompilerType &compiler_type) override {
    return false;
  }

  // TODO: The following functions are left intentionally blank and will be
  // populated in a future patch
  lldb_private::ConstString ConstructDemangledNameFromDWARF(
      const lldb_private::plugin::dwarf::DWARFDIE &die) override {
    return lldb_private::ConstString();
  }

  lldb_private::CompilerDecl GetDeclForUIDFromDWARF(
      const lldb_private::plugin::dwarf::DWARFDIE &die) override {
    return lldb_private::CompilerDecl();
  }

  lldb_private::CompilerDeclContext GetDeclContextForUIDFromDWARF(
      const lldb_private::plugin::dwarf::DWARFDIE &die) override {
    return lldb_private::CompilerDeclContext();
  }

  lldb_private::CompilerDeclContext GetDeclContextContainingUIDFromDWARF(
      const lldb_private::plugin::dwarf::DWARFDIE &die) override {
    return lldb_private::CompilerDeclContext();
  }

  void EnsureAllDIEsInDeclContextHaveBeenParsed(
      lldb_private::CompilerDeclContext decl_context) override {}

  std::string GetDIEClassTemplateParams(
      lldb_private::plugin::dwarf::DWARFDIE die) override {
    return {};
  }

private:
  lldb_private::TypeSystemFortran &m_ast;
};

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFASTPARSERFORTRAN_H
