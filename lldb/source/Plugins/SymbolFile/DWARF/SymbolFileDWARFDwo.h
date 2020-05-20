//===-- SymbolFileDWARFDwo.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_SYMBOLFILEDWARFDWO_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_SYMBOLFILEDWARFDWO_H

#include "SymbolFileDWARF.h"

class SymbolFileDWARFDwo : public SymbolFileDWARF {
  /// LLVM RTTI support.
  static char ID;

public:
  /// LLVM RTTI support.
  /// \{
  bool isA(const void *ClassID) const override {
    return ClassID == &ID || SymbolFileDWARF::isA(ClassID);
  }
  static bool classof(const SymbolFile *obj) { return obj->isA(&ID); }
  /// \}

  SymbolFileDWARFDwo(SymbolFileDWARF &m_base_symbol_file,
                     lldb::ObjectFileSP objfile, uint32_t id);

  ~SymbolFileDWARFDwo() override = default;

  DWARFCompileUnit *GetDWOCompileUnitForHash(uint64_t hash);

  void GetObjCMethods(lldb_private::ConstString class_name,
                      llvm::function_ref<bool(DWARFDIE die)> callback) override;

  llvm::Expected<lldb_private::TypeSystem &>
  GetTypeSystemForLanguage(lldb::LanguageType language) override;

  DWARFDIE
  GetDIE(const DIERef &die_ref) override;

  llvm::Optional<uint32_t> GetDwoNum() override { return GetID() >> 32; }

protected:
  DIEToTypePtr &GetDIEToType() override;

  DIEToVariableSP &GetDIEToVariable() override;

  DIEToClangType &GetForwardDeclDieToClangType() override;

  ClangTypeToDIE &GetForwardDeclClangTypeToDie() override;

  UniqueDWARFASTTypeMap &GetUniqueDWARFASTTypeMap() override;

  lldb::TypeSP FindDefinitionTypeForDWARFDeclContext(
      const DWARFDeclContext &die_decl_ctx) override;

  lldb::TypeSP FindCompleteObjCDefinitionTypeForDIE(
      const DWARFDIE &die, lldb_private::ConstString type_name,
      bool must_be_implementation) override;

  SymbolFileDWARF &GetBaseSymbolFile() { return m_base_symbol_file; }

  /// If this file contains exactly one compile unit, this function will return
  /// it. Otherwise it returns nullptr.
  DWARFCompileUnit *FindSingleCompileUnit();

  SymbolFileDWARF &m_base_symbol_file;
};

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_SYMBOLFILEDWARFDWO_H
