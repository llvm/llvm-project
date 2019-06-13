//===-- SymbolFileDWARFDwo.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARFDwo_SymbolFileDWARFDwo_h_
#define SymbolFileDWARFDwo_SymbolFileDWARFDwo_h_

#include "SymbolFileDWARF.h"

class SymbolFileDWARFDwo : public SymbolFileDWARF {
public:
  SymbolFileDWARFDwo(lldb::ObjectFileSP objfile, DWARFCompileUnit &dwarf_cu);

  ~SymbolFileDWARFDwo() override = default;

  lldb::CompUnitSP ParseCompileUnit(DWARFCompileUnit &dwarf_cu) override;

  DWARFUnit *GetCompileUnit();

  DWARFUnit *
  GetDWARFCompileUnit(lldb_private::CompileUnit *comp_unit) override;

  lldb_private::DWARFExpression::LocationListFormat
  GetLocationListFormat() const override;

  size_t GetObjCMethodDIEOffsets(lldb_private::ConstString class_name,
                                 DIEArray &method_die_offsets) override;

  lldb_private::TypeSystem *
  GetTypeSystemForLanguage(lldb::LanguageType language) override;

  DWARFDIE
  GetDIE(const DIERef &die_ref) override;

  std::unique_ptr<SymbolFileDWARFDwo>
  GetDwoSymbolFileForCompileUnit(DWARFUnit &dwarf_cu,
                                 const DWARFDebugInfoEntry &cu_die) override {
    return nullptr;
  }

  DWARFCompileUnit *GetBaseCompileUnit() override { return &m_base_dwarf_cu; }

protected:
  void LoadSectionData(lldb::SectionType sect_type,
                       lldb_private::DWARFDataExtractor &data) override;

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

  SymbolFileDWARF *GetBaseSymbolFile();

  lldb::ObjectFileSP m_obj_file_sp;
  DWARFCompileUnit &m_base_dwarf_cu;
};

#endif // SymbolFileDWARFDwo_SymbolFileDWARFDwo_h_
