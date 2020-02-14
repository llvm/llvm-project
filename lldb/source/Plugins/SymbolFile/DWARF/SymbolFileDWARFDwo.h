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

  size_t GetObjCMethodDIEOffsets(lldb_private::ConstString class_name,
                                 DIEArray &method_die_offsets) override;

  llvm::Expected<lldb_private::TypeSystem &>
  GetTypeSystemForLanguage(lldb::LanguageType language) override;

  DWARFDIE
  GetDIE(const DIERef &die_ref) override;

  llvm::Optional<uint32_t> GetDwoNum() override { return GetID() >> 32; }

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

  SymbolFileDWARF &GetBaseSymbolFile() { return m_base_symbol_file; }

  /// If this file contains exactly one compile unit, this function will return
  /// it. Otherwise it returns nullptr.
  DWARFCompileUnit *FindSingleCompileUnit();

  SymbolFileDWARF &m_base_symbol_file;
};

#endif // SymbolFileDWARFDwo_SymbolFileDWARFDwo_h_
