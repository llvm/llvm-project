//===-- SymbolFileDWARFDwo.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolFileDWARFDwo.h"

#include "lldb/Core/Section.h"
#include "lldb/Expression/DWARFExpression.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/LLDBAssert.h"

#include "DWARFCompileUnit.h"
#include "DWARFDebugInfo.h"
#include "DWARFUnit.h"

using namespace lldb;
using namespace lldb_private;

SymbolFileDWARFDwo::SymbolFileDWARFDwo(ObjectFileSP objfile,
                                       DWARFCompileUnit &dwarf_cu)
    : SymbolFileDWARF(objfile.get(), objfile->GetSectionList(
                                         /*update_module_section_list*/ false)),
      m_obj_file_sp(objfile), m_base_dwarf_cu(dwarf_cu) {
  SetID(((lldb::user_id_t)dwarf_cu.GetID()) << 32);
}

void SymbolFileDWARFDwo::LoadSectionData(lldb::SectionType sect_type,
                                         DWARFDataExtractor &data) {
  const SectionList *section_list =
      m_obj_file->GetSectionList(false /* update_module_section_list */);
  if (section_list) {
    SectionSP section_sp(section_list->FindSectionByType(sect_type, true));
    if (section_sp) {

      if (m_obj_file->ReadSectionData(section_sp.get(), data) != 0)
        return;

      data.Clear();
    }
  }

  SymbolFileDWARF::LoadSectionData(sect_type, data);
}

lldb::CompUnitSP
SymbolFileDWARFDwo::ParseCompileUnit(DWARFCompileUnit &dwarf_cu) {
  assert(GetCompileUnit() == &dwarf_cu &&
         "SymbolFileDWARFDwo::ParseCompileUnit called with incompatible "
         "compile unit");
  return GetBaseSymbolFile()->ParseCompileUnit(m_base_dwarf_cu);
}

DWARFUnit *SymbolFileDWARFDwo::GetCompileUnit() {
  // Only dwo files with 1 compile unit is supported
  if (GetNumCompileUnits() == 1)
    return DebugInfo()->GetUnitAtIndex(0);
  else
    return nullptr;
}

DWARFUnit *
SymbolFileDWARFDwo::GetDWARFCompileUnit(lldb_private::CompileUnit *comp_unit) {
  return GetCompileUnit();
}

SymbolFileDWARF::DIEToTypePtr &SymbolFileDWARFDwo::GetDIEToType() {
  return GetBaseSymbolFile()->GetDIEToType();
}

SymbolFileDWARF::DIEToVariableSP &SymbolFileDWARFDwo::GetDIEToVariable() {
  return GetBaseSymbolFile()->GetDIEToVariable();
}

SymbolFileDWARF::DIEToClangType &
SymbolFileDWARFDwo::GetForwardDeclDieToClangType() {
  return GetBaseSymbolFile()->GetForwardDeclDieToClangType();
}

SymbolFileDWARF::ClangTypeToDIE &
SymbolFileDWARFDwo::GetForwardDeclClangTypeToDie() {
  return GetBaseSymbolFile()->GetForwardDeclClangTypeToDie();
}

size_t SymbolFileDWARFDwo::GetObjCMethodDIEOffsets(
    lldb_private::ConstString class_name, DIEArray &method_die_offsets) {
  return GetBaseSymbolFile()->GetObjCMethodDIEOffsets(
      class_name, method_die_offsets);
}

UniqueDWARFASTTypeMap &SymbolFileDWARFDwo::GetUniqueDWARFASTTypeMap() {
  return GetBaseSymbolFile()->GetUniqueDWARFASTTypeMap();
}

lldb::TypeSP SymbolFileDWARFDwo::FindDefinitionTypeForDWARFDeclContext(
    const DWARFDeclContext &die_decl_ctx) {
  return GetBaseSymbolFile()->FindDefinitionTypeForDWARFDeclContext(
      die_decl_ctx);
}

lldb::TypeSP SymbolFileDWARFDwo::FindCompleteObjCDefinitionTypeForDIE(
    const DWARFDIE &die, lldb_private::ConstString type_name,
    bool must_be_implementation) {
  return GetBaseSymbolFile()->FindCompleteObjCDefinitionTypeForDIE(
      die, type_name, must_be_implementation);
}

SymbolFileDWARF *SymbolFileDWARFDwo::GetBaseSymbolFile() {
  return m_base_dwarf_cu.GetSymbolFileDWARF();
}

DWARFExpression::LocationListFormat
SymbolFileDWARFDwo::GetLocationListFormat() const {
  return DWARFExpression::SplitDwarfLocationList;
}

TypeSystem *
SymbolFileDWARFDwo::GetTypeSystemForLanguage(LanguageType language) {
  return GetBaseSymbolFile()->GetTypeSystemForLanguage(language);
}

DWARFDIE
SymbolFileDWARFDwo::GetDIE(const DIERef &die_ref) {
  lldbassert(die_ref.cu_offset == m_base_dwarf_cu.GetOffset() ||
             die_ref.cu_offset == DW_INVALID_OFFSET);
  return DebugInfo()->GetDIEForDIEOffset(die_ref.section, die_ref.die_offset);
}
