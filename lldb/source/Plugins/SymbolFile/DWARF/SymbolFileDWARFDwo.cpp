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
#include "llvm/Support/Casting.h"

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
  return GetBaseSymbolFile().ParseCompileUnit(m_base_dwarf_cu);
}

DWARFCompileUnit *SymbolFileDWARFDwo::GetCompileUnit() {
  if (!m_cu)
    m_cu = ComputeCompileUnit();
  return m_cu;
}

DWARFCompileUnit *SymbolFileDWARFDwo::ComputeCompileUnit() {
  DWARFDebugInfo *debug_info = DebugInfo();
  if (!debug_info)
    return nullptr;

  // Right now we only support dwo files with one compile unit. If we don't have
  // type units, we can just check for the unit count.
  if (!debug_info->ContainsTypeUnits() && debug_info->GetNumUnits() == 1)
    return llvm::cast<DWARFCompileUnit>(debug_info->GetUnitAtIndex(0));

  // Otherwise, we have to run through all units, and find the compile unit that
  // way.
  DWARFCompileUnit *cu = nullptr;
  for (size_t i = 0; i < debug_info->GetNumUnits(); ++i) {
    if (auto *candidate =
            llvm::dyn_cast<DWARFCompileUnit>(debug_info->GetUnitAtIndex(i))) {
      if (cu)
        return nullptr; // More that one CU found.
      cu = candidate;
    }
  }
  return cu;
}

DWARFUnit *
SymbolFileDWARFDwo::GetDWARFCompileUnit(lldb_private::CompileUnit *comp_unit) {
  return GetCompileUnit();
}

SymbolFileDWARF::DIEToTypePtr &SymbolFileDWARFDwo::GetDIEToType() {
  return GetBaseSymbolFile().GetDIEToType();
}

SymbolFileDWARF::DIEToVariableSP &SymbolFileDWARFDwo::GetDIEToVariable() {
  return GetBaseSymbolFile().GetDIEToVariable();
}

SymbolFileDWARF::DIEToClangType &
SymbolFileDWARFDwo::GetForwardDeclDieToClangType() {
  return GetBaseSymbolFile().GetForwardDeclDieToClangType();
}

SymbolFileDWARF::ClangTypeToDIE &
SymbolFileDWARFDwo::GetForwardDeclClangTypeToDie() {
  return GetBaseSymbolFile().GetForwardDeclClangTypeToDie();
}

size_t SymbolFileDWARFDwo::GetObjCMethodDIEOffsets(
    lldb_private::ConstString class_name, DIEArray &method_die_offsets) {
  return GetBaseSymbolFile().GetObjCMethodDIEOffsets(class_name,
                                                     method_die_offsets);
}

UniqueDWARFASTTypeMap &SymbolFileDWARFDwo::GetUniqueDWARFASTTypeMap() {
  return GetBaseSymbolFile().GetUniqueDWARFASTTypeMap();
}

lldb::TypeSP SymbolFileDWARFDwo::FindDefinitionTypeForDWARFDeclContext(
    const DWARFDeclContext &die_decl_ctx) {
  return GetBaseSymbolFile().FindDefinitionTypeForDWARFDeclContext(
      die_decl_ctx);
}

lldb::TypeSP SymbolFileDWARFDwo::FindCompleteObjCDefinitionTypeForDIE(
    const DWARFDIE &die, lldb_private::ConstString type_name,
    bool must_be_implementation) {
  return GetBaseSymbolFile().FindCompleteObjCDefinitionTypeForDIE(
      die, type_name, must_be_implementation);
}

SymbolFileDWARF &SymbolFileDWARFDwo::GetBaseSymbolFile() {
  return m_base_dwarf_cu.GetSymbolFileDWARF();
}

DWARFExpression::LocationListFormat
SymbolFileDWARFDwo::GetLocationListFormat() const {
  return DWARFExpression::SplitDwarfLocationList;
}

TypeSystem *
SymbolFileDWARFDwo::GetTypeSystemForLanguage(LanguageType language) {
  return GetBaseSymbolFile().GetTypeSystemForLanguage(language);
}

DWARFDIE
SymbolFileDWARFDwo::GetDIE(const DIERef &die_ref) {
  if (*die_ref.dwo_num() == GetDwoNum())
    return DebugInfo()->GetDIE(die_ref);
  return GetBaseSymbolFile().GetDIE(die_ref);
}
