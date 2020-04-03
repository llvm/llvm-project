//===-- SymbolFileDWARFDwo.cpp --------------------------------------------===//
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

char SymbolFileDWARFDwo::ID;

SymbolFileDWARFDwo::SymbolFileDWARFDwo(SymbolFileDWARF &base_symbol_file,
                                       ObjectFileSP objfile, uint32_t id)
    : SymbolFileDWARF(objfile, objfile->GetSectionList(
                                   /*update_module_section_list*/ false)),
      m_base_symbol_file(base_symbol_file) {
  SetID(user_id_t(id) << 32);

  // Parsing of the dwarf unit index is not thread-safe, so we need to prime it
  // to enable subsequent concurrent lookups.
  m_context.GetAsLLVM().getCUIndex();
}

DWARFCompileUnit *SymbolFileDWARFDwo::GetDWOCompileUnitForHash(uint64_t hash) {
  if (const llvm::DWARFUnitIndex &index = m_context.GetAsLLVM().getCUIndex()) {
    if (const llvm::DWARFUnitIndex::Entry *entry = index.getFromHash(hash)) {
      if (auto *unit_contrib = entry->getContribution())
        return llvm::dyn_cast_or_null<DWARFCompileUnit>(
            DebugInfo().GetUnitAtOffset(DIERef::Section::DebugInfo,
                                        unit_contrib->Offset));
    }
    return nullptr;
  }

  DWARFCompileUnit *cu = FindSingleCompileUnit();
  if (!cu)
    return nullptr;
  if (hash !=
      cu->GetUnitDIEOnly().GetAttributeValueAsUnsigned(DW_AT_GNU_dwo_id, 0))
    return nullptr;
  return cu;
}

DWARFCompileUnit *SymbolFileDWARFDwo::FindSingleCompileUnit() {
  DWARFDebugInfo &debug_info = DebugInfo();

  // Right now we only support dwo files with one compile unit. If we don't have
  // type units, we can just check for the unit count.
  if (!debug_info.ContainsTypeUnits() && debug_info.GetNumUnits() == 1)
    return llvm::cast<DWARFCompileUnit>(debug_info.GetUnitAtIndex(0));

  // Otherwise, we have to run through all units, and find the compile unit that
  // way.
  DWARFCompileUnit *cu = nullptr;
  for (size_t i = 0; i < debug_info.GetNumUnits(); ++i) {
    if (auto *candidate =
            llvm::dyn_cast<DWARFCompileUnit>(debug_info.GetUnitAtIndex(i))) {
      if (cu)
        return nullptr; // More that one CU found.
      cu = candidate;
    }
  }
  return cu;
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

llvm::Expected<TypeSystem &>
SymbolFileDWARFDwo::GetTypeSystemForLanguage(LanguageType language) {
  return GetBaseSymbolFile().GetTypeSystemForLanguage(language);
}

DWARFDIE
SymbolFileDWARFDwo::GetDIE(const DIERef &die_ref) {
  if (die_ref.dwo_num() == GetDwoNum())
    return DebugInfo().GetDIE(die_ref);
  return GetBaseSymbolFile().GetDIE(die_ref);
}
