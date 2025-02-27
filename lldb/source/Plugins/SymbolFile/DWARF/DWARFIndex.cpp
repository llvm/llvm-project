//===-- DWARFIndex.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/SymbolFile/DWARF/DWARFIndex.h"
#include "DWARFDebugInfoEntry.h"
#include "DWARFDeclContext.h"
#include "Plugins/Language/ObjC/ObjCLanguage.h"
#include "Plugins/SymbolFile/DWARF/DWARFDIE.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARF.h"

#include "lldb/Core/Mangled.h"
#include "lldb/Core/Module.h"
#include "lldb/Target/Language.h"

using namespace lldb_private;
using namespace lldb;
using namespace lldb_private::plugin::dwarf;

DWARFIndex::~DWARFIndex() = default;

bool DWARFIndex::ProcessFunctionDIE(
    const Module::LookupInfo &lookup_info, DWARFDIE die,
    const CompilerDeclContext &parent_decl_ctx,
    llvm::function_ref<bool(DWARFDIE die)> callback) {
  llvm::StringRef name = lookup_info.GetLookupName().GetStringRef();
  FunctionNameType name_type_mask = lookup_info.GetNameTypeMask();

  if (!(name_type_mask & eFunctionNameTypeFull)) {
    ConstString name_to_match_against;
    if (const char *mangled_die_name = die.GetMangledName()) {
      name_to_match_against = ConstString(mangled_die_name);
    } else {
      SymbolFileDWARF *symbols = die.GetDWARF();
      if (ConstString demangled_die_name =
              symbols->ConstructFunctionDemangledName(die))
        name_to_match_against = demangled_die_name;
    }

    if (!lookup_info.NameMatchesLookupInfo(name_to_match_against,
                                           lookup_info.GetLanguageType()))
      return true;
  }

  // Exit early if we're searching exclusively for methods or selectors and
  // we have a context specified (no methods in namespaces).
  uint32_t looking_for_nonmethods =
      name_type_mask & ~(eFunctionNameTypeMethod | eFunctionNameTypeSelector);
  if (!looking_for_nonmethods && parent_decl_ctx.IsValid())
    return true;

  // Otherwise, we need to also check that the context matches. If it does not
  // match, we do nothing.
  if (!SymbolFileDWARF::DIEInDeclContext(parent_decl_ctx, die))
    return true;

  // In case of a full match, we just insert everything we find.
  if (name_type_mask & eFunctionNameTypeFull && die.GetMangledName() == name)
    return callback(die);

  // If looking for ObjC selectors, we need to also check if the name is a
  // possible selector.
  if (name_type_mask & eFunctionNameTypeSelector &&
      ObjCLanguage::IsPossibleObjCMethodName(die.GetName()))
    return callback(die);

  bool looking_for_methods = name_type_mask & lldb::eFunctionNameTypeMethod;
  bool looking_for_functions = name_type_mask & lldb::eFunctionNameTypeBase;
  if (looking_for_methods || looking_for_functions) {
    // If we're looking for either methods or functions, we definitely want this
    // die. Otherwise, only keep it if the die type matches what we are
    // searching for.
    if ((looking_for_methods && looking_for_functions) ||
        looking_for_methods == die.IsMethod())
      return callback(die);
  }

  return true;
}

DWARFIndex::DIERefCallbackImpl::DIERefCallbackImpl(
    const DWARFIndex &index, llvm::function_ref<bool(DWARFDIE die)> callback,
    llvm::StringRef name)
    : m_index(index),
      m_dwarf(*llvm::cast<SymbolFileDWARF>(
          index.m_module.GetSymbolFile()->GetBackingSymbolFile())),
      m_callback(callback), m_name(name) {}

bool DWARFIndex::DIERefCallbackImpl::operator()(DIERef ref) const {
  if (DWARFDIE die = m_dwarf.GetDIE(ref))
    return m_callback(die);
  m_index.ReportInvalidDIERef(ref, m_name);
  return true;
}

bool DWARFIndex::DIERefCallbackImpl::operator()(
    const llvm::AppleAcceleratorTable::Entry &entry) const {
  return this->operator()(DIERef(std::nullopt, DIERef::Section::DebugInfo,
                                 *entry.getDIESectionOffset()));
}

void DWARFIndex::ReportInvalidDIERef(DIERef ref, llvm::StringRef name) const {
  m_module.ReportErrorIfModifyDetected(
      "the DWARF debug information has been modified (accelerator table had "
      "bad die {0:x16} for '{1}')\n",
      ref.die_offset(), name.str().c_str());
}

void DWARFIndex::GetFullyQualifiedType(
    const DWARFDeclContext &context,
    llvm::function_ref<bool(DWARFDIE die)> callback) {
  GetTypes(context, [&](DWARFDIE die) {
    return GetFullyQualifiedTypeImpl(context, die, callback);
  });
}

bool DWARFIndex::GetFullyQualifiedTypeImpl(
    const DWARFDeclContext &context, DWARFDIE die,
    llvm::function_ref<bool(DWARFDIE die)> callback) {
  DWARFDeclContext dwarf_decl_ctx = die.GetDWARFDeclContext();
  if (dwarf_decl_ctx == context)
    return callback(die);
  return true;
}

void DWARFIndex::GetTypesWithQuery(
    TypeQuery &query, llvm::function_ref<bool(DWARFDIE die)> callback) {
  GetTypes(query.GetTypeBasename(), [&](DWARFDIE die) {
    return ProcessTypeDIEMatchQuery(query, die, callback);
  });
}

bool DWARFIndex::ProcessTypeDIEMatchQuery(
    TypeQuery &query, DWARFDIE die,
    llvm::function_ref<bool(DWARFDIE die)> callback) {
  // Check the language, but only if we have a language filter.
  if (query.HasLanguage() &&
      !query.LanguageMatches(SymbolFileDWARF::GetLanguageFamily(*die.GetCU())))
    return true; // Keep iterating over index types, language mismatch.

  // Since mangled names are unique, we only need to check if the names are
  // the same.
  if (query.GetSearchByMangledName()) {
    if (die.GetMangledName(/*substitute_name_allowed=*/false) !=
        query.GetTypeBasename().GetStringRef())
      return true; // Keep iterating over index types, mangled name mismatch.
    return callback(die);
  }

  std::vector<lldb_private::CompilerContext> die_context;
  if (query.GetModuleSearch())
    die_context = die.GetDeclContext();
  else
    die_context = die.GetTypeLookupContext();

  if (!query.ContextMatches(die_context))
    return true;
  return callback(die);
}

void DWARFIndex::GetNamespacesWithParents(
    ConstString name, const CompilerDeclContext &parent_decl_ctx,
    llvm::function_ref<bool(DWARFDIE die)> callback) {
  GetNamespaces(name, [&](DWARFDIE die) {
    return ProcessNamespaceDieMatchParents(parent_decl_ctx, die, callback);
  });
}

bool DWARFIndex::ProcessNamespaceDieMatchParents(
    const CompilerDeclContext &parent_decl_ctx, DWARFDIE die,
    llvm::function_ref<bool(DWARFDIE die)> callback) {
  if (!SymbolFileDWARF::DIEInDeclContext(parent_decl_ctx, die))
    return true;
  return callback(die);
}
