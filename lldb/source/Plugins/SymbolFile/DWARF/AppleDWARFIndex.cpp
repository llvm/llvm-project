//===-- AppleDWARFIndex.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/SymbolFile/DWARF/AppleDWARFIndex.h"
#include "Plugins/SymbolFile/DWARF/DWARFDeclContext.h"
#include "Plugins/SymbolFile/DWARF/DWARFUnit.h"
#include "Plugins/SymbolFile/DWARF/LogChannelDWARF.h"

#include "lldb/Core/Module.h"
#include "lldb/Symbol/Function.h"

using namespace lldb_private;
using namespace lldb;
using namespace lldb_private::dwarf;

std::unique_ptr<AppleDWARFIndex> AppleDWARFIndex::Create(
    Module &module, DWARFDataExtractor apple_names,
    DWARFDataExtractor apple_namespaces, DWARFDataExtractor apple_types,
    DWARFDataExtractor apple_objc, DWARFDataExtractor debug_str) {
  auto apple_names_table_up = std::make_unique<DWARFMappedHash::MemoryTable>(
      apple_names, debug_str, ".apple_names");
  if (!apple_names_table_up->IsValid())
    apple_names_table_up.reset();

  auto apple_namespaces_table_up =
      std::make_unique<DWARFMappedHash::MemoryTable>(
          apple_namespaces, debug_str, ".apple_namespaces");
  if (!apple_namespaces_table_up->IsValid())
    apple_namespaces_table_up.reset();

  auto apple_types_table_up = std::make_unique<DWARFMappedHash::MemoryTable>(
      apple_types, debug_str, ".apple_types");
  if (!apple_types_table_up->IsValid())
    apple_types_table_up.reset();

  auto apple_objc_table_up = std::make_unique<DWARFMappedHash::MemoryTable>(
      apple_objc, debug_str, ".apple_objc");
  if (!apple_objc_table_up->IsValid())
    apple_objc_table_up.reset();

  if (apple_names_table_up || apple_namespaces_table_up ||
      apple_types_table_up || apple_objc_table_up)
    return std::make_unique<AppleDWARFIndex>(
        module, std::move(apple_names_table_up),
        std::move(apple_namespaces_table_up), std::move(apple_types_table_up),
        std::move(apple_objc_table_up));

  return nullptr;
}

void AppleDWARFIndex::GetGlobalVariables(
    ConstString basename, llvm::function_ref<bool(DWARFDIE die)> callback) {
  if (!m_apple_names_up)
    return;
  m_apple_names_up->FindByName(
      basename.GetStringRef(),
      DIERefCallback(callback, basename.GetStringRef()));
}

void AppleDWARFIndex::GetGlobalVariables(
    const RegularExpression &regex,
    llvm::function_ref<bool(DWARFDIE die)> callback) {
  if (!m_apple_names_up)
    return;

  DWARFMappedHash::DIEInfoArray hash_data;
  m_apple_names_up->AppendAllDIEsThatMatchingRegex(regex, hash_data);
  // This is not really the DIE name.
  DWARFMappedHash::ExtractDIEArray(hash_data,
                                   DIERefCallback(callback, regex.GetText()));
}

void AppleDWARFIndex::GetGlobalVariables(
    DWARFUnit &cu, llvm::function_ref<bool(DWARFDIE die)> callback) {
  if (!m_apple_names_up)
    return;

  lldbassert(!cu.GetSymbolFileDWARF().GetDwoNum());
  const DWARFUnit &non_skeleton_cu = cu.GetNonSkeletonUnit();
  DWARFMappedHash::DIEInfoArray hash_data;
  m_apple_names_up->AppendAllDIEsInRange(non_skeleton_cu.GetOffset(),
                                         non_skeleton_cu.GetNextUnitOffset(),
                                         hash_data);
  DWARFMappedHash::ExtractDIEArray(hash_data, DIERefCallback(callback));
}

void AppleDWARFIndex::GetObjCMethods(
    ConstString class_name, llvm::function_ref<bool(DWARFDIE die)> callback) {
  if (!m_apple_objc_up)
    return;
  m_apple_objc_up->FindByName(
      class_name.GetStringRef(),
      DIERefCallback(callback, class_name.GetStringRef()));
}

void AppleDWARFIndex::GetCompleteObjCClass(
    ConstString class_name, bool must_be_implementation,
    llvm::function_ref<bool(DWARFDIE die)> callback) {
  if (!m_apple_types_up)
    return;
  m_apple_types_up->FindCompleteObjCClassByName(
      class_name.GetStringRef(),
      DIERefCallback(callback, class_name.GetStringRef()),
      must_be_implementation);
}

void AppleDWARFIndex::GetTypes(
    ConstString name, llvm::function_ref<bool(DWARFDIE die)> callback) {
  if (!m_apple_types_up)
    return;
  m_apple_types_up->FindByName(name.GetStringRef(),
                               DIERefCallback(callback, name.GetStringRef()));
}

void AppleDWARFIndex::GetTypes(
    const DWARFDeclContext &context,
    llvm::function_ref<bool(DWARFDIE die)> callback) {
  if (!m_apple_types_up)
    return;

  Log *log = GetLog(DWARFLog::TypeCompletion | DWARFLog::Lookups);
  const bool has_tag = m_apple_types_up->GetHeader().header_data.ContainsAtom(
      DWARFMappedHash::eAtomTypeTag);
  const bool has_qualified_name_hash =
      m_apple_types_up->GetHeader().header_data.ContainsAtom(
          DWARFMappedHash::eAtomTypeQualNameHash);

  const ConstString type_name(context[0].name);
  const dw_tag_t tag = context[0].tag;
  if (has_tag && has_qualified_name_hash) {
    const char *qualified_name = context.GetQualifiedName();
    const uint32_t qualified_name_hash = llvm::djbHash(qualified_name);
    if (log)
      m_module.LogMessage(log, "FindByNameAndTagAndQualifiedNameHash()");
    m_apple_types_up->FindByNameAndTagAndQualifiedNameHash(
        type_name.GetStringRef(), tag, qualified_name_hash,
        DIERefCallback(callback, type_name.GetStringRef()));
    return;
  }

  if (has_tag) {
    // When searching for a scoped type (for example,
    // "std::vector<int>::const_iterator") searching for the innermost
    // name alone ("const_iterator") could yield many false
    // positives. By searching for the parent type ("vector<int>")
    // first we can avoid extracting type DIEs from object files that
    // would fail the filter anyway.
    if (!has_qualified_name_hash && (context.GetSize() > 1) &&
        (context[1].tag == DW_TAG_class_type ||
         context[1].tag == DW_TAG_structure_type)) {
      if (m_apple_types_up->FindByName(context[1].name,
                                       [&](DIERef ref) { return false; }))
        return;
    }

    if (log)
      m_module.LogMessage(log, "FindByNameAndTag()");
    m_apple_types_up->FindByNameAndTag(
        type_name.GetStringRef(), tag,
        DIERefCallback(callback, type_name.GetStringRef()));
    return;
  }

  m_apple_types_up->FindByName(
      type_name.GetStringRef(),
      DIERefCallback(callback, type_name.GetStringRef()));
}

void AppleDWARFIndex::GetNamespaces(
    ConstString name, llvm::function_ref<bool(DWARFDIE die)> callback) {
  if (!m_apple_namespaces_up)
    return;
  m_apple_namespaces_up->FindByName(
      name.GetStringRef(), DIERefCallback(callback, name.GetStringRef()));
}

void AppleDWARFIndex::GetFunctions(
    const Module::LookupInfo &lookup_info, SymbolFileDWARF &dwarf,
    const CompilerDeclContext &parent_decl_ctx,
    llvm::function_ref<bool(DWARFDIE die)> callback) {
  ConstString name = lookup_info.GetLookupName();
  m_apple_names_up->FindByName(name.GetStringRef(), [&](DIERef die_ref) {
    return ProcessFunctionDIE(lookup_info, die_ref, dwarf, parent_decl_ctx,
                              callback);
  });
}

void AppleDWARFIndex::GetFunctions(
    const RegularExpression &regex,
    llvm::function_ref<bool(DWARFDIE die)> callback) {
  if (!m_apple_names_up)
    return;

  DWARFMappedHash::DIEInfoArray hash_data;
  m_apple_names_up->AppendAllDIEsThatMatchingRegex(regex, hash_data);
  DWARFMappedHash::ExtractDIEArray(hash_data,
                                   DIERefCallback(callback, regex.GetText()));
}

void AppleDWARFIndex::Dump(Stream &s) {
  if (m_apple_names_up)
    s.PutCString(".apple_names index present\n");
  if (m_apple_namespaces_up)
    s.PutCString(".apple_namespaces index present\n");
  if (m_apple_types_up)
    s.PutCString(".apple_types index present\n");
  if (m_apple_objc_up)
    s.PutCString(".apple_objc index present\n");
  // TODO: Dump index contents
}
