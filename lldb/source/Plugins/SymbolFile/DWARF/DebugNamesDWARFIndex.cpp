//===-- DebugNamesDWARFIndex.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/SymbolFile/DWARF/DebugNamesDWARFIndex.h"
#include "Plugins/SymbolFile/DWARF/DWARFDebugInfo.h"
#include "Plugins/SymbolFile/DWARF/DWARFDeclContext.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARFDwo.h"
#include "lldb/Utility/RegularExpression.h"
#include "lldb/Utility/Stream.h"

using namespace lldb_private;
using namespace lldb;

llvm::Expected<std::unique_ptr<DebugNamesDWARFIndex>>
DebugNamesDWARFIndex::Create(Module &module, DWARFDataExtractor debug_names,
                             DWARFDataExtractor debug_str,
                             SymbolFileDWARF &dwarf) {
  auto index_up = std::make_unique<DebugNames>(debug_names.GetAsLLVM(),
                                                debug_str.GetAsLLVM());
  if (llvm::Error E = index_up->extract())
    return std::move(E);

  return std::unique_ptr<DebugNamesDWARFIndex>(new DebugNamesDWARFIndex(
      module, std::move(index_up), debug_names, debug_str, dwarf));
}

llvm::DenseSet<dw_offset_t>
DebugNamesDWARFIndex::GetUnits(const DebugNames &debug_names) {
  llvm::DenseSet<dw_offset_t> result;
  for (const DebugNames::NameIndex &ni : debug_names) {
    for (uint32_t cu = 0; cu < ni.getCUCount(); ++cu)
      result.insert(ni.getCUOffset(cu));
  }
  return result;
}

llvm::Optional<DIERef>
DebugNamesDWARFIndex::ToDIERef(const DebugNames::Entry &entry) {
  llvm::Optional<uint64_t> cu_offset = entry.getCUOffset();
  if (!cu_offset)
    return llvm::None;

  DWARFUnit *cu = m_debug_info.GetUnitAtOffset(DIERef::Section::DebugInfo, *cu_offset);
  if (!cu)
    return llvm::None;

  cu = &cu->GetNonSkeletonUnit();
  if (llvm::Optional<uint64_t> die_offset = entry.getDIEUnitOffset())
    return DIERef(cu->GetSymbolFileDWARF().GetDwoNum(),
                  DIERef::Section::DebugInfo, cu->GetOffset() + *die_offset);

  return llvm::None;
}

void DebugNamesDWARFIndex::Append(const DebugNames::Entry &entry,
                                  DIEArray &offsets) {
  if (llvm::Optional<DIERef> ref = ToDIERef(entry))
    offsets.push_back(*ref);
}

void DebugNamesDWARFIndex::MaybeLogLookupError(llvm::Error error,
                                               const DebugNames::NameIndex &ni,
                                               llvm::StringRef name) {
  // Ignore SentinelErrors, log everything else.
  LLDB_LOG_ERROR(
      LogChannelDWARF::GetLogIfAll(DWARF_LOG_LOOKUPS),
      handleErrors(std::move(error), [](const DebugNames::SentinelError &) {}),
      "Failed to parse index entries for index at {1:x}, name {2}: {0}",
      ni.getUnitOffset(), name);
}

void DebugNamesDWARFIndex::GetGlobalVariables(ConstString basename,
                                              DIEArray &offsets) {
  m_fallback.GetGlobalVariables(basename, offsets);

  for (const DebugNames::Entry &entry :
       m_debug_names_up->equal_range(basename.GetStringRef())) {
    if (entry.tag() != DW_TAG_variable)
      continue;

    Append(entry, offsets);
  }
}

void DebugNamesDWARFIndex::GetGlobalVariables(const RegularExpression &regex,
                                              DIEArray &offsets) {
  m_fallback.GetGlobalVariables(regex, offsets);

  for (const DebugNames::NameIndex &ni: *m_debug_names_up) {
    for (DebugNames::NameTableEntry nte: ni) {
      if (!regex.Execute(nte.getString()))
        continue;

      uint64_t entry_offset = nte.getEntryOffset();
      llvm::Expected<DebugNames::Entry> entry_or = ni.getEntry(&entry_offset);
      for (; entry_or; entry_or = ni.getEntry(&entry_offset)) {
        if (entry_or->tag() != DW_TAG_variable)
          continue;

        Append(*entry_or, offsets);
      }
      MaybeLogLookupError(entry_or.takeError(), ni, nte.getString());
    }
  }
}

void DebugNamesDWARFIndex::GetGlobalVariables(const DWARFUnit &cu,
                                              DIEArray &offsets) {
  m_fallback.GetGlobalVariables(cu, offsets);

  uint64_t cu_offset = cu.GetOffset();
  for (const DebugNames::NameIndex &ni: *m_debug_names_up) {
    for (DebugNames::NameTableEntry nte: ni) {
      uint64_t entry_offset = nte.getEntryOffset();
      llvm::Expected<DebugNames::Entry> entry_or = ni.getEntry(&entry_offset);
      for (; entry_or; entry_or = ni.getEntry(&entry_offset)) {
        if (entry_or->tag() != DW_TAG_variable)
          continue;
        if (entry_or->getCUOffset() != cu_offset)
          continue;

        Append(*entry_or, offsets);
      }
      MaybeLogLookupError(entry_or.takeError(), ni, nte.getString());
    }
  }
}

void DebugNamesDWARFIndex::GetCompleteObjCClass(ConstString class_name,
                                                bool must_be_implementation,
                                                DIEArray &offsets) {
  m_fallback.GetCompleteObjCClass(class_name, must_be_implementation, offsets);

  // Keep a list of incomplete types as fallback for when we don't find the
  // complete type.
  DIEArray incomplete_types;

  for (const DebugNames::Entry &entry :
       m_debug_names_up->equal_range(class_name.GetStringRef())) {
    if (entry.tag() != DW_TAG_structure_type &&
        entry.tag() != DW_TAG_class_type)
      continue;

    llvm::Optional<DIERef> ref = ToDIERef(entry);
    if (!ref)
      continue;

    DWARFUnit *cu = m_debug_info.GetUnit(*ref);
    if (!cu || !cu->Supports_DW_AT_APPLE_objc_complete_type()) {
      incomplete_types.push_back(*ref);
      continue;
    }

    // FIXME: We should return DWARFDIEs so we don't have to resolve it twice.
    DWARFDIE die = m_debug_info.GetDIE(*ref);
    if (!die)
      continue;

    if (die.GetAttributeValueAsUnsigned(DW_AT_APPLE_objc_complete_type, 0)) {
      // If we find the complete version we're done.
      offsets.push_back(*ref);
      return;
    } else {
      incomplete_types.push_back(*ref);
    }
  }

  offsets.insert(offsets.end(), incomplete_types.begin(),
                 incomplete_types.end());
}

void DebugNamesDWARFIndex::GetTypes(ConstString name, DIEArray &offsets) {
  m_fallback.GetTypes(name, offsets);

  for (const DebugNames::Entry &entry :
       m_debug_names_up->equal_range(name.GetStringRef())) {
    if (isType(entry.tag()))
      Append(entry, offsets);
  }
}

void DebugNamesDWARFIndex::GetTypes(const DWARFDeclContext &context,
                                    DIEArray &offsets) {
  m_fallback.GetTypes(context, offsets);

  for (const DebugNames::Entry &entry :
       m_debug_names_up->equal_range(context[0].name)) {
    if (entry.tag() == context[0].tag)
      Append(entry, offsets);
  }
}

void DebugNamesDWARFIndex::GetNamespaces(ConstString name, DIEArray &offsets) {
  m_fallback.GetNamespaces(name, offsets);

  for (const DebugNames::Entry &entry :
       m_debug_names_up->equal_range(name.GetStringRef())) {
    if (entry.tag() == DW_TAG_namespace)
      Append(entry, offsets);
  }
}

void DebugNamesDWARFIndex::GetFunctions(
    ConstString name, SymbolFileDWARF &dwarf,
    const CompilerDeclContext &parent_decl_ctx, uint32_t name_type_mask,
    std::vector<DWARFDIE> &dies) {

  std::vector<DWARFDIE> v;
  m_fallback.GetFunctions(name, dwarf, parent_decl_ctx, name_type_mask, v);

  for (const DebugNames::Entry &entry :
       m_debug_names_up->equal_range(name.GetStringRef())) {
    Tag tag = entry.tag();
    if (tag != DW_TAG_subprogram && tag != DW_TAG_inlined_subroutine)
      continue;

    if (llvm::Optional<DIERef> ref = ToDIERef(entry))
      ProcessFunctionDIE(name.GetStringRef(), *ref, dwarf, parent_decl_ctx,
                         name_type_mask, v);
  }

  std::set<DWARFDebugInfoEntry *> seen;
  for (DWARFDIE die : v)
    if (seen.insert(die.GetDIE()).second)
      dies.push_back(die);
}

void DebugNamesDWARFIndex::GetFunctions(const RegularExpression &regex,
                                        DIEArray &offsets) {
  m_fallback.GetFunctions(regex, offsets);

  for (const DebugNames::NameIndex &ni: *m_debug_names_up) {
    for (DebugNames::NameTableEntry nte: ni) {
      if (!regex.Execute(nte.getString()))
        continue;

      uint64_t entry_offset = nte.getEntryOffset();
      llvm::Expected<DebugNames::Entry> entry_or = ni.getEntry(&entry_offset);
      for (; entry_or; entry_or = ni.getEntry(&entry_offset)) {
        Tag tag = entry_or->tag();
        if (tag != DW_TAG_subprogram && tag != DW_TAG_inlined_subroutine)
          continue;

        Append(*entry_or, offsets);
      }
      MaybeLogLookupError(entry_or.takeError(), ni, nte.getString());
    }
  }
}

void DebugNamesDWARFIndex::Dump(Stream &s) {
  m_fallback.Dump(s);

  std::string data;
  llvm::raw_string_ostream os(data);
  m_debug_names_up->dump(os);
  s.PutCString(os.str());
}
