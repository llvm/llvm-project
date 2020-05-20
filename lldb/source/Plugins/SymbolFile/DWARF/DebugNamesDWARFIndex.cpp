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
#include "lldb/Core/Module.h"
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

bool DebugNamesDWARFIndex::ProcessEntry(
    const DebugNames::Entry &entry,
    llvm::function_ref<bool(DWARFDIE die)> callback, llvm::StringRef name) {
  llvm::Optional<DIERef> ref = ToDIERef(entry);
  if (!ref)
    return true;
  SymbolFileDWARF &dwarf =
      *llvm::cast<SymbolFileDWARF>(m_module.GetSymbolFile());
  DWARFDIE die = dwarf.GetDIE(*ref);
  if (!die)
    return true;
  return callback(die);
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

void DebugNamesDWARFIndex::GetGlobalVariables(
    ConstString basename, llvm::function_ref<bool(DWARFDIE die)> callback) {
  for (const DebugNames::Entry &entry :
       m_debug_names_up->equal_range(basename.GetStringRef())) {
    if (entry.tag() != DW_TAG_variable)
      continue;

    if (!ProcessEntry(entry, callback, basename.GetStringRef()))
      return;
  }

  m_fallback.GetGlobalVariables(basename, callback);
}

void DebugNamesDWARFIndex::GetGlobalVariables(
    const RegularExpression &regex,
    llvm::function_ref<bool(DWARFDIE die)> callback) {
  for (const DebugNames::NameIndex &ni: *m_debug_names_up) {
    for (DebugNames::NameTableEntry nte: ni) {
      if (!regex.Execute(nte.getString()))
        continue;

      uint64_t entry_offset = nte.getEntryOffset();
      llvm::Expected<DebugNames::Entry> entry_or = ni.getEntry(&entry_offset);
      for (; entry_or; entry_or = ni.getEntry(&entry_offset)) {
        if (entry_or->tag() != DW_TAG_variable)
          continue;

        if (!ProcessEntry(*entry_or, callback,
                          llvm::StringRef(nte.getString())))
          return;
      }
      MaybeLogLookupError(entry_or.takeError(), ni, nte.getString());
    }
  }

  m_fallback.GetGlobalVariables(regex, callback);
}

void DebugNamesDWARFIndex::GetGlobalVariables(
    const DWARFUnit &cu, llvm::function_ref<bool(DWARFDIE die)> callback) {
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

        if (!ProcessEntry(*entry_or, callback,
                          llvm::StringRef(nte.getString())))
          return;
      }
      MaybeLogLookupError(entry_or.takeError(), ni, nte.getString());
    }
  }

  m_fallback.GetGlobalVariables(cu, callback);
}

void DebugNamesDWARFIndex::GetCompleteObjCClass(
    ConstString class_name, bool must_be_implementation,
    llvm::function_ref<bool(DWARFDIE die)> callback) {
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

    DWARFDIE die = m_debug_info.GetDIE(*ref);
    if (!die) {
      ReportInvalidDIERef(*ref, class_name.GetStringRef());
      continue;
    }

    if (die.GetAttributeValueAsUnsigned(DW_AT_APPLE_objc_complete_type, 0)) {
      // If we find the complete version we're done.
      callback(die);
      return;
    }
    incomplete_types.push_back(*ref);
  }

  auto dierefcallback = DIERefCallback(callback, class_name.GetStringRef());
  for (DIERef ref : incomplete_types)
    if (!dierefcallback(ref))
      return;

  m_fallback.GetCompleteObjCClass(class_name, must_be_implementation, callback);
}

void DebugNamesDWARFIndex::GetTypes(
    ConstString name, llvm::function_ref<bool(DWARFDIE die)> callback) {
  for (const DebugNames::Entry &entry :
       m_debug_names_up->equal_range(name.GetStringRef())) {
    if (isType(entry.tag())) {
      if (!ProcessEntry(entry, callback, name.GetStringRef()))
        return;
    }
  }

  m_fallback.GetTypes(name, callback);
}

void DebugNamesDWARFIndex::GetTypes(
    const DWARFDeclContext &context,
    llvm::function_ref<bool(DWARFDIE die)> callback) {
  auto name = context[0].name;
  for (const DebugNames::Entry &entry : m_debug_names_up->equal_range(name)) {
    if (entry.tag() == context[0].tag) {
      if (!ProcessEntry(entry, callback, name))
        return;
    }
  }

  m_fallback.GetTypes(context, callback);
}

void DebugNamesDWARFIndex::GetNamespaces(
    ConstString name, llvm::function_ref<bool(DWARFDIE die)> callback) {
  for (const DebugNames::Entry &entry :
       m_debug_names_up->equal_range(name.GetStringRef())) {
    if (entry.tag() == DW_TAG_namespace) {
      if (!ProcessEntry(entry, callback, name.GetStringRef()))
        return;
    }
  }

  m_fallback.GetNamespaces(name, callback);
}

void DebugNamesDWARFIndex::GetFunctions(
    ConstString name, SymbolFileDWARF &dwarf,
    const CompilerDeclContext &parent_decl_ctx, uint32_t name_type_mask,
    llvm::function_ref<bool(DWARFDIE die)> callback) {

  std::set<DWARFDebugInfoEntry *> seen;
  for (const DebugNames::Entry &entry :
       m_debug_names_up->equal_range(name.GetStringRef())) {
    Tag tag = entry.tag();
    if (tag != DW_TAG_subprogram && tag != DW_TAG_inlined_subroutine)
      continue;

    if (llvm::Optional<DIERef> ref = ToDIERef(entry)) {
      if (!ProcessFunctionDIE(name.GetStringRef(), *ref, dwarf, parent_decl_ctx,
                              name_type_mask, [&](DWARFDIE die) {
                                if (!seen.insert(die.GetDIE()).second)
                                  return true;
                                return callback(die);
                              }))
        return;
    }
  }

  m_fallback.GetFunctions(name, dwarf, parent_decl_ctx, name_type_mask,
                          callback);
}

void DebugNamesDWARFIndex::GetFunctions(
    const RegularExpression &regex,
    llvm::function_ref<bool(DWARFDIE die)> callback) {
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

        if (!ProcessEntry(*entry_or, callback,
                          llvm::StringRef(nte.getString())))
          return;
      }
      MaybeLogLookupError(entry_or.takeError(), ni, nte.getString());
    }
  }

  m_fallback.GetFunctions(regex, callback);
}

void DebugNamesDWARFIndex::Dump(Stream &s) {
  m_fallback.Dump(s);

  std::string data;
  llvm::raw_string_ostream os(data);
  m_debug_names_up->dump(os);
  s.PutCString(os.str());
}
