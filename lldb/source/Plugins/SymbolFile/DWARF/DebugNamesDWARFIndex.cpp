//===-- DebugNamesDWARFIndex.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Plugins/SymbolFile/DWARF/DebugNamesDWARFIndex.h"
#include "lldb/Utility/RegularExpression.h"
#include "lldb/Utility/Stream.h"

using namespace lldb_private;
using namespace lldb;

static llvm::DWARFDataExtractor ToLLVM(const DWARFDataExtractor &data) {
  return llvm::DWARFDataExtractor(
      llvm::StringRef(reinterpret_cast<const char *>(data.GetDataStart()),
                      data.GetByteSize()),
      data.GetByteOrder() == eByteOrderLittle, data.GetAddressByteSize());
}

llvm::Expected<std::unique_ptr<DebugNamesDWARFIndex>>
DebugNamesDWARFIndex::Create(Module &module, DWARFDataExtractor debug_names,
                             DWARFDataExtractor debug_str,
                             DWARFDebugInfo *debug_info) {
  auto index_up =
      llvm::make_unique<DebugNames>(ToLLVM(debug_names), ToLLVM(debug_str));
  if (llvm::Error E = index_up->extract())
    return std::move(E);

  return std::unique_ptr<DebugNamesDWARFIndex>(new DebugNamesDWARFIndex(
      module, std::move(index_up), debug_names, debug_str, debug_info));
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

void DebugNamesDWARFIndex::Append(const DebugNames::Entry &entry,
                                  DIEArray &offsets) {
  llvm::Optional<uint64_t> cu_offset = entry.getCUOffset();
  llvm::Optional<uint64_t> die_offset = entry.getDIESectionOffset();
  if (cu_offset && die_offset)
    offsets.emplace_back(*cu_offset, *die_offset);
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

      uint32_t entry_offset = nte.getEntryOffset();
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

void DebugNamesDWARFIndex::GetTypes(ConstString name, DIEArray &offsets) {
  m_fallback.GetTypes(name, offsets);

  for (const DebugNames::Entry &entry :
       m_debug_names_up->equal_range(name.GetStringRef())) {
    if (isType(entry.tag()))
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

void DebugNamesDWARFIndex::Dump(Stream &s) {
  m_fallback.Dump(s);

  std::string data;
  llvm::raw_string_ostream os(data);
  m_debug_names_up->dump(os);
  s.PutCString(os.str());
}
