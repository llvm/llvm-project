//===-- DWARFCompileUnit.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFCompileUnit.h"

#include "lldb/Core/DumpDataExtractor.h"
#include "lldb/Core/Mangled.h"
#include "lldb/Core/Module.h"
#include "lldb/Host/StringConvert.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/LineTable.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/Stream.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/Timer.h"

#include "DWARFDIECollection.h"
#include "DWARFDebugAbbrev.h"
#include "DWARFDebugAranges.h"
#include "DWARFDebugInfo.h"
#include "DWARFFormValue.h"
#include "LogChannelDWARF.h"
#include "NameToDIE.h"
#include "SymbolFileDWARF.h"
#include "SymbolFileDWARFDebugMap.h"
#include "SymbolFileDWARFDwo.h"

using namespace lldb;
using namespace lldb_private;
using namespace std;

extern int g_verbose;

DWARFCompileUnit::DWARFCompileUnit(SymbolFileDWARF *dwarf2Data)
    : m_dwarf2Data(dwarf2Data) {}

DWARFUnitSP DWARFCompileUnit::Extract(SymbolFileDWARF *dwarf2Data,
    lldb::offset_t *offset_ptr) {
  // std::make_shared would require the ctor to be public.
  std::shared_ptr<DWARFCompileUnit> cu_sp(new DWARFCompileUnit(dwarf2Data));
  // Out of memory?
  if (cu_sp.get() == NULL)
    return nullptr;

  const DWARFDataExtractor &debug_info = dwarf2Data->get_debug_info_data();

  cu_sp->m_offset = *offset_ptr;

  if (debug_info.ValidOffset(*offset_ptr)) {
    dw_offset_t abbr_offset;
    const DWARFDebugAbbrev *abbr = dwarf2Data->DebugAbbrev();
    cu_sp->m_length = debug_info.GetDWARFInitialLength(offset_ptr);
    cu_sp->m_is_dwarf64 = debug_info.IsDWARF64();
    cu_sp->m_version = debug_info.GetU16(offset_ptr);
    abbr_offset = debug_info.GetDWARFOffset(offset_ptr);
    cu_sp->m_addr_size = debug_info.GetU8(offset_ptr);

    bool length_OK =
        debug_info.ValidOffset(cu_sp->GetNextCompileUnitOffset() - 1);
    bool version_OK = SymbolFileDWARF::SupportedVersion(cu_sp->m_version);
    bool abbr_offset_OK =
        dwarf2Data->get_debug_abbrev_data().ValidOffset(abbr_offset);
    bool addr_size_OK = (cu_sp->m_addr_size == 4) || (cu_sp->m_addr_size == 8);

    if (length_OK && version_OK && addr_size_OK && abbr_offset_OK &&
        abbr != NULL) {
      cu_sp->m_abbrevs = abbr->GetAbbreviationDeclarationSet(abbr_offset);
      return cu_sp;
    }

    // reset the offset to where we tried to parse from if anything went wrong
    *offset_ptr = cu_sp->m_offset;
  }

  return nullptr;
}

void DWARFCompileUnit::ClearDIEs(bool keep_compile_unit_die) {
  if (m_die_array.size() > 1) {
    // std::vectors never get any smaller when resized to a smaller size,
    // or when clear() or erase() are called, the size will report that it
    // is smaller, but the memory allocated remains intact (call capacity()
    // to see this). So we need to create a temporary vector and swap the
    // contents which will cause just the internal pointers to be swapped
    // so that when "tmp_array" goes out of scope, it will destroy the
    // contents.

    // Save at least the compile unit DIE
    DWARFDebugInfoEntry::collection tmp_array;
    m_die_array.swap(tmp_array);
    if (keep_compile_unit_die)
      m_die_array.push_back(tmp_array.front());
  }

  if (m_dwo_symbol_file)
    m_dwo_symbol_file->GetCompileUnit()->ClearDIEs(keep_compile_unit_die);
}

//----------------------------------------------------------------------
// ParseCompileUnitDIEsIfNeeded
//
// Parses a compile unit and indexes its DIEs if it hasn't already been
// done.
//----------------------------------------------------------------------
size_t DWARFCompileUnit::ExtractDIEsIfNeeded(bool cu_die_only) {
  const size_t initial_die_array_size = m_die_array.size();
  if ((cu_die_only && initial_die_array_size > 0) || initial_die_array_size > 1)
    return 0; // Already parsed

  static Timer::Category func_cat(LLVM_PRETTY_FUNCTION);
  Timer scoped_timer(
      func_cat,
      "%8.8x: DWARFCompileUnit::ExtractDIEsIfNeeded( cu_die_only = %i )",
      m_offset, cu_die_only);

  // Set the offset to that of the first DIE and calculate the start of the
  // next compilation unit header.
  lldb::offset_t offset = GetFirstDIEOffset();
  lldb::offset_t next_cu_offset = GetNextCompileUnitOffset();

  DWARFDebugInfoEntry die;
  // Keep a flat array of the DIE for binary lookup by DIE offset
  if (!cu_die_only) {
    Log *log(
        LogChannelDWARF::GetLogIfAny(DWARF_LOG_DEBUG_INFO | DWARF_LOG_LOOKUPS));
    if (log) {
      m_dwarf2Data->GetObjectFile()->GetModule()->LogMessageVerboseBacktrace(
          log, "DWARFCompileUnit::ExtractDIEsIfNeeded () for compile unit at "
               ".debug_info[0x%8.8x]",
          GetOffset());
    }
  }

  uint32_t depth = 0;
  // We are in our compile unit, parse starting at the offset
  // we were told to parse
  const DWARFDataExtractor &debug_info_data =
      m_dwarf2Data->get_debug_info_data();
  std::vector<uint32_t> die_index_stack;
  die_index_stack.reserve(32);
  die_index_stack.push_back(0);
  bool prev_die_had_children = false;
  DWARFFormValue::FixedFormSizes fixed_form_sizes =
      DWARFFormValue::GetFixedFormSizesForAddressSize(GetAddressByteSize(),
                                                      m_is_dwarf64);
  while (offset < next_cu_offset &&
         die.FastExtract(debug_info_data, this, fixed_form_sizes, &offset)) {
    //        if (log)
    //            log->Printf("0x%8.8x: %*.*s%s%s",
    //                        die.GetOffset(),
    //                        depth * 2, depth * 2, "",
    //                        DW_TAG_value_to_name (die.Tag()),
    //                        die.HasChildren() ? " *" : "");

    const bool null_die = die.IsNULL();
    if (depth == 0) {
      if (initial_die_array_size == 0)
        AddCompileUnitDIE(die);
      uint64_t base_addr = die.GetAttributeValueAsAddress(
          m_dwarf2Data, this, DW_AT_low_pc, LLDB_INVALID_ADDRESS);
      if (base_addr == LLDB_INVALID_ADDRESS)
        base_addr = die.GetAttributeValueAsAddress(m_dwarf2Data, this,
                                                   DW_AT_entry_pc, 0);
      SetBaseAddress(base_addr);
      if (cu_die_only)
        return 1;
    } else {
      if (null_die) {
        if (prev_die_had_children) {
          // This will only happen if a DIE says is has children
          // but all it contains is a NULL tag. Since we are removing
          // the NULL DIEs from the list (saves up to 25% in C++ code),
          // we need a way to let the DIE know that it actually doesn't
          // have children.
          if (!m_die_array.empty())
            m_die_array.back().SetEmptyChildren(true);
        }
      } else {
        die.SetParentIndex(m_die_array.size() - die_index_stack[depth - 1]);

        if (die_index_stack.back())
          m_die_array[die_index_stack.back()].SetSiblingIndex(
              m_die_array.size() - die_index_stack.back());

        // Only push the DIE if it isn't a NULL DIE
        m_die_array.push_back(die);
      }
    }

    if (null_die) {
      // NULL DIE.
      if (!die_index_stack.empty())
        die_index_stack.pop_back();

      if (depth > 0)
        --depth;
      if (depth == 0)
        break; // We are done with this compile unit!

      prev_die_had_children = false;
    } else {
      die_index_stack.back() = m_die_array.size() - 1;
      // Normal DIE
      const bool die_has_children = die.HasChildren();
      if (die_has_children) {
        die_index_stack.push_back(0);
        ++depth;
      }
      prev_die_had_children = die_has_children;
    }
  }

  // Give a little bit of info if we encounter corrupt DWARF (our offset
  // should always terminate at or before the start of the next compilation
  // unit header).
  if (offset > next_cu_offset) {
    m_dwarf2Data->GetObjectFile()->GetModule()->ReportWarning(
        "DWARF compile unit extends beyond its bounds cu 0x%8.8x at "
        "0x%8.8" PRIx64 "\n",
        GetOffset(), offset);
  }

  // Since std::vector objects will double their size, we really need to
  // make a new array with the perfect size so we don't end up wasting
  // space. So here we copy and swap to make sure we don't have any extra
  // memory taken up.

  if (m_die_array.size() < m_die_array.capacity()) {
    DWARFDebugInfoEntry::collection exact_size_die_array(m_die_array.begin(),
                                                         m_die_array.end());
    exact_size_die_array.swap(m_die_array);
  }
  Log *log(LogChannelDWARF::GetLogIfAll(DWARF_LOG_DEBUG_INFO));
  if (log && log->GetVerbose()) {
    StreamString strm;
    Dump(&strm);
    if (m_die_array.empty())
      strm.Printf("error: no DIE for compile unit");
    else
      m_die_array[0].Dump(m_dwarf2Data, this, strm, UINT32_MAX);
    log->PutString(strm.GetString());
  }

  if (!m_dwo_symbol_file)
    return m_die_array.size();

  DWARFUnit *dwo_cu = m_dwo_symbol_file->GetCompileUnit();
  size_t dwo_die_count = dwo_cu->ExtractDIEsIfNeeded(cu_die_only);
  return m_die_array.size() + dwo_die_count -
         1; // We have 2 CU die, but we want to count it only as one
}

void DWARFCompileUnit::AddCompileUnitDIE(DWARFDebugInfoEntry &die) {
  assert(m_die_array.empty() && "Compile unit DIE already added");
  AddDIE(die);

  const DWARFDebugInfoEntry &cu_die = m_die_array.front();
  std::unique_ptr<SymbolFileDWARFDwo> dwo_symbol_file =
      m_dwarf2Data->GetDwoSymbolFileForCompileUnit(*this, cu_die);
  if (!dwo_symbol_file)
    return;

  DWARFUnit *dwo_cu = dwo_symbol_file->GetCompileUnit();
  if (!dwo_cu)
    return; // Can't fetch the compile unit from the dwo file.

  DWARFDIE dwo_cu_die = dwo_cu->GetCompileUnitDIEOnly();
  if (!dwo_cu_die.IsValid())
    return; // Can't fetch the compile unit DIE from the dwo file.

  uint64_t main_dwo_id = cu_die.GetAttributeValueAsUnsigned(
      m_dwarf2Data, this, DW_AT_GNU_dwo_id, 0);
  uint64_t sub_dwo_id =
      dwo_cu_die.GetAttributeValueAsUnsigned(DW_AT_GNU_dwo_id, 0);
  if (main_dwo_id != sub_dwo_id)
    return; // The 2 dwo ID isn't match. Don't use the dwo file as it belongs to
            // a differectn compilation.

  m_dwo_symbol_file = std::move(dwo_symbol_file);

  dw_addr_t addr_base = cu_die.GetAttributeValueAsUnsigned(
      m_dwarf2Data, this, DW_AT_GNU_addr_base, 0);
  dw_addr_t ranges_base = cu_die.GetAttributeValueAsUnsigned(
      m_dwarf2Data, this, DW_AT_GNU_ranges_base, 0);
  dwo_cu->SetAddrBase(addr_base, ranges_base, m_offset);
}

dw_offset_t DWARFCompileUnit::GetAbbrevOffset() const {
  return m_abbrevs ? m_abbrevs->GetOffset() : DW_INVALID_OFFSET;
}

bool DWARFCompileUnit::Verify(Stream *s) const {
  const DWARFDataExtractor &debug_info = m_dwarf2Data->get_debug_info_data();
  bool valid_offset = debug_info.ValidOffset(m_offset);
  bool length_OK = debug_info.ValidOffset(GetNextCompileUnitOffset() - 1);
  bool version_OK = SymbolFileDWARF::SupportedVersion(m_version);
  bool abbr_offset_OK =
      m_dwarf2Data->get_debug_abbrev_data().ValidOffset(GetAbbrevOffset());
  bool addr_size_OK = ((m_addr_size == 4) || (m_addr_size == 8));
  if (valid_offset && length_OK && version_OK && addr_size_OK &&
      abbr_offset_OK) {
    return true;
  } else {
    s->Printf("    0x%8.8x: ", m_offset);
    DumpDataExtractor(m_dwarf2Data->get_debug_info_data(), s, m_offset,
                      lldb::eFormatHex, 1, Size(), 32, LLDB_INVALID_ADDRESS, 0,
                      0);
    s->EOL();
    if (valid_offset) {
      if (!length_OK)
        s->Printf("        The length (0x%8.8x) for this compile unit is too "
                  "large for the .debug_info provided.\n",
                  m_length);
      if (!version_OK)
        s->Printf("        The 16 bit compile unit header version is not "
                  "supported.\n");
      if (!abbr_offset_OK)
        s->Printf("        The offset into the .debug_abbrev section (0x%8.8x) "
                  "is not valid.\n",
                  GetAbbrevOffset());
      if (!addr_size_OK)
        s->Printf("        The address size is unsupported: 0x%2.2x\n",
                  m_addr_size);
    } else
      s->Printf("        The start offset of the compile unit header in the "
                ".debug_info is invalid.\n");
  }
  return false;
}

void DWARFCompileUnit::Dump(Stream *s) const {
  s->Printf("0x%8.8x: Compile Unit: length = 0x%8.8x, version = 0x%4.4x, "
            "abbr_offset = 0x%8.8x, addr_size = 0x%2.2x (next CU at "
            "{0x%8.8x})\n",
            m_offset, m_length, m_version, GetAbbrevOffset(), m_addr_size,
            GetNextCompileUnitOffset());
}

lldb::user_id_t DWARFCompileUnit::GetID() const {
  dw_offset_t local_id =
      m_base_obj_offset != DW_INVALID_OFFSET ? m_base_obj_offset : m_offset;
  if (m_dwarf2Data)
    return DIERef(local_id, local_id).GetUID(m_dwarf2Data);
  else
    return local_id;
}

void DWARFCompileUnit::BuildAddressRangeTable(
    SymbolFileDWARF *dwarf2Data, DWARFDebugAranges *debug_aranges) {
  // This function is usually called if there in no .debug_aranges section
  // in order to produce a compile unit level set of address ranges that
  // is accurate.

  size_t num_debug_aranges = debug_aranges->GetNumRanges();

  // First get the compile unit DIE only and check if it has a DW_AT_ranges
  const DWARFDebugInfoEntry *die = GetCompileUnitDIEPtrOnly();

  const dw_offset_t cu_offset = GetOffset();
  if (die) {
    DWARFRangeList ranges;
    const size_t num_ranges =
        die->GetAttributeAddressRanges(dwarf2Data, this, ranges, false);
    if (num_ranges > 0) {
      // This compile unit has DW_AT_ranges, assume this is correct if it
      // is present since clang no longer makes .debug_aranges by default
      // and it emits DW_AT_ranges for DW_TAG_compile_units. GCC also does
      // this with recent GCC builds.
      for (size_t i = 0; i < num_ranges; ++i) {
        const DWARFRangeList::Entry &range = ranges.GetEntryRef(i);
        debug_aranges->AppendRange(cu_offset, range.GetRangeBase(),
                                   range.GetRangeEnd());
      }

      return; // We got all of our ranges from the DW_AT_ranges attribute
    }
  }
  // We don't have a DW_AT_ranges attribute, so we need to parse the DWARF

  // If the DIEs weren't parsed, then we don't want all dies for all compile
  // units
  // to stay loaded when they weren't needed. So we can end up parsing the DWARF
  // and then throwing them all away to keep memory usage down.
  const bool clear_dies = ExtractDIEsIfNeeded(false) > 1;

  die = DIEPtr();
  if (die)
    die->BuildAddressRangeTable(dwarf2Data, this, debug_aranges);

  if (debug_aranges->GetNumRanges() == num_debug_aranges) {
    // We got nothing from the functions, maybe we have a line tables only
    // situation. Check the line tables and build the arange table from this.
    SymbolContext sc;
    sc.comp_unit = dwarf2Data->GetCompUnitForDWARFCompUnit(this);
    if (sc.comp_unit) {
      SymbolFileDWARFDebugMap *debug_map_sym_file =
          m_dwarf2Data->GetDebugMapSymfile();
      if (debug_map_sym_file == NULL) {
        LineTable *line_table = sc.comp_unit->GetLineTable();

        if (line_table) {
          LineTable::FileAddressRanges file_ranges;
          const bool append = true;
          const size_t num_ranges =
              line_table->GetContiguousFileAddressRanges(file_ranges, append);
          for (uint32_t idx = 0; idx < num_ranges; ++idx) {
            const LineTable::FileAddressRanges::Entry &range =
                file_ranges.GetEntryRef(idx);
            debug_aranges->AppendRange(cu_offset, range.GetRangeBase(),
                                       range.GetRangeEnd());
          }
        }
      } else
        debug_map_sym_file->AddOSOARanges(dwarf2Data, debug_aranges);
    }
  }

  if (debug_aranges->GetNumRanges() == num_debug_aranges) {
    // We got nothing from the functions, maybe we have a line tables only
    // situation. Check the line tables and build the arange table from this.
    SymbolContext sc;
    sc.comp_unit = dwarf2Data->GetCompUnitForDWARFCompUnit(this);
    if (sc.comp_unit) {
      LineTable *line_table = sc.comp_unit->GetLineTable();

      if (line_table) {
        LineTable::FileAddressRanges file_ranges;
        const bool append = true;
        const size_t num_ranges =
            line_table->GetContiguousFileAddressRanges(file_ranges, append);
        for (uint32_t idx = 0; idx < num_ranges; ++idx) {
          const LineTable::FileAddressRanges::Entry &range =
              file_ranges.GetEntryRef(idx);
          debug_aranges->AppendRange(GetOffset(), range.GetRangeBase(),
                                     range.GetRangeEnd());
        }
      }
    }
  }

  // Keep memory down by clearing DIEs if this generate function
  // caused them to be parsed
  if (clear_dies)
    ClearDIEs(true);
}

const DWARFDebugAranges &DWARFCompileUnit::GetFunctionAranges() {
  if (m_func_aranges_ap.get() == NULL) {
    m_func_aranges_ap.reset(new DWARFDebugAranges());
    Log *log(LogChannelDWARF::GetLogIfAll(DWARF_LOG_DEBUG_ARANGES));

    if (log) {
      m_dwarf2Data->GetObjectFile()->GetModule()->LogMessage(
          log, "DWARFCompileUnit::GetFunctionAranges() for compile unit at "
               ".debug_info[0x%8.8x]",
          GetOffset());
    }
    const DWARFDebugInfoEntry *die = DIEPtr();
    if (die)
      die->BuildFunctionAddressRangeTable(m_dwarf2Data, this,
                                          m_func_aranges_ap.get());

    if (m_dwo_symbol_file) {
      DWARFUnit *dwo_cu = m_dwo_symbol_file->GetCompileUnit();
      const DWARFDebugInfoEntry *dwo_die = dwo_cu->DIEPtr();
      if (dwo_die)
        dwo_die->BuildFunctionAddressRangeTable(m_dwo_symbol_file.get(), dwo_cu,
                                                m_func_aranges_ap.get());
    }

    const bool minimize = false;
    m_func_aranges_ap->Sort(minimize);
  }
  return *m_func_aranges_ap.get();
}

DWARFDIE
DWARFCompileUnit::LookupAddress(const dw_addr_t address) {
  if (DIE()) {
    const DWARFDebugAranges &func_aranges = GetFunctionAranges();

    // Re-check the aranges auto pointer contents in case it was created above
    if (!func_aranges.IsEmpty())
      return GetDIE(func_aranges.FindAddress(address));
  }
  return DWARFDIE();
}

size_t DWARFCompileUnit::AppendDIEsWithTag(const dw_tag_t tag,
                                           DWARFDIECollection &dies,
                                           uint32_t depth) const {
  size_t old_size = dies.Size();
  DWARFDebugInfoEntry::const_iterator pos;
  DWARFDebugInfoEntry::const_iterator end = m_die_array.end();
  for (pos = m_die_array.begin(); pos != end; ++pos) {
    if (pos->Tag() == tag)
      dies.Append(DWARFDIE(this, &(*pos)));
  }

  // Return the number of DIEs added to the collection
  return dies.Size() - old_size;
}

// void
// DWARFCompileUnit::AddGlobalDIEByIndex (uint32_t die_idx)
//{
//    m_global_die_indexes.push_back (die_idx);
//}
//
//
// void
// DWARFCompileUnit::AddGlobal (const DWARFDebugInfoEntry* die)
//{
//    // Indexes to all file level global and static variables
//    m_global_die_indexes;
//
//    if (m_die_array.empty())
//        return;
//
//    const DWARFDebugInfoEntry* first_die = &m_die_array[0];
//    const DWARFDebugInfoEntry* end = first_die + m_die_array.size();
//    if (first_die <= die && die < end)
//        m_global_die_indexes.push_back (die - first_die);
//}

void DWARFCompileUnit::ParseProducerInfo() {
  m_producer_version_major = UINT32_MAX;
  m_producer_version_minor = UINT32_MAX;
  m_producer_version_update = UINT32_MAX;

  const DWARFDebugInfoEntry *die = GetCompileUnitDIEPtrOnly();
  if (die) {

    const char *producer_cstr = die->GetAttributeValueAsString(
        m_dwarf2Data, this, DW_AT_producer, NULL);
    if (producer_cstr) {
      RegularExpression llvm_gcc_regex(
          llvm::StringRef("^4\\.[012]\\.[01] \\(Based on Apple "
                          "Inc\\. build [0-9]+\\) \\(LLVM build "
                          "[\\.0-9]+\\)$"));
      if (llvm_gcc_regex.Execute(llvm::StringRef(producer_cstr))) {
        m_producer = eProducerLLVMGCC;
      } else if (strstr(producer_cstr, "clang")) {
        static RegularExpression g_clang_version_regex(
            llvm::StringRef("clang-([0-9]+)\\.([0-9]+)\\.([0-9]+)"));
        RegularExpression::Match regex_match(3);
        if (g_clang_version_regex.Execute(llvm::StringRef(producer_cstr),
                                          &regex_match)) {
          std::string str;
          if (regex_match.GetMatchAtIndex(producer_cstr, 1, str))
            m_producer_version_major =
                StringConvert::ToUInt32(str.c_str(), UINT32_MAX, 10);
          if (regex_match.GetMatchAtIndex(producer_cstr, 2, str))
            m_producer_version_minor =
                StringConvert::ToUInt32(str.c_str(), UINT32_MAX, 10);
          if (regex_match.GetMatchAtIndex(producer_cstr, 3, str))
            m_producer_version_update =
                StringConvert::ToUInt32(str.c_str(), UINT32_MAX, 10);
        }
        m_producer = eProducerClang;
      } else if (strstr(producer_cstr, "GNU"))
        m_producer = eProducerGCC;
    }
  }
  if (m_producer == eProducerInvalid)
    m_producer = eProcucerOther;
}

DWARFProducer DWARFCompileUnit::GetProducer() {
  if (m_producer == eProducerInvalid)
    ParseProducerInfo();
  return m_producer;
}

uint32_t DWARFCompileUnit::GetProducerVersionMajor() {
  if (m_producer_version_major == 0)
    ParseProducerInfo();
  return m_producer_version_major;
}

uint32_t DWARFCompileUnit::GetProducerVersionMinor() {
  if (m_producer_version_minor == 0)
    ParseProducerInfo();
  return m_producer_version_minor;
}

uint32_t DWARFCompileUnit::GetProducerVersionUpdate() {
  if (m_producer_version_update == 0)
    ParseProducerInfo();
  return m_producer_version_update;
}

LanguageType DWARFCompileUnit::GetLanguageType() {
  if (m_language_type != eLanguageTypeUnknown)
    return m_language_type;

  const DWARFDebugInfoEntry *die = GetCompileUnitDIEPtrOnly();
  if (die)
    m_language_type = LanguageTypeFromDWARF(die->GetAttributeValueAsUnsigned(
        m_dwarf2Data, this, DW_AT_language, 0));
  return m_language_type;
}

bool DWARFCompileUnit::GetIsOptimized() {
  if (m_is_optimized == eLazyBoolCalculate) {
    const DWARFDebugInfoEntry *die = GetCompileUnitDIEPtrOnly();
    if (die) {
      m_is_optimized = eLazyBoolNo;
      if (die->GetAttributeValueAsUnsigned(m_dwarf2Data, this,
                                           DW_AT_APPLE_optimized, 0) == 1) {
        m_is_optimized = eLazyBoolYes;
      }
    }
  }
  if (m_is_optimized == eLazyBoolYes) {
    return true;
  } else {
    return false;
  }
}

TypeSystem *DWARFCompileUnit::GetTypeSystem() {
  if (m_dwarf2Data)
    return m_dwarf2Data->GetTypeSystemForLanguage(GetLanguageType());
  else
    return nullptr;
}

void DWARFCompileUnit::SetUserData(void *d) {
  m_user_data = d;
  if (m_dwo_symbol_file)
    m_dwo_symbol_file->GetCompileUnit()->SetUserData(d);
}

void DWARFCompileUnit::SetAddrBase(dw_addr_t addr_base,
                                   dw_addr_t ranges_base,
                                   dw_offset_t base_obj_offset) {
  m_addr_base = addr_base;
  m_ranges_base = ranges_base;
  m_base_obj_offset = base_obj_offset;
}
