//===-- DWARFCompileUnit.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_DWARFCompileUnit_h_
#define SymbolFileDWARF_DWARFCompileUnit_h_

#include "DWARFUnit.h"

class DWARFCompileUnit : public DWARFUnit {
  friend class DWARFUnit;

public:
  static DWARFUnitSP Extract(SymbolFileDWARF *dwarf2Data,
      lldb::offset_t *offset_ptr);

  size_t ExtractDIEsIfNeeded(bool cu_die_only);
  DWARFDIE LookupAddress(const dw_addr_t address);
  size_t AppendDIEsWithTag(const dw_tag_t tag,
                           DWARFDIECollection &matching_dies,
                           uint32_t depth = UINT32_MAX) const;
  bool Verify(lldb_private::Stream *s) const;
  void Dump(lldb_private::Stream *s) const;
  lldb::user_id_t GetID() const;
  dw_offset_t GetAbbrevOffset() const;
  void SetAddrBase(dw_addr_t addr_base, dw_addr_t ranges_base, dw_offset_t base_obj_offset);
  void ClearDIEs(bool keep_compile_unit_die);
  void BuildAddressRangeTable(SymbolFileDWARF *dwarf2Data,
                              DWARFDebugAranges *debug_aranges);

  lldb_private::TypeSystem *GetTypeSystem();

  DWARFDIE
  GetCompileUnitDIEOnly() { return DWARFDIE(this, GetCompileUnitDIEPtrOnly()); }

  DWARFDIE
  DIE() { return DWARFDIE(this, DIEPtr()); }

  void AddDIE(DWARFDebugInfoEntry &die) {
    // The average bytes per DIE entry has been seen to be
    // around 14-20 so lets pre-reserve half of that since
    // we are now stripping the NULL tags.

    // Only reserve the memory if we are adding children of
    // the main compile unit DIE. The compile unit DIE is always
    // the first entry, so if our size is 1, then we are adding
    // the first compile unit child DIE and should reserve
    // the memory.
    if (m_die_array.empty())
      m_die_array.reserve(GetDebugInfoSize() / 24);
    m_die_array.push_back(die);
  }

  void AddCompileUnitDIE(DWARFDebugInfoEntry &die);

  void SetUserData(void *d);

  const DWARFDebugAranges &GetFunctionAranges();

  DWARFProducer GetProducer();

  uint32_t GetProducerVersionMajor();

  uint32_t GetProducerVersionMinor();

  uint32_t GetProducerVersionUpdate();

  lldb::LanguageType GetLanguageType();

  bool GetIsOptimized();

protected:
  virtual DWARFCompileUnit &Data() override { return *this; }
  virtual const DWARFCompileUnit &Data() const override { return *this; }

  SymbolFileDWARF *m_dwarf2Data;
  std::unique_ptr<SymbolFileDWARFDwo> m_dwo_symbol_file;
  const DWARFAbbreviationDeclarationSet *m_abbrevs;
  void *m_user_data = nullptr;
  DWARFDebugInfoEntry::collection
      m_die_array; // The compile unit debug information entry item
  std::unique_ptr<DWARFDebugAranges> m_func_aranges_ap; // A table similar to
                                                        // the .debug_aranges
                                                        // table, but this one
                                                        // points to the exact
                                                        // DW_TAG_subprogram
                                                        // DIEs
  dw_addr_t m_base_addr = 0;
  dw_offset_t m_length;
  uint16_t m_version;
  uint8_t m_addr_size;
  DWARFProducer m_producer = eProducerInvalid;
  uint32_t m_producer_version_major = 0;
  uint32_t m_producer_version_minor = 0;
  uint32_t m_producer_version_update = 0;
  lldb::LanguageType m_language_type = lldb::eLanguageTypeUnknown;
  bool m_is_dwarf64;
  lldb_private::LazyBool m_is_optimized = lldb_private::eLazyBoolCalculate;
  dw_addr_t m_addr_base = 0;     // Value of DW_AT_addr_base
  dw_addr_t m_ranges_base = 0;   // Value of DW_AT_ranges_base
  // If this is a dwo compile unit this is the offset of the base compile unit
  // in the main object file
  dw_offset_t m_base_obj_offset = DW_INVALID_OFFSET;

  void ParseProducerInfo();

private:
  DWARFCompileUnit(SymbolFileDWARF *dwarf2Data);

  const DWARFDebugInfoEntry *GetCompileUnitDIEPtrOnly() {
    ExtractDIEsIfNeeded(true);
    if (m_die_array.empty())
      return NULL;
    return &m_die_array[0];
  }

  const DWARFDebugInfoEntry *DIEPtr() {
    ExtractDIEsIfNeeded(false);
    if (m_die_array.empty())
      return NULL;
    return &m_die_array[0];
  }

  DISALLOW_COPY_AND_ASSIGN(DWARFCompileUnit);
};

#endif // SymbolFileDWARF_DWARFCompileUnit_h_
