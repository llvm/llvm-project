//===-- DWARFUnit.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_DWARFUnit_h_
#define SymbolFileDWARF_DWARFUnit_h_

#include "DWARFDIE.h"
#include "DWARFDebugInfoEntry.h"
#include "lldb/lldb-enumerations.h"

class DWARFUnit;
class DWARFCompileUnit;
class NameToDIE;
class SymbolFileDWARF;
class SymbolFileDWARFDwo;

typedef std::shared_ptr<DWARFUnit> DWARFUnitSP;

enum DWARFProducer {
  eProducerInvalid = 0,
  eProducerClang,
  eProducerGCC,
  eProducerLLVMGCC,
  eProcucerOther
};

class DWARFUnit {
  friend class DWARFCompileUnit;

public:
  virtual ~DWARFUnit();

  size_t ExtractDIEsIfNeeded(bool cu_die_only);
  DWARFDIE LookupAddress(const dw_addr_t address);
  size_t AppendDIEsWithTag(const dw_tag_t tag,
                           DWARFDIECollection &matching_dies,
                           uint32_t depth = UINT32_MAX) const;
  bool Verify(lldb_private::Stream *s) const;
  void Dump(lldb_private::Stream *s) const;
  // Offset of the initial length field.
  dw_offset_t GetOffset() const { return m_offset; }
  lldb::user_id_t GetID() const;
  // Size in bytes of the initial length + compile unit header.
  uint32_t Size() const;
  bool ContainsDIEOffset(dw_offset_t die_offset) const {
    return die_offset >= GetFirstDIEOffset() &&
           die_offset < GetNextCompileUnitOffset();
  }
  dw_offset_t GetFirstDIEOffset() const { return m_offset + Size(); }
  dw_offset_t GetNextCompileUnitOffset() const;
  // Size of the CU data (without initial length and without header).
  size_t GetDebugInfoSize() const;
  // Size of the CU data incl. header but without initial length.
  uint32_t GetLength() const;
  uint16_t GetVersion() const;
  const DWARFAbbreviationDeclarationSet *GetAbbreviations() const;
  dw_offset_t GetAbbrevOffset() const;
  uint8_t GetAddressByteSize() const;
  dw_addr_t GetBaseAddress() const;
  dw_addr_t GetAddrBase() const;
  dw_addr_t GetRangesBase() const;
  void SetAddrBase(dw_addr_t addr_base, dw_addr_t ranges_base, dw_offset_t base_obj_offset);
  void ClearDIEs(bool keep_compile_unit_die);
  void BuildAddressRangeTable(SymbolFileDWARF *dwarf2Data,
                              DWARFDebugAranges *debug_aranges);

  lldb::ByteOrder GetByteOrder() const;

  lldb_private::TypeSystem *GetTypeSystem();

  DWARFFormValue::FixedFormSizes GetFixedFormSizes();

  void SetBaseAddress(dw_addr_t base_addr);

  DWARFDIE
  GetCompileUnitDIEOnly();

  DWARFDIE
  DIE();

  bool HasDIEsParsed() const;

  DWARFDIE GetDIE(dw_offset_t die_offset);

  static uint8_t GetAddressByteSize(const DWARFUnit *cu);

  static bool IsDWARF64(const DWARFUnit *cu);

  static uint8_t GetDefaultAddressSize();

  static void SetDefaultAddressSize(uint8_t addr_size);

  void *GetUserData() const;

  void SetUserData(void *d);

  bool Supports_DW_AT_APPLE_objc_complete_type();

  bool DW_AT_decl_file_attributes_are_invalid();

  bool Supports_unnamed_objc_bitfields();

  void Index(NameToDIE &func_basenames, NameToDIE &func_fullnames,
             NameToDIE &func_methods, NameToDIE &func_selectors,
             NameToDIE &objc_class_selectors, NameToDIE &globals,
             NameToDIE &types, NameToDIE &namespaces);

  SymbolFileDWARF *GetSymbolFileDWARF() const;

  DWARFProducer GetProducer();

  uint32_t GetProducerVersionMajor();

  uint32_t GetProducerVersionMinor();

  uint32_t GetProducerVersionUpdate();

  static lldb::LanguageType LanguageTypeFromDWARF(uint64_t val);

  lldb::LanguageType GetLanguageType();

  bool IsDWARF64() const;

  bool GetIsOptimized();

  SymbolFileDWARFDwo *GetDwoSymbolFile() const;

  dw_offset_t GetBaseObjOffset() const;

protected:
  virtual DWARFCompileUnit &Data() = 0;
  virtual const DWARFCompileUnit &Data() const = 0;

  DWARFUnit();

  static void
  IndexPrivate(DWARFUnit *dwarf_cu, const lldb::LanguageType cu_language,
               const DWARFFormValue::FixedFormSizes &fixed_form_sizes,
               const dw_offset_t cu_offset, NameToDIE &func_basenames,
               NameToDIE &func_fullnames, NameToDIE &func_methods,
               NameToDIE &func_selectors, NameToDIE &objc_class_selectors,
               NameToDIE &globals, NameToDIE &types, NameToDIE &namespaces);

  // Offset of the initial length field.
  dw_offset_t m_offset;

private:
  const DWARFDebugInfoEntry *GetCompileUnitDIEPtrOnly();

  const DWARFDebugInfoEntry *DIEPtr();

  DISALLOW_COPY_AND_ASSIGN(DWARFUnit);
};

#endif // SymbolFileDWARF_DWARFUnit_h_
