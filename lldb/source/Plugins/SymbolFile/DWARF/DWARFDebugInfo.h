//===-- DWARFDebugInfo.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_DWARFDebugInfo_h_
#define SymbolFileDWARF_DWARFDebugInfo_h_

#include <map>
#include <vector>

#include "DWARFDIE.h"
#include "DWARFTypeUnit.h"
#include "DWARFUnit.h"
#include "SymbolFileDWARF.h"
#include "lldb/Core/STLUtils.h"
#include "lldb/lldb-private.h"
#include "llvm/Support/Error.h"

namespace lldb_private {
class DWARFContext;
}

typedef std::multimap<const char *, dw_offset_t, CStringCompareFunctionObject>
    CStringToDIEMap;
typedef CStringToDIEMap::iterator CStringToDIEMapIter;
typedef CStringToDIEMap::const_iterator CStringToDIEMapConstIter;

class DWARFDebugInfo {
public:
  typedef dw_offset_t (*Callback)(SymbolFileDWARF *dwarf2Data,
                                  DWARFUnit *cu,
                                  DWARFDebugInfoEntry *die,
                                  const dw_offset_t next_offset,
                                  const uint32_t depth, void *userData);

  explicit DWARFDebugInfo(SymbolFileDWARF &dwarf,
                          lldb_private::DWARFContext &context);

  size_t GetNumUnits();
  DWARFUnit *GetUnitAtIndex(lldb::user_id_t idx);
  DWARFUnit *GetUnitAtOffset(DIERef::Section section, dw_offset_t cu_offset,
                             uint32_t *idx_ptr = nullptr);
  DWARFUnit *GetUnitContainingDIEOffset(DIERef::Section section,
                                        dw_offset_t die_offset);
  DWARFUnit *GetUnit(const DIERef &die_ref);
  DWARFTypeUnit *GetTypeUnitForHash(uint64_t hash);
  bool ContainsTypeUnits();
  DWARFDIE GetDIEForDIEOffset(DIERef::Section section,
                              dw_offset_t die_offset);
  DWARFDIE GetDIE(const DIERef &die_ref);

  enum {
    eDumpFlag_Verbose = (1 << 0),  // Verbose dumping
    eDumpFlag_ShowForm = (1 << 1), // Show the DW_form type
    eDumpFlag_ShowAncestors =
        (1 << 2) // Show all parent DIEs when dumping single DIEs
  };

  llvm::Expected<DWARFDebugAranges &> GetCompileUnitAranges();

protected:
  typedef std::vector<DWARFUnitSP> UnitColl;

  SymbolFileDWARF &m_dwarf;
  lldb_private::DWARFContext &m_context;
  UnitColl m_units;
  std::unique_ptr<DWARFDebugAranges>
      m_cu_aranges_up; // A quick address to compile unit table

  std::vector<std::pair<uint64_t, uint32_t>> m_type_hash_to_unit_index;

private:
  // All parsing needs to be done partially any managed by this class as
  // accessors are called.
  void ParseUnitHeadersIfNeeded();

  void ParseUnitsFor(DIERef::Section section);

  uint32_t FindUnitIndex(DIERef::Section section, dw_offset_t offset);

  DISALLOW_COPY_AND_ASSIGN(DWARFDebugInfo);
};

#endif // SymbolFileDWARF_DWARFDebugInfo_h_
