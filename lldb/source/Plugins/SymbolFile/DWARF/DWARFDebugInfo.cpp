//===-- DWARFDebugInfo.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolFileDWARF.h"

#include <algorithm>
#include <set>

#include "lldb/Host/PosixApi.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/RegularExpression.h"
#include "lldb/Utility/Stream.h"

#include "DWARFCompileUnit.h"
#include "DWARFContext.h"
#include "DWARFDebugAranges.h"
#include "DWARFDebugInfo.h"
#include "DWARFDebugInfoEntry.h"
#include "DWARFFormValue.h"

using namespace lldb;
using namespace lldb_private;
using namespace std;

// Constructor
DWARFDebugInfo::DWARFDebugInfo(lldb_private::DWARFContext &context)
    : m_dwarf2Data(NULL), m_context(context), m_compile_units(),
      m_cu_aranges_up() {}

// SetDwarfData
void DWARFDebugInfo::SetDwarfData(SymbolFileDWARF *dwarf2Data) {
  m_dwarf2Data = dwarf2Data;
  m_compile_units.clear();
}

llvm::Expected<DWARFDebugAranges &> DWARFDebugInfo::GetCompileUnitAranges() {
  if (m_cu_aranges_up)
    return *m_cu_aranges_up;

  assert(m_dwarf2Data);

  m_cu_aranges_up = llvm::make_unique<DWARFDebugAranges>();
  const DWARFDataExtractor *debug_aranges_data =
      m_context.getOrLoadArangesData();
  if (debug_aranges_data) {
    llvm::Error error = m_cu_aranges_up->extract(*debug_aranges_data);
    if (error)
      return std::move(error);
  }

  // Make a list of all CUs represented by the arange data in the file.
  std::set<dw_offset_t> cus_with_data;
  for (size_t n = 0; n < m_cu_aranges_up->GetNumRanges(); n++) {
    dw_offset_t offset = m_cu_aranges_up->OffsetAtIndex(n);
    if (offset != DW_INVALID_OFFSET)
      cus_with_data.insert(offset);
  }

  // Manually build arange data for everything that wasn't in the
  // .debug_aranges table.
  const size_t num_compile_units = GetNumCompileUnits();
  for (size_t idx = 0; idx < num_compile_units; ++idx) {
    DWARFUnit *cu = GetCompileUnitAtIndex(idx);

    dw_offset_t offset = cu->GetOffset();
    if (cus_with_data.find(offset) == cus_with_data.end())
      cu->BuildAddressRangeTable(m_dwarf2Data, m_cu_aranges_up.get());
  }

  const bool minimize = true;
  m_cu_aranges_up->Sort(minimize);
  return *m_cu_aranges_up;
}

void DWARFDebugInfo::ParseCompileUnitHeadersIfNeeded() {
  if (!m_compile_units.empty())
    return;
  if (!m_dwarf2Data)
    return;

  lldb::offset_t offset = 0;
  const auto &debug_info_data = m_dwarf2Data->get_debug_info_data();

  while (debug_info_data.ValidOffset(offset)) {
    llvm::Expected<DWARFUnitSP> cu_sp = DWARFCompileUnit::extract(
        m_dwarf2Data, m_compile_units.size(), debug_info_data, &offset);

    if (!cu_sp) {
      // FIXME: Propagate this error up.
      llvm::consumeError(cu_sp.takeError());
      return;
    }

    // If it didn't return an error, then it should be returning a valid
    // CompileUnit.
    assert(*cu_sp);

    m_compile_units.push_back(*cu_sp);

    offset = (*cu_sp)->GetNextCompileUnitOffset();
  }
}

size_t DWARFDebugInfo::GetNumCompileUnits() {
  ParseCompileUnitHeadersIfNeeded();
  return m_compile_units.size();
}

DWARFUnit *DWARFDebugInfo::GetCompileUnitAtIndex(user_id_t idx) {
  DWARFUnit *cu = NULL;
  if (idx < GetNumCompileUnits())
    cu = m_compile_units[idx].get();
  return cu;
}

bool DWARFDebugInfo::OffsetLessThanCompileUnitOffset(
    dw_offset_t offset, const DWARFUnitSP &cu_sp) {
  return offset < cu_sp->GetOffset();
}

uint32_t DWARFDebugInfo::FindCompileUnitIndex(dw_offset_t offset) {
  ParseCompileUnitHeadersIfNeeded();

  // llvm::lower_bound is not used as for DIE offsets it would still return
  // index +1 and GetOffset() returning index itself would be a special case.
  auto pos = llvm::upper_bound(m_compile_units, offset,
                               OffsetLessThanCompileUnitOffset);
  uint32_t idx = std::distance(m_compile_units.begin(), pos);
  if (idx == 0)
    return DW_INVALID_OFFSET;
  return idx - 1;
}

DWARFUnit *DWARFDebugInfo::GetCompileUnitAtOffset(dw_offset_t cu_offset,
                                                  uint32_t *idx_ptr) {
  uint32_t idx = FindCompileUnitIndex(cu_offset);
  DWARFUnit *result = GetCompileUnitAtIndex(idx);
  if (result && result->GetOffset() != cu_offset) {
    result = nullptr;
    idx = DW_INVALID_INDEX;
  }
  if (idx_ptr)
    *idx_ptr = idx;
  return result;
}

DWARFUnit *DWARFDebugInfo::GetCompileUnit(const DIERef &die_ref) {
  if (die_ref.cu_offset == DW_INVALID_OFFSET)
    return GetCompileUnitContainingDIEOffset(die_ref.die_offset);
  else
    return GetCompileUnitAtOffset(die_ref.cu_offset);
}

DWARFUnit *
DWARFDebugInfo::GetCompileUnitContainingDIEOffset(dw_offset_t die_offset) {
  uint32_t idx = FindCompileUnitIndex(die_offset);
  DWARFUnit *result = GetCompileUnitAtIndex(idx);
  if (result && !result->ContainsDIEOffset(die_offset))
    return nullptr;
  return result;
}

DWARFDIE
DWARFDebugInfo::GetDIEForDIEOffset(dw_offset_t die_offset) {
  DWARFUnit *cu = GetCompileUnitContainingDIEOffset(die_offset);
  if (cu)
    return cu->GetDIE(die_offset);
  return DWARFDIE();
}

// GetDIE()
//
// Get the DIE (Debug Information Entry) with the specified offset.
DWARFDIE
DWARFDebugInfo::GetDIE(const DIERef &die_ref) {
  DWARFUnit *cu = GetCompileUnit(die_ref);
  if (cu)
    return cu->GetDIE(die_ref.die_offset);
  return DWARFDIE(); // Not found
}

