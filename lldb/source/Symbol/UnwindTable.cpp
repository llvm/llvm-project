//===-- UnwindTable.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/UnwindTable.h"

#include <cstdio>
#include <optional>

#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "lldb/Symbol/ArmUnwindInfo.h"
#include "lldb/Symbol/CallFrameInfo.h"
#include "lldb/Symbol/CompactUnwindInfo.h"
#include "lldb/Symbol/DWARFCallFrameInfo.h"
#include "lldb/Symbol/FuncUnwinders.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/SymbolVendor.h"

// There is one UnwindTable object per ObjectFile. It contains a list of Unwind
// objects -- one per function, populated lazily -- for the ObjectFile. Each
// Unwind object has multiple UnwindPlans for different scenarios.

using namespace lldb;
using namespace lldb_private;

UnwindTable::UnwindTable(Module &module)
    : m_module(module), m_unwinds(), m_scanned_all_unwind_sources(false),
      m_mutex(), m_object_file_unwind_up(), m_eh_frame_up(),
      m_compact_unwind_up(), m_arm_unwind_up() {}

// We can't do some of this initialization when the ObjectFile is running its
// ctor; delay doing it until needed for something.
void UnwindTable::Initialize() {
  if (m_scanned_all_unwind_sources)
    return;

  std::lock_guard<std::mutex> guard(m_mutex);

  if (m_scanned_all_unwind_sources) // check again once we've acquired the lock
    return;

  ObjectFile *object_file = m_module.GetObjectFile();
  if (!object_file)
    return;

  m_scanned_all_unwind_sources = true;

  if (!m_object_file_unwind_up)
    m_object_file_unwind_up = object_file->CreateCallFrameInfo();

  SectionList *sl = m_module.GetSectionList();
  if (!sl)
    return;

  SectionSP sect = sl->FindSectionByType(eSectionTypeEHFrame, true);
  if (!m_eh_frame_up && sect)
    m_eh_frame_up = std::make_unique<DWARFCallFrameInfo>(
        *object_file, sect, DWARFCallFrameInfo::EH);

  sect = sl->FindSectionByType(eSectionTypeDWARFDebugFrame, true);
  if (!m_debug_frame_up && sect)
    m_debug_frame_up = std::make_unique<DWARFCallFrameInfo>(
        *object_file, sect, DWARFCallFrameInfo::DWARF);

  sect = sl->FindSectionByType(eSectionTypeCompactUnwind, true);
  if (!m_compact_unwind_up && sect)
    m_compact_unwind_up =
        std::make_unique<CompactUnwindInfo>(*object_file, sect);

  sect = sl->FindSectionByType(eSectionTypeARMexidx, true);
  if (!m_arm_unwind_up && sect) {
    SectionSP sect_extab = sl->FindSectionByType(eSectionTypeARMextab, true);
    if (sect_extab.get()) {
      m_arm_unwind_up =
          std::make_unique<ArmUnwindInfo>(*object_file, sect, sect_extab);
    }
  }
}

void UnwindTable::ModuleWasUpdated() {
  std::lock_guard<std::mutex> guard(m_mutex);
  m_scanned_all_unwind_sources = false;
  m_unwinds.clear();
}

UnwindTable::~UnwindTable() = default;

AddressRanges UnwindTable::GetAddressRanges(const Address &addr,
                                            const SymbolContext &sc) {
  AddressRange range;

  // First check the unwind info from the object file plugin
  if (m_object_file_unwind_up &&
      m_object_file_unwind_up->GetAddressRange(addr, range))
    return {range};

  // Check the symbol context
  AddressRanges result;
  for (size_t idx = 0;
       sc.GetAddressRange(eSymbolContextFunction | eSymbolContextSymbol, idx,
                          false, range) &&
       range.GetBaseAddress().IsValid();
       ++idx)
    result.push_back(range);
  if (!result.empty())
    return result;

  // Does the eh_frame unwind info has a function bounds for this addr?
  if (m_eh_frame_up && m_eh_frame_up->GetAddressRange(addr, range))
    return {range};

  // Try debug_frame as well
  if (m_debug_frame_up && m_debug_frame_up->GetAddressRange(addr, range))
    return {range};

  return {};
}

static Address GetFunctionOrSymbolAddress(const Address &addr,
                                          const SymbolContext &sc) {
  if (Address result = sc.GetFunctionOrSymbolAddress(); result.IsValid())
    return result;
  return addr;
}

FuncUnwindersSP
UnwindTable::GetFuncUnwindersContainingAddress(const Address &addr,
                                               const SymbolContext &sc) {
  Initialize();

  std::lock_guard<std::mutex> guard(m_mutex);

  // There is an UnwindTable per object file, so we can safely use file handles
  addr_t file_addr = addr.GetFileAddress();
  iterator insert_pos = m_unwinds.upper_bound(file_addr);
  if (insert_pos != m_unwinds.begin()) {
    auto pos = std::prev(insert_pos);
    if (pos->second->ContainsAddress(addr))
      return pos->second;
  }

  Address start_addr = GetFunctionOrSymbolAddress(addr, sc);
  AddressRanges ranges = GetAddressRanges(addr, sc);
  if (ranges.empty())
    return nullptr;

  auto func_unwinder_sp =
      std::make_shared<FuncUnwinders>(*this, start_addr, ranges);
  for (const AddressRange &range : ranges)
    m_unwinds.emplace_hint(insert_pos, range.GetBaseAddress().GetFileAddress(),
                           func_unwinder_sp);
  return func_unwinder_sp;
}

// Ignore any existing FuncUnwinders for this function, create a new one and
// don't add it to the UnwindTable.  This is intended for use by target modules
// show-unwind where we want to create new UnwindPlans, not re-use existing
// ones.
FuncUnwindersSP UnwindTable::GetUncachedFuncUnwindersContainingAddress(
    const Address &addr, const SymbolContext &sc) {
  Initialize();

  Address start_addr = GetFunctionOrSymbolAddress(addr, sc);
  AddressRanges ranges = GetAddressRanges(addr, sc);
  if (ranges.empty())
    return nullptr;

  return std::make_shared<FuncUnwinders>(*this, start_addr, std::move(ranges));
}

void UnwindTable::Dump(Stream &s) {
  std::lock_guard<std::mutex> guard(m_mutex);
  s.Format("UnwindTable for '{0}':\n", m_module.GetFileSpec());
  const_iterator begin = m_unwinds.begin();
  const_iterator end = m_unwinds.end();
  for (const_iterator pos = begin; pos != end; ++pos) {
    s.Printf("[%u] 0x%16.16" PRIx64 "\n", (unsigned)std::distance(begin, pos),
             pos->first);
  }
  s.EOL();
}

lldb_private::CallFrameInfo *UnwindTable::GetObjectFileUnwindInfo() {
  Initialize();
  return m_object_file_unwind_up.get();
}

DWARFCallFrameInfo *UnwindTable::GetEHFrameInfo() {
  Initialize();
  return m_eh_frame_up.get();
}

DWARFCallFrameInfo *UnwindTable::GetDebugFrameInfo() {
  Initialize();
  return m_debug_frame_up.get();
}

CompactUnwindInfo *UnwindTable::GetCompactUnwindInfo() {
  Initialize();
  return m_compact_unwind_up.get();
}

ArmUnwindInfo *UnwindTable::GetArmUnwindInfo() {
  Initialize();
  return m_arm_unwind_up.get();
}

SymbolFile *UnwindTable::GetSymbolFile() { return m_module.GetSymbolFile(); }

ArchSpec UnwindTable::GetArchitecture() { return m_module.GetArchitecture(); }

bool UnwindTable::GetAllowAssemblyEmulationUnwindPlans() {
  if (ObjectFile *object_file = m_module.GetObjectFile())
    return object_file->AllowAssemblyEmulationUnwindPlans();
  return false;
}
