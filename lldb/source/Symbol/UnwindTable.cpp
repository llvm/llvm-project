//===-- UnwindTable.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/UnwindTable.h"

#include <stdio.h>

#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "lldb/Symbol/ArmUnwindInfo.h"
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
    : m_module(module), m_unwinds(), m_initialized(false), m_mutex(),
      m_eh_frame_up(), m_compact_unwind_up(), m_arm_unwind_up() {}

// We can't do some of this initialization when the ObjectFile is running its
// ctor; delay doing it until needed for something.

void UnwindTable::Initialize() {
  if (m_initialized)
    return;

  std::lock_guard<std::mutex> guard(m_mutex);

  if (m_initialized) // check again once we've acquired the lock
    return;
  m_initialized = true;
  ObjectFile *object_file = m_module.GetObjectFile();
  if (!object_file)
    return;

  SectionList *sl = m_module.GetSectionList();
  if (!sl)
    return;

  SectionSP sect = sl->FindSectionByType(eSectionTypeEHFrame, true);
  if (sect.get()) {
    m_eh_frame_up.reset(
        new DWARFCallFrameInfo(*object_file, sect, DWARFCallFrameInfo::EH));
  }

  sect = sl->FindSectionByType(eSectionTypeDWARFDebugFrame, true);
  if (sect) {
    m_debug_frame_up.reset(
        new DWARFCallFrameInfo(*object_file, sect, DWARFCallFrameInfo::DWARF));
  }

  sect = sl->FindSectionByType(eSectionTypeCompactUnwind, true);
  if (sect) {
    m_compact_unwind_up.reset(new CompactUnwindInfo(*object_file, sect));
  }

  sect = sl->FindSectionByType(eSectionTypeARMexidx, true);
  if (sect) {
    SectionSP sect_extab = sl->FindSectionByType(eSectionTypeARMextab, true);
    if (sect_extab.get()) {
      m_arm_unwind_up.reset(new ArmUnwindInfo(*object_file, sect, sect_extab));
    }
  }
}

UnwindTable::~UnwindTable() {}

llvm::Optional<AddressRange> UnwindTable::GetAddressRange(const Address &addr,
                                                          SymbolContext &sc) {
  AddressRange range;

  // First check the symbol context
  if (sc.GetAddressRange(eSymbolContextFunction | eSymbolContextSymbol, 0,
                         false, range) &&
      range.GetBaseAddress().IsValid())
    return range;

  // Does the eh_frame unwind info has a function bounds for this addr?
  if (m_eh_frame_up && m_eh_frame_up->GetAddressRange(addr, range))
    return range;

  // Try debug_frame as well
  if (m_debug_frame_up && m_debug_frame_up->GetAddressRange(addr, range))
    return range;

  return llvm::None;
}

FuncUnwindersSP
UnwindTable::GetFuncUnwindersContainingAddress(const Address &addr,
                                               SymbolContext &sc) {
  Initialize();

  std::lock_guard<std::mutex> guard(m_mutex);

  // There is an UnwindTable per object file, so we can safely use file handles
  addr_t file_addr = addr.GetFileAddress();
  iterator end = m_unwinds.end();
  iterator insert_pos = end;
  if (!m_unwinds.empty()) {
    insert_pos = m_unwinds.lower_bound(file_addr);
    iterator pos = insert_pos;
    if ((pos == m_unwinds.end()) ||
        (pos != m_unwinds.begin() &&
         pos->second->GetFunctionStartAddress() != addr))
      --pos;

    if (pos->second->ContainsAddress(addr))
      return pos->second;
  }

  auto range_or = GetAddressRange(addr, sc);
  if (!range_or)
    return nullptr;

  FuncUnwindersSP func_unwinder_sp(new FuncUnwinders(*this, *range_or));
  m_unwinds.insert(insert_pos,
                   std::make_pair(range_or->GetBaseAddress().GetFileAddress(),
                                  func_unwinder_sp));
  return func_unwinder_sp;
}

// Ignore any existing FuncUnwinders for this function, create a new one and
// don't add it to the UnwindTable.  This is intended for use by target modules
// show-unwind where we want to create new UnwindPlans, not re-use existing
// ones.
FuncUnwindersSP
UnwindTable::GetUncachedFuncUnwindersContainingAddress(const Address &addr,
                                                       SymbolContext &sc) {
  Initialize();

  auto range_or = GetAddressRange(addr, sc);
  if (!range_or)
    return nullptr;

  return std::make_shared<FuncUnwinders>(*this, *range_or);
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

SymbolFile *UnwindTable::GetSymbolFile() {
  if (SymbolVendor *vendor = m_module.GetSymbolVendor())
    return vendor->GetSymbolFile();
  return nullptr;
}

ArchSpec UnwindTable::GetArchitecture() { return m_module.GetArchitecture(); }

bool UnwindTable::GetAllowAssemblyEmulationUnwindPlans() {
  if (ObjectFile *object_file = m_module.GetObjectFile())
    return object_file->AllowAssemblyEmulationUnwindPlans();
  return false;
}
