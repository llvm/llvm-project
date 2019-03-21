//===-- DWARFContext.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFContext.h"

#include "lldb/Core/Section.h"

using namespace lldb;
using namespace lldb_private;

static const DWARFDataExtractor *
GetPointerOrNull(const llvm::Optional<DWARFDataExtractor> &extractor) {
  if (!extractor.hasValue())
    return nullptr;
  return extractor.getPointer();
}

static const DWARFDataExtractor *
LoadOrGetSection(Module &module, SectionType section_type,
                 llvm::Optional<DWARFDataExtractor> &extractor) {
  if (extractor.hasValue())
    return extractor->GetByteSize() > 0 ? extractor.getPointer() : nullptr;

  // Initialize to an empty extractor so that we always take the fast path going
  // forward.
  extractor.emplace();

  const SectionList *section_list = module.GetSectionList();
  if (!section_list)
    return nullptr;

  auto section_sp = section_list->FindSectionByType(section_type, true);
  if (!section_sp)
    return nullptr;

  section_sp->GetSectionData(*extractor);
  return extractor.getPointer();
}

DWARFContext::DWARFContext(Module &module) : m_module(module) {}

const DWARFDataExtractor *DWARFContext::getOrLoadArangesData() {
  return LoadOrGetSection(m_module, eSectionTypeDWARFDebugAranges,
                          m_data_debug_aranges);
}

const DWARFDataExtractor *DWARFContext::getOrLoadDebugLineData() {
  return LoadOrGetSection(m_module, eSectionTypeDWARFDebugLine,
                          m_data_debug_line);
}

const DWARFDataExtractor *DWARFContext::getOrLoadDebugLineStrData() {
  return LoadOrGetSection(m_module, eSectionTypeDWARFDebugLineStr,
                          m_data_debug_line_str);
}

const DWARFDataExtractor *DWARFContext::getOrLoadDebugMacroData() {
  return LoadOrGetSection(m_module, eSectionTypeDWARFDebugMacro,
                          m_data_debug_macro);
}

const DWARFDataExtractor *DWARFContext::getOrLoadDebugLocData() {
  return LoadOrGetSection(m_module, eSectionTypeDWARFDebugLoc,
                          m_data_debug_loc);
}

const DWARFDataExtractor *DWARFContext::getOrLoadDebugLoclistData() {
  return LoadOrGetSection(m_module, eSectionTypeDWARFDebugLocLists,
                          m_data_debug_loclists);
}

const DWARFDataExtractor *DWARFContext::getOrLoadDebugRangesData() {
  return LoadOrGetSection(m_module, eSectionTypeDWARFDebugRanges,
                          m_data_debug_ranges);
}

const DWARFDataExtractor *DWARFContext::getOrLoadDebugRnglistsData() {
  return LoadOrGetSection(m_module, eSectionTypeDWARFDebugRngLists,
                          m_data_debug_rnglists);
}

const DWARFDataExtractor *DWARFContext::getOrLoadDebugFrameData() {
  return LoadOrGetSection(m_module, eSectionTypeDWARFDebugFrame,
                          m_data_debug_frame);
}

const DWARFDataExtractor *DWARFContext::getOrLoadDebugTypesData() {
  return LoadOrGetSection(m_module, eSectionTypeDWARFDebugTypes,
                          m_data_debug_types);
}

const DWARFDataExtractor *DWARFContext::getOrLoadGnuDebugAltlinkData() {
  return LoadOrGetSection(m_module, eSectionTypeDWARFGNUDebugAltLink,
                          m_data_gnu_debug_altlink);
}

const DWARFDataExtractor *DWARFContext::getOrLoadBestDebugLocData() {
  const DWARFDataExtractor *loc = getOrLoadDebugLocData();
  if (loc)
    return loc;

  return getOrLoadDebugLoclistData();
}
