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

static DWARFDataExtractor LoadSection(SectionList *section_list,
                                      SectionType section_type) {
  if (!section_list)
    return DWARFDataExtractor();

  auto section_sp = section_list->FindSectionByType(section_type, true);
  if (!section_sp)
    return DWARFDataExtractor();

  DWARFDataExtractor data;
  section_sp->GetSectionData(data);
  return data;
}

static const DWARFDataExtractor &
LoadOrGetSection(SectionList *section_list, SectionType section_type,
                 llvm::Optional<DWARFDataExtractor> &extractor) {
  if (!extractor)
    extractor = LoadSection(section_list, section_type);
  return *extractor;
}

const DWARFDataExtractor &DWARFContext::getOrLoadArangesData() {
  return LoadOrGetSection(m_main_section_list, eSectionTypeDWARFDebugAranges,
                          m_data_debug_aranges);
}

const DWARFDataExtractor &DWARFContext::getOrLoadDebugInfoData() {
  if (isDwo())
    return LoadOrGetSection(m_dwo_section_list, eSectionTypeDWARFDebugInfoDwo,
                            m_data_debug_info);
  return LoadOrGetSection(m_main_section_list, eSectionTypeDWARFDebugInfo,
                          m_data_debug_info);
}
