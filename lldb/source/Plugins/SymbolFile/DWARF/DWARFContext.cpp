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

static DWARFDataExtractor LoadSection(Module &module,
                                      SectionType section_type) {
  SectionList *section_list = module.GetSectionList();
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
LoadOrGetSection(Module &module, SectionType section_type,
                 llvm::Optional<DWARFDataExtractor> &extractor) {
  if (!extractor)
    extractor = LoadSection(module, section_type);
  return *extractor;
}

DWARFContext::DWARFContext(Module &module) : m_module(module) {}

const DWARFDataExtractor &DWARFContext::getOrLoadArangesData() {
  return LoadOrGetSection(m_module, eSectionTypeDWARFDebugAranges,
                          m_data_debug_aranges);
}
