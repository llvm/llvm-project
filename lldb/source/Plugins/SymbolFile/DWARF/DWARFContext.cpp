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
