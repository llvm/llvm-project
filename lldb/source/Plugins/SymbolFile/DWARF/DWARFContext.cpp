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

const DWARFDataExtractor &
DWARFContext::LoadOrGetSection(SectionType main_section_type,
                               llvm::Optional<SectionType> dwo_section_type,
                               SectionData &data) {
  llvm::call_once(data.flag, [&] {
    if (dwo_section_type && isDwo())
      data.data = LoadSection(m_dwo_section_list, *dwo_section_type);
    else
      data.data = LoadSection(m_main_section_list, main_section_type);
  });
  return data.data;
}

const DWARFDataExtractor &DWARFContext::getOrLoadAbbrevData() {
  return LoadOrGetSection(eSectionTypeDWARFDebugAbbrev,
                          eSectionTypeDWARFDebugAbbrevDwo, m_data_debug_abbrev);
}

const DWARFDataExtractor &DWARFContext::getOrLoadArangesData() {
  return LoadOrGetSection(eSectionTypeDWARFDebugAranges, llvm::None,
                          m_data_debug_aranges);
}

const DWARFDataExtractor &DWARFContext::getOrLoadAddrData() {
  return LoadOrGetSection(eSectionTypeDWARFDebugAddr, llvm::None,
                          m_data_debug_addr);
}

const DWARFDataExtractor &DWARFContext::getOrLoadDebugInfoData() {
  return LoadOrGetSection(eSectionTypeDWARFDebugInfo,
                          eSectionTypeDWARFDebugInfoDwo, m_data_debug_info);
}

const DWARFDataExtractor &DWARFContext::getOrLoadLineData() {
  return LoadOrGetSection(eSectionTypeDWARFDebugLine, llvm::None,
                          m_data_debug_line);
}

const DWARFDataExtractor &DWARFContext::getOrLoadLineStrData() {
  return LoadOrGetSection(eSectionTypeDWARFDebugLineStr, llvm::None,
                          m_data_debug_line_str);
}

const DWARFDataExtractor &DWARFContext::getOrLoadMacroData() {
  return LoadOrGetSection(eSectionTypeDWARFDebugMacro, llvm::None,
                          m_data_debug_macro);
}

const DWARFDataExtractor &DWARFContext::getOrLoadRangesData() {
  return LoadOrGetSection(eSectionTypeDWARFDebugRanges, llvm::None,
                          m_data_debug_ranges);
}

const DWARFDataExtractor &DWARFContext::getOrLoadRngListsData() {
  return LoadOrGetSection(eSectionTypeDWARFDebugRngLists, llvm::None,
                          m_data_debug_rnglists);
}

const DWARFDataExtractor &DWARFContext::getOrLoadStrData() {
  return LoadOrGetSection(eSectionTypeDWARFDebugStr,
                          eSectionTypeDWARFDebugStrDwo, m_data_debug_str);
}

const DWARFDataExtractor &DWARFContext::getOrLoadStrOffsetsData() {
  return LoadOrGetSection(eSectionTypeDWARFDebugStrOffsets,
                          eSectionTypeDWARFDebugStrOffsetsDwo,
                          m_data_debug_str_offsets);
}

const DWARFDataExtractor &DWARFContext::getOrLoadDebugTypesData() {
  return LoadOrGetSection(eSectionTypeDWARFDebugTypes,
                          eSectionTypeDWARFDebugTypesDwo, m_data_debug_types);
}

llvm::DWARFContext &DWARFContext::GetAsLLVM() {
  if (!m_llvm_context) {
    llvm::StringMap<std::unique_ptr<llvm::MemoryBuffer>> section_map;
    uint8_t addr_size = 0;

    auto AddSection = [&](Section &section) {
      DataExtractor section_data;
      section.GetSectionData(section_data);

      // Set the address size the first time we see it.
      if (addr_size == 0)
        addr_size = section_data.GetByteSize();

      llvm::StringRef data = llvm::toStringRef(section_data.GetData());
      llvm::StringRef name = section.GetName().GetStringRef();
      if (name.startswith("."))
        name = name.drop_front();
      section_map.try_emplace(
          name, llvm::MemoryBuffer::getMemBuffer(data, name, false));
    };

    if (m_main_section_list) {
      for (auto &section : *m_main_section_list)
        AddSection(*section);
    }

    if (m_dwo_section_list) {
      for (auto &section : *m_dwo_section_list)
        AddSection(*section);
    }

    m_llvm_context = llvm::DWARFContext::create(section_map, addr_size);
  }
  return *m_llvm_context;
}
