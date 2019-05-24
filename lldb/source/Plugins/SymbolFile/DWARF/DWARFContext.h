//===-- DWARFContext.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_SYMBOLFILE_DWARF_DWARFCONTEXT_H
#define LLDB_PLUGINS_SYMBOLFILE_DWARF_DWARFCONTEXT_H

#include "DWARFDataExtractor.h"
#include "lldb/Core/Section.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Threading.h"
#include <memory>

namespace lldb_private {
class DWARFContext {
private:
  SectionList *m_main_section_list;
  SectionList *m_dwo_section_list;

  struct SectionData {
    llvm::once_flag flag;
    DWARFDataExtractor data;
  };

  SectionData m_data_debug_abbrev;
  SectionData m_data_debug_addr;
  SectionData m_data_debug_aranges;
  SectionData m_data_debug_info;
  SectionData m_data_debug_line;
  SectionData m_data_debug_line_str;
  SectionData m_data_debug_macro;
  SectionData m_data_debug_str;
  SectionData m_data_debug_str_offsets;
  SectionData m_data_debug_types;

  bool isDwo() { return m_dwo_section_list != nullptr; }

  const DWARFDataExtractor &
  LoadOrGetSection(lldb::SectionType main_section_type,
                   llvm::Optional<lldb::SectionType> dwo_section_type,
                   SectionData &data);

public:
  explicit DWARFContext(SectionList *main_section_list,
                        SectionList *dwo_section_list)
      : m_main_section_list(main_section_list),
        m_dwo_section_list(dwo_section_list) {}

  const DWARFDataExtractor &getOrLoadAbbrevData();
  const DWARFDataExtractor &getOrLoadAddrData();
  const DWARFDataExtractor &getOrLoadArangesData();
  const DWARFDataExtractor &getOrLoadDebugInfoData();
  const DWARFDataExtractor &getOrLoadLineData();
  const DWARFDataExtractor &getOrLoadLineStrData();
  const DWARFDataExtractor &getOrLoadMacroData();
  const DWARFDataExtractor &getOrLoadStrData();
  const DWARFDataExtractor &getOrLoadStrOffsetsData();
  const DWARFDataExtractor &getOrLoadDebugTypesData();
};
} // namespace lldb_private

#endif
