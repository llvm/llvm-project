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
#include <memory>

namespace lldb_private {
class DWARFContext {
private:
  SectionList *m_main_section_list;
  SectionList *m_dwo_section_list;

  llvm::Optional<DWARFDataExtractor> m_data_debug_aranges;
  llvm::Optional<DWARFDataExtractor> m_data_debug_info;

  bool isDwo() { return m_dwo_section_list != nullptr; }

public:
  explicit DWARFContext(SectionList *main_section_list,
                        SectionList *dwo_section_list)
      : m_main_section_list(main_section_list),
        m_dwo_section_list(dwo_section_list) {}

  const DWARFDataExtractor &getOrLoadArangesData();
  const DWARFDataExtractor &getOrLoadDebugInfoData();
};
} // namespace lldb_private

#endif
