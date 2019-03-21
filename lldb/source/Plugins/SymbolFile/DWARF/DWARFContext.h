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
#include "lldb/Core/Module.h"
#include "llvm/ADT/Optional.h"
#include <memory>

namespace lldb_private {
class DWARFContext {
private:
  Module &m_module;
  llvm::Optional<DWARFDataExtractor> m_data_debug_aranges;
  llvm::Optional<DWARFDataExtractor> m_data_debug_frame;
  llvm::Optional<DWARFDataExtractor> m_data_debug_line;
  llvm::Optional<DWARFDataExtractor> m_data_debug_line_str;
  llvm::Optional<DWARFDataExtractor> m_data_debug_macro;
  llvm::Optional<DWARFDataExtractor> m_data_debug_loc;
  llvm::Optional<DWARFDataExtractor> m_data_debug_loclists;
  llvm::Optional<DWARFDataExtractor> m_data_debug_ranges;
  llvm::Optional<DWARFDataExtractor> m_data_debug_rnglists;
  llvm::Optional<DWARFDataExtractor> m_data_debug_types;
  llvm::Optional<DWARFDataExtractor> m_data_gnu_debug_altlink;

public:
  explicit DWARFContext(Module &module);

  const DWARFDataExtractor *getOrLoadArangesData();
  const DWARFDataExtractor *getOrLoadDebugLineData();
  const DWARFDataExtractor *getOrLoadDebugLineStrData();
  const DWARFDataExtractor *getOrLoadDebugMacroData();
  const DWARFDataExtractor *getOrLoadDebugLocData();
  const DWARFDataExtractor *getOrLoadDebugLoclistData();
  const DWARFDataExtractor *getOrLoadDebugRangesData();
  const DWARFDataExtractor *getOrLoadDebugRnglistsData();
  const DWARFDataExtractor *getOrLoadDebugFrameData();
  const DWARFDataExtractor *getOrLoadDebugTypesData();
  const DWARFDataExtractor *getOrLoadGnuDebugAltlinkData();

  const DWARFDataExtractor *getOrLoadBestDebugLocData();
};
} // namespace lldb_private

#endif
