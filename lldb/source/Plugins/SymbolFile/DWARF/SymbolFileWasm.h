//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_SYMBOLFILEWASM_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_SYMBOLFILEWASM_H

#include "SymbolFileDWARF.h"

namespace lldb_private::plugin {
namespace dwarf {
class SymbolFileWasm : public SymbolFileDWARF {
public:
  SymbolFileWasm(lldb::ObjectFileSP objfile_sp, SectionList *dwo_section_list);

  ~SymbolFileWasm() override;

  lldb::offset_t GetVendorDWARFOpcodeSize(const DataExtractor &data,
                                          const lldb::offset_t data_offset,
                                          const uint8_t op) const override;

  bool ParseVendorDWARFOpcode(uint8_t op, const DataExtractor &opcodes,
                              lldb::offset_t &offset, RegisterContext *reg_ctx,
                              lldb::RegisterKind reg_kind,
                              std::vector<Value> &stack) const override;
};
} // namespace dwarf
} // namespace lldb_private::plugin

#endif
