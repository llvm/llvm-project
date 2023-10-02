//===-- DWARFTypeUnit.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFTYPEUNIT_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFTYPEUNIT_H

#include "DWARFUnit.h"
#include "llvm/Support/Error.h"

namespace llvm {
class DWARFAbbreviationDeclarationSet;
}

class DWARFTypeUnit : public DWARFUnit {
public:
  void BuildAddressRangeTable(DWARFDebugAranges *debug_aranges) override {}

  void Dump(lldb_private::Stream *s) const override;

  uint64_t GetTypeHash() { return m_header.GetTypeHash(); }

  dw_offset_t GetTypeOffset() { return GetOffset() + m_header.GetTypeOffset(); }

  static bool classof(const DWARFUnit *unit) { return unit->IsTypeUnit(); }

private:
  DWARFTypeUnit(SymbolFileDWARF &dwarf, lldb::user_id_t uid,
                const DWARFUnitHeader &header,
                const llvm::DWARFAbbreviationDeclarationSet &abbrevs,
                DIERef::Section section, bool is_dwo)
      : DWARFUnit(dwarf, uid, header, abbrevs, section, is_dwo) {}

  friend class DWARFUnit;
};

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFTYPEUNIT_H
