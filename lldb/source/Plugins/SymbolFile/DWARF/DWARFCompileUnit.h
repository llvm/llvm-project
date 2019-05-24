//===-- DWARFCompileUnit.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_DWARFCompileUnit_h_
#define SymbolFileDWARF_DWARFCompileUnit_h_

#include "DWARFUnit.h"
#include "llvm/Support/Error.h"

class DWARFCompileUnit : public DWARFUnit {
public:
  void Dump(lldb_private::Stream *s) const override;

private:
  DWARFCompileUnit(SymbolFileDWARF *dwarf, lldb::user_id_t uid,
                   const DWARFUnitHeader &header,
                   const DWARFAbbreviationDeclarationSet &abbrevs,
                   DIERef::Section section)
      : DWARFUnit(dwarf, uid, header, abbrevs, section) {}

  DISALLOW_COPY_AND_ASSIGN(DWARFCompileUnit);

  friend class DWARFUnit;
};

#endif // SymbolFileDWARF_DWARFCompileUnit_h_
