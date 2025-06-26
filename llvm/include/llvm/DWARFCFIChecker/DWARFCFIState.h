//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares DWARFCFIState class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_DWARFCFICHECKER_UNWINDINFOSTATE_H
#define LLVM_DWARFCFICHECKER_UNWINDINFOSTATE_H

#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include <optional>

namespace llvm {

using DWARFRegNum = uint32_t;

class DWARFCFIState {
public:
  DWARFCFIState(MCContext *Context) : Context(Context) {};

  std::optional<dwarf::UnwindTable::const_iterator> getCurrentUnwindRow() const;
  void update(const MCCFIInstruction &Directive);

private:
  MCContext *Context;
  dwarf::UnwindTable::RowContainer Table;

  std::optional<dwarf::CFIProgram> convert(MCCFIInstruction Directive);
};

} // namespace llvm

#endif
