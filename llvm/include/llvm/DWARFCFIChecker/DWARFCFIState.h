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

/// This class is used to maintain a CFI state and history, referred to as an
/// unwinding table, during CFI analysis. The table is private, meaning the only
/// way to modify it is to append a new row by updating it with a CFI directive,
/// and the only way to read from it is to read the last row (i.e., current CFI
/// state) from the table. The fetched row is constant and should not be
/// modified or deleted.
class DWARFCFIState {
public:
  DWARFCFIState(MCContext *Context) : Context(Context) {};
  ~DWARFCFIState();

  std::optional<const dwarf::UnwindRow *> getCurrentUnwindRow() const;
  void update(const MCCFIInstruction &Directive);

private:
  MCContext *Context;
  std::vector<dwarf::UnwindRow *> Table;

  std::optional<dwarf::CFIProgram> convert(MCCFIInstruction Directive);
};

} // namespace llvm

#endif
