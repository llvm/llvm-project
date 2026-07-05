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

#include "llvm/DebugInfo/DWARF/LowLevel/DWARFUnwindTable.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/Support/Compiler.h"
#include <optional>

namespace llvm {

using DWARFRegNum = uint32_t;

/// This class is used to maintain a CFI state, referred to as an unwinding row,
/// during CFI analysis. The only way to modify the state is by updating it with
/// a CFI directive.
class DWARFCFIState {
public:
  DWARFCFIState(MCContext *Context) : Context(Context), IsInitiated(false) {};

  LLVM_ABI std::optional<dwarf::UnwindRow> getCurrentUnwindRow() const;

  /// This method updates the state by applying \p Directive to the current
  /// state. If the directive is not supported by the checker or any error
  /// happens while applying the CFI directive, a warning or error is reported
  /// to the user, and the directive is ignored, leaving the state unchanged.
  LLVM_ABI void update(const MCCFIInstruction &Directive);

private:
  dwarf::CFIProgram convert(MCCFIInstruction Directive);

private:
  dwarf::UnwindRow Row;
  MCContext *Context;
  bool IsInitiated;
};

} // namespace llvm

#endif
