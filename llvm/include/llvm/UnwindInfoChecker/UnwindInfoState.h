//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares UnwindInfoState class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNWINDINFOCHECKER_UNWINDINFOSTATE_H
#define LLVM_UNWINDINFOCHECKER_UNWINDINFOSTATE_H

#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include <cstdint>
#include <optional>
namespace llvm {

using DWARFRegNum = uint32_t;

class UnwindInfoState {
public:
  UnwindInfoState(MCContext *Context) : Context(Context) {};

  std::optional<dwarf::UnwindTable::const_iterator> getCurrentUnwindRow() const;
  void update(const MCCFIInstruction &CFIDirective);

private:
  MCContext *Context;
  dwarf::UnwindTable::RowContainer Table;

  std::optional<dwarf::CFIProgram> convert(MCCFIInstruction CFIDirective);
};
} // namespace llvm

#endif
