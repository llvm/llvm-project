//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares UnwindInfoAnalysis class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNWINDINFOCHECKER_UNWINDINFOANALYSIS_H
#define LLVM_UNWINDINFOCHECKER_UNWINDINFOANALYSIS_H

#include "UnwindInfoState.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include <set>

namespace llvm {

class UnwindInfoAnalysis {
  MCContext *Context;
  MCInstrInfo const &MCII;
  MCRegisterInfo const *MCRI;
  UnwindInfoState State;
  bool IsEH;

public:
  UnwindInfoAnalysis(MCContext *Context, MCInstrInfo const &MCII, bool IsEH,
                     ArrayRef<MCCFIInstruction> Prologue);

  void update(const MCInst &Inst, ArrayRef<MCCFIInstruction> CFIDirectives);

private:
  void checkRegDiff(const MCInst &Inst, DWARFRegNum Reg,
                    const dwarf::UnwindTable::const_iterator &PrevRow,
                    const dwarf::UnwindTable::const_iterator &NextRow,
                    const std::set<DWARFRegNum> &Reads,
                    const std::set<DWARFRegNum> &Writes);

  void checkCFADiff(const MCInst &Inst,
                    const dwarf::UnwindTable::const_iterator &PrevRow,
                    const dwarf::UnwindTable::const_iterator &NextRow,
                    const std::set<DWARFRegNum> &Reads,
                    const std::set<DWARFRegNum> &Writes);
};

} // namespace llvm
#endif
