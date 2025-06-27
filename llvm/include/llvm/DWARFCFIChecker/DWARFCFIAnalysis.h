//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares DWARFCFIAnalysis class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_DWARFCFICHECKER_DWARFCFIANALYSIS_H
#define LLVM_DWARFCFICHECKER_DWARFCFIANALYSIS_H

#include "DWARFCFIState.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallSet.h"
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

namespace llvm {

class DWARFCFIAnalysis {
  MCContext *Context;
  MCInstrInfo const &MCII;
  MCRegisterInfo const *MCRI;
  DWARFCFIState State;
  bool IsEH;

public:
  DWARFCFIAnalysis(MCContext *Context, MCInstrInfo const &MCII, bool IsEH,
                   ArrayRef<MCCFIInstruction> Prologue);

  void update(const MCInst &Inst, ArrayRef<MCCFIInstruction> Directives);

private:
  void checkRegDiff(const MCInst &Inst, DWARFRegNum Reg,
                    const dwarf::UnwindRow *PrevRow,
                    const dwarf::UnwindRow *NextRow,
                    const SmallSet<DWARFRegNum, 4> &Reads,
                    const SmallSet<DWARFRegNum, 4> &Writes);

  void checkCFADiff(const MCInst &Inst, const dwarf::UnwindRow *PrevRow,
                    const dwarf::UnwindRow *NextRow,
                    const SmallSet<DWARFRegNum, 4> &Reads,
                    const SmallSet<DWARFRegNum, 4> &Writes);
};

} // namespace llvm

#endif
