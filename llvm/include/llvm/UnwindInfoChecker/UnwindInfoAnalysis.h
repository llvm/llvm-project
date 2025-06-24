//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file declares UnwindInfoAnalysis class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNWINDINFOCHECKER_UNWINDINFOANALYSIS_H
#define LLVM_UNWINDINFOCHECKER_UNWINDINFOANALYSIS_H

#include "UnwindInfoHistory.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegister.h"
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
  UnwindInfoHistory State;
  bool IsEH;

private:
  // The CFI analysis only keeps track and cares about super registers, not the
  // subregisters. All reads to/writes from subregisters and considered the same
  // operation to super registers.
  // TODO Other operations like loading and stores are considered only if they
  // TODO are exactly doing the operation to or from a super register. As an
  // TODO example, if you spill a sub register to stack, the CFI analysis does
  // TODO not consider that a register spilling.
  bool isSuperReg(MCPhysReg Reg);

  SmallVector<std::pair<MCPhysReg, MCRegisterClass const *>> getAllSuperRegs();

  MCPhysReg getSuperReg(MCPhysReg Reg);

public:
  UnwindInfoAnalysis(MCContext *Context, MCInstrInfo const &MCII, bool IsEH,
                     ArrayRef<MCCFIInstruction> PrologueCFIDirectives);

  void update(const MCInst &Inst, ArrayRef<MCCFIInstruction> CFIDirectives);

private:
  void checkRegDiff(const MCInst &Inst, DWARFRegType Reg,
                    const dwarf::UnwindTable::const_iterator &PrevRow,
                    const dwarf::UnwindTable::const_iterator &NextRow,
                    const std::set<DWARFRegType> &Reads,
                    const std::set<DWARFRegType> &Writes);

  void checkCFADiff(const MCInst &Inst,
                    const dwarf::UnwindTable::const_iterator &PrevRow,
                    const dwarf::UnwindTable::const_iterator &NextRow,
                    const std::set<DWARFRegType> &Reads,
                    const std::set<DWARFRegType> &Writes);
};

} // namespace llvm
#endif