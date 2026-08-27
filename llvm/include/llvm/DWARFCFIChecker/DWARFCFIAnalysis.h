//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares `DWARFCFIAnalysis` class.
/// `DWARFCFIAnalysis` is a minimal implementation of a DWARF CFI checker
/// described in this link:
/// https://discourse.llvm.org/t/rfc-dwarf-cfi-validation/86936
///
/// The goal of the checker is to validate DWARF CFI directives using the
/// prologue directives and the machine instructions. The main proposed
/// algorithm validates the directives by comparing the CFI state in each
/// instruction with the state achieved by abstract execution of the instruction
/// on the CFI state. However, the current version implemented here is a simple
/// conditional check based on the registers modified by each instruction.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_DWARFCFICHECKER_DWARFCFIANALYSIS_H
#define LLVM_DWARFCFICHECKER_DWARFCFIANALYSIS_H

#include "DWARFCFIState.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/DWARF/LowLevel/DWARFUnwindTable.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

/// `DWARFCFIAnalysis` validates the DWARF Call Frame Information one machine
/// instruction at a time. This class maintains an internal CFI state
/// initialized with the prologue directives and updated with each instruction's
/// associated directives. In each update, it checks if the machine
/// instruction changes the CFI state in a way that matches the changes
/// from the CFI directives. This checking may results in errors and warnings.
///
/// In current stage, the analysis is only aware of what registers the
/// instruction modifies. If the modification is happening to a sub-register,
/// the analysis considers the super-register is modified.
///
/// In each update, for each register (or CFA), the following cases can happen:
/// 1. The unwinding rule is not changed:
///   a. The registers involved in this rule are not modified: the analysis
///      proceeds without emitting error or warning.
///   b. The registers involved in this rule are modified: it emits an error.
/// 2. The unwinding rule is changed:
///   a. The rule is structurally modified (i.e., the location is changed): It
///      emits a warning.
///   b. The rule is structurally the same, but the register set is changed: it
///      emits a warning.
///   c. The rule is structurally the same, using the same set of registers, but
///      the offset is changed:
///      i. If the registers included in the rule are modified as well: It
///         emits a warning.
///     ii. If the registers included in the rule are not modified: It emits an
///         error.
///
/// The analysis only checks the CFA unwinding rule when the rule is a register
/// plus some offset. Therefore, for CFA, only cases 1, 2.b, and 2.c are
/// checked, and in all other case(s), a warning is emitted.
class DWARFCFIAnalysis {
public:
  LLVM_ABI DWARFCFIAnalysis(MCContext *Context, MCInstrInfo const &MCII,
                            bool IsEH, ArrayRef<MCCFIInstruction> Prologue);

  LLVM_ABI void update(const MCInst &Inst,
                       ArrayRef<MCCFIInstruction> Directives);

private:
  void checkRegDiff(const MCInst &Inst, DWARFRegNum Reg,
                    const dwarf::UnwindRow &PrevRow,
                    const dwarf::UnwindRow &NextRow,
                    const SmallSet<DWARFRegNum, 4> &Writes);

  void checkCFADiff(const MCInst &Inst, const dwarf::UnwindRow &PrevRow,
                    const dwarf::UnwindRow &NextRow,
                    const SmallSet<DWARFRegNum, 4> &Writes);

private:
  DWARFCFIState State;
  MCContext *Context;
  MCInstrInfo const &MCII;
  MCRegisterInfo const *MCRI;
  bool IsEH;
};

} // namespace llvm

#endif
