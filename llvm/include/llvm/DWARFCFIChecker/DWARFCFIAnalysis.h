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
/// `DWARFCFIAnalysis` is an implementation of a DWARF CFI checker
/// described in this link:
/// https://discourse.llvm.org/t/rfc-dwarf-cfi-validation/86936
///
/// The goal of the checker is to validate DWARF CFI directives using the
/// prologue directives and the machine instructions. It compares the CFI state
/// after each instruction with the state obtained by abstract execution of the
/// instruction. The abstract execution uses the instruction semantics from
/// `MCInstrAnalysis`. When those are not available for an instruction, the
/// checker falls back to a simpler check based on the registers the
/// instruction modifies.
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
#include "llvm/MC/MCInstrAnalysis.h"
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
/// Each update starts with an abstract execution of the instruction on the
/// previous CFI state, using the semantics from `MCInstrAnalysis`
/// (`evaluateDefinedValue`, `evaluateStoreAddress`, and `getStoredValue`). It
/// collects the rules that are still valid for each register and the CFA after
/// the instruction:
/// - A rule stays valid if the instruction does not modify the registers it
///   uses.
/// - If the instruction sets a register to the CFA register plus a constant,
///   the CFA can be described with that register. For example, after
///   `push %rax`, `CFA = %rsp + 8` becomes `CFA = %rsp + 16`, and after
///   `mov %rsp, %rbp`, `CFA = %rbp + 8` is also valid.
/// - If a register is stored at a CFA-relative address, the register's rule can
///   become that stack slot. If the slot is loaded back into a register, the
///   rule can become that register.
///
/// The rule coming from the CFI directives is then compared with this set. If
/// it is in the set, nothing is reported. If it only differs from one of them
/// in its offset (e.g. `CFA + 15` instead of `CFA + 16`), an error with the
/// expected offset is reported. Otherwise, the checks below are used.
///
/// The semantics are sound but may be incomplete, so the checks below only
/// rely on the set of registers the instruction modifies. A write to a
/// sub-register is treated as a write to its super-register.
///
/// For each register (or CFA), the following cases can happen:
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
  /// \p MCIA provides the instruction semantics. It can be null, in which case
  /// only the checks based on modified registers are done.
  LLVM_ABI DWARFCFIAnalysis(MCContext *Context, MCInstrInfo const &MCII,
                            const MCInstrAnalysis *MCIA, bool IsEH,
                            ArrayRef<MCCFIInstruction> Prologue);

  LLVM_ABI void update(const MCInst &Inst,
                       ArrayRef<MCCFIInstruction> Directives);

private:
  /// Returns the new value of \p Reg after \p Inst, written in terms of the
  /// state before \p Inst, or std::nullopt if it is not known.
  std::optional<MCSemExpr>
  getDefinedValue(const MCInst &Inst, DWARFRegNum Reg,
                  const SmallSet<DWARFRegNum, 4> &Writes) const;

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
  const MCInstrAnalysis *MCIA;
  MCRegisterInfo const *MCRI;
  bool IsEH;
};

} // namespace llvm

#endif
