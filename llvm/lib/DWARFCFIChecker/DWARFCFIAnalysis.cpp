//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DWARFCFIChecker/DWARFCFIAnalysis.h"
#include "Registers.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/DWARFCFIChecker/DWARFCFIState.h"
#include "llvm/DebugInfo/DWARF/LowLevel/DWARFUnwindTable.h"
#include "llvm/MC/MCAsmInfo.h"
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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include <optional>

using namespace llvm;

struct CFARegOffsetInfo {
  DWARFRegNum Reg;
  int64_t Offset;

  CFARegOffsetInfo(DWARFRegNum Reg, int64_t Offset)
      : Reg(Reg), Offset(Offset) {}

  bool operator==(const CFARegOffsetInfo &RHS) const {
    return Reg == RHS.Reg && Offset == RHS.Offset;
  }
};

static std::optional<CFARegOffsetInfo>
getCFARegOffsetInfo(const dwarf::UnwindRow &UnwindRow) {
  auto CFALocation = UnwindRow.getCFAValue();
  if (CFALocation.getLocation() !=
      dwarf::UnwindLocation::Location::RegPlusOffset)
    return std::nullopt;

  return CFARegOffsetInfo(CFALocation.getRegister(), CFALocation.getOffset());
}

static SmallSet<DWARFRegNum, 4>
getUnwindRuleRegSet(const dwarf::UnwindRow &UnwindRow, DWARFRegNum Reg) {
  auto MaybeLoc = UnwindRow.getRegisterLocations().getRegisterLocation(Reg);
  assert(MaybeLoc && "the register should be included in the unwinding row");
  auto Loc = *MaybeLoc;

  switch (Loc.getLocation()) {
  case dwarf::UnwindLocation::Location::Unspecified:
  case dwarf::UnwindLocation::Location::Undefined:
  case dwarf::UnwindLocation::Location::Constant:
  case dwarf::UnwindLocation::Location::CFAPlusOffset:
    // [CFA + offset] does not depend on any register because the CFA value is
    // constant throughout the entire frame; only the way to calculate it might
    // change.
  case dwarf::UnwindLocation::Location::DWARFExpr:
    // TODO: Expressions are not supported yet, but if they were to be
    // supported, all the registers used in an expression should extracted and
    // returned here.
    return {};
  case dwarf::UnwindLocation::Location::Same:
    return {Reg};
  case dwarf::UnwindLocation::Location::RegPlusOffset:
    return {Loc.getRegister()};
  }
  llvm_unreachable("Unknown dwarf::UnwindLocation::Location enum");
}

DWARFCFIAnalysis::DWARFCFIAnalysis(MCContext *Context, MCInstrInfo const &MCII,
                                   bool IsEH,
                                   ArrayRef<MCCFIInstruction> Prologue)
    : State(Context), Context(Context), MCII(MCII),
      MCRI(Context->getRegisterInfo()), IsEH(IsEH) {

  for (auto LLVMReg : getTrackingRegs(MCRI)) {
    if (MCRI->get(LLVMReg).IsArtificial || MCRI->get(LLVMReg).IsConstant)
      continue;

    DWARFRegNum Reg = MCRI->getDwarfRegNum(LLVMReg, IsEH);
    // TODO: this should be `undefined` instead of `same_value`, but because
    // initial frame state doesn't have any directives about callee saved
    // registers, every register is tracked. After initial frame state is
    // corrected, this should be changed.
    State.update(MCCFIInstruction::createSameValue(nullptr, Reg));
  }

  // TODO: Ignoring PC should be in the initial frame state.
  State.update(MCCFIInstruction::createUndefined(
      nullptr, MCRI->getDwarfRegNum(MCRI->getProgramCounter(), IsEH)));

  for (auto &&InitialFrameStateCFIDirective :
       Context->getAsmInfo()->getInitialFrameState())
    State.update(InitialFrameStateCFIDirective);

  auto MaybeCurrentRow = State.getCurrentUnwindRow();
  assert(MaybeCurrentRow && "there should be at least one row");
  auto MaybeCFA = getCFARegOffsetInfo(*MaybeCurrentRow);
  assert(MaybeCFA &&
         "the CFA information should be describable in [reg + offset] in here");
  auto CFA = *MaybeCFA;

  // TODO: CFA register callee value is CFA's value, this should be in initial
  // frame state.
  State.update(MCCFIInstruction::createOffset(nullptr, CFA.Reg, 0));

  // Applying the prologue after default assumptions to overwrite them.
  for (auto &&Directive : Prologue)
    State.update(Directive);
}

void DWARFCFIAnalysis::update(const MCInst &Inst,
                              ArrayRef<MCCFIInstruction> Directives) {
  const MCInstrDesc &MCInstInfo = MCII.get(Inst.getOpcode());

  auto MaybePrevRow = State.getCurrentUnwindRow();
  assert(MaybePrevRow && "the analysis should have initialized the "
                         "state with at least one row by now");
  auto PrevRow = *MaybePrevRow;

  for (auto &&Directive : Directives)
    State.update(Directive);

  SmallSet<DWARFRegNum, 4> Writes;
  for (unsigned I = 0; I < MCInstInfo.NumImplicitDefs; I++)
    Writes.insert(MCRI->getDwarfRegNum(
        getSuperReg(MCRI, MCInstInfo.implicit_defs()[I]), IsEH));

  for (unsigned I = 0; I < Inst.getNumOperands(); I++) {
    auto &&Op = Inst.getOperand(I);
    if (Op.isReg()) {
      if (I < MCInstInfo.getNumDefs())
        Writes.insert(
            MCRI->getDwarfRegNum(getSuperReg(MCRI, Op.getReg()), IsEH));
    }
  }

  auto MaybeNextRow = State.getCurrentUnwindRow();
  assert(MaybeNextRow && "previous row existed, so should the current row");
  auto NextRow = *MaybeNextRow;

  checkCFADiff(Inst, PrevRow, NextRow, Writes);

  for (auto LLVMReg : getTrackingRegs(MCRI)) {
    DWARFRegNum Reg = MCRI->getDwarfRegNum(LLVMReg, IsEH);

    checkRegDiff(Inst, Reg, PrevRow, NextRow, Writes);
  }
}

void DWARFCFIAnalysis::checkRegDiff(const MCInst &Inst, DWARFRegNum Reg,
                                    const dwarf::UnwindRow &PrevRow,
                                    const dwarf::UnwindRow &NextRow,
                                    const SmallSet<DWARFRegNum, 4> &Writes) {
  auto MaybePrevLoc = PrevRow.getRegisterLocations().getRegisterLocation(Reg);
  auto MaybeNextLoc = NextRow.getRegisterLocations().getRegisterLocation(Reg);

  // All the tracked registers are added during initiation. So if a register is
  // not added, should stay the same during execution and vice versa.
  if (!MaybePrevLoc) {
    assert(!MaybeNextLoc && "the register unwind info suddenly appeared here");
    return;
  }
  assert(MaybeNextLoc && "the register unwind info suddenly vanished here");

  auto PrevLoc = MaybePrevLoc.value();
  auto NextLoc = MaybeNextLoc.value();

  auto MaybeLLVMReg = MCRI->getLLVMRegNum(Reg, IsEH);
  if (!MaybeLLVMReg) {
    if (!(PrevLoc == NextLoc))
      Context->reportWarning(
          Inst.getLoc(),
          formatv("the dwarf register {0} does not have a LLVM number, but its "
                  "unwind info changed. Ignoring this change",
                  Reg));
    return;
  }
  const char *RegName = MCRI->getName(*MaybeLLVMReg);

  // Each case is annotated with its corresponding number as described in
  // `llvm/include/llvm/DWARFCFIChecker/DWARFCFIAnalysis.h`.

  // TODO: Expressions are not supported yet, but if they were to be supported,
  // note that structure equality for expressions is defined as follows: Two
  // expressions are structurally equal if they become the same after you
  // replace every operand with a placeholder.

  if (PrevLoc == NextLoc) { // Case 1
    for (DWARFRegNum UsedReg : getUnwindRuleRegSet(PrevRow, Reg))
      if (Writes.count(UsedReg)) { // Case 1.b
        auto MaybeLLVMUsedReg = MCRI->getLLVMRegNum(UsedReg, IsEH);
        assert(MaybeLLVMUsedReg && "instructions will always write to a "
                                   "register that has an LLVM register number");
        Context->reportError(
            Inst.getLoc(),
            formatv("changed register {1}, that register {0}'s unwinding rule "
                    "uses, but there is no CFI directives about it",
                    RegName, MCRI->getName(*MaybeLLVMUsedReg)));
        return;
      }
    return; // Case 1.a
  }
  // Case 2
  if (PrevLoc.getLocation() != NextLoc.getLocation()) { // Case 2.a
    Context->reportWarning(
        Inst.getLoc(),
        formatv("validating changes happening to register {0} unwinding "
                "rule structure is not implemented yet",
                RegName));
    return;
  }
  auto &&PrevRegSet = getUnwindRuleRegSet(PrevRow, Reg);
  if (PrevRegSet != getUnwindRuleRegSet(NextRow, Reg)) { // Case 2.b
    Context->reportWarning(
        Inst.getLoc(),
        formatv("validating changes happening to register {0} unwinding "
                "rule register set is not implemented yet",
                RegName));
    return;
  }
  // Case 2.c
  for (DWARFRegNum UsedReg : PrevRegSet)
    if (Writes.count(UsedReg)) { // Case 2.c.i
      Context->reportWarning(
          Inst.getLoc(),
          formatv("register {0} unwinding rule's offset is changed, and one of "
                  "the rule's registers is modified, but validating the "
                  "modification amount is not implemented yet",
                  RegName));
      return;
    }
  // Case 2.c.ii
  Context->reportError(
      Inst.getLoc(), formatv("register {0} unwinding rule's offset is changed, "
                             "but not any of the rule's registers are modified",
                             RegName));
}

void DWARFCFIAnalysis::checkCFADiff(const MCInst &Inst,
                                    const dwarf::UnwindRow &PrevRow,
                                    const dwarf::UnwindRow &NextRow,
                                    const SmallSet<DWARFRegNum, 4> &Writes) {

  auto MaybePrevCFA = getCFARegOffsetInfo(PrevRow);
  auto MaybeNextCFA = getCFARegOffsetInfo(NextRow);

  if (!MaybePrevCFA) {
    if (MaybeNextCFA) {
      Context->reportWarning(Inst.getLoc(),
                             "CFA rule changed to [reg + offset], this "
                             "transition will not be checked");
      return;
    }

    Context->reportWarning(Inst.getLoc(),
                           "CFA rule is not [reg + offset], not checking it");
    return;
  }

  if (!MaybeNextCFA) {
    Context->reportWarning(Inst.getLoc(),
                           "CFA rule changed from [reg + offset], this "
                           "transition will not be checked");
    return;
  }

  auto PrevCFA = *MaybePrevCFA;
  auto NextCFA = *MaybeNextCFA;

  auto MaybeLLVMPrevReg = MCRI->getLLVMRegNum(PrevCFA.Reg, IsEH);
  const char *PrevCFARegName =
      MaybeLLVMPrevReg ? MCRI->getName(*MaybeLLVMPrevReg) : "";
  auto MaybeLLVMNextReg = MCRI->getLLVMRegNum(NextCFA.Reg, IsEH);
  const char *NextCFARegName =
      MaybeLLVMNextReg ? MCRI->getName(*MaybeLLVMNextReg) : "";

  if (PrevCFA == NextCFA) {         // Case 1
    if (!Writes.count(PrevCFA.Reg)) // Case 1.a
      return;
    // Case 1.b
    Context->reportError(
        Inst.getLoc(),
        formatv("modified CFA register {0} but not changed CFA rule",
                PrevCFARegName));
    return;
  }

  if (PrevCFA.Reg != NextCFA.Reg) { // Case 2.b
    Context->reportWarning(
        Inst.getLoc(),
        formatv("CFA register changed from register {0} to register {1}, "
                "validating this change is not implemented yet",
                PrevCFARegName, NextCFARegName));
    return;
  }
  // Case 2.c
  if (Writes.count(PrevCFA.Reg)) { // Case 2.c.i
    Context->reportWarning(
        Inst.getLoc(), formatv("CFA offset is changed from {0} to {1}, and CFA "
                               "register {2} is modified, but validating the "
                               "modification amount is not implemented yet",
                               PrevCFA.Offset, NextCFA.Offset, PrevCFARegName));
    return;
  }
  // Case 2.c.ii
  Context->reportError(
      Inst.getLoc(),
      formatv("did not modify CFA register {0} but changed CFA rule",
              PrevCFARegName));
}
