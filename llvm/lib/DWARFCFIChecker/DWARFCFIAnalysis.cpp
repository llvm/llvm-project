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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/DWARFCFIChecker/DWARFCFIState.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
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
};

static std::optional<CFARegOffsetInfo>
getCFARegOffsetInfo(const dwarf::UnwindRow *UnwindRow) {
  auto CFALocation = UnwindRow->getCFAValue();
  if (CFALocation.getLocation() !=
      dwarf::UnwindLocation::Location::RegPlusOffset) {
    return std::nullopt;
  }

  return CFARegOffsetInfo(CFALocation.getRegister(), CFALocation.getOffset());
}

static std::optional<DWARFRegNum>
getUnwindRuleRefReg(const dwarf::UnwindRow *UnwindRow, DWARFRegNum Reg) {
  auto MaybeLoc = UnwindRow->getRegisterLocations().getRegisterLocation(Reg);
  assert(MaybeLoc &&
         "The register should be tracked inside the register states");
  auto Loc = *MaybeLoc;

  switch (Loc.getLocation()) {
  case dwarf::UnwindLocation::Location::Undefined:
  case dwarf::UnwindLocation::Location::Constant:
  case dwarf::UnwindLocation::Location::Unspecified:
  case dwarf::UnwindLocation::Location::DWARFExpr:
    // TODO: here should look into expr and find the registers.
    return std::nullopt;
  case dwarf::UnwindLocation::Location::Same:
    return Reg;
  case dwarf::UnwindLocation::Location::RegPlusOffset:
    return Loc.getRegister();
  case dwarf::UnwindLocation::Location::CFAPlusOffset:
    auto MaybeCFA = getCFARegOffsetInfo(UnwindRow);
    if (MaybeCFA)
      return MaybeCFA->Reg;
    return std::nullopt;
  }
}

DWARFCFIAnalysis::DWARFCFIAnalysis(MCContext *Context, MCInstrInfo const &MCII,
                                   bool IsEH,
                                   ArrayRef<MCCFIInstruction> Prologue)
    : Context(Context), MCII(MCII), MCRI(Context->getRegisterInfo()),
      State(Context), IsEH(IsEH) {

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
       Context->getAsmInfo()->getInitialFrameState()) {
    State.update(InitialFrameStateCFIDirective);
  }

  auto MaybeCurrentRow = State.getCurrentUnwindRow();
  assert(MaybeCurrentRow && "there should be at least one row");
  auto MaybeCFA = getCFARegOffsetInfo(*MaybeCurrentRow);
  assert(MaybeCFA &&
         "the CFA information should be describable in [Reg + Offset] in here");
  auto CFA = *MaybeCFA;

  // TODO: CFA register callee value is CFA's value, this should be in initial
  // frame state.
  State.update(MCCFIInstruction::createOffset(nullptr, CFA.Reg, 0));

  // Applying the prologue after default assumptions to overwrite them.
  for (auto &&Directive : Prologue) {
    State.update(Directive);
  }
}

void DWARFCFIAnalysis::update(const MCInst &Inst,
                              ArrayRef<MCCFIInstruction> Directives) {
  const MCInstrDesc &MCInstInfo = MCII.get(Inst.getOpcode());

  auto MaybePrevRow = State.getCurrentUnwindRow();
  assert(MaybePrevRow && "The analysis should have initialized the "
                         "history with at least one row by now");
  const dwarf::UnwindRow *PrevRow = MaybePrevRow.value();

  for (auto &&Directive : Directives)
    State.update(Directive);

  SmallSet<DWARFRegNum, 4> Writes, Reads;
  for (unsigned I = 0; I < MCInstInfo.NumImplicitUses; I++)
    Reads.insert(MCRI->getDwarfRegNum(
        getSuperReg(MCRI, MCInstInfo.implicit_uses()[I]), IsEH));
  for (unsigned I = 0; I < MCInstInfo.NumImplicitDefs; I++)
    Writes.insert(MCRI->getDwarfRegNum(
        getSuperReg(MCRI, MCInstInfo.implicit_defs()[I]), IsEH));

  for (unsigned I = 0; I < Inst.getNumOperands(); I++) {
    auto &&Op = Inst.getOperand(I);
    if (Op.isReg()) {
      if (I < MCInstInfo.getNumDefs())
        Writes.insert(
            MCRI->getDwarfRegNum(getSuperReg(MCRI, Op.getReg()), IsEH));
      else if (Op.getReg())
        Reads.insert(
            MCRI->getDwarfRegNum(getSuperReg(MCRI, Op.getReg()), IsEH));
    }
  }

  auto MaybeNextRow = State.getCurrentUnwindRow();
  assert(MaybeNextRow && "Prev row existed, so should the current row.");
  const dwarf::UnwindRow *NextRow = *MaybeNextRow;

  checkCFADiff(Inst, PrevRow, NextRow, Reads, Writes);

  for (auto LLVMReg : getTrackingRegs(MCRI)) {
    DWARFRegNum Reg = MCRI->getDwarfRegNum(LLVMReg, IsEH);

    checkRegDiff(Inst, Reg, PrevRow, NextRow, Reads, Writes);
  }
}

void DWARFCFIAnalysis::checkRegDiff(const MCInst &Inst, DWARFRegNum Reg,
                                    const dwarf::UnwindRow *PrevRow,
                                    const dwarf::UnwindRow *NextRow,
                                    const SmallSet<DWARFRegNum, 4> &Reads,
                                    const SmallSet<DWARFRegNum, 4> &Writes) {
  auto MaybePrevLoc = PrevRow->getRegisterLocations().getRegisterLocation(Reg);
  auto MaybeNextLoc = NextRow->getRegisterLocations().getRegisterLocation(Reg);

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

  auto &&MaybePrevRefReg = getUnwindRuleRefReg(PrevRow, Reg);
  std::optional<MCPhysReg> PrevRefLLVMReg =
      (MaybePrevRefReg ? MCRI->getLLVMRegNum(MaybePrevRefReg.value(), IsEH)
                       : std::nullopt);

  if (!(PrevLoc == NextLoc)) {
    if (PrevLoc.getLocation() == NextLoc.getLocation()) {
      Context->reportWarning(
          Inst.getLoc(),
          formatv(
              "unknown change happened to register {0} unwinding rule values",
              RegName));
      //! FIXME: Check if the register is changed or not
      return;
    }

    Context->reportWarning(
        Inst.getLoc(),
        formatv(
            "unknown change happened to register {0} unwinding rule structure",
            RegName));
    return;
  }

  switch (PrevLoc.getLocation()) {
  case dwarf::UnwindLocation::Same:
  case dwarf::UnwindLocation::RegPlusOffset:
    assert(MaybePrevRefReg &&
           "when the unwinding rule is the same value, or reg plus offset, "
           "there should always exist a reference register");
    if (Writes.count(MaybePrevRefReg.value())) {
      Context->reportError(
          Inst.getLoc(),
          formatv("changed register {1}, that register {0}'s unwinding rule "
                  "uses, but there is no CFI directives about it",
                  RegName, MCRI->getName(*PrevRefLLVMReg)));
      return;
    }
    break;
  case dwarf::UnwindLocation::DWARFExpr:
    // TODO: Expressions are not supported yet, but if wanted to be supported,
    // all the registers used in an expression should extracted and checked if
    // the instruction modifies them or not.
  default:
    // Everything may be ok
    break;
  }
}

void DWARFCFIAnalysis::checkCFADiff(const MCInst &Inst,
                                    const dwarf::UnwindRow *PrevRow,
                                    const dwarf::UnwindRow *NextRow,
                                    const SmallSet<DWARFRegNum, 4> &Reads,
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

  auto MaybeLLVMReg = MCRI->getLLVMRegNum(PrevCFA.Reg, IsEH);
  const char *PrevCFARegName = MaybeLLVMReg ? MCRI->getName(*MaybeLLVMReg) : "";

  if (PrevCFA.Reg != NextCFA.Reg) {
    //! FIXME: warn here
    return;
  }

  if (PrevCFA.Offset == NextCFA.Offset) {
    if (!Writes.count(PrevCFA.Reg))
      return;
    Context->reportError(
        Inst.getLoc(),
        formatv("modified CFA register ({0}) but not changed CFA rule",
                PrevCFARegName));
  }

  // The offset is changed.
  if (Writes.count(PrevCFA.Reg))
    return;

  Context->reportError(
      Inst.getLoc(),
      formatv("did not modify CFA register ({0}) but changed CFA rule",
              PrevCFARegName));
}
