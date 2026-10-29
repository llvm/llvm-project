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
#include "llvm/ADT/STLExtras.h"
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
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSemExpr.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
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

/// Returns the DWARF number of \p Reg. Sub-registers are rejected, because the
/// analysis tracks super-registers and a sub-register only holds part of the
/// value.
static std::optional<DWARFRegNum> getDWARFSuperReg(const MCRegisterInfo *MCRI,
                                                   MCRegister Reg, bool IsEH) {
  if (!Reg.isValid() || !isSuperReg(MCRI, Reg))
    return std::nullopt;
  int DWARFReg = MCRI->getDwarfRegNum(Reg, IsEH);
  if (DWARFReg < 0)
    return std::nullopt;
  return DWARFReg;
}

/// Matches an address of the form `Reg + Offset`.
static std::optional<CFARegOffsetInfo>
getRegPlusOffset(const MCRegisterInfo *MCRI, const MCSemAddrExpr &Expr,
                 bool IsEH) {
  if (Expr.isConstant() || Expr.getScale() != 1)
    return std::nullopt;
  auto MaybeReg = getDWARFSuperReg(MCRI, Expr.getReg(), IsEH);
  if (!MaybeReg)
    return std::nullopt;
  return CFARegOffsetInfo(*MaybeReg, Expr.getOffset());
}

/// Matches a value of the form `Reg + Offset`.
static std::optional<CFARegOffsetInfo>
getRegPlusOffset(const MCRegisterInfo *MCRI, const MCSemExpr &Expr, bool IsEH) {
  if (Expr.isConstant() || Expr.getScale() != 1 || !Expr.getLeaf().isReg())
    return std::nullopt;
  auto MaybeReg = getDWARFSuperReg(MCRI, Expr.getLeaf().getReg(), IsEH);
  if (!MaybeReg)
    return std::nullopt;
  return CFARegOffsetInfo(*MaybeReg, Expr.getOffset());
}

/// Matches a plain load `mem[Reg + Offset]` and returns its address.
static std::optional<CFARegOffsetInfo>
getLoadAddress(const MCRegisterInfo *MCRI, const MCSemExpr &Expr, bool IsEH) {
  if (Expr.isConstant() || Expr.getScale() != 1 || Expr.getOffset() != 0 ||
      !Expr.getLeaf().isMem())
    return std::nullopt;
  return getRegPlusOffset(MCRI, Expr.getLeaf().getAddr(), IsEH);
}

/// Returns the register that holds the caller's value of \p Reg: \p Reg
/// itself for `same_value`, or the other register for `register`.
static std::optional<DWARFRegNum>
getCallerValueReg(const dwarf::UnwindLocation &Loc, DWARFRegNum Reg) {
  switch (Loc.getLocation()) {
  case dwarf::UnwindLocation::Location::Same:
    return Reg;
  case dwarf::UnwindLocation::Location::RegPlusOffset:
    if (!Loc.getDereference() && Loc.getOffset() == 0 && !Loc.hasAddressSpace())
      return Loc.getRegister();
    return std::nullopt;
  default:
    return std::nullopt;
  }
}

/// The rule for \p Reg when its caller's value is held in \p ValueReg.
static dwarf::UnwindLocation getRegisterRule(DWARFRegNum Reg,
                                             DWARFRegNum ValueReg) {
  if (Reg == ValueReg)
    return dwarf::UnwindLocation::createSame();
  return dwarf::UnwindLocation::createIsRegisterPlusOffset(ValueReg, 0);
}

/// Returns true if the two rules only differ in their offset, for example
/// `CFA + 8` and `CFA + 16`.
static bool isStructurallySimilar(const dwarf::UnwindLocation &LHS,
                                  const dwarf::UnwindLocation &RHS) {
  if (LHS.getLocation() != RHS.getLocation() ||
      LHS.getDereference() != RHS.getDereference())
    return false;
  switch (LHS.getLocation()) {
  case dwarf::UnwindLocation::Location::CFAPlusOffset:
    return true;
  case dwarf::UnwindLocation::Location::RegPlusOffset:
    return LHS.getRegister() == RHS.getRegister();
  default:
    return false;
  }
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
                                   const MCInstrAnalysis *MCIA, bool IsEH,
                                   ArrayRef<MCCFIInstruction> Prologue)
    : State(Context), Context(Context), MCII(MCII), MCIA(MCIA),
      MCRI(Context->getRegisterInfo()), IsEH(IsEH) {

  for (auto LLVMReg : getTrackingRegs(MCRI)) {
    if (MCRI->get(LLVMReg).IsArtificial || MCRI->get(LLVMReg).IsConstant)
      continue;

    DWARFRegNum Reg = MCRI->getDwarfRegNum(LLVMReg, IsEH);
    // Based on dwarf documentation, the default rule for all columns before
    // interpretation of the initial instructions is the undefined rule.

    // For now, this tool depends on the user to write a prologue that
    // establishes the rules required for proper validation of the rest of the
    // function.
    State.update(MCCFIInstruction::createUndefined(nullptr, Reg));
  }

  State.update(MCCFIInstruction::createUndefined(
      nullptr, MCRI->getDwarfRegNum(MCRI->getProgramCounter(), IsEH)));

  for (auto &&InitialFrameStateCFIDirective :
       Context->getAsmInfo().getInitialFrameState())
    State.update(InitialFrameStateCFIDirective);

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

std::optional<MCSemExpr> DWARFCFIAnalysis::getDefinedValue(
    const MCInst &Inst, DWARFRegNum Reg,
    const SmallSet<DWARFRegNum, 4> &Writes) const {
  // The semantics can be incomplete, so `Writes` stays the source of truth for
  // which registers are modified. Only ask about those.
  if (!MCIA || !Writes.count(Reg))
    return std::nullopt;
  auto MaybeLLVMReg = MCRI->getLLVMRegNum(Reg, IsEH);
  if (!MaybeLLVMReg)
    return std::nullopt;
  return MCIA->evaluateDefinedValue(Inst, *MaybeLLVMReg);
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

  // Abstract execution. Collect the rules that are valid for the register
  // after this instruction. Everything, including the CFA, is read from the
  // state before the instruction.
  SmallVector<dwarf::UnwindLocation> PossibleNextLocs;
  auto MaybePrevCFA = getCFARegOffsetInfo(PrevRow);
  if (auto MaybeValueReg = getCallerValueReg(PrevLoc, Reg)) {
    // The caller's value is in a register.
    DWARFRegNum ValueReg = *MaybeValueReg;

    if (!Writes.count(ValueReg))
      PossibleNextLocs.push_back(PrevLoc);

    // Spill: if `ValueReg` is stored at `CFAReg + N`, the caller's value is now
    // at `CFA + (N - CFAOffset)`.
    if (MCIA && MaybePrevCFA) {
      auto StoreAddrs = MCIA->evaluateStoreAddress(Inst);
      auto StoredValues = MCIA->getStoredValue(Inst);
      if (StoreAddrs.size() == StoredValues.size()) {
        for (auto &&[Addr, Value] : zip_equal(StoreAddrs, StoredValues)) {
          auto MaybeAddr = getRegPlusOffset(MCRI, Addr, IsEH);
          auto MaybeValue = getRegPlusOffset(MCRI, Value, IsEH);
          if (!MaybeAddr || !MaybeValue ||
              MaybeAddr->Reg != MaybePrevCFA->Reg ||
              MaybeValue->Reg != ValueReg || MaybeValue->Offset != 0)
            continue;
          int64_t SlotOffset = MaybeAddr->Offset - MaybePrevCFA->Offset;
          if (isInt<32>(SlotOffset))
            PossibleNextLocs.push_back(
                dwarf::UnwindLocation::createAtCFAPlusOffset(SlotOffset));
        }
      }
    }

    // Copy: `ValueReg` is moved to another register.
    for (DWARFRegNum DefReg : Writes)
      if (auto MaybeDefValue = getDefinedValue(Inst, DefReg, Writes))
        if (auto MaybeDef = getRegPlusOffset(MCRI, *MaybeDefValue, IsEH);
            MaybeDef && MaybeDef->Reg == ValueReg && MaybeDef->Offset == 0)
          PossibleNextLocs.push_back(getRegisterRule(Reg, DefReg));
  } else if (PrevLoc.getLocation() ==
                 dwarf::UnwindLocation::Location::CFAPlusOffset &&
             PrevLoc.getDereference()) {
    // The caller's value is on the stack at `CFA + Offset`. Any store might
    // overwrite it, since the list of stored addresses can be incomplete.
    if (!MCII.get(Inst.getOpcode()).mayStore())
      PossibleNextLocs.push_back(PrevLoc);

    // Restore: the slot is loaded back into a register.
    if (MaybePrevCFA)
      for (DWARFRegNum DefReg : Writes)
        if (auto MaybeDefValue = getDefinedValue(Inst, DefReg, Writes))
          if (auto MaybeLoad = getLoadAddress(MCRI, *MaybeDefValue, IsEH);
              MaybeLoad && MaybeLoad->Reg == MaybePrevCFA->Reg &&
              MaybeLoad->Offset - MaybePrevCFA->Offset == PrevLoc.getOffset())
            PossibleNextLocs.push_back(getRegisterRule(Reg, DefReg));
  }

  if (is_contained(PossibleNextLocs, NextLoc)) // Match
    return;
  // An unchanged rule that is not in the set may have been invalidated by the
  // instruction. The fallback checks below report that.
  if (!(PrevLoc == NextLoc)) {
    for (auto &&PossibleNextLoc : PossibleNextLocs) {
      if (!isStructurallySimilar(PossibleNextLoc, NextLoc))
        continue;
      Context->reportError(
          Inst.getLoc(),
          formatv("expected register {0} unwinding rule's offset to be {1}, "
                  "but the CFI directives set it to {2}",
                  RegName, PossibleNextLoc.getOffset(), NextLoc.getOffset()));
      return;
    }
  }

  // The abstract execution could not decide, so fall back to the checks based
  // on the set of modified registers.

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

  // Abstract execution. Collect the CFA rules that are valid after this
  // instruction, reading everything from the state before it.
  SmallVector<CFARegOffsetInfo> PossibleNextCFAs;
  if (!Writes.count(PrevCFA.Reg))
    PossibleNextCFAs.push_back(PrevCFA);
  // If `Reg := CFAReg + C`, then `CFA = CFAReg + Offset = Reg + (Offset - C)`.
  // This covers push/pop on the stack pointer (Reg == CFAReg) and copying the
  // stack pointer into the frame pointer.
  for (DWARFRegNum DefReg : Writes)
    if (auto MaybeDefValue = getDefinedValue(Inst, DefReg, Writes))
      if (auto MaybeDef = getRegPlusOffset(MCRI, *MaybeDefValue, IsEH);
          MaybeDef && MaybeDef->Reg == PrevCFA.Reg)
        PossibleNextCFAs.emplace_back(DefReg,
                                      PrevCFA.Offset - MaybeDef->Offset);

  if (is_contained(PossibleNextCFAs, NextCFA)) // Match
    return;
  // An unchanged rule that is not in the set may have been invalidated by the
  // instruction. The fallback checks below report that.
  if (!(PrevCFA == NextCFA)) {
    for (auto &&PossibleNextCFA : PossibleNextCFAs) {
      if (PossibleNextCFA.Reg != NextCFA.Reg)
        continue;
      Context->reportError(
          Inst.getLoc(),
          formatv("expected CFA offset {0} from CFA register {1}, but the CFI "
                  "directives set it to {2}",
                  PossibleNextCFA.Offset, NextCFARegName, NextCFA.Offset));
      return;
    }
  }

  // The abstract execution could not decide, so fall back to the checks based
  // on the set of modified registers.

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
