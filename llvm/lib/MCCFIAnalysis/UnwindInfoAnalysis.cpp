#include "llvm/MCCFIAnalysis/UnwindInfoAnalysis.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
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
#include "llvm/MCCFIAnalysis/UnwindInfoHistory.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstdint>
#include <optional>
#include <set>

using namespace llvm;

struct CFARegOffsetInfo {
  DWARFRegType Reg;
  int64_t Offset;

  CFARegOffsetInfo(DWARFRegType Reg, int64_t Offset)
      : Reg(Reg), Offset(Offset) {}
};

static std::optional<CFARegOffsetInfo>
getCFARegOffsetInfo(const dwarf::UnwindTable::const_iterator &UnwindRow) {
  auto CFALocation = UnwindRow->getCFAValue();
  if (CFALocation.getLocation() !=
      dwarf::UnwindLocation::Location::RegPlusOffset) {
    return std::nullopt;
  }

  return CFARegOffsetInfo(CFALocation.getRegister(), CFALocation.getOffset());
}

static std::optional<DWARFRegType> getReferenceRegisterForUnwindInfoOfRegister(
    const dwarf::UnwindTable::const_iterator &UnwindRow, DWARFRegType Reg) {
  auto UnwinLoc = UnwindRow->getRegisterLocations().getRegisterLocation(Reg);
  assert(UnwinLoc &&
         "The register should be tracked inside the register states");

  switch (UnwinLoc->getLocation()) {
  case dwarf::UnwindLocation::Location::Undefined:
  case dwarf::UnwindLocation::Location::Constant:
  case dwarf::UnwindLocation::Location::Unspecified:
  case dwarf::UnwindLocation::Location::DWARFExpr:
    // TODO here should look into expr and find the registers.
    return std::nullopt;
  case dwarf::UnwindLocation::Location::Same:
    return Reg;
  case dwarf::UnwindLocation::Location::RegPlusOffset:
    return UnwinLoc->getRegister();
  case dwarf::UnwindLocation::Location::CFAPlusOffset:
    // TODO check if it's ok to assume CFA is always depending on other
    // TODO register, if yes assert it here!
    return UnwindRow->getCFAValue().getRegister();
  }
}

// TODO remove it, it's just for debug purposes.
void printUntilNextLine(const char *Str) {
  for (int I = 0; Str[I] != '\0' && Str[I] != '\n'; I++)
    dbgs() << Str[I];
}

bool UnwindInfoAnalysis::isSuperReg(MCPhysReg Reg) {
  return MCRI->superregs(Reg).empty();
}

SmallVector<std::pair<MCPhysReg, MCRegisterClass const *>>
UnwindInfoAnalysis::getAllSuperRegs() {
  std::map<MCPhysReg, MCRegisterClass const *> SuperRegs;
  for (auto &&RegClass : MCRI->regclasses()) {
    for (unsigned I = 0; I < RegClass.getNumRegs(); I++) {
      MCPhysReg Reg = RegClass.getRegister(I);
      if (!isSuperReg(Reg) || MCRI->isArtificial(Reg) || MCRI->isConstant(Reg))
        continue;
      SuperRegs[Reg] = &RegClass;
    }
  }

  SmallVector<std::pair<MCPhysReg, MCRegisterClass const *>> SuperRegWithClass;
  for (auto &&[Reg, RegClass] : SuperRegs)
    SuperRegWithClass.emplace_back(Reg, RegClass);
  return SuperRegWithClass;
}

MCPhysReg UnwindInfoAnalysis::getSuperReg(MCPhysReg Reg) {
  if (isSuperReg(Reg))
    return Reg;
  for (auto SuperReg : MCRI->superregs(Reg)) {
    if (isSuperReg(SuperReg))
      return SuperReg;
  }

  llvm_unreachable("Should either be a super reg, or have a super reg");
}

UnwindInfoAnalysis::UnwindInfoAnalysis(
    MCContext *Context, MCInstrInfo const &MCII, MCInstrAnalysis *MCIA,
    bool IsEH, ArrayRef<MCCFIInstruction> PrologueCFIDirectives)
    : Context(Context), MCII(MCII), MCRI(Context->getRegisterInfo()),
      State(Context), IsEH(IsEH) {

  // TODO These all should be handled by setting .cfi_same_value for only callee
  // TODO saved registers inside `getInitialFrameState`. Instead now what's
  // TODO happening is that the analysis .cfi_same_value all the registers and
  // TODO remove PC (undefined), and RSP (CFA) from them.
  for (auto &&[Reg, RegClass] : getAllSuperRegs()) {
    if (MCRI->get(Reg).IsArtificial || MCRI->get(Reg).IsConstant)
      continue;

    DWARFRegType DwarfReg = MCRI->getDwarfRegNum(Reg, IsEH);
    State.update(MCCFIInstruction::createSameValue(nullptr, DwarfReg));
  }

  State.update(MCCFIInstruction::createUndefined(
      nullptr, MCRI->getDwarfRegNum(MCRI->getProgramCounter(), IsEH)));

  for (auto &&InitialFrameStateCFIDirective :
       Context->getAsmInfo()->getInitialFrameState()) {
    State.update(InitialFrameStateCFIDirective);
  }

  auto MaybeLastRow = State.getCurrentUnwindRow();
  assert(MaybeLastRow && "there should be at least one row");
  auto LastRow = *MaybeLastRow;

  auto MaybeCFA = getCFARegOffsetInfo(LastRow);
  assert(MaybeCFA &&
         "the CFA information should be describable in [Reg + Offset] in here");
  auto CFA = *MaybeCFA;

  State.update(MCCFIInstruction::createOffset(nullptr, CFA.Reg,
                                              0)); // sp's old value is CFA

  // Applying the prologue after default assumptions to overwrite them.
  for (auto &&PrologueCFIDirective : PrologueCFIDirectives) {
    State.update(PrologueCFIDirective);
  }
}

void UnwindInfoAnalysis::update(const MCInst &Inst,
                                ArrayRef<MCCFIInstruction> CFIDirectives) {
  const MCInstrDesc &MCInstInfo = MCII.get(Inst.getOpcode());

  auto MaybePrevUnwindRow = State.getCurrentUnwindRow();
  assert(MaybePrevUnwindRow && "The analysis should have initialized the "
                               "history with at least one row by now");
  auto PrevUnwindRow = MaybePrevUnwindRow.value();

  for (auto &&CFIDirective : CFIDirectives)
    State.update(CFIDirective);

  std::set<DWARFRegType> Writes, Reads;
  for (unsigned I = 0; I < MCInstInfo.NumImplicitUses; I++)
    Reads.insert(
        MCRI->getDwarfRegNum(getSuperReg(MCInstInfo.implicit_uses()[I]), IsEH));
  for (unsigned I = 0; I < MCInstInfo.NumImplicitDefs; I++)
    Writes.insert(
        MCRI->getDwarfRegNum(getSuperReg(MCInstInfo.implicit_defs()[I]), IsEH));

  for (unsigned I = 0; I < Inst.getNumOperands(); I++) {
    auto &&Operand = Inst.getOperand(I);
    if (Operand.isReg()) {
      if (I < MCInstInfo.getNumDefs())
        Writes.insert(
            MCRI->getDwarfRegNum(getSuperReg(Operand.getReg()), IsEH));
      else if (Operand.getReg())
        Reads.insert(MCRI->getDwarfRegNum(getSuperReg(Operand.getReg()), IsEH));
    }
  }

  auto MaybeCurrentUnwindRow = State.getCurrentUnwindRow();
  assert(MaybeCurrentUnwindRow &&
         "Prev row existed, so should the current row.");
  auto CurrentUnwindRow = *MaybeCurrentUnwindRow;

  checkCFADiff(Inst, PrevUnwindRow, CurrentUnwindRow, Reads, Writes);

  for (auto [LLVMReg, _] : getAllSuperRegs()) {
    DWARFRegType Reg = MCRI->getDwarfRegNum(LLVMReg, IsEH);

    auto MaybePrevUnwindLoc =
        PrevUnwindRow->getRegisterLocations().getRegisterLocation(Reg);
    auto MaybeNextUnwindLoc =
        CurrentUnwindRow->getRegisterLocations().getRegisterLocation(Reg);

    if (!MaybePrevUnwindLoc) {
      assert(!MaybeNextUnwindLoc && "The register unwind info suddenly "
                                    "appeared here, ignoring this change");
      continue;
    }

    assert(MaybeNextUnwindLoc && "The register unwind info suddenly vanished "
                                 "here, ignoring this change");

    auto PrevUnwindLoc = MaybePrevUnwindLoc.value();
    auto NextUnwindLoc = MaybeNextUnwindLoc.value();

    checkRegDiff(Inst, Reg, PrevUnwindRow, CurrentUnwindRow, PrevUnwindLoc,
                 NextUnwindLoc, Reads, Writes);
  }
}

void UnwindInfoAnalysis::checkRegDiff(
    const MCInst &Inst, DWARFRegType Reg,
    const dwarf::UnwindTable::const_iterator &PrevRow,
    const dwarf::UnwindTable::const_iterator &NextRow,
    const dwarf::UnwindLocation &PrevRegLoc,
    const dwarf::UnwindLocation &NextRegLoc,
    const std::set<DWARFRegType> &Reads, const std::set<DWARFRegType> &Writes) {
  auto MaybeRegLLVM = MCRI->getLLVMRegNum(Reg, IsEH);
  if (!MaybeRegLLVM) {
    assert(PrevRegLoc == NextRegLoc &&
           "The dwarf register does not have a LLVM number, so the unwind info "
           "for it should not change");
    return;
  }

  const char *RegLLVMName = MCRI->getName(MaybeRegLLVM.value());

  auto &&MaybePrevRefReg =
      getReferenceRegisterForUnwindInfoOfRegister(PrevRow, Reg);

  std::optional<MCPhysReg> PrevRefRegLLVM =
      (MaybePrevRefReg ? MCRI->getLLVMRegNum(MaybePrevRefReg.value(), IsEH)
                       : std::nullopt);
  std::optional<MCPhysReg> NextRefRegLLVM =
      (MaybePrevRefReg ? MCRI->getLLVMRegNum(MaybePrevRefReg.value(), IsEH)
                       : std::nullopt);

  if (PrevRegLoc == NextRegLoc) {
    switch (PrevRegLoc.getLocation()) {
    case dwarf::UnwindLocation::Same:
    case dwarf::UnwindLocation::RegPlusOffset:
      if (Writes.count(MaybePrevRefReg.value())) {
        Context->reportError(
            Inst.getLoc(),
            formatv("This instruction changes %{1}, that %{0} unwinding rule "
                    "uses, but there is no CFI directives about it",
                    RegLLVMName, MCRI->getName(PrevRefRegLLVM.value())));
        return;
      }
      break;
      // TODO think about what to do with expressions here.
      // TODO the expressions be solved more precise, by looking into the
      // TODO expression, finding registers that are used and checking if the
      // TODO instruction is changing them or not.
    default:
      // Everything may be ok
      break;
    }
    return;
  }

  if (PrevRegLoc.getLocation() == NextRegLoc.getLocation()) {
    // Everything may be ok
    return;
  }

  if (PrevRegLoc.getLocation() == dwarf::UnwindLocation::Undefined) {
    Context->reportError(
        Inst.getLoc(),
        "Changed %{0} unwinding rule from `undefined` to something else");
    return;
  }

  Context->reportWarning(
      Inst.getLoc(),
      formatv("Unknown change happened to %{0} unwinding rule", RegLLVMName));
  // Everything may be ok
  return;
}

void UnwindInfoAnalysis::checkCFADiff(
    const MCInst &Inst, const dwarf::UnwindTable::const_iterator &PrevRow,
    const dwarf::UnwindTable::const_iterator &NextRow,
    const std::set<DWARFRegType> &Reads, const std::set<DWARFRegType> &Writes) {

  auto MaybePrevCFA = getCFARegOffsetInfo(PrevRow);
  auto MaybeNextCFA = getCFARegOffsetInfo(NextRow);

  if (!MaybePrevCFA) {
    if (MaybeNextCFA) {
      Context->reportWarning(
          Inst.getLoc(),
          "CFA rule changed to [reg + offset], not checking the change");
      return;
    }

    Context->reportWarning(Inst.getLoc(),
                           "CFA rule is not [reg + offset], not checking it");
    return;
  }

  if (!MaybeNextCFA) {
    Context->reportWarning(
        Inst.getLoc(),
        "CFA rule changed from [reg + offset], not checking the change");
    return;
  }

  auto PrevCFA = *MaybePrevCFA;
  auto NextCFA = *MaybeNextCFA;

  const char *PrevCFARegName =
      MCRI->getName(MCRI->getLLVMRegNum(PrevCFA.Reg, IsEH).value());

  if (PrevCFA.Reg == NextCFA.Reg) {
    if (PrevCFA.Offset == NextCFA.Offset) {
      if (Writes.count(PrevCFA.Reg)) {
        Context->reportError(
            Inst.getLoc(),
            formatv("This instruction modifies CFA register (%{0}) "
                    "but CFA rule is not changed",
                    PrevCFARegName));
        return;
      }

      // Everything is ok!
      return;
    }

    // The offset is changed.
    if (!Writes.count(PrevCFA.Reg)) {
      Context->reportError(
          Inst.getLoc(),
          formatv("This instruction does not modifies CFA register (%{0}) "
                  "but CFA rule is changed",
                  PrevCFARegName));
    }

    // The CFA register value is changed, and the offset is changed as well,
    // everything may be ok.
    return;
  }

  // The CFA register is changed, everything may be ok.
}
