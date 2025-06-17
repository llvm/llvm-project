#include "llvm/MCCFIAnalysis/CFIAnalysis.h"
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
#include "llvm/MCCFIAnalysis/CFIState.h"
#include "llvm/MCCFIAnalysis/ExtendedMCInstrAnalysis.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <set>

using namespace llvm;

// TODO remove it, it's just for debug purposes.
void printUntilNextLine(const char *Str) {
  for (int I = 0; Str[I] != '\0' && Str[I] != '\n'; I++)
    dbgs() << Str[I];
}

bool CFIAnalysis::isSuperReg(MCPhysReg Reg) {
  return MCRI->superregs(Reg).empty();
}

SmallVector<std::pair<MCPhysReg, MCRegisterClass const *>>
CFIAnalysis::getAllSuperRegs() {
  std::map<MCPhysReg, MCRegisterClass const *> SuperRegs;
  for (auto &&RegClass : MCRI->regclasses()) {
    for (unsigned I = 0; I < RegClass.getNumRegs(); I++) {
      MCPhysReg Reg = RegClass.getRegister(I);
      if (!isSuperReg(Reg))
        continue;
      SuperRegs[Reg] = &RegClass;
    }
  }

  SmallVector<std::pair<MCPhysReg, MCRegisterClass const *>> SuperRegWithClass;
  for (auto &&[Reg, RegClass] : SuperRegs)
    SuperRegWithClass.emplace_back(Reg, RegClass);
  return SuperRegWithClass;
}

MCPhysReg CFIAnalysis::getSuperReg(MCPhysReg Reg) {
  if (isSuperReg(Reg))
    return Reg;
  for (auto SuperReg : MCRI->superregs(Reg)) {
    if (isSuperReg(SuperReg))
      return SuperReg;
  }

  llvm_unreachable("Should either be a super reg, or have a super reg");
}

CFIAnalysis::CFIAnalysis(MCContext *Context, MCInstrInfo const &MCII,
                         MCInstrAnalysis *MCIA, bool IsEH,
                         ArrayRef<MCCFIInstruction> PrologueCFIDirectives)
    : Context(Context), MCII(MCII), MCRI(Context->getRegisterInfo()),
      State(Context), IsEH(IsEH) {
  EMCIA.reset(new ExtendedMCInstrAnalysis(Context, MCII, MCIA));

  for (auto &&[Reg, RegClass] : getAllSuperRegs()) {
    if (MCRI->get(Reg).IsArtificial || MCRI->get(Reg).IsConstant)
      continue;

    DWARFRegType DwarfReg = MCRI->getDwarfRegNum(Reg, IsEH);
    // TODO is it OK to create fake CFI directives for doing this?
    State.apply(MCCFIInstruction::createSameValue(nullptr, DwarfReg));
  }

  for (auto &&InitialFrameStateCFIDirective :
       Context->getAsmInfo()->getInitialFrameState()) {
    State.apply(InitialFrameStateCFIDirective);
  }

  // TODO these are temporay added to make things work.
  // Setup the basic information:
  State.apply(MCCFIInstruction::createUndefined(
      nullptr,
      MCRI->getDwarfRegNum(MCRI->getProgramCounter(),
                           IsEH))); // TODO for now, we don't care about the PC
  // TODO
  auto LastRow = State.getLastRow();
  assert(LastRow && "there should be at least one row");
  // TODO assert that CFA is there, and based on register.
  // TODO

  State.apply(MCCFIInstruction::createOffset(
      nullptr, LastRow->getCFAValue().getRegister(),
      0)); // sp's old value is CFA

  State.apply(MCCFIInstruction::createUndefined(
      nullptr, MCRI->getDwarfRegNum(EMCIA->getFlagsReg(),
                                    IsEH))); // Flags cannot be caller-saved

  // Applying the prologue after default assumptions to overwrite them.
  for (auto &&PrologueCFIDirective : PrologueCFIDirectives) {
    State.apply(PrologueCFIDirective);
  }
}

void CFIAnalysis::update(const MCInst &Inst,
                         ArrayRef<MCCFIInstruction> CFIDirectives) {
  const MCInstrDesc &MCInstInfo = MCII.get(Inst.getOpcode());
  CFIState AfterState(State);
  for (auto &&CFIDirective : CFIDirectives)
    AfterState.apply(CFIDirective);

  //! if (!AfterState.apply(CFIDirective))
  //!     Context->reportWarning(
  //!         CFIDirective.getLoc(),
  //!         "I don't support this CFI directive, I assume this does nothing "
  //!         "(which will probably break other things)");

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

  checkCFADiff(Inst, State, AfterState, Reads, Writes);

  for (auto [LLVMReg, _] : getAllSuperRegs()) {
    DWARFRegType Reg = MCRI->getDwarfRegNum(LLVMReg, IsEH);

    auto PrevUnwindLoc =
        State.getLastRow()->getRegisterLocations().getRegisterLocation(Reg);
    auto NextUnwindLoc =
        AfterState.getLastRow()->getRegisterLocations().getRegisterLocation(
            Reg);

    // TODO think more if these assertions are OK
    // TODO maybe change them to error messages.
    if (!PrevUnwindLoc) {
      assert(
          !NextUnwindLoc &&
          "A CFI directive should not define a new register out of thin air.");
      continue;
    }
    assert(NextUnwindLoc &&
           "A CFI directive should not delete a register state.");

    checkRegDiff(Inst, Reg, State, AfterState, PrevUnwindLoc.value(),
                 NextUnwindLoc.value(), Reads, Writes);
  }

  State = AfterState;
}

void CFIAnalysis::checkRegDiff(
    const MCInst &Inst, DWARFRegType Reg, const CFIState &PrevState,
    const CFIState &NextState,
    const dwarf::UnwindLocation &PrevRegState, // TODO maybe re-name
    const dwarf::UnwindLocation &NextRegState, // TODO maybe re-name
    const std::set<DWARFRegType> &Reads, const std::set<DWARFRegType> &Writes) {
  auto RegLLVMOpt = MCRI->getLLVMRegNum(Reg, IsEH);
  if (RegLLVMOpt == std::nullopt) {
    assert(PrevRegState == NextRegState);
    return;
  }
  const char *RegLLVMName = MCRI->getName(RegLLVMOpt.value());

  auto &&PrevRefReg =
      PrevState.getReferenceRegisterForCallerValueOfRegister(Reg);

  std::optional<MCPhysReg> PrevRefRegLLVM =
      (PrevRefReg != std::nullopt
           ? std::make_optional(
                 MCRI->getLLVMRegNum(PrevRefReg.value(), IsEH).value())
           : std::nullopt);
  std::optional<MCPhysReg> NextRefRegLLVM =
      (PrevRefReg != std::nullopt
           ? std::make_optional(
                 MCRI->getLLVMRegNum(PrevRefReg.value(), IsEH).value())
           : std::nullopt);

  // TODO Again getting CFA register out of UnwindRow
  // TODO consider either making it a help function.
  // TODO Also, again you have to re-look in the assumption
  // TODO that CFA is depending on another register, maybe
  // TODO it's not true. And if it's always true, consider
  // TODO asserting it (maybe in the helper function).
  MCPhysReg PrevStateCFARegLLVM =
      MCRI->getLLVMRegNum(PrevState.getLastRow()->getCFAValue().getRegister(),
                          IsEH)
          .value();

  { // try generate
    // Widen
    std::vector<dwarf::UnwindLocation> PossibleNextRegStates;

    { // stay the same
      bool CanStayTheSame = true;

      // TODO make sure it has meaning in `Undefined` or `Unspecified` case.
      // TODO make it smarter, if it's in the memory and the instruction does
      // TODO not store anything and the register (if any) to access that part
      // TODO is not changed, then it's OK.
      if (PrevRegState.getDereference())
        CanStayTheSame = false;

      if (PrevRefReg && Writes.count(PrevRefReg.value())) {
        CanStayTheSame = false;
      }

      if (PrevRegState.getLocation() == dwarf::UnwindLocation::DWARFExpr) {
        // TODO this can be solved more precise, by looking into the expression,
        // TODO finding registers that are used and checking if the instruction
        // TODO is changing them or not.
        CanStayTheSame = false;
      }

      if (CanStayTheSame)
        PossibleNextRegStates.push_back(PrevRegState);
    }

    // TODO In this stage of program, there is not possible next stages, it's
    // TODO either the same or nothing. Then maybe I have to refactor it back to
    // TODO the primitive design. And when got the semantic info, get back to
    // TODO this code.
    for (auto &&PossibleNextRegState : PossibleNextRegStates) {
      if (PossibleNextRegState == NextRegState) {
        // Everything is ok
        return;
      }
    }

    for (auto &&PossibleNextRegState : PossibleNextRegStates) {
      if (PossibleNextRegState.getLocation() != NextRegState.getLocation())
        continue;

      // TODO again, as said above, does this happen in the current primitive
      // TODO design?
      if (PossibleNextRegState.getLocation() ==
          dwarf::UnwindLocation::CFAPlusOffset) {
        Context->reportError(
            Inst.getLoc(),
            formatv("Expected %{0} unwinding rule should be [CFA + {1}] but "
                    "based CFI directives are [CFA + {2}]",
                    RegLLVMName, PossibleNextRegState.getOffset(),
                    NextRegState.getOffset()));
      }
    }
  }
  // Either couldn't generate, or the programmer changed the state to
  // something that couldn't be matched to any of the generated states. So
  // it falls back into read/write checks.

  if (PrevRegState == NextRegState) {
    switch (PrevRegState.getLocation()) {
    case dwarf::UnwindLocation::Same:
    case dwarf::UnwindLocation::RegPlusOffset:
      if (Writes.count(PrevRefReg.value())) {
        Context->reportError(
            Inst.getLoc(),
            formatv("This instruction changes %{1}, that %{0} unwinding rule "
                    "uses, but there is no CFI directives about it",
                    RegLLVMName, MCRI->getName(PrevRefRegLLVM.value())));
        return;
      }
      break;
      // TODO think about what to do with expressions here.
    default:
      // Everything may be ok
      break;
    }
    return;
  }

  if (PrevRegState.getLocation() == NextRegState.getLocation()) {
    // Everything may be ok
    return;
  }

  if (PrevRegState.getLocation() == dwarf::UnwindLocation::Undefined) {
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

void CFIAnalysis::checkCFADiff(const MCInst &Inst, const CFIState &PrevState,
                               const CFIState &NextState,
                               const std::set<DWARFRegType> &Reads,
                               const std::set<DWARFRegType> &Writes) {
  // TODO again getting the CFA register in the bad way.
  const DWARFRegType PrevCFAReg =
      PrevState.getLastRow()->getCFAValue().getRegister();
  const int32_t PrevCFAOffset =
      PrevState.getLastRow()->getCFAValue().getOffset();

  const DWARFRegType NextCFAReg =
      NextState.getLastRow()->getCFAValue().getRegister();
  const int32_t NextCFAOffset =
      NextState.getLastRow()->getCFAValue().getOffset();

  const char *PrevCFARegName =
      MCRI->getName(MCRI->getLLVMRegNum(PrevCFAReg, IsEH).value());

  if (PrevCFAReg == NextCFAReg) {
    if (PrevCFAOffset == NextCFAOffset) {
      if (Writes.count(PrevCFAReg)) {
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

    if (!Writes.count(PrevCFAReg)) {
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
