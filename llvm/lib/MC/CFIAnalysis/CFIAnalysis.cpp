#include "llvm/MC/MCCFIAnalysis/CFIAnalysis.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCFIAnalysis/ExtendedMCInstrAnalysis.h"
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
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
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

CFIAnalysis::CFIAnalysis(MCContext &Context, MCInstrInfo const &MCII,
                         MCInstrAnalysis *MCIA,
                         ArrayRef<MCCFIInstruction> PrologueCFIDirectives)
    : Context(Context), MCII(MCII), MCRI(Context.getRegisterInfo()) {
  EMCIA.reset(new ExtendedMCInstrAnalysis(Context, MCII, MCIA));

  // TODO check what should be passed as EH? I am putting false everywhere.
  for (auto &&[Reg, RegClass] : getAllSuperRegs()) {
    if (MCRI->get(Reg).IsArtificial || MCRI->get(Reg).IsConstant)
      continue;

    DWARFRegType DwarfReg = MCRI->getDwarfRegNum(Reg, false);
    State.RegisterCFIStates[DwarfReg] = RegisterCFIState::createSameValue();
  }

  for (auto &&InitialFrameStateCFIDirective :
       Context.getAsmInfo()->getInitialFrameState()) {
    State.apply(InitialFrameStateCFIDirective);
  }

  // TODO these are temporay added to make things work.
  // Setup the basic information:
  State.RegisterCFIStates[MCRI->getDwarfRegNum(MCRI->getProgramCounter(),
                                               false)] =
      RegisterCFIState::createUndefined(); // TODO for now, we don't care
                                           // about the PC
  State.RegisterCFIStates[MCRI->getDwarfRegNum(State.CFARegister,
                                               false)] =
      RegisterCFIState::createOffsetFromCFAVal(0); // sp's old value is CFA

  // State.RegisterCFIStates[MCRI->getDwarfRegNum(EMCIA->getFlagsReg(), false)] =
  //     RegisterCFIState::createUndefined(); // Flags cannot be caller-saved

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
    if (!AfterState.apply(CFIDirective))
      Context.reportWarning(
          CFIDirective.getLoc(),
          "I don't support this CFI directive, I assume this does nothing "
          "(which will probably break other things)");

  std::set<DWARFRegType> Writes, Reads;
  for (unsigned I = 0; I < MCInstInfo.NumImplicitUses; I++)
    Reads.insert(MCRI->getDwarfRegNum(
        getSuperReg(MCInstInfo.implicit_uses()[I]), false));
  for (unsigned I = 0; I < MCInstInfo.NumImplicitDefs; I++)
    Writes.insert(MCRI->getDwarfRegNum(
        getSuperReg(MCInstInfo.implicit_defs()[I]), false));

  for (unsigned I = 0; I < Inst.getNumOperands(); I++) {
    auto &&Operand = Inst.getOperand(I);
    if (Operand.isReg()) {
      if (I < MCInstInfo.getNumDefs())
        Writes.insert(
            MCRI->getDwarfRegNum(getSuperReg(Operand.getReg()), false));
      else if (Operand.getReg())
        Reads.insert(
            MCRI->getDwarfRegNum(getSuperReg(Operand.getReg()), false));
    }
  }

  checkCFADiff(Inst, State, AfterState, Reads, Writes);

  for (auto &&[Reg, RegState] : State.RegisterCFIStates) {
    assert(AfterState.RegisterCFIStates.count(Reg) &&
           "Registers' state should not be deleted by CFI instruction.");
    checkRegDiff(Inst, Reg, State, AfterState, RegState,
                 AfterState.RegisterCFIStates[Reg], Reads, Writes);
  }

  State = AfterState;
}

void CFIAnalysis::checkRegDiff(const MCInst &Inst, DWARFRegType Reg,
                               const CFIState &PrevState,
                               const CFIState &NextState,
                               const RegisterCFIState &PrevRegState,
                               const RegisterCFIState &NextRegState,
                               const std::set<DWARFRegType> &Reads,
                               const std::set<DWARFRegType> &Writes) {
  auto RegLLVMOpt = MCRI->getLLVMRegNum(Reg, false);
  if (RegLLVMOpt == std::nullopt) {
    assert(PrevRegState == NextRegState);
    return;
  }
  MCPhysReg RegLLVM = RegLLVMOpt.value();

  auto &&PrevRefReg =
      PrevState.getReferenceRegisterForCallerValueOfRegister(Reg);

  std::optional<MCPhysReg> PrevRefRegLLVM =
      (PrevRefReg != std::nullopt
           ? std::make_optional(
                 MCRI->getLLVMRegNum(PrevRefReg.value(), false).value())
           : std::nullopt);
  std::optional<MCPhysReg> NextRefRegLLVM =
      (PrevRefReg != std::nullopt
           ? std::make_optional(
                 MCRI->getLLVMRegNum(PrevRefReg.value(), false).value())
           : std::nullopt);

  MCPhysReg PrevStateCFARegLLVM =
      MCRI->getLLVMRegNum(PrevState.CFARegister, false).value();

  { // try generate
    // Widen
    std::vector<RegisterCFIState> PossibleNextRegStates;

    { // stay the same
      bool CanStayTheSame = false;

      switch (PrevRegState.RetrieveApproach) {
      case RegisterCFIState::Undefined:
      case RegisterCFIState::OffsetFromCFAVal:
        CanStayTheSame = true;
        break;
      case RegisterCFIState::SameValue:
      case RegisterCFIState::AnotherRegister:
        CanStayTheSame = !Writes.count(PrevRefReg.value());
        break;
      case RegisterCFIState::OffsetFromCFAAddr:
      case RegisterCFIState::Other:
        // cannot be sure
        break;
      }

      if (CanStayTheSame)
        PossibleNextRegStates.push_back(PrevRegState);
    }

    for (auto &&PossibleNextRegState : PossibleNextRegStates) {
      if (PossibleNextRegState == NextRegState) {
        // Everything is ok
        return;
      }
    }

    for (auto &&PossibleNextRegState : PossibleNextRegStates) {
      if (PossibleNextRegState.RetrieveApproach !=
          NextRegState.RetrieveApproach)
        continue;

      if (PossibleNextRegState.RetrieveApproach ==
          RegisterCFIState::OffsetFromCFAAddr) {
        Context.reportError(
            Inst.getLoc(),
            formatv(
                "Expected caller's value of reg#{0} should be at offset {1} "
                "of CFA but the CFI directives say it's in {2}",
                RegLLVM, PossibleNextRegState.Info.OffsetFromCFA,
                NextRegState.Info.OffsetFromCFA));
      }
    }
  }
  // Either couldn't generate, or the programmer changed the state to
  // something that couldn't be matched to any of the generated states. So
  // it falls back into read/write checks.

  if (PrevRegState == NextRegState) {
    switch (PrevRegState.RetrieveApproach) {
    case RegisterCFIState::SameValue:
    case RegisterCFIState::AnotherRegister:
      if (Writes.count(PrevRefReg.value())) {
        Context.reportError(
            Inst.getLoc(),
            formatv("Reg#{0} caller's value is in reg#{1} which is changed "
                    "by this instruction, but not changed in CFI directives",
                    RegLLVM, PrevRefRegLLVM.value()));
        return;
      }
      break;
    default:
      // Everything may be ok
      break;
    }
    return;
  }

  if (PrevRegState.RetrieveApproach == NextRegState.RetrieveApproach) {
    // Everything may be ok
    return;
  }

  if (PrevRegState.RetrieveApproach == RegisterCFIState::Undefined) {
    Context.reportError(Inst.getLoc(),
                        "Cannot change a register CFI information from "
                        "undefined to something else.");
    return;
  }

  Context.reportWarning(Inst.getLoc(),
                        formatv("The reg#{0} CFI state is changed, but I "
                                "don't have any idea how.",
                                RegLLVM));
  // Everything may be ok
  return;
}

void CFIAnalysis::checkCFADiff(const MCInst &Inst, const CFIState &PrevState,
                               const CFIState &NextState,
                               const std::set<DWARFRegType> &Reads,
                               const std::set<DWARFRegType> &Writes) {
  const char *PrevCFARegName =
      MCRI->getName(MCRI->getLLVMRegNum(PrevState.CFARegister, false).value());

  if (PrevState.CFARegister == NextState.CFARegister) {
    if (PrevState.CFAOffset == NextState.CFAOffset) {
      if (Writes.count(PrevState.CFARegister)) {
        Context.reportError(
            Inst.getLoc(),
            formatv("Missing CFI directive for the CFA offset adjustment. CFA "
                    "register ({0}) is modified by this instruction.",
                    PrevCFARegName));
        return;
      }

      // Everything is ok!
      return;
    }
    // The offset is changed.

    if (!Writes.count(PrevState.CFARegister)) {
      Context.reportError(
          Inst.getLoc(),
          formatv("Wrong adjustment to CFA offset. CFA register ({0}) is not "
                  "modified by this instruction.",
                  PrevCFARegName));
    }

    // The CFA register value is changed, and the offset is changed as well,
    // everything may be ok.
    return;
  }

  // The CFA register is changed, everything may be ok.
}
