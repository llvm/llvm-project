#ifndef LLVM_TOOLS_LLVM_MC_CFI_ANALYSIS_H
#define LLVM_TOOLS_LLVM_MC_CFI_ANALYSIS_H

#include "CFIState.h"
#include "ExtendedMCInstrAnalysis.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
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
#include <cstdint>
#include <memory>
#include <optional>
#include <set>

namespace llvm {

// TODO remove it, it's just for debug purposes.
void printUntilNextLine(const char *Str) {
  for (int I = 0; Str[I] != '\0' && Str[I] != '\n'; I++)
    dbgs() << Str[I];
}

class CFIAnalysis {
  MCContext &Context;
  MCInstrInfo const &MCII;
  MCRegisterInfo const *MCRI;
  std::unique_ptr<ExtendedMCInstrAnalysis> EMCIA;
  CFIState State;

private:
  // The CFI analysis only keeps track and cares about super registers, not the
  // subregisters. All reads to/writes from subregisters and considered the same
  // operation to super registers. Other operations like loading and stores are
  // considered only if they are exactly doing the operation to or from a super
  // register.
  // As en example, if you spill a sub register to stack, the CFI analysis does
  // not consider that a register spilling.
  bool isSuperReg(MCPhysReg Reg) { return MCRI->superregs(Reg).empty(); }

  std::set<MCPhysReg> getAllSuperRegs() {
    std::set<MCPhysReg> SuperRegs;
    for (auto &&RegClass : MCRI->regclasses()) {
      for (unsigned I = 0; I < RegClass.getNumRegs(); I++) {
        MCPhysReg Reg = RegClass.getRegister(I);
        if (!isSuperReg(Reg))
          continue;
        SuperRegs.insert(Reg);
      }
    }

    return SuperRegs;
  }

  MCPhysReg getSuperReg(MCPhysReg Reg) {
    if (isSuperReg(Reg))
      return Reg;
    for (auto SuperReg : MCRI->superregs(Reg)) {
      if (isSuperReg(SuperReg))
        return SuperReg;
    }

    llvm_unreachable("Should either be a super reg, or have a super reg");
  }

public:
  CFIAnalysis(MCContext &Context, MCInstrInfo const &MCII,
              MCInstrAnalysis *MCIA,
              ArrayRef<MCCFIInstruction> PrologueCFIDirectives)
      : Context(Context), MCII(MCII), MCRI(Context.getRegisterInfo()) {
    EMCIA.reset(new ExtendedMCInstrAnalysis(Context, MCII, MCIA));

    // TODO CFA offset should be the slot size, but for now I don't have any
    // access to it, maybe can be read from the prologue
    // TODO check what should be passed as EH?
    State = CFIState(MCRI->getDwarfRegNum(EMCIA->getStackPointer(), false), 8);
    for (MCPhysReg I : getAllSuperRegs()) {
      DWARFRegType DwarfI = MCRI->getDwarfRegNum(I, false);
      State.RegisterCFIStates[DwarfI] = RegisterCFIState::createSameValue();
    }

    // TODO these are temporay added to make things work.
    // Setup the basic information:
    State.RegisterCFIStates[MCRI->getDwarfRegNum(MCRI->getProgramCounter(),
                                                 false)] =
        RegisterCFIState::createUndefined(); // For now, we don't care about the
                                             // PC
    State.RegisterCFIStates[MCRI->getDwarfRegNum(EMCIA->getStackPointer(),
                                                 false)] =
        RegisterCFIState::createOffsetFromCFAVal(0); // sp's old value is CFA

    State.RegisterCFIStates[MCRI->getDwarfRegNum(EMCIA->getFlagsReg(), false)] =
        RegisterCFIState::createUndefined(); // Flags cannot be caller-saved

    // Applying the prologue after default assumptions to overwrite them.
    for (auto &&PrologueCFIDirective : PrologueCFIDirectives) {
      State.apply(PrologueCFIDirective);
    }
  }

  void update(MCInst &Inst, ArrayRef<MCCFIInstruction> CFIDirectives) {
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

private:
  void checkRegDiff(const MCInst &Inst, DWARFRegType Reg,
                    const CFIState &PrevState, const CFIState &NextState,
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
    auto &&NextRefReg =
        NextState.getReferenceRegisterForCallerValueOfRegister(Reg);

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
      { // storing to offset from CFA
        if (PrevRegState.RetrieveApproach == RegisterCFIState::SameValue ||
            PrevRegState.RetrieveApproach ==
                RegisterCFIState::AnotherRegister) {
          int64_t OffsetFromCFAReg;
          if (EMCIA->doStoreFromReg(Inst, PrevRefRegLLVM.value(),
                                    PrevStateCFARegLLVM, OffsetFromCFAReg)) {
            PossibleNextRegStates.push_back(
                RegisterCFIState::createOffsetFromCFAAddr(OffsetFromCFAReg -
                                                          PrevState.CFAOffset));
          }
        }
      }

      { // loading from an offset from CFA
        if (PrevRegState.RetrieveApproach ==
            RegisterCFIState::OffsetFromCFAAddr) {
          int64_t OffsetFromCFAReg;
          MCPhysReg ToRegLLVM;
          if (EMCIA->doLoadFromReg(Inst, PrevStateCFARegLLVM, OffsetFromCFAReg,
                                   ToRegLLVM) &&
              OffsetFromCFAReg - PrevState.CFAOffset ==
                  PrevRegState.Info.OffsetFromCFA) {
            DWARFRegType ToReg = MCRI->getDwarfRegNum(ToRegLLVM, false);
            if (ToReg == Reg) {
              PossibleNextRegStates.push_back(
                  RegisterCFIState::createSameValue());
            } else {
              PossibleNextRegStates.push_back(
                  RegisterCFIState::createAnotherRegister(ToReg));
            }
          }
        }
      }

      { // moved from reg to other reg
        if (PrevRegState.RetrieveApproach == RegisterCFIState::SameValue ||
            PrevRegState.RetrieveApproach ==
                RegisterCFIState::AnotherRegister) {

          int Diff;
          MCPhysReg PossibleRegLLVM;
          if (EMCIA->isInConstantDistanceOfEachOther(
                  Inst, PossibleRegLLVM, PrevRefRegLLVM.value(), Diff)) {
            DWARFRegType PossibleReg =
                MCRI->getDwarfRegNum(PossibleRegLLVM, false);
            if (Diff == 0) {
              if (PossibleReg == Reg) {
                PossibleNextRegStates.push_back(
                    RegisterCFIState::createSameValue());
              } else {
                PossibleNextRegStates.push_back(
                    RegisterCFIState::createAnotherRegister(PossibleReg));
              }
            }
          }
        }
      }

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

  void checkCFADiff(const MCInst &Inst, const CFIState &PrevState,
                    const CFIState &NextState,
                    const std::set<DWARFRegType> &Reads,
                    const std::set<DWARFRegType> &Writes) {
    MCPhysReg PrevCFAPhysReg =
        MCRI->getLLVMRegNum(PrevState.CFARegister, false).value();
    MCPhysReg NextCFAPhysReg =
        MCRI->getLLVMRegNum(NextState.CFARegister, false).value();

    { // try generate
      // Widen
      std::vector<std::pair<DWARFRegType, int>> PossibleNextCFAStates;
      { // no change
        if (!Writes.count(PrevState.CFARegister)) {
          PossibleNextCFAStates.emplace_back(PrevState.CFARegister,
                                             PrevState.CFAOffset);
        }
      }

      { // const change
        int64_t HowMuch;
        if (EMCIA->doesConstantChange(Inst, PrevCFAPhysReg, HowMuch)) {
          PossibleNextCFAStates.emplace_back(PrevState.CFARegister,
                                             PrevState.CFAOffset - HowMuch);
        }
      }

      { // constant distance with each other
        int Diff;
        MCPhysReg PossibleNewCFAReg;
        if (EMCIA->isInConstantDistanceOfEachOther(Inst, PossibleNewCFAReg,
                                                   PrevCFAPhysReg, Diff)) {
          PossibleNextCFAStates.emplace_back(
              MCRI->getDwarfRegNum(PossibleNewCFAReg, false),
              PrevState.CFAOffset - Diff);
        }
      }

      for (auto &&[PossibleNextCFAReg, PossibleNextCFAOffset] :
           PossibleNextCFAStates) {
        if (PossibleNextCFAReg != NextState.CFARegister)
          continue;

        if (PossibleNextCFAOffset == NextState.CFAOffset) {
          // Everything is ok!
          return;
        }

        Context.reportError(Inst.getLoc(),
                            formatv("Expected CFA [reg: {0}, offset: {1}] but "
                                    "got [reg: {2}, offset: {3}].",
                                    NextCFAPhysReg, PossibleNextCFAOffset,
                                    NextCFAPhysReg, NextState.CFAOffset));
        return;
      }
    }

    // Either couldn't generate, or did, but the programmer wants to change
    // the source of register for CFA to something not expected by the
    // generator. So it falls back into read/write checks.

    if (PrevState.CFARegister == NextState.CFARegister) {
      if (PrevState.CFAOffset == NextState.CFAOffset) {
        if (Writes.count(PrevState.CFARegister)) {
          Context.reportError(
              Inst.getLoc(),
              formatv("This instruction changes reg#{0}, which is "
                      "the CFA register, but the CFI directives do not.",
                      PrevCFAPhysReg));
          return;
        }

        // Everything is ok!
        return;
      }
      // The offset is changed.

      if (!Writes.count(PrevState.CFARegister)) {
        Context.reportError(
            Inst.getLoc(),
            formatv(
                "You changed the CFA offset, but there is no modification to "
                "the CFA register happening in this instruction."));
      }

      Context.reportWarning(
          Inst.getLoc(),
          "I don't know what the instruction did, but it changed the CFA "
          "reg's "
          "value, and the offset is changed as well by the CFI directives.");
      // Everything may be ok!
      return;
    }
    // The CFA register is changed
    Context.reportWarning(
        Inst.getLoc(), "The CFA register is changed to something, and I don't "
                       "have any idea on the new register relevance to CFA. I "
                       "assume CFA is preserved.");
    // Everything may be ok!
  }
};

} // namespace llvm
#endif