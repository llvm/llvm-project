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

    // Applying the prologue after default assumptions to overwrite them.
    for (auto &&PrologueCFIDirective : PrologueCFIDirectives) {
      State.apply(PrologueCFIDirective);
    }
  }

  bool doesConstantChange(const MCInst &Inst, MCPhysReg Reg, int64_t &HowMuch) {
    if (EMCIA->isPush(Inst) && Reg == EMCIA->getStackPointer()) {
      // TODO should get the stack direction here, now it assumes that it goes
      // down.
      HowMuch = -EMCIA->getPushSize(Inst);
      return true;
    }

    if (EMCIA->isPop(Inst) && Reg == EMCIA->getStackPointer()) {
      // TODO should get the stack direction here, now it assumes that it goes
      // down.
      HowMuch = EMCIA->getPushSize(Inst);
      return true;
    }

    return false;
  }

  // Tries to guess Reg1's value in a form of Reg2 (before Inst's execution) +
  // Diff.
  bool isInConstantDistanceOfEachOther(const MCInst &Inst, MCPhysReg &Reg1,
                                       MCPhysReg Reg2, int &Diff) {
    {
      MCPhysReg From;
      MCPhysReg To;
      if (EMCIA->isRegToRegMove(Inst, From, To) && From == Reg2) {
        Reg1 = To;
        Diff = 0;
        return true;
      }
    }

    return false;
  }

  bool doStoreFromReg(const MCInst &Inst, MCPhysReg StoringReg,
                      MCPhysReg FromReg, int64_t &Offset) {
    if (EMCIA->isPush(Inst) && FromReg == EMCIA->getStackPointer()) {
      // TODO should get the stack direction here, now it assumes that it goes
      // down.
      Offset = -EMCIA->getPushSize(Inst);
      return true;
    }

    {
      bool IsLoad;
      bool IsStore;
      bool IsStoreFromReg;
      MCPhysReg SrcReg;
      int32_t SrcImm;
      uint16_t StackPtrReg;
      int64_t StackOffset;
      uint8_t Size;
      bool IsSimple;
      bool IsIndexed;
      if (EMCIA->isStackAccess(Inst, IsLoad, IsStore, IsStoreFromReg, SrcReg,
                               SrcImm, StackPtrReg, StackOffset, Size, IsSimple,
                               IsIndexed)) {
        // TODO make sure that simple means that it's store and does nothing
        // more.
        if (IsStore && IsSimple && StackPtrReg == FromReg && IsStoreFromReg &&
            SrcReg == StoringReg) {
          Offset = StackOffset;
          return true;
        }
      }
    }

    return false;
  }

  bool doLoadFromReg(const MCInst &Inst, MCPhysReg FromReg, int64_t &Offset,
                     MCPhysReg &LoadingReg) {
    if (EMCIA->isPop(Inst) && FromReg == EMCIA->getStackPointer()) {
      // TODO should get the stack direction here, now it assumes that it goes
      // down.
      Offset = 0;
      LoadingReg = Inst.getOperand(0).getReg();
      return true;
    }

    {
      bool IsLoad;
      bool IsStore;
      bool IsStoreFromReg;
      MCPhysReg SrcReg;
      int32_t SrcImm;
      uint16_t StackPtrReg;
      int64_t StackOffset;
      uint8_t Size;
      bool IsSimple;
      bool IsIndexed;
      if (EMCIA->isStackAccess(Inst, IsLoad, IsStore, IsStoreFromReg, SrcReg,
                               SrcImm, StackPtrReg, StackOffset, Size, IsSimple,
                               IsIndexed)) {
        // TODO make sure that simple means that it's store and does nothing
        // more.
        if (IsLoad && IsSimple && StackPtrReg == FromReg) {
          Offset = StackOffset;
          LoadingReg = SrcReg;
          return true;
        }
      }
    }

    {
      if (EMCIA->isMoveMem2Reg(Inst)) {
        auto X86MemAccess = EMCIA->evaluateX86MemoryOperand(Inst).value();
        if (X86MemAccess.BaseRegNum == FromReg &&
            (X86MemAccess.ScaleImm == 0 || X86MemAccess.IndexRegNum == 0) &&
            !X86MemAccess.DispExpr) {
          LoadingReg = Inst.getOperand(0).getReg();
          Offset = X86MemAccess.DispImm;
          return true;
        }
      }
    }

    return false;
  }

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
          if (doStoreFromReg(Inst, PrevRefRegLLVM.value(), PrevStateCFARegLLVM,
                             OffsetFromCFAReg)) {
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
          if (doLoadFromReg(Inst, PrevStateCFARegLLVM, OffsetFromCFAReg,
                            ToRegLLVM) &&
              OffsetFromCFAReg == PrevRegState.Info.OffsetFromCFA) {
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
          if (isInConstantDistanceOfEachOther(Inst, PossibleRegLLVM,
                                              PrevRefRegLLVM.value(), Diff)) {
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
        if (doesConstantChange(Inst, PrevCFAPhysReg, HowMuch)) {
          PossibleNextCFAStates.emplace_back(PrevState.CFARegister,
                                             PrevState.CFAOffset - HowMuch);
        }
      }

      { // constant distance with each other
        int Diff;
        MCPhysReg PossibleNewCFAReg;
        if (isInConstantDistanceOfEachOther(Inst, PossibleNewCFAReg,
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

    ///////////// being diffing the CFI states
    checkCFADiff(Inst, State, AfterState, Reads, Writes);

    for (auto &&[Reg, RegState] : State.RegisterCFIStates) {
      assert(AfterState.RegisterCFIStates.count(Reg) &&
             "Registers' state should not be deleted by CFI instruction.");
      checkRegDiff(Inst, Reg, State, AfterState, RegState,
                   AfterState.RegisterCFIStates[Reg], Reads, Writes);
    }
    ///////////// end diffing the CFI states

    // dbgs() << "^^^^^^^^^^^^^^^^ InsCFIs ^^^^^^^^^^^^^^^^\n";
    // printUntilNextLine(Inst.getLoc().getPointer());
    // dbgs() << "\n";
    // dbgs() << "Op code: " << Inst.getOpcode() << "\n";
    // dbgs() << "--------------Operands Info--------------\n";
    // auto DefCount = MCInstInfo.getNumDefs();
    // for (unsigned I = 0; I < Inst.getNumOperands(); I++) {
    //   dbgs() << "Operand #" << I << ", which will be "
    //          << (I < DefCount ? "defined" : "used") << ", is a";
    //   if (I == EMCIA->getMemoryOperandNo(Inst)) {
    //     dbgs() << " memory access, details are:\n";
    //     auto X86MemoryOperand = EMCIA->evaluateX86MemoryOperand(Inst);
    //     dbgs() << "  Base Register{ reg#" << X86MemoryOperand->BaseRegNum
    //            << " }";
    //     dbgs() << " + (Index Register{ reg#" <<
    //     X86MemoryOperand->IndexRegNum
    //            << " }";
    //     dbgs() << " * Scale{ value " << X86MemoryOperand->ScaleImm << " }";
    //     dbgs() << ") + Displacement{ "
    //            << (X86MemoryOperand->DispExpr
    //                    ? "an expression"
    //                    : "value " + itostr(X86MemoryOperand->DispImm))
    //            << " }\n";
    //     // TODO, it's not correct always, it cannot be assumed (or should
    //     be
    //     // checked) that memory operands are flatten into 4 operands in
    //     MCInc. I += 4; continue;
    //   }
    //   auto &&Operand = Inst.getOperand(I);
    //   if (Operand.isImm()) {
    //     dbgs() << "n immediate with value " << Operand.getImm() << "\n";
    //     continue;
    //   }
    //   if (Operand.isReg()) {
    //     dbgs() << " reg#" << Operand.getReg() << "\n";
    //     continue;
    //   }
    //   assert(Operand.isExpr());
    //   dbgs() << "n unknown expression \n";
    // }
    // if (MCInstInfo.NumImplicitDefs) {
    //   dbgs() << "implicitly defines: { ";
    //   for (unsigned I = 0; I < MCInstInfo.NumImplicitDefs; I++) {
    //     dbgs() << "reg#" << MCInstInfo.implicit_defs()[I] << " ";
    //   }
    //   dbgs() << "}\n";
    // }
    // if (MCInstInfo.NumImplicitUses) {
    //   dbgs() << "implicitly uses: { ";
    //   for (unsigned I = 0; I < MCInstInfo.NumImplicitUses; I++) {
    //     dbgs() << "reg#" << MCInstInfo.implicit_uses()[I] << " ";
    //   }
    //   dbgs() << "}\n";
    // }
    // dbgs() << "----------------Move Info----------------\n";
    // {   // move
    //   { // Reg2Reg
    //     MCPhysReg From, To;
    //     if (EMCIA->isRegToRegMove(Inst, From, To)) {
    //       dbgs() << "It's a reg to reg move.\nFrom reg#" << From << " to
    //       reg
    //       #"
    //              << To << "\n";
    //     } else if (MCInstInfo.isMoveReg()) {
    //       dbgs() << "It's reg to reg move from MCInstInfo view but not from
    //       "
    //                 "MCPlus view.\n";
    //     }
    //   }
    //   if (EMCIA->isConditionalMove(Inst)) {
    //     dbgs() << "Its a conditional move.\n";
    //   }
    //   if (EMCIA->isMoveMem2Reg(Inst)) {
    //     dbgs() << "It's a move from memory to register\n";
    //     assert(EMCIA->getMemoryOperandNo(Inst) == 1);
    //   }
    // }

    // dbgs() << "---------------Stack Info----------------\n";
    // { // stack
    //   int32_t SrcImm = 0;
    //   MCPhysReg Reg = 0;
    //   uint16_t StackPtrReg = 0;
    //   int64_t StackOffset = 0;
    //   uint8_t Size = 0;
    //   bool IsSimple = false;
    //   bool IsIndexed = false;
    //   bool IsLoad = false;
    //   bool IsStore = false;
    //   bool IsStoreFromReg = false;
    //   if (EMCIA->isStackAccess(Inst, IsLoad, IsStore, IsStoreFromReg, Reg,
    //                            SrcImm, StackPtrReg, StackOffset, Size,
    //                            IsSimple, IsIndexed)) {
    //     dbgs() << "This instruction accesses the stack, the details
    //     are:\n"; dbgs() << "  Source immediate: " << SrcImm << "\n"; dbgs()
    //     << "  Source reg#" << Reg << "\n"; dbgs() << "  Stack pointer:
    //     reg#" << StackPtrReg << "\n"; dbgs() << "  Stack offset: " <<
    //     StackOffset << "\n"; dbgs() << "  size: " << (int)Size << "\n";
    //     dbgs() << "  Is simple: " << (IsSimple ? "yes" : "no") << "\n";
    //     dbgs() << "  Is indexed: " << (IsIndexed ? "yes" : "no") << "\n";
    //     dbgs() << "  Is load: " << (IsLoad ? "yes" : "no") << "\n";
    //     dbgs() << "  Is store: " << (IsStore ? "yes" : "no") << "\n";
    //     dbgs() << "  Is store from reg: " << (IsStoreFromReg ? "yes" :
    //     "no")
    //            << "\n";
    //   }
    //   if (EMCIA->isPush(Inst)) {
    //     dbgs() << "This is a push instruction with size "
    //            << EMCIA->getPushSize(Inst) << "\n";
    //   }
    //   if (EMCIA->isPop(Inst)) {
    //     dbgs() << "This is a pop instruction with size "
    //            << EMCIA->getPopSize(Inst) << "\n";
    //   }
    // }

    // dbgs() << "---------------Arith Info----------------\n";
    // { // arith
    //   /* MutableArrayRef<MCInst> MAR = {Inst};
    //   if (MCPB->matchAdd(MCPB->matchAnyOperand(), MCPB->matchAnyOperand())
    //           ->match(*MCRI, *MCPB, MutableArrayRef<MCInst>(), -1)) {
    //     dbgs() << "It is an addition instruction.\n";
    //   } else */
    //   if (MCInstInfo.isAdd()) {
    //     dbgs() << "It is an addition from MCInstInfo view but not from
    //     MCPlus"
    //               "view.\n";
    //   }
    //   if (EMCIA->isSUB(Inst)) {
    //     dbgs() << "This is a subtraction.\n";
    //   }
    // }

    // dbgs() << "-----------------------------------------\n";
    // dbgs() << "The CFA register is: " << CFAReg << "\n";
    // dbgs() << "The instruction does " << (ChangedCFA ? "" : "NOT ")
    //        << "change the CFA.\n";
    // dbgs() << "The CFI directives does " << (GoingToChangeCFA ? "" : "NOT
    // ")
    //        << "change the CFA.\n";
    // dbgs() << "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n";

    State = AfterState;
  }

private:
};

} // namespace llvm
#endif