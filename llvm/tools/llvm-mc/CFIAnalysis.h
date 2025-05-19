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

public:
  CFIAnalysis(MCContext &Context, MCInstrInfo const &MCII,
              MCInstrAnalysis *MCIA)
      : Context(Context), MCII(MCII), MCRI(Context.getRegisterInfo()) {
    // TODO it should look at the prologue directives and setup the
    // registers' previous value state here, but for now, it's assumed that all
    // values are by default `samevalue`.
    EMCIA.reset(new ExtendedMCInstrAnalysis(Context, MCII, MCIA));

    // TODO CFA offset should be the slot size, but for now I don't have any
    // access to it, maybe can be read from the prologue
    // TODO check what should be passed as EH?
    State = CFIState(MCRI->getDwarfRegNum(EMCIA->getStackPointer(), false), 8);
    for (unsigned I = 0; I < MCRI->getNumRegs(); I++) {
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

  void checkCFADiff(const MCInst &Inst, const CFIState &PrevState,
                    const CFIState &NextState,
                    const std::set<DWARFRegType> &Reads,
                    const std::set<DWARFRegType> &Writes) {
    MCPhysReg PrevCFAPhysReg =
        MCRI->getLLVMRegNum(PrevState.CFARegister, false).value();
    MCPhysReg NextCFAPhysReg =
        MCRI->getLLVMRegNum(NextState.CFARegister, false).value();

    // Try to guess the next state. (widen)
    std::vector<std::pair<DWARFRegType, int>> PossibleNextCFAStates;
    {   // try generate
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

    // Either couldn't generate, or did, but the programmer wants to change the
    // source of register for CFA to something not expected by the generator. So
    // it falls back into read/write checks.

    if (PrevState.CFARegister == NextState.CFARegister) {
      if (PrevState.CFAOffset == NextState.CFAOffset) {
        if (Writes.count(PrevState.CFARegister)) {
          Context.reportError(
              Inst.getLoc(),
              formatv("This instruction changes reg#{0}, which is "
                      "the CFA register, but the CFI directives do not.",
                      PrevCFAPhysReg));
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
          "I don't know what the instruction did, but it changed the CFA reg's "
          "value, and the offset is changed as well by the CFI directives.");
      // Everything may be ok!
      return;
    }
    // The CFA register is changed
    Context.reportWarning(
        Inst.getLoc(), "The CFA register is changed to something, and I don't "
                       "have any idea on the new register relevance to CFA.");
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
      Reads.insert(MCRI->getDwarfRegNum(MCInstInfo.implicit_uses()[I], false));
    for (unsigned I = 0; I < MCInstInfo.NumImplicitDefs; I++)
      Writes.insert(MCRI->getDwarfRegNum(MCInstInfo.implicit_defs()[I], false));

    for (unsigned I = 0; I < Inst.getNumOperands(); I++) {
      auto &&Operand = Inst.getOperand(I);
      if (Operand.isReg()) {
        if (I < MCInstInfo.getNumDefs())
          Writes.insert(MCRI->getDwarfRegNum(Operand.getReg(), false));
        else
          Reads.insert(MCRI->getDwarfRegNum(Operand.getReg(), false));
      }
    }
    ///////////// being diffing the CFI states
    checkCFADiff(Inst, State, AfterState, Reads, Writes);
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
    //     dbgs() << " + (Index Register{ reg#" << X86MemoryOperand->IndexRegNum
    //            << " }";
    //     dbgs() << " * Scale{ value " << X86MemoryOperand->ScaleImm << " }";
    //     dbgs() << ") + Displacement{ "
    //            << (X86MemoryOperand->DispExpr
    //                    ? "an expression"
    //                    : "value " + itostr(X86MemoryOperand->DispImm))
    //            << " }\n";
    //     // TODO, it's not correct always, it cannot be assumed (or should be
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
    //       dbgs() << "It's a reg to reg move.\nFrom reg#" << From << " to reg
    //       #"
    //              << To << "\n";
    //     } else if (MCInstInfo.isMoveReg()) {
    //       dbgs() << "It's reg to reg move from MCInstInfo view but not from "
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
    //     dbgs() << "This instruction accesses the stack, the details are:\n";
    //     dbgs() << "  Source immediate: " << SrcImm << "\n";
    //     dbgs() << "  Source reg#" << Reg << "\n";
    //     dbgs() << "  Stack pointer: reg#" << StackPtrReg << "\n";
    //     dbgs() << "  Stack offset: " << StackOffset << "\n";
    //     dbgs() << "  size: " << (int)Size << "\n";
    //     dbgs() << "  Is simple: " << (IsSimple ? "yes" : "no") << "\n";
    //     dbgs() << "  Is indexed: " << (IsIndexed ? "yes" : "no") << "\n";
    //     dbgs() << "  Is load: " << (IsLoad ? "yes" : "no") << "\n";
    //     dbgs() << "  Is store: " << (IsStore ? "yes" : "no") << "\n";
    //     dbgs() << "  Is store from reg: " << (IsStoreFromReg ? "yes" : "no")
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
    // dbgs() << "The CFI directives does " << (GoingToChangeCFA ? "" : "NOT ")
    //        << "change the CFA.\n";
    // dbgs() << "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n";

    State = AfterState;
  }

private:
};

} // namespace llvm
#endif