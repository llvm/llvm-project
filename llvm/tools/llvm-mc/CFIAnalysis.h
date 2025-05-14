#ifndef LLVM_TOOLS_LLVM_MC_CFI_ANALYSIS_H
#define LLVM_TOOLS_LLVM_MC_CFI_ANALYSIS_H

#include "CFIState.h"
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
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <set>

namespace llvm {

bolt::MCPlusBuilder *createMCPlusBuilder(const Triple::ArchType Arch,
                                         const MCInstrAnalysis *Analysis,
                                         const MCInstrInfo *Info,
                                         const MCRegisterInfo *RegInfo,
                                         const MCSubtargetInfo *STI) {
  dbgs() << "arch: " << Arch << ", and expected " << Triple::x86_64 << "\n";
  if (Arch == Triple::x86_64)
    return bolt::createX86MCPlusBuilder(Analysis, Info, RegInfo, STI);

  // if (Arch == Triple::aarch64)
  //   return createAArch64MCPlusBuilder(Analysis, Info, RegInfo, STI);

  // if (Arch == Triple::riscv64)
  //   return createRISCVMCPlusBuilder(Analysis, Info, RegInfo, STI);

  llvm_unreachable("architecture unsupported by MCPlusBuilder");
}

// TODO remove it, it's just for debug purposes.
void printUntilNextLine(const char *Str) {
  for (int I = 0; Str[I] != '\0' && Str[I] != '\n'; I++)
    dbgs() << Str[I];
}

class CFIAnalysis {
  MCContext &Context;
  MCInstrInfo const &MCII;
  std::unique_ptr<bolt::MCPlusBuilder> MCPB;
  MCRegisterInfo const *MCRI;
  CFIState State;

public:
  CFIAnalysis(MCContext &Context, MCInstrInfo const &MCII,
              MCInstrAnalysis *MCIA)
      : Context(Context), MCII(MCII), MCRI(Context.getRegisterInfo()) {
    // TODO it should look at the prologue directives and setup the
    // registers' previous value state here, but for now, it's assumed that all
    // values are by default `samevalue`.
    MCPB.reset(createMCPlusBuilder(Context.getTargetTriple().getArch(), MCIA,
                                   &MCII, Context.getRegisterInfo(),
                                   Context.getSubtargetInfo()));

    // TODO CFA offset should be the slot size, but for now I don't have any
    // access to it, maybe can be read from the prologue
    // TODO check what should be passed as EH?
    State = CFIState(MCRI->getDwarfRegNum(MCPB->getStackPointer(), false), 8);
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
    State.RegisterCFIStates[MCRI->getDwarfRegNum(MCPB->getStackPointer(),
                                                 false)] =
        RegisterCFIState::createOffsetFromCFAVal(0); // sp's old value is CFA
  }

  bool doesConstantChange(const MCInst &Inst, MCPhysReg Reg, int64_t &HowMuch) {
    if (MCPB->isPush(Inst) && Reg == MCPB->getStackPointer()) {
      // TODO should get the stack direction here, now it assumes that it goes
      // down.
      HowMuch = -MCPB->getPushSize(Inst);
      return true;
    }

    if (MCPB->isPop(Inst) && Reg == MCPB->getStackPointer()) {
      // TODO should get the stack direction here, now it assumes that it goes
      // down.
      HowMuch = MCPB->getPushSize(Inst);
      return true;
    }

    return false;
  }

  // Tries to guess Reg1's value in a form of Reg2 (before Inst's execution) +
  // Diff.
  bool isInConstantDistanceOfEachOther(const MCInst &Inst, MCPhysReg Reg1,
                                       MCPhysReg Reg2, int &Diff) {
    {
      MCPhysReg From;
      MCPhysReg To;
      if (MCPB->isRegToRegMove(Inst, From, To) && From == Reg2 && To == Reg1) {
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
    if (PrevState.CFARegister == NextState.CFARegister) {
      if (PrevState.CFAOffset == NextState.CFAOffset) {
        if (Writes.count(PrevState.CFARegister)) {
          Context.reportWarning(
              Inst.getLoc(),
              formatv("This instruction changes reg#{0}, which is "
                      "the CFA register, but the CFI directives do not.",
                      MCRI->getLLVMRegNum(PrevState.CFARegister, false)));
        }
        return;
      }
      // The offset is changed.
      if (!Writes.count(PrevState.CFARegister)) {
        Context.reportWarning(
            Inst.getLoc(),
            formatv(
                "You changed the CFA offset, but there is no modification to "
                "the CFA register happening in this instruction."));
      }
      int64_t HowMuch;
      if (!doesConstantChange(
              Inst, MCRI->getLLVMRegNum(PrevState.CFARegister, false).value(),
              HowMuch)) {
        Context.reportWarning(
            Inst.getLoc(),
            formatv("The CFA register changed, but I don't know how, finger "
                    "crossed the CFI directives are correct."));
        return;
      }
      // we know it is changed by HowMuch, so the CFA offset should be changed
      // by -HowMuch, i.e. AfterState.Offset - State.Offset = -HowMuch
      if (NextState.CFAOffset - PrevState.CFAOffset != -HowMuch) {
        Context.reportError(
            Inst.getLoc(),
            formatv("The CFA offset is changed by {0}, but "
                    "the directives changed it by {1}. (to be exact, the new "
                    "offset should be {2}, but it is {3})",
                    -HowMuch, NextState.CFAOffset - PrevState.CFAOffset,
                    PrevState.CFAOffset - HowMuch, NextState.CFAOffset));
        return;
      }

      // Everything OK!
      return;
    }
    // The CFA register is changed.
    // TODO move it up
    MCPhysReg OldCFAReg = MCRI->getLLVMRegNum(PrevState.CFARegister, false).value();
    MCPhysReg NewCFAReg =
        MCRI->getLLVMRegNum(NextState.CFARegister, false).value();
    if (!Writes.count(NextState.CFARegister)) {
      Context.reportWarning(
          Inst.getLoc(),
          formatv(
              "The new CFA register reg#{0}'s value is not assigned by this "
              "instruction, try to move the new CFA def to where this "
              "value is changed, now I can't infer if this change is "
              "correct or not.",
              NewCFAReg));
      return;
    }
    // Because CFA should is the CFA always stays the same:
    int OffsetDiff = PrevState.CFAOffset - NextState.CFAOffset;
    int Diff;
    if (!isInConstantDistanceOfEachOther(Inst, NewCFAReg, OldCFAReg, Diff)) {
      Context.reportWarning(
          Inst.getLoc(),
          formatv("Based on this instruction I cannot infer that the new and "
                  "old CFA registers are in {0} distance of each other. I "
                  "hope it's ok.",
                  OffsetDiff));
      return;
    }
    if (Diff != OffsetDiff) {
      Context.reportError(
          Inst.getLoc(), formatv("After changing the CFA register, the CFA "
                                 "offset should be {0}, but it is {1}.",
                                 PrevState.CFAOffset - Diff, NextState.CFAOffset));
      return;
    }
    // Everything is OK!
  }

  void update(const MCDwarfFrameInfo &DwarfFrame, MCInst &Inst,
              std::pair<unsigned, unsigned>
                  CFIDirectivesRange) { // FIXME this should not be a pair,
                                        // but an ArrayRef
    ArrayRef<MCCFIInstruction> CFIDirectives(DwarfFrame.Instructions);
    CFIDirectives = CFIDirectives.drop_front(CFIDirectivesRange.first)
                        .drop_back(DwarfFrame.Instructions.size() -
                                   CFIDirectivesRange.second);

    const MCInstrDesc &MCInstInfo = MCII.get(Inst.getOpcode());
    CFIState AfterState(State);
    for (auto &&CFIDirective : CFIDirectives)
      if (!AfterState.apply(CFIDirective))
        Context.reportWarning(
            CFIDirective.getLoc(),
            "I don't support this CFI directive, I assume this does nothing "
            "(which will probably break other things)");
    auto CFAReg = MCRI->getLLVMRegNum(DwarfFrame.CurrentCfaRegister, false);
    assert(DwarfFrame.CurrentCfaRegister == AfterState.CFARegister &&
           "Checking if the CFA tracking is working");

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

    bool ChangedCFA = Writes.count(CFAReg->id());
    bool GoingToChangeCFA = false;
    for (auto CFIDirective : CFIDirectives) {
      auto Op = CFIDirective.getOperation();
      GoingToChangeCFA |= (Op == MCCFIInstruction::OpDefCfa ||
                           Op == MCCFIInstruction::OpDefCfaOffset ||
                           Op == MCCFIInstruction::OpDefCfaRegister ||
                           Op == MCCFIInstruction::OpAdjustCfaOffset ||
                           Op == MCCFIInstruction::OpLLVMDefAspaceCfa);
    }
    if (ChangedCFA && !GoingToChangeCFA) {
      Context.reportError(Inst.getLoc(),
                          "The instruction changes CFA register value, but the "
                          "CFI directives don't update the CFA.");
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
    //   if (I == MCPB->getMemoryOperandNo(Inst)) {
    //     dbgs() << " memory access, details are:\n";
    //     auto X86MemoryOperand = MCPB->evaluateX86MemoryOperand(Inst);
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
    //     if (MCPB->isRegToRegMove(Inst, From, To)) {
    //       dbgs() << "It's a reg to reg move.\nFrom reg#" << From << " to
    //       reg#"
    //              << To << "\n";
    //     } else if (MCInstInfo.isMoveReg()) {
    //       dbgs() << "It's reg to reg move from MCInstInfo view but not from "
    //                 "MCPlus view.\n";
    //     }
    //   }
    //   if (MCPB->isConditionalMove(Inst)) {
    //     dbgs() << "Its a conditional move.\n";
    //   }
    //   if (MCPB->isMoveMem2Reg(Inst)) {
    //     dbgs() << "It's a move from memory to register\n";
    //     assert(MCPB->getMemoryOperandNo(Inst) == 1);
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
    //   if (MCPB->isStackAccess(Inst, IsLoad, IsStore, IsStoreFromReg, Reg,
    //                           SrcImm, StackPtrReg, StackOffset, Size,
    //                           IsSimple, IsIndexed)) {
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
    //   if (MCPB->isPush(Inst)) {
    //     dbgs() << "This is a push instruction with size "
    //            << MCPB->getPushSize(Inst) << "\n";
    //   }
    //   if (MCPB->isPop(Inst)) {
    //     dbgs() << "This is a pop instruction with size "
    //            << MCPB->getPopSize(Inst) << "\n";
    //   }
    // }

    // dbgs() << "---------------Arith Info----------------\n";
    // { // arith
    //   MutableArrayRef<MCInst> MAR = {Inst};
    //   if (MCPB->matchAdd(MCPB->matchAnyOperand(), MCPB->matchAnyOperand())
    //           ->match(*MCRI, *MCPB, MutableArrayRef<MCInst>(), -1)) {
    //     dbgs() << "It is an addition instruction.\n";
    //   } else if (MCInstInfo.isAdd()) {
    //     dbgs() << "It is an addition from MCInstInfo view but not from MCPlus
    //     "
    //               "view.\n";
    //   }
    //   if (MCPB->isSUB(Inst)) {
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