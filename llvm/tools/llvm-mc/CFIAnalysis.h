#ifndef LLVM_TOOLS_LLVM_MC_CFI_ANALYSIS_H
#define LLVM_TOOLS_LLVM_MC_CFI_ANALYSIS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/Debug.h"

namespace llvm {

// TODO remove it, it's just for debug purposes.
void printUntilNextLine(const char *Str) {
  for (int I = 0; Str[I] != '\0' && Str[I] != '\n'; I++)
    dbgs() << Str[I];
}

class CFIAnalysis {
  MCContext &Context;
  MCInstrInfo const &MCII;

public:
  CFIAnalysis(MCContext &Context, MCInstrInfo const &MCII)
      : Context(Context), MCII(MCII) {}

  void update(const MCDwarfFrameInfo &DwarfFrame, const MCInst &Inst,
              ArrayRef<MCCFIInstruction> CFIDirectives) {
    dbgs() << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n";
    printUntilNextLine(Inst.getLoc().getPointer());
    dbgs() << "\n";
    dbgs() << "codes: ";
    Inst.print(dbgs());
    dbgs() << "\n";
    dbgs() << "------------------------------\n";
    auto *RI = Context.getRegisterInfo();
    auto CFAReg = RI->getLLVMRegNum(DwarfFrame.CurrentCfaRegister, false);
    dbgs() << "The CFA register is: " << CFAReg << "\n";
    bool GoingToChangeCFA = false;
    for (auto CFIDirective : CFIDirectives) {
      auto Op = CFIDirective.getOperation();
      GoingToChangeCFA |= (Op == MCCFIInstruction::OpDefCfa ||
                           Op == MCCFIInstruction::OpDefCfaOffset ||
                           Op == MCCFIInstruction::OpDefCfaRegister ||
                           Op == MCCFIInstruction::OpAdjustCfaOffset ||
                           Op == MCCFIInstruction::OpLLVMDefAspaceCfa);
    }
    dbgs() << "------------------------------\n";
    bool ChangedCFA = false;
    for (int I = 0; I < Inst.getNumOperands(); I++) {
      auto &&Operand = Inst.getOperand(I);
      if (!Operand.isReg())
        continue;
      if (MCII.get(Inst.getOpcode())
              .hasDefOfPhysReg(Inst, Operand.getReg(), *RI)) {
        dbgs() << "This instruction modifies: " << Operand.getReg().id()
               << "\n";
        if (Operand.getReg() == CFAReg.value())
          ChangedCFA = true;
      }
    }
    dbgs() << "------------------------------\n";
    dbgs() << "The instruction DOES " << (ChangedCFA ? "" : "NOT ")
           << "change the CFA.\n";
    dbgs() << "The CFI directives DOES " << (GoingToChangeCFA ? "" : "NOT ")
           << "change the CFA.\n";
    if (ChangedCFA && !GoingToChangeCFA) {
      Context.reportError(Inst.getLoc(),
                          "The instruction changes CFA register value, but the "
                          "CFI directives don't update the CFA.");
    }
    // TODO needs more work
    // if (!ChangedCFA && GoingToChangeCFA) {
    //   Context.reportError(
    //       Inst.getLoc(),
    //       "The instruction doesn't change CFA register value, but the "
    //       "CFI directives update the CFA.");
    // }
    dbgs() << "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n";
    ChangedCFA = false;
  }
};

} // namespace llvm
#endif