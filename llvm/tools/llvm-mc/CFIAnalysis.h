#ifndef LLVM_TOOLS_LLVM_MC_CFI_ANALYSIS_H
#define LLVM_TOOLS_LLVM_MC_CFI_ANALYSIS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/Debug.h"
#include <ostream>
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

public:
  CFIAnalysis(MCContext &Context, MCInstrInfo const &MCII)
      : Context(Context), MCII(MCII) {
    // TODO it should look at the poluge directives and setup the
    // registers' previous value state here, but for now, it's assumed that all
    // values are by default `samevalue`.
  }

  void update(const MCDwarfFrameInfo &DwarfFrame, const MCInst &Inst,
              ArrayRef<MCCFIInstruction> CFIDirectives) {

    auto MCInstInfo = MCII.get(Inst.getOpcode());
    auto *RI = Context.getRegisterInfo();
    auto CFAReg = RI->getLLVMRegNum(DwarfFrame.CurrentCfaRegister, false);

    std::set<int> Writes, Reads; // TODO reads is not ready for now
    // FIXME this way of extracting uses is buggy:
    // for (unsigned I = 0; I < MCInstInfo.NumImplicitUses; I++)
    //   Reads.insert(MCInstInfo.implicit_uses()[I]);
    // for (unsigned I = 0; I < MCInstInfo.NumImplicitDefs; I++)
    //   Writes.insert(MCInstInfo.implicit_defs()[I]);

    for (unsigned I = 0; I < Inst.getNumOperands(); I++) {
      auto &&Operand = Inst.getOperand(I);
      if (Operand.isReg()) {
        // TODO it is not percise, maybe this operand is for output, then it
        // means that there is no read happening here.
        Reads.insert(Operand.getReg());
      }

      if (Operand.isExpr()) {
        // TODO maybe the argument is not a register, but is a expression like
        // `34(sp)` that has a register in it. Check if this works or not and if
        // no change it somehow that it count that register as reads or writes
        // too.
      }

      if (Operand.isReg() &&
          MCInstInfo.hasDefOfPhysReg(Inst, Operand.getReg(), *RI))
        Writes.insert(Operand.getReg().id());
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

    dbgs() << "^^^^^^^^^^^^^^^^ InsCFIs ^^^^^^^^^^^^^^^^\n";
    printUntilNextLine(Inst.getLoc().getPointer());
    dbgs() << "\n";
    dbgs() << "-----------------------------------------\n";
    dbgs() << "writes into: { ";
    for (auto Reg : Writes) {
      dbgs() << (int)Reg << " ";
    }
    dbgs() << "}\n";
    dbgs() << "-----------------------------------------\n";
    dbgs() << "The CFA register is: " << CFAReg << "\n";
    dbgs() << "The instruction does " << (ChangedCFA ? "" : "NOT ")
           << "change the CFA.\n";
    dbgs() << "The CFI directives does " << (GoingToChangeCFA ? "" : "NOT ")
           << "change the CFA.\n";
    dbgs() << "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n";
    ChangedCFA = false;
  }
};

} // namespace llvm
#endif