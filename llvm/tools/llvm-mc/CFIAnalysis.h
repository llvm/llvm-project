#ifndef LLVM_TOOLS_LLVM_MC_CFI_ANALYSIS_H
#define LLVM_TOOLS_LLVM_MC_CFI_ANALYSIS_H

#include "bolt/Core/MCPlusBuilder.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include <memory>
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

public:
  CFIAnalysis(MCContext &Context, MCInstrInfo const &MCII,
              MCInstrAnalysis *MCIA)
      : Context(Context), MCII(MCII) {
    // TODO it should look at the poluge directives and setup the
    // registers' previous value state here, but for now, it's assumed that all
    // values are by default `samevalue`.
    MCPB.reset(createMCPlusBuilder(Context.getTargetTriple().getArch(), MCIA,
                                   &MCII, Context.getRegisterInfo(),
                                   Context.getSubtargetInfo()));
  }

  void update(const MCDwarfFrameInfo &DwarfFrame, const MCInst &Inst,
              ArrayRef<MCCFIInstruction> CFIDirectives) {

    const MCInstrDesc &MCInstInfo = MCII.get(Inst.getOpcode());
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
    dbgs() << "isMoveMem2Reg: " << MCPB->isMoveMem2Reg(Inst) << "\n";
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