#ifndef LLVM_TOOLS_LLVM_MC_CFI_ANALYSIS_H
#define LLVM_TOOLS_LLVM_MC_CFI_ANALYSIS_H

#include "bolt/Core/MCPlusBuilder.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
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

  void update(const MCDwarfFrameInfo &DwarfFrame, MCInst &Inst,
              ArrayRef<MCCFIInstruction> CFIDirectives) {

    const MCInstrDesc &MCInstInfo = MCII.get(Inst.getOpcode());
    auto *RI = Context.getRegisterInfo();
    auto CFAReg = RI->getLLVMRegNum(DwarfFrame.CurrentCfaRegister, false);

    std::set<int> Writes, Reads; // TODO reads is not ready for now
    // FIXME this way of extracting uses is buggy:
    // for (unsigned I = 0; I < MCInstInfo.NumImplicitUses; I++)
    //   Reads.insert(MCInstInfo.implicit_uses()[I]);
    for (unsigned I = 0; I < MCInstInfo.NumImplicitDefs; I++)
      Writes.insert(MCInstInfo.implicit_defs()[I]);

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
    dbgs() << "Op code: " << Inst.getOpcode() << "\n";
    dbgs() << "--------------Operands Info--------------\n";
    auto DefCount = MCInstInfo.getNumDefs();
    for (unsigned I = 0; I < Inst.getNumOperands(); I++) {
      dbgs() << "Operand #" << I << ", which will be "
             << (I < DefCount ? "defined" : "used") << ", is a";
      if (I == MCPB->getMemoryOperandNo(Inst)) {
        dbgs() << " memory access, details are:\n";
        auto X86MemoryOperand = MCPB->evaluateX86MemoryOperand(Inst);
        dbgs() << "  Base Register{ reg#" << X86MemoryOperand->BaseRegNum
               << " }";
        dbgs() << " + (Index Register{ reg#" << X86MemoryOperand->IndexRegNum
               << " }";
        dbgs() << " * Scale{ value " << X86MemoryOperand->ScaleImm << " }";
        dbgs() << ") + Displacement{ "
               << (X86MemoryOperand->DispExpr
                       ? "an expression"
                       : "value " + itostr(X86MemoryOperand->DispImm))
               << " }\n";
        // TODO, it's not correct always, it cannot be assumed (or should be
        // checked) that memory operands are flatten into 4 operands in MCInc.
        I += 4;
        continue;
      }
      auto &&Operand = Inst.getOperand(I);
      if (Operand.isImm()) {
        dbgs() << "n immediate with value " << Operand.getImm() << "\n";
        continue;
      }
      if (Operand.isReg()) {
        dbgs() << " reg#" << Operand.getReg() << "\n";
        continue;
      }
      assert(Operand.isExpr());
      dbgs() << "n unknown expression \n";
    }
    if (MCInstInfo.NumImplicitDefs) {
      dbgs() << "implicitly defines: { ";
      for (unsigned I = 0; I < MCInstInfo.NumImplicitDefs; I++) {
        dbgs() << "reg#" << MCInstInfo.implicit_defs()[I] << " ";
      }
      dbgs() << "}\n";
    }
    if (MCInstInfo.NumImplicitUses) {
      dbgs() << "implicitly uses: { ";
      for (unsigned I = 0; I < MCInstInfo.NumImplicitUses; I++) {
        dbgs() << "reg#" << MCInstInfo.implicit_uses()[I] << " ";
      }
      dbgs() << "}\n";
    }
    dbgs() << "----------------Move Info----------------\n";
    {   // move
      { // Reg2Reg
        MCPhysReg From, To;
        if (MCPB->isRegToRegMove(Inst, From, To)) {
          dbgs() << "It's a reg to reg move.\nFrom reg#" << From << " to reg#"
                 << To << "\n";
        } else if (MCInstInfo.isMoveReg()) {
          dbgs() << "It's reg to reg move from MCInstInfo view but not from "
                    "MCPlus view.\n";
        }
      }
      if (MCPB->isConditionalMove(Inst)) {
        dbgs() << "Its a conditional move.\n";
      }
      if (MCPB->isMoveMem2Reg(Inst)) {
        dbgs() << "It's a move from memory to register\n";
        assert(MCPB->getMemoryOperandNo(Inst) == 1);
      }
    }

    dbgs() << "---------------Stack Info----------------\n";
    { // stack
      int32_t SrcImm = 0;
      MCPhysReg Reg = 0;
      uint16_t StackPtrReg = 0;
      int64_t StackOffset = 0;
      uint8_t Size = 0;
      bool IsSimple = false;
      bool IsIndexed = false;
      bool IsLoad = false;
      bool IsStore = false;
      bool IsStoreFromReg = false;
      if (MCPB->isStackAccess(Inst, IsLoad, IsStore, IsStoreFromReg, Reg,
                              SrcImm, StackPtrReg, StackOffset, Size, IsSimple,
                              IsIndexed)) {
        dbgs() << "This instruction accesses the stack, the details are:\n";
        dbgs() << "  Source immediate: " << SrcImm << "\n";
        dbgs() << "  Source reg#" << Reg << "\n";
        dbgs() << "  Stack pointer: reg#" << StackPtrReg << "\n";
        dbgs() << "  Stack offset: " << StackOffset << "\n";
        dbgs() << "  size: " << (int)Size << "\n";
        dbgs() << "  Is simple: " << (IsSimple ? "yes" : "no") << "\n";
        dbgs() << "  Is indexed: " << (IsIndexed ? "yes" : "no") << "\n";
        dbgs() << "  Is load: " << (IsLoad ? "yes" : "no") << "\n";
        dbgs() << "  Is store: " << (IsStore ? "yes" : "no") << "\n";
        dbgs() << "  Is store from reg: " << (IsStoreFromReg ? "yes" : "no")
               << "\n";
      }
      if (MCPB->isPush(Inst)) {
        dbgs() << "This is a push instruction with size "
               << MCPB->getPushSize(Inst) << "\n";
      }
      if (MCPB->isPop(Inst)) {
        dbgs() << "This is a pop instruction with size "
               << MCPB->getPopSize(Inst) << "\n";
      }
    }

    dbgs() << "---------------Arith Info----------------\n";
    { // arith
      MutableArrayRef<MCInst> MAR = {Inst};
      if (MCPB->matchAdd(MCPB->matchAnyOperand(), MCPB->matchAnyOperand())
              ->match(*RI, *MCPB, MutableArrayRef<MCInst>(), -1)) {
        dbgs() << "It is an addition instruction.\n";
      } else if (MCInstInfo.isAdd()) {
        dbgs() << "It is an addition from MCInstInfo view but not from MCPlus "
                  "view.\n";
      }
      if (MCPB->isSUB(Inst)) {
        dbgs() << "This is a subtraction.\n";
      }
    }

    // dbgs() << "-----------------------------------------\n";
    // dbgs() << "The CFA register is: " << CFAReg << "\n";
    // dbgs() << "The instruction does " << (ChangedCFA ? "" : "NOT ")
    //        << "change the CFA.\n";
    // dbgs() << "The CFI directives does " << (GoingToChangeCFA ? "" : "NOT ")
    //        << "change the CFA.\n";
    dbgs() << "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n";
    ChangedCFA = false;
  }
};

} // namespace llvm
#endif