//===-- ParasolAsmPrinter.cpp - Parasol LLVM Assembly Printer -------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to GAS-format CPU0 assembly language.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/ParasolInstPrinter.h"
#include "ParasolInstrInfo.h"
#include "ParasolTargetMachine.h"
#include "TargetInfo/ParasolTargetInfo.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "parasol-asm-printer"

namespace llvm {
class ParasolAsmPrinter : public AsmPrinter {
public:
  explicit ParasolAsmPrinter(TargetMachine &TM,
                             std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)) {}

  virtual StringRef getPassName() const override {
    return "Parasol Assembly Printer";
  }

  void emitInstruction(const MachineInstr *MI) override;

  // This function must be present as it is internally used by the
  // auto-generated function emitPseudoExpansionLowering to expand pseudo
  // instruction
  void EmitToStreamer(MCStreamer &S, const MCInst &Inst);
  // Auto-generated function in ParasolGenMCPseudoLowering.inc
  bool emitPseudoExpansionLowering(MCStreamer &OutStreamer,
                                   const MachineInstr *MI);

private:
  void LowerInstruction(const MachineInstr *MI, MCInst &OutMI) const;
  MCOperand LowerOperand(const MachineOperand &MO) const;
  MCOperand LowerSymbolOperand(const MachineOperand &MO, MCSymbol *Sym) const;
};
} // namespace llvm

// Simple pseudo-instructions have their lowering (with expansion to real
// instructions) auto-generated.
#include "ParasolGenMCPseudoLowering.inc"
void ParasolAsmPrinter::EmitToStreamer(MCStreamer &S, const MCInst &Inst) {
  AsmPrinter::EmitToStreamer(*OutStreamer, Inst);
}

void ParasolAsmPrinter::emitInstruction(const MachineInstr *MI) {
  // Do any auto-generated pseudo lowerings.
  if (emitPseudoExpansionLowering(*OutStreamer, MI))
    return;

  MCInst TmpInst;
  LowerInstruction(MI, TmpInst);
  EmitToStreamer(*OutStreamer, TmpInst);
}

void ParasolAsmPrinter::LowerInstruction(const MachineInstr *MI,
                                         MCInst &OutMI) const {
  OutMI.setOpcode(MI->getOpcode());

  for (const MachineOperand &MO : MI->operands()) {
    MCOperand MCOp = LowerOperand(MO);
    if (MCOp.isValid())
      OutMI.addOperand(MCOp);
  }
}

MCOperand ParasolAsmPrinter::LowerOperand(const MachineOperand &MO) const {
  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    // Ignore all implicit register operands.
    if (MO.isImplicit()) {
      break;
    }
    return MCOperand::createReg(MO.getReg());

  case MachineOperand::MO_Immediate:
    return MCOperand::createImm(MO.getImm());

  case MachineOperand::MO_MachineBasicBlock:
    return LowerSymbolOperand(MO, MO.getMBB()->getSymbol());

  case MachineOperand::MO_GlobalAddress:
    return LowerSymbolOperand(MO, getSymbol(MO.getGlobal()));

  case MachineOperand::MO_BlockAddress:
    return LowerSymbolOperand(MO, GetBlockAddressSymbol(MO.getBlockAddress()));

  case MachineOperand::MO_ExternalSymbol:
    return LowerSymbolOperand(MO, GetExternalSymbolSymbol(MO.getSymbolName()));

  case MachineOperand::MO_ConstantPoolIndex:
    return LowerSymbolOperand(MO, GetCPISymbol(MO.getIndex()));

  case MachineOperand::MO_RegisterMask:
    break;

  default:
    report_fatal_error("unknown operand type");
  }

  return MCOperand();
}

MCOperand ParasolAsmPrinter::LowerSymbolOperand(const MachineOperand &MO,
                                                MCSymbol *Sym) const {
  MCContext &Ctx = OutContext;

  const MCExpr *Expr =
      MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_None, Ctx);

  if (!MO.isJTI() && !MO.isMBB() && MO.getOffset())
    Expr = MCBinaryExpr::createAdd(
        Expr, MCConstantExpr::create(MO.getOffset(), Ctx), Ctx);

  return MCOperand::createExpr(Expr);
}

// Force static initialization.
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeParasolAsmPrinter() {
  RegisterAsmPrinter<ParasolAsmPrinter> X(getTheParasolTarget());
}
