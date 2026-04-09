//===-- LX32AsmPrinter.cpp - LX32 Assembly Printer ------------------------===//
//
// Part of the LX32 Project
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "LX32InstrInfo.h"
#include "LX32TargetMachine.h"

#include "../TargetInfo/LX32TargetInfo.h"

#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

class LX32MCInstLower {
  AsmPrinter &Printer;
  MCContext &Ctx;

public:
  explicit LX32MCInstLower(AsmPrinter &Printer, MCContext &Ctx)
      : Printer(Printer), Ctx(Ctx) {}

  const MCExpr *lowerSymbolOperand(const MachineOperand &MO) const {
    const MCSymbol *Sym = nullptr;
    int64_t Offset = 0;

    switch (MO.getType()) {
    default:
      return nullptr;
    case MachineOperand::MO_GlobalAddress:
      Sym = Printer.getSymbol(MO.getGlobal());
      Offset = MO.getOffset();
      break;
    case MachineOperand::MO_MachineBasicBlock:
      Sym = MO.getMBB()->getSymbol();
      break;
    case MachineOperand::MO_ExternalSymbol:
      Sym = Printer.GetExternalSymbolSymbol(MO.getSymbolName());
      break;
    case MachineOperand::MO_BlockAddress:
      Sym = Printer.GetBlockAddressSymbol(MO.getBlockAddress());
      Offset = MO.getOffset();
      break;
    case MachineOperand::MO_ConstantPoolIndex:
      Sym = Printer.GetCPISymbol(MO.getIndex());
      Offset = MO.getOffset();
      break;
    case MachineOperand::MO_JumpTableIndex:
      Sym = Printer.GetJTISymbol(MO.getIndex());
      break;
    }

    const MCExpr *Expr = MCSymbolRefExpr::create(Sym, Ctx);
    if (Offset != 0)
      Expr = MCBinaryExpr::createAdd(
          Expr, MCConstantExpr::create(Offset, Ctx), Ctx);
    return Expr;
  }

  bool lowerOperand(const MachineOperand &MO, MCOperand &MCOp) const {
    switch (MO.getType()) {
    default:
      return false;
    case MachineOperand::MO_Register:
      if (MO.isImplicit())
        return false;
      MCOp = MCOperand::createReg(MO.getReg());
      return true;
    case MachineOperand::MO_Immediate:
      MCOp = MCOperand::createImm(MO.getImm());
      return true;
    case MachineOperand::MO_CImmediate:
      MCOp = MCOperand::createImm(MO.getCImm()->getSExtValue());
      return true;
    case MachineOperand::MO_FPImmediate:
      return false;
    case MachineOperand::MO_MachineBasicBlock:
    case MachineOperand::MO_GlobalAddress:
    case MachineOperand::MO_ExternalSymbol:
    case MachineOperand::MO_BlockAddress:
    case MachineOperand::MO_ConstantPoolIndex:
    case MachineOperand::MO_JumpTableIndex: {
      const MCExpr *Expr = lowerSymbolOperand(MO);
      if (!Expr)
        return false;
      MCOp = MCOperand::createExpr(Expr);
      return true;
    }
    }
  }

  void lower(const MachineInstr *MI, MCInst &OutMI) const {
    OutMI.setOpcode(MI->getOpcode());
    for (const MachineOperand &MO : MI->operands()) {
      MCOperand MCOp;
      if (lowerOperand(MO, MCOp))
        OutMI.addOperand(MCOp);
    }
  }
};

class LX32AsmPrinter : public AsmPrinter {
  LX32MCInstLower MCILower;

public:
  explicit LX32AsmPrinter(TargetMachine &TM,
                          std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)), MCILower(*this, OutContext) {}

  StringRef getPassName() const override { return "LX32 Assembly Printer"; }

  void emitInstruction(const MachineInstr *MI) override {
    switch (MI->getOpcode()) {
    default:
      break;
    case LX32::PseudoRET: {
      MCInst Ret;
      Ret.setOpcode(LX32::JALR);
      Ret.addOperand(MCOperand::createReg(LX32::X0));
      Ret.addOperand(MCOperand::createReg(LX32::X1));
      Ret.addOperand(MCOperand::createImm(0));
      EmitToStreamer(*OutStreamer, Ret);
      return;
    }
    case LX32::PseudoNOP: {
      MCInst Nop;
      Nop.setOpcode(LX32::ADDI);
      Nop.addOperand(MCOperand::createReg(LX32::X0));
      Nop.addOperand(MCOperand::createReg(LX32::X0));
      Nop.addOperand(MCOperand::createImm(0));
      EmitToStreamer(*OutStreamer, Nop);
      return;
    }
    case LX32::PseudoCALL: {
      // Support both call forms used by the backend:
      //  1) register-held target  -> jalr ra, rs1, 0
      //  2) direct symbol target  -> jal  ra, symbol
      Register Base = 0;
      MCOperand TargetSym;
      bool HasTargetSym = false;
      for (const MachineOperand &MO : MI->operands()) {
        if (MO.isReg() && MO.getReg() != 0 && !MO.isImplicit()) {
          Base = MO.getReg();
          break;
        }
        if (!HasTargetSym &&
            (MO.isGlobal() || MO.isSymbol() || MO.isMBB() ||
             MO.isBlockAddress() || MO.isCPI() || MO.isJTI())) {
          HasTargetSym = MCILower.lowerOperand(MO, TargetSym);
        }
      }

      if (Base) {
        MCInst Call;
        Call.setOpcode(LX32::JALR);
        Call.addOperand(MCOperand::createReg(LX32::X1));
        Call.addOperand(MCOperand::createReg(Base));
        Call.addOperand(MCOperand::createImm(0));
        EmitToStreamer(*OutStreamer, Call);
        return;
      }

      if (HasTargetSym) {
        MCInst Call;
        Call.setOpcode(LX32::JAL);
        Call.addOperand(MCOperand::createReg(LX32::X1));
        Call.addOperand(TargetSym);
        EmitToStreamer(*OutStreamer, Call);
        return;
      }

      std::string Dump;
      raw_string_ostream OS(Dump);
      MI->print(OS);
      report_fatal_error(Twine("lx32: malformed PseudoCALL (missing callable target operand): ") +
                         OS.str());
    }
    case LX32::PseudoLA: {
      // Minimal symbol materialization for asm path.
      MCOperand RD, SymOp;
      if (!MCILower.lowerOperand(MI->getOperand(0), RD) ||
          !MCILower.lowerOperand(MI->getOperand(1), SymOp))
        report_fatal_error("lx32: failed to lower PseudoLA operands");

      MCInst Auipc;
      Auipc.setOpcode(LX32::AUIPC);
      Auipc.addOperand(RD);
      Auipc.addOperand(SymOp);
      EmitToStreamer(*OutStreamer, Auipc);

      MCInst Addi;
      Addi.setOpcode(LX32::ADDI);
      Addi.addOperand(RD);
      Addi.addOperand(RD);
      Addi.addOperand(SymOp);
      EmitToStreamer(*OutStreamer, Addi);
      return;
    }
    }

    MCInst TmpInst;
    MCILower.lower(MI, TmpInst);
    EmitToStreamer(*OutStreamer, TmpInst);
  }
};

} // end anonymous namespace

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeLX32AsmPrinter() {
  RegisterAsmPrinter<LX32AsmPrinter> X(getTheLX32TargetInfo());
}





