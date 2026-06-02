//===-- EZHAsmPrinter.cpp - EZH LLVM assembly writer ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EZHConstantPoolValue.h"
#include "EZHMCInstLower.h"
#include "EZHTargetMachine.h"
#include "MCTargetDesc/EZHInstPrinter.h"
#include "MCTargetDesc/EZHMCTargetDesc.h"
#include "TargetInfo/EZHTargetInfo.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

namespace {
class EZHAsmPrinter : public AsmPrinter {

public:
  explicit EZHAsmPrinter(TargetMachine &TM,
                         std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer), ID) {}

  StringRef getPassName() const override { return "EZH Assembly Printer"; }

  void emitInstruction(const MachineInstr *MI) override;
  void emitFunctionBodyEnd() override;
  void emitBasicBlockEnd(const MachineBasicBlock &MBB) override;
  void emitMachineConstantPoolValue(MachineConstantPoolValue *MCPV) override;
  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       const char *ExtraCode, raw_ostream &OS) override;

  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                             const char *ExtraCode, raw_ostream &OS) override;

  void emitConstantPool() override {
    // Do nothing. We handle constant pool emission via CONSTPOOL_ENTRY in
    // Constant Island pass.
  }
  void emitJumpTableInfo() override {
    // Do nothing. We handle jump table emission via CONSTPOOL_ENTRY in Constant
    // Island pass.
  }

  static char ID;
};
} // end of anonymous namespace

void EZHAsmPrinter::emitFunctionBodyEnd() {}

void EZHAsmPrinter::emitBasicBlockEnd(const MachineBasicBlock &MBB) {}

void EZHAsmPrinter::emitMachineConstantPoolValue(
    MachineConstantPoolValue *MCPV) {
  auto *CPV = static_cast<const EZHConstantPoolValue *>(MCPV);

  if (CPV->isJumpTable()) {
    SmallString<32> SymName;
    raw_svector_ostream OS(SymName);
    OS << ".LJTI" << MF->getFunctionNumber() << "_" << CPV->getJumpTableIndex();
    MCSymbol *Sym = OutContext.getOrCreateSymbol(SymName.str());
    OutStreamer->emitValue(MCSymbolRefExpr::create(Sym, OutContext), 4);
  } else if (CPV->isBlockAddress()) {
    MCSymbol *Sym = GetBlockAddressSymbol(CPV->getBlockAddress());
    OutStreamer->emitValue(MCSymbolRefExpr::create(Sym, OutContext), 4);
  } else if (CPV->isExtSymbol()) {
    MCSymbol *Sym = OutContext.getOrCreateSymbol(CPV->getExtSymbol());
    OutStreamer->emitValue(MCSymbolRefExpr::create(Sym, OutContext), 4);
  } else if (CPV->isMachineBasicBlock()) {
    OutStreamer->emitValue(
        MCSymbolRefExpr::create(CPV->getMachineBasicBlock()->getSymbol(),
                                OutContext),
        4);
  } else if (CPV->isGlobalValue()) {
    const MCExpr *Expr =
        MCSymbolRefExpr::create(getSymbol(CPV->getGlobalValue()), OutContext);
    if (CPV->getOffset() != 0) {
      Expr = MCBinaryExpr::createAdd(
          Expr, MCConstantExpr::create(CPV->getOffset(), OutContext),
          OutContext);
    }
    OutStreamer->emitValue(Expr, 4);
  } else if (CPV->isConstantPoolIndex()) {
    MCSymbol *Sym = GetCPISymbol(CPV->getConstantPoolIndex());
    OutStreamer->emitValue(MCSymbolRefExpr::create(Sym, OutContext), 4);
  }
}

void EZHAsmPrinter::emitInstruction(const MachineInstr *MI) {

  if (MI->getOpcode() == EZH::CONSTPOOL_ENTRY) {
    const MachineOperand &MO0 = MI->getOperand(0);
    const MachineOperand &MO1 = MI->getOperand(1);

    MCSymbol *Sym = MO0.getMCSymbol();
    OutStreamer->emitLabel(Sym);

    if (MO1.isGlobal()) {
      OutStreamer->emitValue(
          MCSymbolRefExpr::create(getSymbol(MO1.getGlobal()), OutContext), 4);
    } else if (MO1.isImm()) {
      OutStreamer->emitValue(MCConstantExpr::create(MO1.getImm(), OutContext),
                             4);
    } else if (MO1.isCPI()) {
      int CPI = MO1.getIndex();
      const MachineConstantPool *MCP =
          MI->getParent()->getParent()->getConstantPool();
      const std::vector<MachineConstantPoolEntry> &Constants =
          MCP->getConstants();
      const MachineConstantPoolEntry &CPE = Constants[CPI];
      if (CPE.isMachineConstantPoolEntry()) {
        emitMachineConstantPoolValue(CPE.Val.MachineCPVal);
      } else {
        emitGlobalConstant(MI->getParent()->getParent()->getDataLayout(),
                           CPE.Val.ConstVal);
      }
    } else if (MO1.isSymbol()) {
      OutStreamer->emitValue(
          MCSymbolRefExpr::create(
              OutContext.getOrCreateSymbol(MO1.getSymbolName()), OutContext),
          4);
    } else if (MO1.isJTI()) {
      int JTI = MO1.getIndex();
      const MachineJumpTableInfo *MJTI =
          MI->getParent()->getParent()->getJumpTableInfo();
      const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
      const std::vector<MachineBasicBlock *> &MBBs = JT[JTI].MBBs;
      for (MachineBasicBlock *MBB : MBBs) {
        const MCExpr *Expr =
            MCSymbolRefExpr::create(MBB->getSymbol(), OutContext);
        if (TM.getRelocationModel() == Reloc::PIC_) {
          const MCExpr *Base = MCSymbolRefExpr::create(Sym, OutContext);
          const MCExpr *Diff = MCBinaryExpr::createSub(Expr, Base, OutContext);
          OutStreamer->emitValue(Diff, 4);
        } else {
          OutStreamer->emitValue(Expr, 4);
        }
      }
    }
    return;
  }

  if (MI->getOpcode() == EZH::LOAD_CONSTANT) {
    Register Rd = MI->getOperand(0).getReg();
    const MachineOperand &MO = MI->getOperand(1);

    MCSymbol *Sym = nullptr;
    if (MO.isMCSymbol()) {
      Sym = MO.getMCSymbol();
    } else if (MO.isCPI()) {
      Sym = GetCPISymbol(MO.getIndex());
    } else if (MO.isImm()) {
      int64_t Imm = MO.getImm();
      if (isInt<11>(Imm)) {
        EmitToStreamer(*OutStreamer,
                       MCInstBuilder(EZH::MOVri__).addReg(Rd).addImm(Imm));
        return;
      } else {
        llvm_unreachable("Immediate constant too large for E_LOAD_IMM!");
      }
    } else {
      llvm_unreachable("Unexpected operand type for LOAD_CONSTANT!");
    }

    // e_ldr Rd, pc, Sym
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(EZH::LDR).addReg(Rd).addReg(EZH::PC).addExpr(
                       MCSymbolRefExpr::create(Sym, OutContext)));
    return;
  }

  if (MI->getOpcode() == EZH::RET) {
    // Intercept standard RET and print as a safe register-move return (e_mov
    // pc, ra). This is 100% immune to bit-slice interrupt RA corruption!
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(EZH::MOVrr__).addReg(EZH::PC).addReg(EZH::RA));
    return;
  }

  EZHMCInstLower MCInstLowering(OutContext, *this);
  MCInst TmpInst;
  MCInstLowering.Lower(MI, TmpInst);
  EmitToStreamer(*OutStreamer, TmpInst);
}

bool EZHAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                    const char *ExtraCode, raw_ostream &OS) {
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[0] == 'c') {
      const MachineOperand &MO = MI->getOperand(OpNo);
      if (MO.isImm()) {
        OS << MO.getImm();
        return false;
      }
      if (MO.isGlobal()) {
        OS << getSymbol(MO.getGlobal())->getName();
        if (MO.getOffset() != 0)
          OS << "+" << MO.getOffset();
        return false;
      }
    }
    return AsmPrinter::PrintAsmOperand(MI, OpNo, ExtraCode, OS);
  }

  const MachineOperand &MO = MI->getOperand(OpNo);
  if (MO.isReg()) {
    OS << EZHInstPrinter::getRegisterName(MO.getReg());
    return false;
  }
  if (MO.isImm()) {
    OS << MO.getImm();
    return false;
  }
  if (MO.isGlobal()) {
    OS << getSymbol(MO.getGlobal())->getName();
    if (MO.getOffset() != 0)
      OS << "+" << MO.getOffset();
    return false;
  }
  if (MO.isSymbol()) {
    OS << MO.getSymbolName();
    return false;
  }
  if (MO.isFI()) {
    OS << MO.getIndex();
    return false;
  }
  if (MO.isMBB()) {
    OS << MO.getMBB()->getSymbol()->getName();
    return false;
  }
  if (MO.isBlockAddress()) {
    OS << GetBlockAddressSymbol(MO.getBlockAddress())->getName();
    return false;
  }
  return AsmPrinter::PrintAsmOperand(MI, OpNo, ExtraCode, OS);
}

bool EZHAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                                          const char *ExtraCode,
                                          raw_ostream &OS) {
  if (ExtraCode && ExtraCode[0])
    return true;
  const MachineOperand &MO = MI->getOperand(OpNo);
  if (MO.isReg()) {
    OS << EZHInstPrinter::getRegisterName(MO.getReg());
    return false;
  }
  return AsmPrinter::PrintAsmMemoryOperand(MI, OpNo, ExtraCode, OS);
}

char EZHAsmPrinter::ID = 0;

INITIALIZE_PASS(EZHAsmPrinter, "ezh-asm-printer", "EZH Assembly Printer", false,
                false)

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeEZHAsmPrinter() {
  RegisterAsmPrinter<EZHAsmPrinter> X(getTheEZHTarget());
}
