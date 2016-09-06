//===- AMDGPUMCInstLower.cpp - Lower AMDGPU MachineInstr to an MCInst -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Code to lower AMDGPU MachineInstrs to their corresponding MCInst.
//
//===----------------------------------------------------------------------===//
//

#include "AMDGPUMCInstLower.h"
#include "AMDGPUAsmPrinter.h"
#include "AMDGPUSubtarget.h"
#include "AMDGPUTargetMachine.h"
#include "InstPrinter/AMDGPUInstPrinter.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include <algorithm>

using namespace llvm;

AMDGPUMCInstLower::AMDGPUMCInstLower(MCContext &ctx, const AMDGPUSubtarget &st):
  Ctx(ctx), ST(st) { }

static MCSymbolRefExpr::VariantKind getVariantKind(unsigned MOFlags) {
  switch (MOFlags) {
  default: return MCSymbolRefExpr::VK_None;
  case SIInstrInfo::MO_GOTPCREL: return MCSymbolRefExpr::VK_GOTPCREL;
  }
}

void AMDGPUMCInstLower::lower(const MachineInstr *MI, MCInst &OutMI) const {

  int MCOpcode = ST.getInstrInfo()->pseudoToMCOpcode(MI->getOpcode());

  if (MCOpcode == -1) {
    LLVMContext &C = MI->getParent()->getParent()->getFunction()->getContext();
    C.emitError("AMDGPUMCInstLower::lower - Pseudo instruction doesn't have "
                "a target-specific version: " + Twine(MI->getOpcode()));
  }

  OutMI.setOpcode(MCOpcode);

  for (const MachineOperand &MO : MI->explicit_operands()) {
    MCOperand MCOp;
    switch (MO.getType()) {
    default:
      llvm_unreachable("unknown operand type");
    case MachineOperand::MO_Immediate:
      MCOp = MCOperand::createImm(MO.getImm());
      break;
    case MachineOperand::MO_Register:
      MCOp = MCOperand::createReg(AMDGPU::getMCReg(MO.getReg(), ST));
      break;
    case MachineOperand::MO_MachineBasicBlock:
      MCOp = MCOperand::createExpr(MCSymbolRefExpr::create(
                                   MO.getMBB()->getSymbol(), Ctx));
      break;
    case MachineOperand::MO_GlobalAddress: {
      const GlobalValue *GV = MO.getGlobal();
      MCSymbol *Sym = Ctx.getOrCreateSymbol(StringRef(GV->getName()));
      const MCExpr *SymExpr =
          MCSymbolRefExpr::create(Sym, getVariantKind(MO.getTargetFlags()),Ctx);
      const MCExpr *Expr = MCBinaryExpr::createAdd(SymExpr,
          MCConstantExpr::create(MO.getOffset(), Ctx), Ctx);
      MCOp = MCOperand::createExpr(Expr);
      break;
    }
    case MachineOperand::MO_ExternalSymbol: {
      MCSymbol *Sym = Ctx.getOrCreateSymbol(StringRef(MO.getSymbolName()));
      Sym->setExternal(true);
      const MCSymbolRefExpr *Expr = MCSymbolRefExpr::create(Sym, Ctx);
      MCOp = MCOperand::createExpr(Expr);
      break;
    }
    }
    OutMI.addOperand(MCOp);
  }
}

void AMDGPUAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  const AMDGPUSubtarget &STI = MF->getSubtarget<AMDGPUSubtarget>();
  AMDGPUMCInstLower MCInstLowering(OutContext, STI);

  StringRef Err;
  if (!STI.getInstrInfo()->verifyInstruction(*MI, Err)) {
    LLVMContext &C = MI->getParent()->getParent()->getFunction()->getContext();
    C.emitError("Illegal instruction detected: " + Err);
    MI->dump();
  }

  if (MI->isBundle()) {
    const MachineBasicBlock *MBB = MI->getParent();
    MachineBasicBlock::const_instr_iterator I = ++MI->getIterator();
    while (I != MBB->instr_end() && I->isInsideBundle()) {
      EmitInstruction(&*I);
      ++I;
    }
  } else {
    // We don't want SI_MASK_BRANCH/SI_RETURN encoded. They are placeholder
    // terminator instructions and should only be printed as comments.
    if (MI->getOpcode() == AMDGPU::SI_MASK_BRANCH) {
      if (isVerbose()) {
        SmallVector<char, 16> BBStr;
        raw_svector_ostream Str(BBStr);

        const MachineBasicBlock *MBB = MI->getOperand(0).getMBB();
        const MCSymbolRefExpr *Expr
          = MCSymbolRefExpr::create(MBB->getSymbol(), OutContext);
        Expr->print(Str, MAI);
        OutStreamer->emitRawComment(" mask branch " + BBStr);
      }

      return;
    }

    if (MI->getOpcode() == AMDGPU::SI_RETURN) {
      if (isVerbose())
        OutStreamer->emitRawComment(" return");
      return;
    }

    MCInst TmpInst;
    MCInstLowering.lower(MI, TmpInst);
    EmitToStreamer(*OutStreamer, TmpInst);

    if (STI.dumpCode()) {
      // Disassemble instruction/operands to text.
      DisasmLines.resize(DisasmLines.size() + 1);
      std::string &DisasmLine = DisasmLines.back();
      raw_string_ostream DisasmStream(DisasmLine);

      AMDGPUInstPrinter InstPrinter(*TM.getMCAsmInfo(),
                                    *STI.getInstrInfo(),
                                    *STI.getRegisterInfo());
      InstPrinter.printInst(&TmpInst, DisasmStream, StringRef(), STI);

      // Disassemble instruction/operands to hex representation.
      SmallVector<MCFixup, 4> Fixups;
      SmallVector<char, 16> CodeBytes;
      raw_svector_ostream CodeStream(CodeBytes);

      auto &ObjStreamer = static_cast<MCObjectStreamer&>(*OutStreamer);
      MCCodeEmitter &InstEmitter = ObjStreamer.getAssembler().getEmitter();
      InstEmitter.encodeInstruction(TmpInst, CodeStream, Fixups,
                                    MF->getSubtarget<MCSubtargetInfo>());
      HexLines.resize(HexLines.size() + 1);
      std::string &HexLine = HexLines.back();
      raw_string_ostream HexStream(HexLine);

      for (size_t i = 0; i < CodeBytes.size(); i += 4) {
        unsigned int CodeDWord = *(unsigned int *)&CodeBytes[i];
        HexStream << format("%s%08X", (i > 0 ? " " : ""), CodeDWord);
      }

      DisasmStream.flush();
      DisasmLineMaxLen = std::max(DisasmLineMaxLen, DisasmLine.size());
    }
  }
}
