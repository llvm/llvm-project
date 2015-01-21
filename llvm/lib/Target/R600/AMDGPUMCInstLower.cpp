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
#include "AMDGPUTargetMachine.h"
#include "InstPrinter/AMDGPUInstPrinter.h"
#include "R600InstrInfo.h"
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
  Ctx(ctx), ST(st)
{ }

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
      MCOp = MCOperand::CreateImm(MO.getImm());
      break;
    case MachineOperand::MO_Register:
      MCOp = MCOperand::CreateReg(MO.getReg());
      break;
    case MachineOperand::MO_MachineBasicBlock:
      MCOp = MCOperand::CreateExpr(MCSymbolRefExpr::Create(
                                   MO.getMBB()->getSymbol(), Ctx));
      break;
    case MachineOperand::MO_GlobalAddress: {
      const GlobalValue *GV = MO.getGlobal();
      MCSymbol *Sym = Ctx.GetOrCreateSymbol(StringRef(GV->getName()));
      MCOp = MCOperand::CreateExpr(MCSymbolRefExpr::Create(Sym, Ctx));
      break;
    }
    case MachineOperand::MO_TargetIndex: {
      assert(MO.getIndex() == AMDGPU::TI_CONSTDATA_START);
      MCSymbol *Sym = Ctx.GetOrCreateSymbol(StringRef(END_OF_TEXT_LABEL_NAME));
      const MCSymbolRefExpr *Expr = MCSymbolRefExpr::Create(Sym, Ctx);
      MCOp = MCOperand::CreateExpr(Expr);
      break;
    }
    case MachineOperand::MO_ExternalSymbol: {
      MCSymbol *Sym = Ctx.GetOrCreateSymbol(StringRef(MO.getSymbolName()));
      const MCSymbolRefExpr *Expr = MCSymbolRefExpr::Create(Sym, Ctx);
      MCOp = MCOperand::CreateExpr(Expr);
      break;
    }
    }
    OutMI.addOperand(MCOp);
  }
}

void AMDGPUAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  AMDGPUMCInstLower MCInstLowering(OutContext,
                               MF->getTarget().getSubtarget<AMDGPUSubtarget>());

#ifdef _DEBUG
  StringRef Err;
  if (!TM.getSubtargetImpl()->getInstrInfo()->verifyInstruction(MI, Err)) {
    errs() << "Warning: Illegal instruction detected: " << Err << "\n";
    MI->dump();
  }
#endif
  if (MI->isBundle()) {
    const MachineBasicBlock *MBB = MI->getParent();
    MachineBasicBlock::const_instr_iterator I = MI;
    ++I;
    while (I != MBB->end() && I->isInsideBundle()) {
      EmitInstruction(I);
      ++I;
    }
  } else {
    MCInst TmpInst;
    MCInstLowering.lower(MI, TmpInst);
    EmitToStreamer(OutStreamer, TmpInst);

    if (DisasmEnabled) {
      // Disassemble instruction/operands to text.
      DisasmLines.resize(DisasmLines.size() + 1);
      std::string &DisasmLine = DisasmLines.back();
      raw_string_ostream DisasmStream(DisasmLine);

      AMDGPUInstPrinter InstPrinter(*TM.getMCAsmInfo(),
                                    *TM.getSubtargetImpl()->getInstrInfo(),
                                    *TM.getSubtargetImpl()->getRegisterInfo());
      InstPrinter.printInst(&TmpInst, DisasmStream, StringRef());

      // Disassemble instruction/operands to hex representation.
      SmallVector<MCFixup, 4> Fixups;
      SmallVector<char, 16> CodeBytes;
      raw_svector_ostream CodeStream(CodeBytes);

      MCObjectStreamer &ObjStreamer = (MCObjectStreamer &)OutStreamer;
      MCCodeEmitter &InstEmitter = ObjStreamer.getAssembler().getEmitter();
      InstEmitter.EncodeInstruction(TmpInst, CodeStream, Fixups,
                                    TM.getSubtarget<MCSubtargetInfo>());
      CodeStream.flush();

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
