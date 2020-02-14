//===-- VEAsmPrinter.cpp - VE LLVM assembly writer ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to GAS-format VE assembly language.
//
//===----------------------------------------------------------------------===//

#include "InstPrinter/VEInstPrinter.h"
#include "MCTargetDesc/VEMCExpr.h"
#include "MCTargetDesc/VETargetStreamer.h"
#include "VE.h"
#include "VEInstrInfo.h"
#include "VETargetMachine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "ve-asmprinter"

namespace {
class VEAsmPrinter : public AsmPrinter {
  VETargetStreamer &getTargetStreamer() {
    return static_cast<VETargetStreamer &>(*OutStreamer->getTargetStreamer());
  }

public:
  explicit VEAsmPrinter(TargetMachine &TM, std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)) {}

  StringRef getPassName() const override { return "VE Assembly Printer"; }

  void lowerGETGOTAndEmitMCInsts(const MachineInstr *MI,
                                 const MCSubtargetInfo &STI);
  void lowerGETFunPLTAndEmitMCInsts(const MachineInstr *MI,
                                    const MCSubtargetInfo &STI);

  void emitInstruction(const MachineInstr *MI) override;

  static const char *getRegisterName(unsigned RegNo) {
    return VEInstPrinter::getRegisterName(RegNo);
  }
};
} // end of anonymous namespace

static MCOperand createVEMCOperand(VEMCExpr::VariantKind Kind, MCSymbol *Sym,
                                   MCContext &OutContext) {
  const MCSymbolRefExpr *MCSym = MCSymbolRefExpr::create(Sym, OutContext);
  const VEMCExpr *expr = VEMCExpr::create(Kind, MCSym, OutContext);
  return MCOperand::createExpr(expr);
}

static MCOperand createGOTRelExprOp(VEMCExpr::VariantKind Kind,
                                    MCSymbol *GOTLabel, MCContext &OutContext) {
  const MCSymbolRefExpr *GOT = MCSymbolRefExpr::create(GOTLabel, OutContext);
  const VEMCExpr *expr = VEMCExpr::create(Kind, GOT, OutContext);
  return MCOperand::createExpr(expr);
}

static void emitSIC(MCStreamer &OutStreamer, MCOperand &RD,
                    const MCSubtargetInfo &STI) {
  MCInst SICInst;
  SICInst.setOpcode(VE::SIC);
  SICInst.addOperand(RD);
  OutStreamer.emitInstruction(SICInst, STI);
}

static void emitLEAzzi(MCStreamer &OutStreamer, MCOperand &Imm, MCOperand &RD,
                       const MCSubtargetInfo &STI) {
  MCInst LEAInst;
  LEAInst.setOpcode(VE::LEAzzi);
  LEAInst.addOperand(RD);
  LEAInst.addOperand(Imm);
  OutStreamer.emitInstruction(LEAInst, STI);
}

static void emitLEASLzzi(MCStreamer &OutStreamer, MCOperand &Imm, MCOperand &RD,
                         const MCSubtargetInfo &STI) {
  MCInst LEASLInst;
  LEASLInst.setOpcode(VE::LEASLzzi);
  LEASLInst.addOperand(RD);
  LEASLInst.addOperand(Imm);
  OutStreamer.emitInstruction(LEASLInst, STI);
}

static void emitLEAzii(MCStreamer &OutStreamer, MCOperand &RS1, MCOperand &Imm,
                       MCOperand &RD, const MCSubtargetInfo &STI) {
  MCInst LEAInst;
  LEAInst.setOpcode(VE::LEAzii);
  LEAInst.addOperand(RD);
  LEAInst.addOperand(RS1);
  LEAInst.addOperand(Imm);
  OutStreamer.emitInstruction(LEAInst, STI);
}

static void emitLEASLrri(MCStreamer &OutStreamer, MCOperand &RS1,
                         MCOperand &RS2, MCOperand &Imm, MCOperand &RD,
                         const MCSubtargetInfo &STI) {
  MCInst LEASLInst;
  LEASLInst.setOpcode(VE::LEASLrri);
  LEASLInst.addOperand(RS1);
  LEASLInst.addOperand(RS2);
  LEASLInst.addOperand(RD);
  LEASLInst.addOperand(Imm);
  OutStreamer.emitInstruction(LEASLInst, STI);
}

static void emitBinary(MCStreamer &OutStreamer, unsigned Opcode, MCOperand &RS1,
                       MCOperand &Src2, MCOperand &RD,
                       const MCSubtargetInfo &STI) {
  MCInst Inst;
  Inst.setOpcode(Opcode);
  Inst.addOperand(RD);
  Inst.addOperand(RS1);
  Inst.addOperand(Src2);
  OutStreamer.emitInstruction(Inst, STI);
}

static void emitANDrm0(MCStreamer &OutStreamer, MCOperand &RS1, MCOperand &Imm,
                       MCOperand &RD, const MCSubtargetInfo &STI) {
  emitBinary(OutStreamer, VE::ANDrm0, RS1, Imm, RD, STI);
}

static void emitHiLo(MCStreamer &OutStreamer, MCSymbol *GOTSym,
                     VEMCExpr::VariantKind HiKind, VEMCExpr::VariantKind LoKind,
                     MCOperand &RD, MCContext &OutContext,
                     const MCSubtargetInfo &STI) {

  MCOperand hi = createVEMCOperand(HiKind, GOTSym, OutContext);
  MCOperand lo = createVEMCOperand(LoKind, GOTSym, OutContext);
  MCOperand ci32 = MCOperand::createImm(32);
  emitLEAzzi(OutStreamer, lo, RD, STI);
  emitANDrm0(OutStreamer, RD, ci32, RD, STI);
  emitLEASLzzi(OutStreamer, hi, RD, STI);
}

void VEAsmPrinter::lowerGETGOTAndEmitMCInsts(const MachineInstr *MI,
                                             const MCSubtargetInfo &STI) {
  MCSymbol *GOTLabel =
      OutContext.getOrCreateSymbol(Twine("_GLOBAL_OFFSET_TABLE_"));

  const MachineOperand &MO = MI->getOperand(0);
  MCOperand MCRegOP = MCOperand::createReg(MO.getReg());

  if (!isPositionIndependent()) {
    // Just load the address of GOT to MCRegOP.
    switch (TM.getCodeModel()) {
    default:
      llvm_unreachable("Unsupported absolute code model");
    case CodeModel::Small:
    case CodeModel::Medium:
    case CodeModel::Large:
      emitHiLo(*OutStreamer, GOTLabel, VEMCExpr::VK_VE_HI32,
               VEMCExpr::VK_VE_LO32, MCRegOP, OutContext, STI);
      break;
    }
    return;
  }

  MCOperand RegGOT = MCOperand::createReg(VE::SX15); // GOT
  MCOperand RegPLT = MCOperand::createReg(VE::SX16); // PLT

  // lea %got, _GLOBAL_OFFSET_TABLE_@PC_LO(-24)
  // and %got, %got, (32)0
  // sic %plt
  // lea.sl %got, _GLOBAL_OFFSET_TABLE_@PC_HI(%got, %plt)
  MCOperand cim24 = MCOperand::createImm(-24);
  MCOperand loImm =
      createGOTRelExprOp(VEMCExpr::VK_VE_PC_LO32, GOTLabel, OutContext);
  emitLEAzii(*OutStreamer, cim24, loImm, MCRegOP, STI);
  MCOperand ci32 = MCOperand::createImm(32);
  emitANDrm0(*OutStreamer, MCRegOP, ci32, MCRegOP, STI);
  emitSIC(*OutStreamer, RegPLT, STI);
  MCOperand hiImm =
      createGOTRelExprOp(VEMCExpr::VK_VE_PC_HI32, GOTLabel, OutContext);
  emitLEASLrri(*OutStreamer, RegGOT, RegPLT, hiImm, MCRegOP, STI);
}

void VEAsmPrinter::lowerGETFunPLTAndEmitMCInsts(const MachineInstr *MI,
                                                const MCSubtargetInfo &STI) {
  const MachineOperand &MO = MI->getOperand(0);
  MCOperand MCRegOP = MCOperand::createReg(MO.getReg());
  const MachineOperand &Addr = MI->getOperand(1);
  MCSymbol *AddrSym = nullptr;

  switch (Addr.getType()) {
  default:
    llvm_unreachable("<unknown operand type>");
    return;
  case MachineOperand::MO_MachineBasicBlock:
    report_fatal_error("MBB is not supported yet");
    return;
  case MachineOperand::MO_ConstantPoolIndex:
    report_fatal_error("ConstantPool is not supported yet");
    return;
  case MachineOperand::MO_ExternalSymbol:
    AddrSym = GetExternalSymbolSymbol(Addr.getSymbolName());
    break;
  case MachineOperand::MO_GlobalAddress:
    AddrSym = getSymbol(Addr.getGlobal());
    break;
  }

  if (!isPositionIndependent()) {
    llvm_unreachable("Unsupported uses of %plt in not PIC code");
    return;
  }

  MCOperand RegPLT = MCOperand::createReg(VE::SX16); // PLT

  // lea %dst, %plt_lo(func)(-24)
  // and %dst, %dst, (32)0
  // sic %plt                            ; FIXME: is it safe to use %plt here?
  // lea.sl %dst, %plt_hi(func)(%dst, %plt)
  MCOperand cim24 = MCOperand::createImm(-24);
  MCOperand loImm =
      createGOTRelExprOp(VEMCExpr::VK_VE_PLT_LO32, AddrSym, OutContext);
  emitLEAzii(*OutStreamer, cim24, loImm, MCRegOP, STI);
  MCOperand ci32 = MCOperand::createImm(32);
  emitANDrm0(*OutStreamer, MCRegOP, ci32, MCRegOP, STI);
  emitSIC(*OutStreamer, RegPLT, STI);
  MCOperand hiImm =
      createGOTRelExprOp(VEMCExpr::VK_VE_PLT_HI32, AddrSym, OutContext);
  emitLEASLrri(*OutStreamer, MCRegOP, RegPLT, hiImm, MCRegOP, STI);
}

void VEAsmPrinter::emitInstruction(const MachineInstr *MI) {

  switch (MI->getOpcode()) {
  default:
    break;
  case TargetOpcode::DBG_VALUE:
    // FIXME: Debug Value.
    return;
  case VE::GETGOT:
    lowerGETGOTAndEmitMCInsts(MI, getSubtargetInfo());
    return;
  case VE::GETFUNPLT:
    lowerGETFunPLTAndEmitMCInsts(MI, getSubtargetInfo());
    return;
  }

  MachineBasicBlock::const_instr_iterator I = MI->getIterator();
  MachineBasicBlock::const_instr_iterator E = MI->getParent()->instr_end();
  do {
    MCInst TmpInst;
    LowerVEMachineInstrToMCInst(&*I, TmpInst, *this);
    EmitToStreamer(*OutStreamer, TmpInst);
  } while ((++I != E) && I->isInsideBundle()); // Delay slot check.
}

// Force static initialization.
extern "C" void LLVMInitializeVEAsmPrinter() {
  RegisterAsmPrinter<VEAsmPrinter> X(getTheVETarget());
}
