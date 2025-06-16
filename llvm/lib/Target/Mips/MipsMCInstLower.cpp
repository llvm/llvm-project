//===- MipsMCInstLower.cpp - Convert Mips MachineInstr to MCInst ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains code to lower Mips MachineInstrs to their corresponding
// MCInst records.
//
//===----------------------------------------------------------------------===//

#include "MipsMCInstLower.h"
#include "MCTargetDesc/MipsBaseInfo.h"
#include "MCTargetDesc/MipsMCAsmInfo.h"
#include "MipsAsmPrinter.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

MipsMCInstLower::MipsMCInstLower(MipsAsmPrinter &asmprinter)
  : AsmPrinter(asmprinter) {}

void MipsMCInstLower::Initialize(MCContext *C) {
  Ctx = C;
}

MCOperand MipsMCInstLower::LowerSymbolOperand(const MachineOperand &MO,
                                              MachineOperandType MOTy,
                                              int64_t Offset) const {
  Mips::Specifier TargetKind = Mips::S_None;
  bool IsGpOff = false;
  const MCSymbol *Symbol;
  SmallString<128> Name;
  unsigned TargetFlags = MO.getTargetFlags();

  if (TargetFlags & MipsII::MO_DLLIMPORT) {
    // Handle dllimport linkage
    Name += "__imp_";
    TargetFlags &= ~MipsII::MO_DLLIMPORT;
  }

  switch (TargetFlags) {
  default:
    llvm_unreachable("Invalid target flag!");
  case MipsII::MO_NO_FLAG:
    break;
  case MipsII::MO_GPREL:
    TargetKind = Mips::S_GPREL;
    break;
  case MipsII::MO_GOT_CALL:
    TargetKind = Mips::S_GOT_CALL;
    break;
  case MipsII::MO_GOT:
    TargetKind = Mips::S_GOT;
    break;
  case MipsII::MO_ABS_HI:
    TargetKind = Mips::S_HI;
    break;
  case MipsII::MO_ABS_LO:
    TargetKind = Mips::S_LO;
    break;
  case MipsII::MO_TLSGD:
    TargetKind = Mips::S_TLSGD;
    break;
  case MipsII::MO_TLSLDM:
    TargetKind = Mips::S_TLSLDM;
    break;
  case MipsII::MO_DTPREL_HI:
    TargetKind = Mips::S_DTPREL_HI;
    break;
  case MipsII::MO_DTPREL_LO:
    TargetKind = Mips::S_DTPREL_LO;
    break;
  case MipsII::MO_GOTTPREL:
    TargetKind = Mips::S_GOTTPREL;
    break;
  case MipsII::MO_TPREL_HI:
    TargetKind = Mips::S_TPREL_HI;
    break;
  case MipsII::MO_TPREL_LO:
    TargetKind = Mips::S_TPREL_LO;
    break;
  case MipsII::MO_GPOFF_HI:
    TargetKind = Mips::S_HI;
    IsGpOff = true;
    break;
  case MipsII::MO_GPOFF_LO:
    TargetKind = Mips::S_LO;
    IsGpOff = true;
    break;
  case MipsII::MO_GOT_DISP:
    TargetKind = Mips::S_GOT_DISP;
    break;
  case MipsII::MO_GOT_HI16:
    TargetKind = Mips::S_GOT_HI16;
    break;
  case MipsII::MO_GOT_LO16:
    TargetKind = Mips::S_GOT_LO16;
    break;
  case MipsII::MO_GOT_PAGE:
    TargetKind = Mips::S_GOT_PAGE;
    break;
  case MipsII::MO_GOT_OFST:
    TargetKind = Mips::S_GOT_OFST;
    break;
  case MipsII::MO_HIGHER:
    TargetKind = Mips::S_HIGHER;
    break;
  case MipsII::MO_HIGHEST:
    TargetKind = Mips::S_HIGHEST;
    break;
  case MipsII::MO_CALL_HI16:
    TargetKind = Mips::S_CALL_HI16;
    break;
  case MipsII::MO_CALL_LO16:
    TargetKind = Mips::S_CALL_LO16;
    break;
  case MipsII::MO_JALR:
    return MCOperand();
  }

  switch (MOTy) {
  case MachineOperand::MO_MachineBasicBlock:
    Symbol = MO.getMBB()->getSymbol();
    break;

  case MachineOperand::MO_GlobalAddress:
    AsmPrinter.getNameWithPrefix(Name, MO.getGlobal());
    Symbol = Ctx->getOrCreateSymbol(Name);
    Offset += MO.getOffset();
    break;

  case MachineOperand::MO_BlockAddress:
    Symbol = AsmPrinter.GetBlockAddressSymbol(MO.getBlockAddress());
    Offset += MO.getOffset();
    break;

  case MachineOperand::MO_ExternalSymbol:
    Symbol = AsmPrinter.GetExternalSymbolSymbol(MO.getSymbolName());
    Offset += MO.getOffset();
    break;

  case MachineOperand::MO_MCSymbol:
    Symbol = MO.getMCSymbol();
    Offset += MO.getOffset();
    break;

  case MachineOperand::MO_JumpTableIndex:
    Symbol = AsmPrinter.GetJTISymbol(MO.getIndex());
    break;

  case MachineOperand::MO_ConstantPoolIndex:
    Symbol = AsmPrinter.GetCPISymbol(MO.getIndex());
    Offset += MO.getOffset();
    break;

  default:
    llvm_unreachable("<unknown operand type>");
  }

  const MCExpr *Expr = MCSymbolRefExpr::create(Symbol, *Ctx);

  if (Offset) {
    // Note: Offset can also be negative
    Expr = MCBinaryExpr::createAdd(Expr, MCConstantExpr::create(Offset, *Ctx),
                                   *Ctx);
  }

  if (IsGpOff)
    Expr = MipsMCExpr::createGpOff(TargetKind, Expr, *Ctx);
  else if (TargetKind != Mips::S_None)
    Expr = MipsMCExpr::create(TargetKind, Expr, *Ctx);

  return MCOperand::createExpr(Expr);
}

MCOperand MipsMCInstLower::LowerOperand(const MachineOperand &MO,
                                        int64_t offset) const {
  MachineOperandType MOTy = MO.getType();

  switch (MOTy) {
  default: llvm_unreachable("unknown operand type");
  case MachineOperand::MO_Register:
    // Ignore all implicit register operands.
    if (MO.isImplicit()) break;
    return MCOperand::createReg(MO.getReg());
  case MachineOperand::MO_Immediate:
    return MCOperand::createImm(MO.getImm() + offset);
  case MachineOperand::MO_MachineBasicBlock:
  case MachineOperand::MO_GlobalAddress:
  case MachineOperand::MO_ExternalSymbol:
  case MachineOperand::MO_MCSymbol:
  case MachineOperand::MO_JumpTableIndex:
  case MachineOperand::MO_ConstantPoolIndex:
  case MachineOperand::MO_BlockAddress:
    return LowerSymbolOperand(MO, MOTy, offset);
  case MachineOperand::MO_RegisterMask:
    break;
 }

  return MCOperand();
}

MCOperand MipsMCInstLower::createSub(MachineBasicBlock *BB1,
                                     MachineBasicBlock *BB2,
                                     Mips::Specifier Kind) const {
  const MCSymbolRefExpr *Sym1 = MCSymbolRefExpr::create(BB1->getSymbol(), *Ctx);
  const MCSymbolRefExpr *Sym2 = MCSymbolRefExpr::create(BB2->getSymbol(), *Ctx);
  const MCBinaryExpr *Sub = MCBinaryExpr::createSub(Sym1, Sym2, *Ctx);

  return MCOperand::createExpr(MipsMCExpr::create(Kind, Sub, *Ctx));
}

void MipsMCInstLower::
lowerLongBranchLUi(const MachineInstr *MI, MCInst &OutMI) const {
  OutMI.setOpcode(Mips::LUi);

  // Lower register operand.
  OutMI.addOperand(LowerOperand(MI->getOperand(0)));

  Mips::Specifier Spec;
  unsigned TargetFlags = MI->getOperand(1).getTargetFlags();
  switch (TargetFlags) {
  case MipsII::MO_HIGHEST:
    Spec = Mips::S_HIGHEST;
    break;
  case MipsII::MO_HIGHER:
    Spec = Mips::S_HIGHER;
    break;
  case MipsII::MO_ABS_HI:
    Spec = Mips::S_HI;
    break;
  case MipsII::MO_ABS_LO:
    Spec = Mips::S_LO;
    break;
  default:
    report_fatal_error("Unexpected flags for lowerLongBranchLUi");
  }

  if (MI->getNumOperands() == 2) {
    const MCExpr *Expr =
        MCSymbolRefExpr::create(MI->getOperand(1).getMBB()->getSymbol(), *Ctx);
    const auto *MipsExpr = MipsMCExpr::create(Spec, Expr, *Ctx);
    OutMI.addOperand(MCOperand::createExpr(MipsExpr));
  } else if (MI->getNumOperands() == 3) {
    // Create %hi($tgt-$baltgt).
    OutMI.addOperand(createSub(MI->getOperand(1).getMBB(),
                               MI->getOperand(2).getMBB(), Spec));
  }
}

void MipsMCInstLower::lowerLongBranchADDiu(const MachineInstr *MI,
                                           MCInst &OutMI, int Opcode) const {
  OutMI.setOpcode(Opcode);

  Mips::Specifier Spec;
  unsigned TargetFlags = MI->getOperand(2).getTargetFlags();
  switch (TargetFlags) {
  case MipsII::MO_HIGHEST:
    Spec = Mips::S_HIGHEST;
    break;
  case MipsII::MO_HIGHER:
    Spec = Mips::S_HIGHER;
    break;
  case MipsII::MO_ABS_HI:
    Spec = Mips::S_HI;
    break;
  case MipsII::MO_ABS_LO:
    Spec = Mips::S_LO;
    break;
  default:
    report_fatal_error("Unexpected flags for lowerLongBranchADDiu");
  }

  // Lower two register operands.
  for (unsigned I = 0, E = 2; I != E; ++I) {
    const MachineOperand &MO = MI->getOperand(I);
    OutMI.addOperand(LowerOperand(MO));
  }

  if (MI->getNumOperands() == 3) {
    // Lower register operand.
    const MCExpr *Expr =
        MCSymbolRefExpr::create(MI->getOperand(2).getMBB()->getSymbol(), *Ctx);
    const auto *MipsExpr = MipsMCExpr::create(Spec, Expr, *Ctx);
    OutMI.addOperand(MCOperand::createExpr(MipsExpr));
  } else if (MI->getNumOperands() == 4) {
    // Create %lo($tgt-$baltgt) or %hi($tgt-$baltgt).
    OutMI.addOperand(createSub(MI->getOperand(2).getMBB(),
                               MI->getOperand(3).getMBB(), Spec));
  }
}

bool MipsMCInstLower::lowerLongBranch(const MachineInstr *MI,
                                      MCInst &OutMI) const {
  switch (MI->getOpcode()) {
  default:
    return false;
  case Mips::LONG_BRANCH_LUi:
  case Mips::LONG_BRANCH_LUi2Op:
  case Mips::LONG_BRANCH_LUi2Op_64:
    lowerLongBranchLUi(MI, OutMI);
    return true;
  case Mips::LONG_BRANCH_ADDiu:
  case Mips::LONG_BRANCH_ADDiu2Op:
    lowerLongBranchADDiu(MI, OutMI, Mips::ADDiu);
    return true;
  case Mips::LONG_BRANCH_DADDiu:
  case Mips::LONG_BRANCH_DADDiu2Op:
    lowerLongBranchADDiu(MI, OutMI, Mips::DADDiu);
    return true;
  }
}

void MipsMCInstLower::Lower(const MachineInstr *MI, MCInst &OutMI) const {
  if (lowerLongBranch(MI, OutMI))
    return;

  OutMI.setOpcode(MI->getOpcode());

  for (const MachineOperand &MO : MI->operands()) {
    MCOperand MCOp = LowerOperand(MO);

    if (MCOp.isValid())
      OutMI.addOperand(MCOp);
  }
}
