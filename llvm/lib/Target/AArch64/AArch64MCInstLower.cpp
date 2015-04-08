//==-- AArch64MCInstLower.cpp - Convert AArch64 MachineInstr to an MCInst --==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains code to lower AArch64 MachineInstrs to their corresponding
// MCInst records.
//
//===----------------------------------------------------------------------===//

#include "AArch64MCInstLower.h"
#include "MCTargetDesc/AArch64MCExpr.h"
#include "Utils/AArch64BaseInfo.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

extern cl::opt<bool> EnableAArch64ELFLocalDynamicTLSGeneration;

AArch64MCInstLower::AArch64MCInstLower(MCContext &ctx, AsmPrinter &printer)
    : Ctx(ctx), Printer(printer), TargetTriple(printer.getTargetTriple()) {}

MCSymbol *
AArch64MCInstLower::GetGlobalAddressSymbol(const MachineOperand &MO) const {
  return Printer.getSymbol(MO.getGlobal());
}

MCSymbol *
AArch64MCInstLower::GetExternalSymbolSymbol(const MachineOperand &MO) const {
  return Printer.GetExternalSymbolSymbol(MO.getSymbolName());
}

MCOperand AArch64MCInstLower::lowerSymbolOperandDarwin(const MachineOperand &MO,
                                                       MCSymbol *Sym) const {
  // FIXME: We would like an efficient form for this, so we don't have to do a
  // lot of extra uniquing.
  MCSymbolRefExpr::VariantKind RefKind = MCSymbolRefExpr::VK_None;
  if ((MO.getTargetFlags() & AArch64II::MO_GOT) != 0) {
    if ((MO.getTargetFlags() & AArch64II::MO_FRAGMENT) == AArch64II::MO_PAGE)
      RefKind = MCSymbolRefExpr::VK_GOTPAGE;
    else if ((MO.getTargetFlags() & AArch64II::MO_FRAGMENT) ==
             AArch64II::MO_PAGEOFF)
      RefKind = MCSymbolRefExpr::VK_GOTPAGEOFF;
    else
      llvm_unreachable("Unexpected target flags with MO_GOT on GV operand");
  } else if ((MO.getTargetFlags() & AArch64II::MO_TLS) != 0) {
    if ((MO.getTargetFlags() & AArch64II::MO_FRAGMENT) == AArch64II::MO_PAGE)
      RefKind = MCSymbolRefExpr::VK_TLVPPAGE;
    else if ((MO.getTargetFlags() & AArch64II::MO_FRAGMENT) ==
             AArch64II::MO_PAGEOFF)
      RefKind = MCSymbolRefExpr::VK_TLVPPAGEOFF;
    else
      llvm_unreachable("Unexpected target flags with MO_TLS on GV operand");
  } else {
    if ((MO.getTargetFlags() & AArch64II::MO_FRAGMENT) == AArch64II::MO_PAGE)
      RefKind = MCSymbolRefExpr::VK_PAGE;
    else if ((MO.getTargetFlags() & AArch64II::MO_FRAGMENT) ==
             AArch64II::MO_PAGEOFF)
      RefKind = MCSymbolRefExpr::VK_PAGEOFF;
  }
  const MCExpr *Expr = MCSymbolRefExpr::Create(Sym, RefKind, Ctx);
  if (!MO.isJTI() && MO.getOffset())
    Expr = MCBinaryExpr::CreateAdd(
        Expr, MCConstantExpr::Create(MO.getOffset(), Ctx), Ctx);
  return MCOperand::CreateExpr(Expr);
}

MCOperand AArch64MCInstLower::lowerSymbolOperandELF(const MachineOperand &MO,
                                                    MCSymbol *Sym) const {
  uint32_t RefFlags = 0;

  if (MO.getTargetFlags() & AArch64II::MO_GOT)
    RefFlags |= AArch64MCExpr::VK_GOT;
  else if (MO.getTargetFlags() & AArch64II::MO_TLS) {
    TLSModel::Model Model;
    if (MO.isGlobal()) {
      const GlobalValue *GV = MO.getGlobal();
      Model = Printer.TM.getTLSModel(GV);
      if (!EnableAArch64ELFLocalDynamicTLSGeneration &&
          Model == TLSModel::LocalDynamic)
        Model = TLSModel::GeneralDynamic;

    } else {
      assert(MO.isSymbol() &&
             StringRef(MO.getSymbolName()) == "_TLS_MODULE_BASE_" &&
             "unexpected external TLS symbol");
      // The general dynamic access sequence is used to get the
      // address of _TLS_MODULE_BASE_.
      Model = TLSModel::GeneralDynamic;
    }
    switch (Model) {
    case TLSModel::InitialExec:
      RefFlags |= AArch64MCExpr::VK_GOTTPREL;
      break;
    case TLSModel::LocalExec:
      RefFlags |= AArch64MCExpr::VK_TPREL;
      break;
    case TLSModel::LocalDynamic:
      RefFlags |= AArch64MCExpr::VK_DTPREL;
      break;
    case TLSModel::GeneralDynamic:
      RefFlags |= AArch64MCExpr::VK_TLSDESC;
      break;
    }
  } else {
    // No modifier means this is a generic reference, classified as absolute for
    // the cases where it matters (:abs_g0: etc).
    RefFlags |= AArch64MCExpr::VK_ABS;
  }

  if ((MO.getTargetFlags() & AArch64II::MO_FRAGMENT) == AArch64II::MO_PAGE)
    RefFlags |= AArch64MCExpr::VK_PAGE;
  else if ((MO.getTargetFlags() & AArch64II::MO_FRAGMENT) ==
           AArch64II::MO_PAGEOFF)
    RefFlags |= AArch64MCExpr::VK_PAGEOFF;
  else if ((MO.getTargetFlags() & AArch64II::MO_FRAGMENT) == AArch64II::MO_G3)
    RefFlags |= AArch64MCExpr::VK_G3;
  else if ((MO.getTargetFlags() & AArch64II::MO_FRAGMENT) == AArch64II::MO_G2)
    RefFlags |= AArch64MCExpr::VK_G2;
  else if ((MO.getTargetFlags() & AArch64II::MO_FRAGMENT) == AArch64II::MO_G1)
    RefFlags |= AArch64MCExpr::VK_G1;
  else if ((MO.getTargetFlags() & AArch64II::MO_FRAGMENT) == AArch64II::MO_G0)
    RefFlags |= AArch64MCExpr::VK_G0;
  else if ((MO.getTargetFlags() & AArch64II::MO_FRAGMENT) == AArch64II::MO_HI12)
    RefFlags |= AArch64MCExpr::VK_HI12;

  if (MO.getTargetFlags() & AArch64II::MO_NC)
    RefFlags |= AArch64MCExpr::VK_NC;

  const MCExpr *Expr =
      MCSymbolRefExpr::Create(Sym, MCSymbolRefExpr::VK_None, Ctx);
  if (!MO.isJTI() && MO.getOffset())
    Expr = MCBinaryExpr::CreateAdd(
        Expr, MCConstantExpr::Create(MO.getOffset(), Ctx), Ctx);

  AArch64MCExpr::VariantKind RefKind;
  RefKind = static_cast<AArch64MCExpr::VariantKind>(RefFlags);
  Expr = AArch64MCExpr::Create(Expr, RefKind, Ctx);

  return MCOperand::CreateExpr(Expr);
}

MCOperand AArch64MCInstLower::LowerSymbolOperand(const MachineOperand &MO,
                                                 MCSymbol *Sym) const {
  if (TargetTriple.isOSDarwin())
    return lowerSymbolOperandDarwin(MO, Sym);

  assert(TargetTriple.isOSBinFormatELF() && "Expect Darwin or ELF target");
  return lowerSymbolOperandELF(MO, Sym);
}

bool AArch64MCInstLower::lowerOperand(const MachineOperand &MO,
                                      MCOperand &MCOp) const {
  switch (MO.getType()) {
  default:
    llvm_unreachable("unknown operand type");
  case MachineOperand::MO_Register:
    // Ignore all implicit register operands.
    if (MO.isImplicit())
      return false;
    MCOp = MCOperand::CreateReg(MO.getReg());
    break;
  case MachineOperand::MO_RegisterMask:
    // Regmasks are like implicit defs.
    return false;
  case MachineOperand::MO_Immediate:
    MCOp = MCOperand::CreateImm(MO.getImm());
    break;
  case MachineOperand::MO_MachineBasicBlock:
    MCOp = MCOperand::CreateExpr(
        MCSymbolRefExpr::Create(MO.getMBB()->getSymbol(), Ctx));
    break;
  case MachineOperand::MO_GlobalAddress:
    MCOp = LowerSymbolOperand(MO, GetGlobalAddressSymbol(MO));
    break;
  case MachineOperand::MO_ExternalSymbol:
    MCOp = LowerSymbolOperand(MO, GetExternalSymbolSymbol(MO));
    break;
  case MachineOperand::MO_JumpTableIndex:
    MCOp = LowerSymbolOperand(MO, Printer.GetJTISymbol(MO.getIndex()));
    break;
  case MachineOperand::MO_ConstantPoolIndex:
    MCOp = LowerSymbolOperand(MO, Printer.GetCPISymbol(MO.getIndex()));
    break;
  case MachineOperand::MO_BlockAddress:
    MCOp = LowerSymbolOperand(
        MO, Printer.GetBlockAddressSymbol(MO.getBlockAddress()));
    break;
  }
  return true;
}

void AArch64MCInstLower::Lower(const MachineInstr *MI, MCInst &OutMI) const {
  OutMI.setOpcode(MI->getOpcode());

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MCOperand MCOp;
    if (lowerOperand(MI->getOperand(i), MCOp))
      OutMI.addOperand(MCOp);
  }
}
