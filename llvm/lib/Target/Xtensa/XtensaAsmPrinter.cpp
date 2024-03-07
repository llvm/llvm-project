//===- XtensaAsmPrinter.cpp Xtensa LLVM Assembly Printer ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to GAS-format Xtensa assembly language.
//
//===----------------------------------------------------------------------===//

#include "XtensaAsmPrinter.h"
#include "MCTargetDesc/XtensaMCExpr.h"
#include "TargetInfo/XtensaTargetInfo.h"
#include "XtensaConstantPoolValue.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

static MCSymbolRefExpr::VariantKind
getModifierVariantKind(XtensaCP::XtensaCPModifier Modifier) {
  switch (Modifier) {
  case XtensaCP::no_modifier:
    return MCSymbolRefExpr::VK_None;
  case XtensaCP::TPOFF:
    return MCSymbolRefExpr::VK_TPOFF;
  }
  report_fatal_error("Invalid XtensaCPModifier!");
}

void XtensaAsmPrinter::emitInstruction(const MachineInstr *MI) {
  MCInst LoweredMI;
  lowerToMCInst(MI, LoweredMI);
  EmitToStreamer(*OutStreamer, LoweredMI);
}

void XtensaAsmPrinter::emitMachineConstantPoolValue(
    MachineConstantPoolValue *MCPV) {
  XtensaConstantPoolValue *ACPV = static_cast<XtensaConstantPoolValue *>(MCPV);
  MCSymbol *MCSym;

  assert(ACPV->isExtSymbol() && "unrecognized constant pool value");

  XtensaConstantPoolSymbol *XtensaSym = cast<XtensaConstantPoolSymbol>(ACPV);
  const char *Sym = XtensaSym->getSymbol();
  std::string SymName(Sym);
  if (XtensaSym->isPrivateLinkage())
    SymName = ".L" + SymName;
  MCSym = GetExternalSymbolSymbol(StringRef(SymName));

  MCSymbol *LblSym = GetCPISymbol(ACPV->getLabelId());
  // TODO find a better way to check whether we emit data to .s file
  if (OutStreamer->hasRawTextSupport()) {
    std::string SymName("\t.literal ");
    SymName += LblSym->getName();
    SymName += ", ";
    SymName += MCSym->getName();

    StringRef Modifier = ACPV->getModifierText();
    SymName += Modifier;

    OutStreamer->emitRawText(StringRef(SymName));
  } else {
    MCSymbolRefExpr::VariantKind VK =
        getModifierVariantKind(ACPV->getModifier());

    if (ACPV->getModifier() != XtensaCP::no_modifier) {
      std::string SymName(MCSym->getName());
      MCSym = GetExternalSymbolSymbol(StringRef(SymName));
    }

    const MCExpr *Expr = MCSymbolRefExpr::create(MCSym, VK, OutContext);
    uint64_t Size = getDataLayout().getTypeAllocSize(ACPV->getType());
    OutStreamer->emitCodeAlignment(
        Align(4), OutStreamer->getContext().getSubtargetInfo());
    OutStreamer->emitLabel(LblSym);
    OutStreamer->emitValue(Expr, Size);
  }
}

void XtensaAsmPrinter::emitMachineConstantPoolEntry(
    const MachineConstantPoolEntry &CPE, int i) {
  if (CPE.isMachineConstantPoolEntry()) {
    XtensaConstantPoolValue *ACPV =
        static_cast<XtensaConstantPoolValue *>(CPE.Val.MachineCPVal);
    ACPV->setLabelId(i);
    emitMachineConstantPoolValue(CPE.Val.MachineCPVal);
  } else {
    MCSymbol *LblSym = GetCPISymbol(i);
    // TODO find a better way to check whether we emit data to .s file
    if (OutStreamer->hasRawTextSupport()) {
      std::string str("\t.literal ");
      str += LblSym->getName();
      str += ", ";
      const Constant *C = CPE.Val.ConstVal;

      if (const auto *CFP = dyn_cast<ConstantFP>(C)) {
        str += toString(CFP->getValueAPF().bitcastToAPInt(), 10, true);
      } else if (const auto *CI = dyn_cast<ConstantInt>(C)) {
        str += toString(CI->getValue(), 10, true);
      } else {
        report_fatal_error(
            "This constant type is not supported yet in constantpool");
      }

      OutStreamer->emitRawText(StringRef(str));
    } else {
      OutStreamer->emitCodeAlignment(
          Align(4), OutStreamer->getContext().getSubtargetInfo());
      OutStreamer->emitLabel(LblSym);
      emitGlobalConstant(getDataLayout(), CPE.Val.ConstVal);
    }
  }
}

// EmitConstantPool - Print to the current output stream assembly
// representations of the constants in the constant pool MCP. This is
// used to print out constants which have been "spilled to memory" by
// the code generator.
void XtensaAsmPrinter::emitConstantPool() {
  const Function &F = MF->getFunction();
  const MachineConstantPool *MCP = MF->getConstantPool();
  const std::vector<MachineConstantPoolEntry> &CP = MCP->getConstants();
  if (CP.empty())
    return;

  for (unsigned i = 0, e = CP.size(); i != e; ++i) {
    const MachineConstantPoolEntry &CPE = CP[i];

    if (i == 0) {
      if (OutStreamer->hasRawTextSupport()) {
        OutStreamer->switchSection(
            getObjFileLowering().SectionForGlobal(&F, TM));
        OutStreamer->emitRawText(StringRef("\t.literal_position\n"));
      } else {
        MCSectionELF *CS =
            (MCSectionELF *)getObjFileLowering().SectionForGlobal(&F, TM);
        std::string CSectionName = CS->getName().str();
        std::size_t Pos = CSectionName.find(".text");
        std::string SectionName;
        if (Pos != std::string::npos) {
          if (Pos > 0)
            SectionName = CSectionName.substr(0, Pos + 5);
          else
            SectionName = "";
          SectionName += ".literal";
          SectionName += CSectionName.substr(Pos + 5);
        } else {
          SectionName = CSectionName;
          SectionName += ".literal";
        }

        MCSectionELF *S =
            OutContext.getELFSection(SectionName, ELF::SHT_PROGBITS,
                                     ELF::SHF_EXECINSTR | ELF::SHF_ALLOC);
        S->setAlignment(Align(4));
        OutStreamer->switchSection(S);
      }
    }

    emitMachineConstantPoolEntry(CPE, i);
  }
}

MCSymbol *
XtensaAsmPrinter::GetConstantPoolIndexSymbol(const MachineOperand &MO) const {
  // Create a symbol for the name.
  return GetCPISymbol(MO.getIndex());
}

MCOperand
XtensaAsmPrinter::LowerSymbolOperand(const MachineOperand &MO,
                                     MachineOperand::MachineOperandType MOTy,
                                     unsigned Offset) const {
  const MCSymbol *Symbol;
  XtensaMCExpr::VariantKind Kind = XtensaMCExpr::VK_Xtensa_None;

  switch (MOTy) {
  case MachineOperand::MO_GlobalAddress:
    Symbol = getSymbol(MO.getGlobal());
    Offset += MO.getOffset();
    break;
  case MachineOperand::MO_ConstantPoolIndex:
    Symbol = GetConstantPoolIndexSymbol(MO);
    Offset += MO.getOffset();
    break;
  default:
    report_fatal_error("<unknown operand type>");
  }

  const MCExpr *ME =
      MCSymbolRefExpr::create(Symbol, MCSymbolRefExpr::VK_None, OutContext);
  ME = XtensaMCExpr::create(ME, Kind, OutContext);

  if (Offset) {
    // Assume offset is never negative.
    assert(Offset > 0);

    const MCConstantExpr *OffsetExpr =
        MCConstantExpr::create(Offset, OutContext);
    ME = MCBinaryExpr::createAdd(ME, OffsetExpr, OutContext);
  }

  return MCOperand::createExpr(ME);
}

MCOperand XtensaAsmPrinter::lowerOperand(const MachineOperand &MO,
                                         unsigned Offset) const {
  MachineOperand::MachineOperandType MOTy = MO.getType();

  switch (MOTy) {
  case MachineOperand::MO_Register:
    // Ignore all implicit register operands.
    if (MO.isImplicit())
      break;
    return MCOperand::createReg(MO.getReg());
  case MachineOperand::MO_Immediate:
    return MCOperand::createImm(MO.getImm() + Offset);
  case MachineOperand::MO_RegisterMask:
    break;
  case MachineOperand::MO_GlobalAddress:
  case MachineOperand::MO_ConstantPoolIndex:
    return LowerSymbolOperand(MO, MOTy, Offset);
  default:
    report_fatal_error("unknown operand type");
  }

  return MCOperand();
}

void XtensaAsmPrinter::lowerToMCInst(const MachineInstr *MI,
                                     MCInst &OutMI) const {
  OutMI.setOpcode(MI->getOpcode());

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    MCOperand MCOp = lowerOperand(MO);

    if (MCOp.isValid())
      OutMI.addOperand(MCOp);
  }
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeXtensaAsmPrinter() {
  RegisterAsmPrinter<XtensaAsmPrinter> A(getTheXtensaTarget());
}
