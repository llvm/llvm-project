//===- XtensaAsmPrinter.cpp Xtensa LLVM Assembly Printer ------------------===//
//
//                     The LLVM Compiler Infrastructure
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
#include "MCTargetDesc/XtensaInstPrinter.h"
#include "XtensaConstantPoolValue.h"
#include "XtensaMCInstLower.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

static MCSymbolRefExpr::VariantKind
getModifierVariantKind(XtensaCP::XtensaCPModifier Modifier) {
  switch (Modifier) {
  case XtensaCP::no_modifier:
    return MCSymbolRefExpr::VK_None;
  case XtensaCP::TPOFF:
    return MCSymbolRefExpr::VK_TPOFF;
  }
  llvm_unreachable("Invalid XtensaCPModifier!");
}

void XtensaAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  XtensaMCInstLower Lower(MF->getContext(), *this);
  MCInst LoweredMI;
  unsigned Opc = MI->getOpcode();

  switch (Opc) {
  case Xtensa::BR_JT: {
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(Xtensa::JX).addReg(MI->getOperand(0).getReg()));
    return;
  }
  }
  Lower.lower(MI, LoweredMI);
  EmitToStreamer(*OutStreamer, LoweredMI);
}

/// EmitConstantPool - Print to the current output stream assembly
/// representations of the constants in the constant pool MCP. This is
/// used to print out constants which have been "spilled to memory" by
/// the code generator.
void XtensaAsmPrinter::EmitConstantPool() {
  const Function &F = MF->getFunction();
  const MachineConstantPool *MCP = MF->getConstantPool();
  const std::vector<MachineConstantPoolEntry> &CP = MCP->getConstants();
  if (CP.empty())
    return;

  for (unsigned i = 0, e = CP.size(); i != e; ++i) {
    const MachineConstantPoolEntry &CPE = CP[i];

    if (i == 0) {
      if (OutStreamer->hasRawTextSupport()) {
        OutStreamer->SwitchSection(
            getObjFileLowering().SectionForGlobal(&F, TM));
        OutStreamer->EmitRawText("\t.literal_position\n");
      } else {
        MCSectionELF *CS =
            (MCSectionELF *)getObjFileLowering().SectionForGlobal(&F, TM);
        std::string CSectionName = CS->getSectionName();
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
        S->setAlignment(4);
        OutStreamer->SwitchSection(S);
      }
    }

    if (CPE.isMachineConstantPoolEntry()) {
      XtensaConstantPoolValue *ACPV =
          static_cast<XtensaConstantPoolValue *>(CPE.Val.MachineCPVal);
      ACPV->setLabelId(i);
      EmitMachineConstantPoolValue(CPE.Val.MachineCPVal);
    } else {
      MCSymbol *LblSym = GetCPISymbol(i);
      // TODO find a better way to check whether we emit data to .s file
      if (OutStreamer->hasRawTextSupport()) {
        std::string str("\t.literal ");
        str += LblSym->getName();
        str += ", ";
        const Constant *C = CPE.Val.ConstVal;

        Type *Ty = C->getType();
        if (const auto *CFP = dyn_cast<ConstantFP>(C)) {
          str += CFP->getValueAPF().bitcastToAPInt().toString(10, true);
        } else if (const auto *CI = dyn_cast<ConstantInt>(C)) {
          str += CI->getValue().toString(10, true);
        } else if (isa<PointerType>(Ty)) {
          const MCExpr *ME = lowerConstant(C);
          const MCSymbolRefExpr &SRE = cast<MCSymbolRefExpr>(*ME);
          const MCSymbol &Sym = SRE.getSymbol();
          str += Sym.getName();
        } else {
          unsigned NumElements;
          if (isa<VectorType>(Ty))
            NumElements = Ty->getVectorNumElements();
          else
            NumElements = Ty->getArrayNumElements();

          for (unsigned I = 0; I < NumElements; I++) {
            const Constant *CAE = C->getAggregateElement(I);
            if (I > 0)
              str += ", ";
            if (const auto *CFP = dyn_cast<ConstantFP>(CAE)) {
              str += CFP->getValueAPF().bitcastToAPInt().toString(10, true);
            } else if (const auto *CI = dyn_cast<ConstantInt>(CAE)) {
              str += CI->getValue().toString(10, true);
            }
          }
        }

        OutStreamer->EmitRawText(str);
      } else {
        OutStreamer->EmitLabel(LblSym);
        EmitGlobalConstant(getDataLayout(), CPE.Val.ConstVal);
      }
    }
  }
}

void XtensaAsmPrinter::EmitMachineConstantPoolValue(
    MachineConstantPoolValue *MCPV) {
  XtensaConstantPoolValue *ACPV = static_cast<XtensaConstantPoolValue *>(MCPV);

  MCSymbol *MCSym;
  if (ACPV->isBlockAddress()) {
    const BlockAddress *BA =
        cast<XtensaConstantPoolConstant>(ACPV)->getBlockAddress();
    MCSym = GetBlockAddressSymbol(BA);
  } else if (ACPV->isGlobalValue()) {
    const GlobalValue *GV = cast<XtensaConstantPoolConstant>(ACPV)->getGV();
    // TODO some modifiers
    MCSym = getSymbol(GV);
  } else if (ACPV->isMachineBasicBlock()) {
    const MachineBasicBlock *MBB = cast<XtensaConstantPoolMBB>(ACPV)->getMBB();
    MCSym = MBB->getSymbol();
  } else if (ACPV->isJumpTable()) {
    unsigned idx = cast<XtensaConstantPoolJumpTable>(ACPV)->getIndex();
    MCSym = this->GetJTISymbol(idx, false);
  } else {
    assert(ACPV->isExtSymbol() && "unrecognized constant pool value");
    XtensaConstantPoolSymbol *XtensaSym = cast<XtensaConstantPoolSymbol>(ACPV);
    const char *Sym = XtensaSym->getSymbol();
    // TODO it's a trick to distinguish static references and generated rodata
    // references Some clear method required
    {
      std::string SymName(Sym);
      if (XtensaSym->isPrivateLinkage())
        SymName = ".L" + SymName;
      MCSym = GetExternalSymbolSymbol(StringRef(SymName));
    }
  }

  MCSymbol *LblSym = GetCPISymbol(ACPV->getLabelId());
  // TODO find a better way to check whether we emit data to .s file
  if (OutStreamer->hasRawTextSupport()) {
    std::string SymName("\t.literal ");
    SymName += LblSym->getName();
    SymName += ", ";
    SymName += MCSym->getName();

    StringRef Modifier = ACPV->getModifierText();
    SymName += Modifier;

    OutStreamer->EmitRawText(SymName);
  } else {
    MCSymbolRefExpr::VariantKind VK =
        getModifierVariantKind(ACPV->getModifier());

    if (ACPV->getModifier() != XtensaCP::no_modifier) {
      std::string SymName(MCSym->getName());
      MCSym = GetExternalSymbolSymbol(StringRef(SymName));
    }

    const MCExpr *Expr = MCSymbolRefExpr::create(MCSym, VK, OutContext);
    uint64_t Size = getDataLayout().getTypeAllocSize(ACPV->getType());
    OutStreamer->EmitLabel(LblSym);
    OutStreamer->EmitValue(Expr, Size);
  }
}

void XtensaAsmPrinter::printOperand(const MachineInstr *MI, int OpNo,
                                    raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(OpNo);
  // TODO look at target flags MO.getTargetFlags() to see if we should wrap this
  // operand
  switch (MO.getType()) {
  case MachineOperand::MO_Register:
  case MachineOperand::MO_Immediate: {
    XtensaMCInstLower Lower(MF->getContext(), *this);
    MCOperand MC(Lower.lowerOperand(MI->getOperand(OpNo)));
    XtensaInstPrinter::printOperand(MC, O);
    break;
  }
  case MachineOperand::MO_GlobalAddress:
    O << *getSymbol(MO.getGlobal());
    break;
  default:
    llvm_unreachable("<unknown operand type>");
  }

  if (MO.getTargetFlags()) {
    O << ")";
  }
}

bool XtensaAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                       const char *ExtraCode, raw_ostream &O) {
  if (ExtraCode && *ExtraCode == 'n') {
    if (!MI->getOperand(OpNo).isImm())
      return true;
    O << -int64_t(MI->getOperand(OpNo).getImm());
  } else {
    printOperand(MI, OpNo, O);
  }
  return false;
}

bool XtensaAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                             unsigned OpNo,
                                             const char *ExtraCode,
                                             raw_ostream &OS) {
  XtensaInstPrinter::printAddress(MI->getOperand(OpNo).getReg(),
                                  MI->getOperand(OpNo + 1).getImm(), OS);
  return false;
}

void XtensaAsmPrinter::printMemOperand(const MachineInstr *MI, int opNum,
                                       raw_ostream &OS) {
  OS << '%'
     << XtensaInstPrinter::getRegisterName(MI->getOperand(opNum).getReg());
  OS << "(";
  OS << MI->getOperand(opNum + 1).getImm();
  OS << ")";
}

// Force static initialization.
extern "C" void LLVMInitializeXtensaAsmPrinter() {
  RegisterAsmPrinter<XtensaAsmPrinter> A(TheXtensaTarget);
}
