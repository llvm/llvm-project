//===-- PPCMCInstLower.cpp - Convert PPC MachineInstr to an MCInst --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains code to lower PPC MachineInstrs to their corresponding
// MCInst records.
//
//===----------------------------------------------------------------------===//

#include "PPC.h"
#include "PPCSubtarget.h"
#include "MCTargetDesc/PPCMCExpr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
using namespace llvm;

static MachineModuleInfoMachO &getMachOMMI(AsmPrinter &AP) {
  return AP.MMI->getObjFileInfo<MachineModuleInfoMachO>();
}


static MCSymbol *GetSymbolFromOperand(const MachineOperand &MO, AsmPrinter &AP){
  const TargetMachine &TM = AP.TM;
  Mangler *Mang = AP.Mang;
  const DataLayout *DL = TM.getDataLayout();
  MCContext &Ctx = AP.OutContext;
  bool isDarwin = TM.getSubtarget<PPCSubtarget>().isDarwin();

  SmallString<128> Name;
  StringRef Suffix;
  if (MO.getTargetFlags() == PPCII::MO_PLT_OR_STUB) {
    if (isDarwin)
      Suffix = "$stub";
  } else if (MO.getTargetFlags() & PPCII::MO_NLP_FLAG)
    Suffix = "$non_lazy_ptr";

  if (!Suffix.empty())
    Name += DL->getPrivateGlobalPrefix();

  unsigned PrefixLen = Name.size();

  if (!MO.isGlobal()) {
    assert(MO.isSymbol() && "Isn't a symbol reference");
    Mang->getNameWithPrefix(Name, MO.getSymbolName());
  } else {
    const GlobalValue *GV = MO.getGlobal();
    TM.getNameWithPrefix(Name, GV, *Mang);
  }

  unsigned OrigLen = Name.size() - PrefixLen;

  Name += Suffix;
  MCSymbol *Sym = Ctx.GetOrCreateSymbol(Name.str());
  StringRef OrigName = StringRef(Name).substr(PrefixLen, OrigLen);

  // If the target flags on the operand changes the name of the symbol, do that
  // before we return the symbol.
  if (MO.getTargetFlags() == PPCII::MO_PLT_OR_STUB && isDarwin) {
    MachineModuleInfoImpl::StubValueTy &StubSym =
      getMachOMMI(AP).getFnStubEntry(Sym);
    if (StubSym.getPointer())
      return Sym;
    
    if (MO.isGlobal()) {
      StubSym =
      MachineModuleInfoImpl::
      StubValueTy(AP.getSymbol(MO.getGlobal()),
                  !MO.getGlobal()->hasInternalLinkage());
    } else {
      StubSym =
      MachineModuleInfoImpl::
      StubValueTy(Ctx.GetOrCreateSymbol(OrigName), false);
    }
    return Sym;
  }

  // If the symbol reference is actually to a non_lazy_ptr, not to the symbol,
  // then add the suffix.
  if (MO.getTargetFlags() & PPCII::MO_NLP_FLAG) {
    MachineModuleInfoMachO &MachO = getMachOMMI(AP);
    
    MachineModuleInfoImpl::StubValueTy &StubSym =
      (MO.getTargetFlags() & PPCII::MO_NLP_HIDDEN_FLAG) ? 
         MachO.getHiddenGVStubEntry(Sym) : MachO.getGVStubEntry(Sym);
    
    if (!StubSym.getPointer()) {
      assert(MO.isGlobal() && "Extern symbol not handled yet");
      StubSym = MachineModuleInfoImpl::
                   StubValueTy(AP.getSymbol(MO.getGlobal()),
                               !MO.getGlobal()->hasInternalLinkage());
    }
    return Sym;
  }
  
  return Sym;
}

static MCOperand GetSymbolRef(const MachineOperand &MO, const MCSymbol *Symbol,
                              AsmPrinter &Printer, bool isDarwin) {
  MCContext &Ctx = Printer.OutContext;
  MCSymbolRefExpr::VariantKind RefKind = MCSymbolRefExpr::VK_None;

  unsigned access = MO.getTargetFlags() & PPCII::MO_ACCESS_MASK;

  switch (access) {
    case PPCII::MO_TPREL_LO:
      RefKind = MCSymbolRefExpr::VK_PPC_TPREL_LO;
      break;
    case PPCII::MO_TPREL_HA:
      RefKind = MCSymbolRefExpr::VK_PPC_TPREL_HA;
      break;
    case PPCII::MO_DTPREL_LO:
      RefKind = MCSymbolRefExpr::VK_PPC_DTPREL_LO;
      break;
    case PPCII::MO_TLSLD_LO:
      RefKind = MCSymbolRefExpr::VK_PPC_GOT_TLSLD_LO;
      break;
    case PPCII::MO_TOC_LO:
      RefKind = MCSymbolRefExpr::VK_PPC_TOC_LO;
      break;
    case PPCII::MO_TLS:
      RefKind = MCSymbolRefExpr::VK_PPC_TLS;
      break;
    case PPCII::MO_TLSGD:
      RefKind = MCSymbolRefExpr::VK_PPC_TLSGD;
      break;
    case PPCII::MO_TLSLD:
      RefKind = MCSymbolRefExpr::VK_PPC_TLSLD;
      break;
  }

  if (MO.getTargetFlags() == PPCII::MO_PLT_OR_STUB && !isDarwin)
    RefKind = MCSymbolRefExpr::VK_PLT;

  const MCExpr *Expr = MCSymbolRefExpr::Create(Symbol, RefKind, Ctx);

  if (!MO.isJTI() && MO.getOffset())
    Expr = MCBinaryExpr::CreateAdd(Expr,
                                   MCConstantExpr::Create(MO.getOffset(), Ctx),
                                   Ctx);

  // Subtract off the PIC base if required.
  if (MO.getTargetFlags() & PPCII::MO_PIC_FLAG) {
    const MachineFunction *MF = MO.getParent()->getParent()->getParent();
    
    const MCExpr *PB = MCSymbolRefExpr::Create(MF->getPICBaseSymbol(), Ctx);
    Expr = MCBinaryExpr::CreateSub(Expr, PB, Ctx);
  }

  // Add ha16() / lo16() markers if required.
  switch (access) {
    case PPCII::MO_LO:
      Expr = PPCMCExpr::CreateLo(Expr, isDarwin, Ctx);
      break;
    case PPCII::MO_HA:
      Expr = PPCMCExpr::CreateHa(Expr, isDarwin, Ctx);
      break;
  }

  return MCOperand::CreateExpr(Expr);
}

void llvm::LowerPPCMachineInstrToMCInst(const MachineInstr *MI, MCInst &OutMI,
                                        AsmPrinter &AP, bool isDarwin) {
  OutMI.setOpcode(MI->getOpcode());
  
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    
    MCOperand MCOp;
    switch (MO.getType()) {
    default:
      MI->dump();
      llvm_unreachable("unknown operand type");
    case MachineOperand::MO_Register:
      assert(!MO.getSubReg() && "Subregs should be eliminated!");
      MCOp = MCOperand::CreateReg(MO.getReg());
      break;
    case MachineOperand::MO_Immediate:
      MCOp = MCOperand::CreateImm(MO.getImm());
      break;
    case MachineOperand::MO_MachineBasicBlock:
      MCOp = MCOperand::CreateExpr(MCSymbolRefExpr::Create(
                                      MO.getMBB()->getSymbol(), AP.OutContext));
      break;
    case MachineOperand::MO_GlobalAddress:
    case MachineOperand::MO_ExternalSymbol:
      MCOp = GetSymbolRef(MO, GetSymbolFromOperand(MO, AP), AP, isDarwin);
      break;
    case MachineOperand::MO_JumpTableIndex:
      MCOp = GetSymbolRef(MO, AP.GetJTISymbol(MO.getIndex()), AP, isDarwin);
      break;
    case MachineOperand::MO_ConstantPoolIndex:
      MCOp = GetSymbolRef(MO, AP.GetCPISymbol(MO.getIndex()), AP, isDarwin);
      break;
    case MachineOperand::MO_BlockAddress:
      MCOp = GetSymbolRef(MO,AP.GetBlockAddressSymbol(MO.getBlockAddress()),AP,
                          isDarwin);
      break;
    case MachineOperand::MO_RegisterMask:
      continue;
    }
    
    OutMI.addOperand(MCOp);
  }
}
