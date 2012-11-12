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
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Target/Mangler.h"
#include "llvm/ADT/SmallString.h"
using namespace llvm;

static MachineModuleInfoMachO &getMachOMMI(AsmPrinter &AP) {
  return AP.MMI->getObjFileInfo<MachineModuleInfoMachO>();
}


static MCSymbol *GetSymbolFromOperand(const MachineOperand &MO, AsmPrinter &AP){
  MCContext &Ctx = AP.OutContext;

  SmallString<128> Name;
  if (!MO.isGlobal()) {
    assert(MO.isSymbol() && "Isn't a symbol reference");
    Name += AP.MAI->getGlobalPrefix();
    Name += MO.getSymbolName();
  } else {    
    const GlobalValue *GV = MO.getGlobal();
    bool isImplicitlyPrivate = false;
    if (MO.getTargetFlags() == PPCII::MO_DARWIN_STUB ||
        (MO.getTargetFlags() & PPCII::MO_NLP_FLAG))
      isImplicitlyPrivate = true;
    
    AP.Mang->getNameWithPrefix(Name, GV, isImplicitlyPrivate);
  }
  
  // If the target flags on the operand changes the name of the symbol, do that
  // before we return the symbol.
  if (MO.getTargetFlags() == PPCII::MO_DARWIN_STUB) {
    Name += "$stub";
    MCSymbol *Sym = Ctx.GetOrCreateSymbol(Name.str());
    MachineModuleInfoImpl::StubValueTy &StubSym =
      getMachOMMI(AP).getFnStubEntry(Sym);
    if (StubSym.getPointer())
      return Sym;
    
    if (MO.isGlobal()) {
      StubSym =
      MachineModuleInfoImpl::
      StubValueTy(AP.Mang->getSymbol(MO.getGlobal()),
                  !MO.getGlobal()->hasInternalLinkage());
    } else {
      Name.erase(Name.end()-5, Name.end());
      StubSym =
      MachineModuleInfoImpl::
      StubValueTy(Ctx.GetOrCreateSymbol(Name.str()), false);
    }
    return Sym;
  }

  // If the symbol reference is actually to a non_lazy_ptr, not to the symbol,
  // then add the suffix.
  if (MO.getTargetFlags() & PPCII::MO_NLP_FLAG) {
    Name += "$non_lazy_ptr";
    MCSymbol *Sym = Ctx.GetOrCreateSymbol(Name.str());
  
    MachineModuleInfoMachO &MachO = getMachOMMI(AP);
    
    MachineModuleInfoImpl::StubValueTy &StubSym =
      (MO.getTargetFlags() & PPCII::MO_NLP_HIDDEN_FLAG) ? 
         MachO.getHiddenGVStubEntry(Sym) : MachO.getGVStubEntry(Sym);
    
    if (StubSym.getPointer() == 0) {
      assert(MO.isGlobal() && "Extern symbol not handled yet");
      StubSym = MachineModuleInfoImpl::
                   StubValueTy(AP.Mang->getSymbol(MO.getGlobal()),
                               !MO.getGlobal()->hasInternalLinkage());
    }
    return Sym;
  }
  
  return Ctx.GetOrCreateSymbol(Name.str());
}

static MCOperand GetSymbolRef(const MachineOperand &MO, const MCSymbol *Symbol,
                              AsmPrinter &Printer, bool isDarwin) {
  MCContext &Ctx = Printer.OutContext;
  MCSymbolRefExpr::VariantKind RefKind = MCSymbolRefExpr::VK_None;

  unsigned access = MO.getTargetFlags() & PPCII::MO_ACCESS_MASK;

  switch (access) {
    case PPCII::MO_HA16: RefKind = isDarwin ? 
                           MCSymbolRefExpr::VK_PPC_DARWIN_HA16 : 
                           MCSymbolRefExpr::VK_PPC_GAS_HA16; 
                         break;
    case PPCII::MO_LO16: RefKind = isDarwin ? 
                           MCSymbolRefExpr::VK_PPC_DARWIN_LO16 : 
                           MCSymbolRefExpr::VK_PPC_GAS_LO16; 
                         break;
    case PPCII::MO_TPREL16_HA: RefKind = MCSymbolRefExpr::VK_PPC_TPREL16_HA;
                               break;
    case PPCII::MO_TPREL16_LO: RefKind = MCSymbolRefExpr::VK_PPC_TPREL16_LO;
                               break;
   }

  // FIXME: This isn't right, but we don't have a good way to express this in
  // the MC Level, see below.
  if (MO.getTargetFlags() & PPCII::MO_PIC_FLAG)
    RefKind = MCSymbolRefExpr::VK_None;
  
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
    // FIXME: We have no way to make the result be VK_PPC_LO16/VK_PPC_HA16,
    // since it is not a symbol!
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
