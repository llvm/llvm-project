//===-- PPCAsmPrinter.cpp - Print machine instrs to PowerPC assembly ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to PowerPC assembly language. This printer is
// the output mechanism used by `llc'.
//
// Documentation at http://developer.apple.com/documentation/DeveloperTools/
// Reference/Assembler/ASMIntroduction/chapter_1_section_1.html
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asmprinter"
#include "PPC.h"
#include "InstPrinter/PPCInstPrinter.h"
#include "MCTargetDesc/PPCPredicates.h"
#include "MCTargetDesc/PPCMCExpr.h"
#include "PPCSubtarget.h"
#include "PPCTargetMachine.h"
#include "PPCTargetStreamer.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/DebugInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
using namespace llvm;

namespace {
  class PPCAsmPrinter : public AsmPrinter {
  protected:
    MapVector<MCSymbol*, MCSymbol*> TOC;
    const PPCSubtarget &Subtarget;
    uint64_t TOCLabelID;
  public:
    explicit PPCAsmPrinter(TargetMachine &TM, MCStreamer &Streamer)
      : AsmPrinter(TM, Streamer),
        Subtarget(TM.getSubtarget<PPCSubtarget>()), TOCLabelID(0) {}

    virtual const char *getPassName() const {
      return "PowerPC Assembly Printer";
    }

    MCSymbol *lookUpOrCreateTOCEntry(MCSymbol *Sym);

    virtual void EmitInstruction(const MachineInstr *MI);

    void printOperand(const MachineInstr *MI, unsigned OpNo, raw_ostream &O);

    bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                         unsigned AsmVariant, const char *ExtraCode,
                         raw_ostream &O);
    bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                               unsigned AsmVariant, const char *ExtraCode,
                               raw_ostream &O);
  };

  /// PPCLinuxAsmPrinter - PowerPC assembly printer, customized for Linux
  class PPCLinuxAsmPrinter : public PPCAsmPrinter {
  public:
    explicit PPCLinuxAsmPrinter(TargetMachine &TM, MCStreamer &Streamer)
      : PPCAsmPrinter(TM, Streamer) {}

    virtual const char *getPassName() const {
      return "Linux PPC Assembly Printer";
    }

    bool doFinalization(Module &M);

    virtual void EmitFunctionEntryLabel();

    void EmitFunctionBodyEnd();
  };

  /// PPCDarwinAsmPrinter - PowerPC assembly printer, customized for Darwin/Mac
  /// OS X
  class PPCDarwinAsmPrinter : public PPCAsmPrinter {
  public:
    explicit PPCDarwinAsmPrinter(TargetMachine &TM, MCStreamer &Streamer)
      : PPCAsmPrinter(TM, Streamer) {}

    virtual const char *getPassName() const {
      return "Darwin PPC Assembly Printer";
    }

    bool doFinalization(Module &M);
    void EmitStartOfAsmFile(Module &M);

    void EmitFunctionStubs(const MachineModuleInfoMachO::SymbolListTy &Stubs);
  };
} // end of anonymous namespace

/// stripRegisterPrefix - This method strips the character prefix from a
/// register name so that only the number is left.  Used by for linux asm.
static const char *stripRegisterPrefix(const char *RegName) {
  switch (RegName[0]) {
    case 'r':
    case 'f':
    case 'v': return RegName + 1;
    case 'c': if (RegName[1] == 'r') return RegName + 2;
  }
  
  return RegName;
}

void PPCAsmPrinter::printOperand(const MachineInstr *MI, unsigned OpNo,
                                 raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(OpNo);
  
  switch (MO.getType()) {
  case MachineOperand::MO_Register: {
    const char *RegName = PPCInstPrinter::getRegisterName(MO.getReg());
    // Linux assembler (Others?) does not take register mnemonics.
    // FIXME - What about special registers used in mfspr/mtspr?
    if (!Subtarget.isDarwin()) RegName = stripRegisterPrefix(RegName);
    O << RegName;
    return;
  }
  case MachineOperand::MO_Immediate:
    O << MO.getImm();
    return;

  case MachineOperand::MO_MachineBasicBlock:
    O << *MO.getMBB()->getSymbol();
    return;
  case MachineOperand::MO_JumpTableIndex:
    O << MAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber()
      << '_' << MO.getIndex();
    // FIXME: PIC relocation model
    return;
  case MachineOperand::MO_ConstantPoolIndex:
    O << MAI->getPrivateGlobalPrefix() << "CPI" << getFunctionNumber()
      << '_' << MO.getIndex();
    return;
  case MachineOperand::MO_BlockAddress:
    O << *GetBlockAddressSymbol(MO.getBlockAddress());
    return;
  case MachineOperand::MO_ExternalSymbol: {
    // Computing the address of an external symbol, not calling it.
    if (TM.getRelocationModel() == Reloc::Static) {
      O << *GetExternalSymbolSymbol(MO.getSymbolName());
      return;
    }

    MCSymbol *NLPSym = 
      OutContext.GetOrCreateSymbol(StringRef(MAI->getGlobalPrefix())+
                                   MO.getSymbolName()+"$non_lazy_ptr");
    MachineModuleInfoImpl::StubValueTy &StubSym = 
      MMI->getObjFileInfo<MachineModuleInfoMachO>().getGVStubEntry(NLPSym);
    if (StubSym.getPointer() == 0)
      StubSym = MachineModuleInfoImpl::
        StubValueTy(GetExternalSymbolSymbol(MO.getSymbolName()), true);
    
    O << *NLPSym;
    return;
  }
  case MachineOperand::MO_GlobalAddress: {
    // Computing the address of a global symbol, not calling it.
    const GlobalValue *GV = MO.getGlobal();
    MCSymbol *SymToPrint;

    // External or weakly linked global variables need non-lazily-resolved stubs
    if (TM.getRelocationModel() != Reloc::Static &&
        (GV->isDeclaration() || GV->isWeakForLinker())) {
      if (!GV->hasHiddenVisibility()) {
        SymToPrint = GetSymbolWithGlobalValueBase(GV, "$non_lazy_ptr");
        MachineModuleInfoImpl::StubValueTy &StubSym = 
          MMI->getObjFileInfo<MachineModuleInfoMachO>()
            .getGVStubEntry(SymToPrint);
        if (StubSym.getPointer() == 0)
          StubSym = MachineModuleInfoImpl::
            StubValueTy(getSymbol(GV), !GV->hasInternalLinkage());
      } else if (GV->isDeclaration() || GV->hasCommonLinkage() ||
                 GV->hasAvailableExternallyLinkage()) {
        SymToPrint = GetSymbolWithGlobalValueBase(GV, "$non_lazy_ptr");
        
        MachineModuleInfoImpl::StubValueTy &StubSym = 
          MMI->getObjFileInfo<MachineModuleInfoMachO>().
                    getHiddenGVStubEntry(SymToPrint);
        if (StubSym.getPointer() == 0)
          StubSym = MachineModuleInfoImpl::
            StubValueTy(getSymbol(GV), !GV->hasInternalLinkage());
      } else {
        SymToPrint = getSymbol(GV);
      }
    } else {
      SymToPrint = getSymbol(GV);
    }
    
    O << *SymToPrint;

    printOffset(MO.getOffset(), O);
    return;
  }

  default:
    O << "<unknown operand type: " << MO.getType() << ">";
    return;
  }
}

/// PrintAsmOperand - Print out an operand for an inline asm expression.
///
bool PPCAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                    unsigned AsmVariant,
                                    const char *ExtraCode, raw_ostream &O) {
  // Does this asm operand have a single letter operand modifier?
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0) return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    default:
      // See if this is a generic print operand
      return AsmPrinter::PrintAsmOperand(MI, OpNo, AsmVariant, ExtraCode, O);
    case 'c': // Don't print "$" before a global var name or constant.
      break; // PPC never has a prefix.
    case 'L': // Write second word of DImode reference.
      // Verify that this operand has two consecutive registers.
      if (!MI->getOperand(OpNo).isReg() ||
          OpNo+1 == MI->getNumOperands() ||
          !MI->getOperand(OpNo+1).isReg())
        return true;
      ++OpNo;   // Return the high-part.
      break;
    case 'I':
      // Write 'i' if an integer constant, otherwise nothing.  Used to print
      // addi vs add, etc.
      if (MI->getOperand(OpNo).isImm())
        O << "i";
      return false;
    }
  }

  printOperand(MI, OpNo, O);
  return false;
}

// At the moment, all inline asm memory operands are a single register.
// In any case, the output of this routine should always be just one
// assembler operand.

bool PPCAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                                          unsigned AsmVariant,
                                          const char *ExtraCode,
                                          raw_ostream &O) {
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0) return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    default: return true;  // Unknown modifier.
    case 'y': // A memory reference for an X-form instruction
      {
        const char *RegName = "r0";
        if (!Subtarget.isDarwin()) RegName = stripRegisterPrefix(RegName);
        O << RegName << ", ";
        printOperand(MI, OpNo, O);
        return false;
      }
    }
  }

  assert(MI->getOperand(OpNo).isReg());
  O << "0(";
  printOperand(MI, OpNo, O);
  O << ")";
  return false;
}


/// lookUpOrCreateTOCEntry -- Given a symbol, look up whether a TOC entry
/// exists for it.  If not, create one.  Then return a symbol that references
/// the TOC entry.
MCSymbol *PPCAsmPrinter::lookUpOrCreateTOCEntry(MCSymbol *Sym) {

  MCSymbol *&TOCEntry = TOC[Sym];

  // To avoid name clash check if the name already exists.
  while (TOCEntry == 0) {
    if (OutContext.LookupSymbol(Twine(MAI->getPrivateGlobalPrefix()) +
                                "C" + Twine(TOCLabelID++)) == 0) {
      TOCEntry = GetTempSymbol("C", TOCLabelID);
    }
  }

  return TOCEntry;
}


/// EmitInstruction -- Print out a single PowerPC MI in Darwin syntax to
/// the current output stream.
///
void PPCAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  MCInst TmpInst;
  
  // Lower multi-instruction pseudo operations.
  switch (MI->getOpcode()) {
  default: break;
  case TargetOpcode::DBG_VALUE:
    llvm_unreachable("Should be handled target independently");
  case PPC::MovePCtoLR:
  case PPC::MovePCtoLR8: {
    // Transform %LR = MovePCtoLR
    // Into this, where the label is the PIC base: 
    //     bl L1$pb
    // L1$pb:
    MCSymbol *PICBase = MF->getPICBaseSymbol();
    
    // Emit the 'bl'.
    OutStreamer.EmitInstruction(MCInstBuilder(PPC::BL)
      // FIXME: We would like an efficient form for this, so we don't have to do
      // a lot of extra uniquing.
      .addExpr(MCSymbolRefExpr::Create(PICBase, OutContext)));
    
    // Emit the label.
    OutStreamer.EmitLabel(PICBase);
    return;
  }
  case PPC::LDtocJTI:
  case PPC::LDtocCPT:
  case PPC::LDtoc: {
    // Transform %X3 = LDtoc <ga:@min1>, %X2
    LowerPPCMachineInstrToMCInst(MI, TmpInst, *this, Subtarget.isDarwin());

    // Change the opcode to LD, and the global address operand to be a
    // reference to the TOC entry we will synthesize later.
    TmpInst.setOpcode(PPC::LD);
    const MachineOperand &MO = MI->getOperand(1);

    // Map symbol -> label of TOC entry
    assert(MO.isGlobal() || MO.isCPI() || MO.isJTI());
    MCSymbol *MOSymbol = 0;
    if (MO.isGlobal())
      MOSymbol = getSymbol(MO.getGlobal());
    else if (MO.isCPI())
      MOSymbol = GetCPISymbol(MO.getIndex());
    else if (MO.isJTI())
      MOSymbol = GetJTISymbol(MO.getIndex());

    MCSymbol *TOCEntry = lookUpOrCreateTOCEntry(MOSymbol);

    const MCExpr *Exp =
      MCSymbolRefExpr::Create(TOCEntry, MCSymbolRefExpr::VK_PPC_TOC,
                              OutContext);
    TmpInst.getOperand(1) = MCOperand::CreateExpr(Exp);
    OutStreamer.EmitInstruction(TmpInst);
    return;
  }
      
  case PPC::ADDIStocHA: {
    // Transform %Xd = ADDIStocHA %X2, <ga:@sym>
    LowerPPCMachineInstrToMCInst(MI, TmpInst, *this, Subtarget.isDarwin());

    // Change the opcode to ADDIS8.  If the global address is external,
    // has common linkage, is a function address, or is a jump table
    // address, then generate a TOC entry and reference that.  Otherwise
    // reference the symbol directly.
    TmpInst.setOpcode(PPC::ADDIS8);
    const MachineOperand &MO = MI->getOperand(2);
    assert((MO.isGlobal() || MO.isCPI() || MO.isJTI()) &&
           "Invalid operand for ADDIStocHA!");
    MCSymbol *MOSymbol = 0;
    bool IsExternal = false;
    bool IsFunction = false;
    bool IsCommon = false;
    bool IsAvailExt = false;

    if (MO.isGlobal()) {
      const GlobalValue *GValue = MO.getGlobal();
      const GlobalAlias *GAlias = dyn_cast<GlobalAlias>(GValue);
      const GlobalValue *RealGValue = GAlias ?
        GAlias->resolveAliasedGlobal(false) : GValue;
      MOSymbol = getSymbol(RealGValue);
      const GlobalVariable *GVar = dyn_cast<GlobalVariable>(RealGValue);
      IsExternal = GVar && !GVar->hasInitializer();
      IsCommon = GVar && RealGValue->hasCommonLinkage();
      IsFunction = !GVar;
      IsAvailExt = GVar && RealGValue->hasAvailableExternallyLinkage();
    } else if (MO.isCPI())
      MOSymbol = GetCPISymbol(MO.getIndex());
    else if (MO.isJTI())
      MOSymbol = GetJTISymbol(MO.getIndex());

    if (IsExternal || IsFunction || IsCommon || IsAvailExt || MO.isJTI() ||
        TM.getCodeModel() == CodeModel::Large)
      MOSymbol = lookUpOrCreateTOCEntry(MOSymbol);

    const MCExpr *Exp =
      MCSymbolRefExpr::Create(MOSymbol, MCSymbolRefExpr::VK_PPC_TOC_HA,
                              OutContext);
    TmpInst.getOperand(2) = MCOperand::CreateExpr(Exp);
    OutStreamer.EmitInstruction(TmpInst);
    return;
  }
  case PPC::LDtocL: {
    // Transform %Xd = LDtocL <ga:@sym>, %Xs
    LowerPPCMachineInstrToMCInst(MI, TmpInst, *this, Subtarget.isDarwin());

    // Change the opcode to LD.  If the global address is external, has
    // common linkage, or is a jump table address, then reference the
    // associated TOC entry.  Otherwise reference the symbol directly.
    TmpInst.setOpcode(PPC::LD);
    const MachineOperand &MO = MI->getOperand(1);
    assert((MO.isGlobal() || MO.isJTI() || MO.isCPI()) &&
           "Invalid operand for LDtocL!");
    MCSymbol *MOSymbol = 0;

    if (MO.isJTI())
      MOSymbol = lookUpOrCreateTOCEntry(GetJTISymbol(MO.getIndex()));
    else if (MO.isCPI()) {
      MOSymbol = GetCPISymbol(MO.getIndex());
      if (TM.getCodeModel() == CodeModel::Large)
        MOSymbol = lookUpOrCreateTOCEntry(MOSymbol);
    }
    else if (MO.isGlobal()) {
      const GlobalValue *GValue = MO.getGlobal();
      const GlobalAlias *GAlias = dyn_cast<GlobalAlias>(GValue);
      const GlobalValue *RealGValue = GAlias ?
        GAlias->resolveAliasedGlobal(false) : GValue;
      MOSymbol = getSymbol(RealGValue);
      const GlobalVariable *GVar = dyn_cast<GlobalVariable>(RealGValue);
    
      if (!GVar || !GVar->hasInitializer() || RealGValue->hasCommonLinkage() ||
          RealGValue->hasAvailableExternallyLinkage() ||
          TM.getCodeModel() == CodeModel::Large)
        MOSymbol = lookUpOrCreateTOCEntry(MOSymbol);
    }

    const MCExpr *Exp =
      MCSymbolRefExpr::Create(MOSymbol, MCSymbolRefExpr::VK_PPC_TOC_LO,
                              OutContext);
    TmpInst.getOperand(1) = MCOperand::CreateExpr(Exp);
    OutStreamer.EmitInstruction(TmpInst);
    return;
  }
  case PPC::ADDItocL: {
    // Transform %Xd = ADDItocL %Xs, <ga:@sym>
    LowerPPCMachineInstrToMCInst(MI, TmpInst, *this, Subtarget.isDarwin());

    // Change the opcode to ADDI8.  If the global address is external, then
    // generate a TOC entry and reference that.  Otherwise reference the
    // symbol directly.
    TmpInst.setOpcode(PPC::ADDI8);
    const MachineOperand &MO = MI->getOperand(2);
    assert((MO.isGlobal() || MO.isCPI()) && "Invalid operand for ADDItocL");
    MCSymbol *MOSymbol = 0;
    bool IsExternal = false;
    bool IsFunction = false;

    if (MO.isGlobal()) {
      const GlobalValue *GValue = MO.getGlobal();
      const GlobalAlias *GAlias = dyn_cast<GlobalAlias>(GValue);
      const GlobalValue *RealGValue = GAlias ?
        GAlias->resolveAliasedGlobal(false) : GValue;
      MOSymbol = getSymbol(RealGValue);
      const GlobalVariable *GVar = dyn_cast<GlobalVariable>(RealGValue);
      IsExternal = GVar && !GVar->hasInitializer();
      IsFunction = !GVar;
    } else if (MO.isCPI())
      MOSymbol = GetCPISymbol(MO.getIndex());

    if (IsFunction || IsExternal || TM.getCodeModel() == CodeModel::Large)
      MOSymbol = lookUpOrCreateTOCEntry(MOSymbol);

    const MCExpr *Exp =
      MCSymbolRefExpr::Create(MOSymbol, MCSymbolRefExpr::VK_PPC_TOC_LO,
                              OutContext);
    TmpInst.getOperand(2) = MCOperand::CreateExpr(Exp);
    OutStreamer.EmitInstruction(TmpInst);
    return;
  }
  case PPC::ADDISgotTprelHA: {
    // Transform: %Xd = ADDISgotTprelHA %X2, <ga:@sym>
    // Into:      %Xd = ADDIS8 %X2, sym@got@tlsgd@ha
    assert(Subtarget.isPPC64() && "Not supported for 32-bit PowerPC");
    const MachineOperand &MO = MI->getOperand(2);
    const GlobalValue *GValue = MO.getGlobal();
    MCSymbol *MOSymbol = getSymbol(GValue);
    const MCExpr *SymGotTprel =
      MCSymbolRefExpr::Create(MOSymbol, MCSymbolRefExpr::VK_PPC_GOT_TPREL_HA,
                              OutContext);
    OutStreamer.EmitInstruction(MCInstBuilder(PPC::ADDIS8)
                                .addReg(MI->getOperand(0).getReg())
                                .addReg(PPC::X2)
                                .addExpr(SymGotTprel));
    return;
  }
  case PPC::LDgotTprelL: {
    // Transform %Xd = LDgotTprelL <ga:@sym>, %Xs
    LowerPPCMachineInstrToMCInst(MI, TmpInst, *this, Subtarget.isDarwin());

    // Change the opcode to LD.
    TmpInst.setOpcode(PPC::LD);
    const MachineOperand &MO = MI->getOperand(1);
    const GlobalValue *GValue = MO.getGlobal();
    MCSymbol *MOSymbol = getSymbol(GValue);
    const MCExpr *Exp =
      MCSymbolRefExpr::Create(MOSymbol, MCSymbolRefExpr::VK_PPC_GOT_TPREL_LO,
                              OutContext);
    TmpInst.getOperand(1) = MCOperand::CreateExpr(Exp);
    OutStreamer.EmitInstruction(TmpInst);
    return;
  }
  case PPC::ADDIStlsgdHA: {
    // Transform: %Xd = ADDIStlsgdHA %X2, <ga:@sym>
    // Into:      %Xd = ADDIS8 %X2, sym@got@tlsgd@ha
    assert(Subtarget.isPPC64() && "Not supported for 32-bit PowerPC");
    const MachineOperand &MO = MI->getOperand(2);
    const GlobalValue *GValue = MO.getGlobal();
    MCSymbol *MOSymbol = getSymbol(GValue);
    const MCExpr *SymGotTlsGD =
      MCSymbolRefExpr::Create(MOSymbol, MCSymbolRefExpr::VK_PPC_GOT_TLSGD_HA,
                              OutContext);
    OutStreamer.EmitInstruction(MCInstBuilder(PPC::ADDIS8)
                                .addReg(MI->getOperand(0).getReg())
                                .addReg(PPC::X2)
                                .addExpr(SymGotTlsGD));
    return;
  }
  case PPC::ADDItlsgdL: {
    // Transform: %Xd = ADDItlsgdL %Xs, <ga:@sym>
    // Into:      %Xd = ADDI8 %Xs, sym@got@tlsgd@l
    assert(Subtarget.isPPC64() && "Not supported for 32-bit PowerPC");
    const MachineOperand &MO = MI->getOperand(2);
    const GlobalValue *GValue = MO.getGlobal();
    MCSymbol *MOSymbol = getSymbol(GValue);
    const MCExpr *SymGotTlsGD =
      MCSymbolRefExpr::Create(MOSymbol, MCSymbolRefExpr::VK_PPC_GOT_TLSGD_LO,
                              OutContext);
    OutStreamer.EmitInstruction(MCInstBuilder(PPC::ADDI8)
                                .addReg(MI->getOperand(0).getReg())
                                .addReg(MI->getOperand(1).getReg())
                                .addExpr(SymGotTlsGD));
    return;
  }
  case PPC::GETtlsADDR: {
    // Transform: %X3 = GETtlsADDR %X3, <ga:@sym>
    // Into:      BL8_NOP_TLS __tls_get_addr(sym@tlsgd)
    assert(Subtarget.isPPC64() && "Not supported for 32-bit PowerPC");

    StringRef Name = "__tls_get_addr";
    MCSymbol *TlsGetAddr = OutContext.GetOrCreateSymbol(Name);
    const MCSymbolRefExpr *TlsRef = 
      MCSymbolRefExpr::Create(TlsGetAddr, MCSymbolRefExpr::VK_None, OutContext);
    const MachineOperand &MO = MI->getOperand(2);
    const GlobalValue *GValue = MO.getGlobal();
    MCSymbol *MOSymbol = getSymbol(GValue);
    const MCExpr *SymVar =
      MCSymbolRefExpr::Create(MOSymbol, MCSymbolRefExpr::VK_PPC_TLSGD,
                              OutContext);
    OutStreamer.EmitInstruction(MCInstBuilder(PPC::BL8_NOP_TLS)
                                .addExpr(TlsRef)
                                .addExpr(SymVar));
    return;
  }
  case PPC::ADDIStlsldHA: {
    // Transform: %Xd = ADDIStlsldHA %X2, <ga:@sym>
    // Into:      %Xd = ADDIS8 %X2, sym@got@tlsld@ha
    assert(Subtarget.isPPC64() && "Not supported for 32-bit PowerPC");
    const MachineOperand &MO = MI->getOperand(2);
    const GlobalValue *GValue = MO.getGlobal();
    MCSymbol *MOSymbol = getSymbol(GValue);
    const MCExpr *SymGotTlsLD =
      MCSymbolRefExpr::Create(MOSymbol, MCSymbolRefExpr::VK_PPC_GOT_TLSLD_HA,
                              OutContext);
    OutStreamer.EmitInstruction(MCInstBuilder(PPC::ADDIS8)
                                .addReg(MI->getOperand(0).getReg())
                                .addReg(PPC::X2)
                                .addExpr(SymGotTlsLD));
    return;
  }
  case PPC::ADDItlsldL: {
    // Transform: %Xd = ADDItlsldL %Xs, <ga:@sym>
    // Into:      %Xd = ADDI8 %Xs, sym@got@tlsld@l
    assert(Subtarget.isPPC64() && "Not supported for 32-bit PowerPC");
    const MachineOperand &MO = MI->getOperand(2);
    const GlobalValue *GValue = MO.getGlobal();
    MCSymbol *MOSymbol = getSymbol(GValue);
    const MCExpr *SymGotTlsLD =
      MCSymbolRefExpr::Create(MOSymbol, MCSymbolRefExpr::VK_PPC_GOT_TLSLD_LO,
                              OutContext);
    OutStreamer.EmitInstruction(MCInstBuilder(PPC::ADDI8)
                                .addReg(MI->getOperand(0).getReg())
                                .addReg(MI->getOperand(1).getReg())
                                .addExpr(SymGotTlsLD));
    return;
  }
  case PPC::GETtlsldADDR: {
    // Transform: %X3 = GETtlsldADDR %X3, <ga:@sym>
    // Into:      BL8_NOP_TLS __tls_get_addr(sym@tlsld)
    assert(Subtarget.isPPC64() && "Not supported for 32-bit PowerPC");

    StringRef Name = "__tls_get_addr";
    MCSymbol *TlsGetAddr = OutContext.GetOrCreateSymbol(Name);
    const MCSymbolRefExpr *TlsRef = 
      MCSymbolRefExpr::Create(TlsGetAddr, MCSymbolRefExpr::VK_None, OutContext);
    const MachineOperand &MO = MI->getOperand(2);
    const GlobalValue *GValue = MO.getGlobal();
    MCSymbol *MOSymbol = getSymbol(GValue);
    const MCExpr *SymVar =
      MCSymbolRefExpr::Create(MOSymbol, MCSymbolRefExpr::VK_PPC_TLSLD,
                              OutContext);
    OutStreamer.EmitInstruction(MCInstBuilder(PPC::BL8_NOP_TLS)
                                .addExpr(TlsRef)
                                .addExpr(SymVar));
    return;
  }
  case PPC::ADDISdtprelHA: {
    // Transform: %Xd = ADDISdtprelHA %X3, <ga:@sym>
    // Into:      %Xd = ADDIS8 %X3, sym@dtprel@ha
    assert(Subtarget.isPPC64() && "Not supported for 32-bit PowerPC");
    const MachineOperand &MO = MI->getOperand(2);
    const GlobalValue *GValue = MO.getGlobal();
    MCSymbol *MOSymbol = getSymbol(GValue);
    const MCExpr *SymDtprel =
      MCSymbolRefExpr::Create(MOSymbol, MCSymbolRefExpr::VK_PPC_DTPREL_HA,
                              OutContext);
    OutStreamer.EmitInstruction(MCInstBuilder(PPC::ADDIS8)
                                .addReg(MI->getOperand(0).getReg())
                                .addReg(PPC::X3)
                                .addExpr(SymDtprel));
    return;
  }
  case PPC::ADDIdtprelL: {
    // Transform: %Xd = ADDIdtprelL %Xs, <ga:@sym>
    // Into:      %Xd = ADDI8 %Xs, sym@dtprel@l
    assert(Subtarget.isPPC64() && "Not supported for 32-bit PowerPC");
    const MachineOperand &MO = MI->getOperand(2);
    const GlobalValue *GValue = MO.getGlobal();
    MCSymbol *MOSymbol = getSymbol(GValue);
    const MCExpr *SymDtprel =
      MCSymbolRefExpr::Create(MOSymbol, MCSymbolRefExpr::VK_PPC_DTPREL_LO,
                              OutContext);
    OutStreamer.EmitInstruction(MCInstBuilder(PPC::ADDI8)
                                .addReg(MI->getOperand(0).getReg())
                                .addReg(MI->getOperand(1).getReg())
                                .addExpr(SymDtprel));
    return;
  }
  case PPC::MFOCRF:
  case PPC::MFOCRF8:
    if (!Subtarget.hasMFOCRF()) {
      // Transform: %R3 = MFOCRF %CR7
      // Into:      %R3 = MFCR   ;; cr7
      unsigned NewOpcode =
        MI->getOpcode() == PPC::MFOCRF ? PPC::MFCR : PPC::MFCR8;
      OutStreamer.AddComment(PPCInstPrinter::
                             getRegisterName(MI->getOperand(1).getReg()));
      OutStreamer.EmitInstruction(MCInstBuilder(NewOpcode)
                                  .addReg(MI->getOperand(0).getReg()));
      return;
    }
    break;
  case PPC::MTOCRF:
  case PPC::MTOCRF8:
    if (!Subtarget.hasMFOCRF()) {
      // Transform: %CR7 = MTOCRF %R3
      // Into:      MTCRF mask, %R3 ;; cr7
      unsigned NewOpcode =
        MI->getOpcode() == PPC::MTOCRF ? PPC::MTCRF : PPC::MTCRF8;
      unsigned Mask = 0x80 >> OutContext.getRegisterInfo()
                              ->getEncodingValue(MI->getOperand(0).getReg());
      OutStreamer.AddComment(PPCInstPrinter::
                             getRegisterName(MI->getOperand(0).getReg()));
      OutStreamer.EmitInstruction(MCInstBuilder(NewOpcode)
                                  .addImm(Mask)
                                  .addReg(MI->getOperand(1).getReg()));
      return;
    }
    break;
  case PPC::LD:
  case PPC::STD:
  case PPC::LWA_32:
  case PPC::LWA: {
    // Verify alignment is legal, so we don't create relocations
    // that can't be supported.
    // FIXME:  This test is currently disabled for Darwin.  The test
    // suite shows a handful of test cases that fail this check for
    // Darwin.  Those need to be investigated before this sanity test
    // can be enabled for those subtargets.
    if (!Subtarget.isDarwin()) {
      unsigned OpNum = (MI->getOpcode() == PPC::STD) ? 2 : 1;
      const MachineOperand &MO = MI->getOperand(OpNum);
      if (MO.isGlobal() && MO.getGlobal()->getAlignment() < 4)
        llvm_unreachable("Global must be word-aligned for LD, STD, LWA!");
    }
    // Now process the instruction normally.
    break;
  }
  }

  LowerPPCMachineInstrToMCInst(MI, TmpInst, *this, Subtarget.isDarwin());
  OutStreamer.EmitInstruction(TmpInst);
}

void PPCLinuxAsmPrinter::EmitFunctionEntryLabel() {
  if (!Subtarget.isPPC64())  // linux/ppc32 - Normal entry label.
    return AsmPrinter::EmitFunctionEntryLabel();
    
  // Emit an official procedure descriptor.
  MCSectionSubPair Current = OutStreamer.getCurrentSection();
  const MCSectionELF *Section = OutStreamer.getContext().getELFSection(".opd",
      ELF::SHT_PROGBITS, ELF::SHF_WRITE | ELF::SHF_ALLOC,
      SectionKind::getReadOnly());
  OutStreamer.SwitchSection(Section);
  OutStreamer.EmitLabel(CurrentFnSym);
  OutStreamer.EmitValueToAlignment(8);
  MCSymbol *Symbol1 = 
    OutContext.GetOrCreateSymbol(".L." + Twine(CurrentFnSym->getName()));
  // Generates a R_PPC64_ADDR64 (from FK_DATA_8) relocation for the function
  // entry point.
  OutStreamer.EmitValue(MCSymbolRefExpr::Create(Symbol1, OutContext),
			8 /*size*/);
  MCSymbol *Symbol2 = OutContext.GetOrCreateSymbol(StringRef(".TOC."));
  // Generates a R_PPC64_TOC relocation for TOC base insertion.
  OutStreamer.EmitValue(MCSymbolRefExpr::Create(Symbol2,
                        MCSymbolRefExpr::VK_PPC_TOCBASE, OutContext),
                        8/*size*/);
  // Emit a null environment pointer.
  OutStreamer.EmitIntValue(0, 8 /* size */);
  OutStreamer.SwitchSection(Current.first, Current.second);

  MCSymbol *RealFnSym = OutContext.GetOrCreateSymbol(
                          ".L." + Twine(CurrentFnSym->getName()));
  OutStreamer.EmitLabel(RealFnSym);
  CurrentFnSymForSize = RealFnSym;
}


bool PPCLinuxAsmPrinter::doFinalization(Module &M) {
  const DataLayout *TD = TM.getDataLayout();

  bool isPPC64 = TD->getPointerSizeInBits() == 64;

  PPCTargetStreamer &TS =
      static_cast<PPCTargetStreamer &>(OutStreamer.getTargetStreamer());

  if (isPPC64 && !TOC.empty()) {
    const MCSectionELF *Section = OutStreamer.getContext().getELFSection(".toc",
        ELF::SHT_PROGBITS, ELF::SHF_WRITE | ELF::SHF_ALLOC,
        SectionKind::getReadOnly());
    OutStreamer.SwitchSection(Section);

    for (MapVector<MCSymbol*, MCSymbol*>::iterator I = TOC.begin(),
         E = TOC.end(); I != E; ++I) {
      OutStreamer.EmitLabel(I->second);
      MCSymbol *S = OutContext.GetOrCreateSymbol(I->first->getName());
      TS.emitTCEntry(*S);
    }
  }

  MachineModuleInfoELF &MMIELF =
    MMI->getObjFileInfo<MachineModuleInfoELF>();

  MachineModuleInfoELF::SymbolListTy Stubs = MMIELF.GetGVStubList();
  if (!Stubs.empty()) {
    OutStreamer.SwitchSection(getObjFileLowering().getDataSection());
    for (unsigned i = 0, e = Stubs.size(); i != e; ++i) {
      // L_foo$stub:
      OutStreamer.EmitLabel(Stubs[i].first);
      //   .long _foo
      OutStreamer.EmitValue(MCSymbolRefExpr::Create(Stubs[i].second.getPointer(),
                                                    OutContext),
                            isPPC64 ? 8 : 4/*size*/);
    }

    Stubs.clear();
    OutStreamer.AddBlankLine();
  }

  return AsmPrinter::doFinalization(M);
}

/// EmitFunctionBodyEnd - Print the traceback table before the .size
/// directive.
///
void PPCLinuxAsmPrinter::EmitFunctionBodyEnd() {
  // Only the 64-bit target requires a traceback table.  For now,
  // we only emit the word of zeroes that GDB requires to find
  // the end of the function, and zeroes for the eight-byte
  // mandatory fields.
  // FIXME: We should fill in the eight-byte mandatory fields as described in
  // the PPC64 ELF ABI (this is a low-priority item because GDB does not
  // currently make use of these fields).
  if (Subtarget.isPPC64()) {
    OutStreamer.EmitIntValue(0, 4/*size*/);
    OutStreamer.EmitIntValue(0, 8/*size*/);
  }
}

void PPCDarwinAsmPrinter::EmitStartOfAsmFile(Module &M) {
  static const char *const CPUDirectives[] = {
    "",
    "ppc",
    "ppc440",
    "ppc601",
    "ppc602",
    "ppc603",
    "ppc7400",
    "ppc750",
    "ppc970",
    "ppcA2",
    "ppce500mc",
    "ppce5500",
    "power3",
    "power4",
    "power5",
    "power5x",
    "power6",
    "power6x",
    "power7",
    "ppc64",
    "ppc64le"
  };

  unsigned Directive = Subtarget.getDarwinDirective();
  if (Subtarget.hasMFOCRF() && Directive < PPC::DIR_970)
    Directive = PPC::DIR_970;
  if (Subtarget.hasAltivec() && Directive < PPC::DIR_7400)
    Directive = PPC::DIR_7400;
  if (Subtarget.isPPC64() && Directive < PPC::DIR_64)
    Directive = PPC::DIR_64;
  assert(Directive <= PPC::DIR_64 && "Directive out of range.");
  
  // FIXME: This is a total hack, finish mc'izing the PPC backend.
  if (OutStreamer.hasRawTextSupport()) {
    assert(Directive < array_lengthof(CPUDirectives) &&
           "CPUDirectives[] might not be up-to-date!");
    OutStreamer.EmitRawText("\t.machine " + Twine(CPUDirectives[Directive]));
  }

  // Prime text sections so they are adjacent.  This reduces the likelihood a
  // large data or debug section causes a branch to exceed 16M limit.
  const TargetLoweringObjectFileMachO &TLOFMacho = 
    static_cast<const TargetLoweringObjectFileMachO &>(getObjFileLowering());
  OutStreamer.SwitchSection(TLOFMacho.getTextCoalSection());
  if (TM.getRelocationModel() == Reloc::PIC_) {
    OutStreamer.SwitchSection(
           OutContext.getMachOSection("__TEXT", "__picsymbolstub1",
                                      MCSectionMachO::S_SYMBOL_STUBS |
                                      MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS,
                                      32, SectionKind::getText()));
  } else if (TM.getRelocationModel() == Reloc::DynamicNoPIC) {
    OutStreamer.SwitchSection(
           OutContext.getMachOSection("__TEXT","__symbol_stub1",
                                      MCSectionMachO::S_SYMBOL_STUBS |
                                      MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS,
                                      16, SectionKind::getText()));
  }
  OutStreamer.SwitchSection(getObjFileLowering().getTextSection());
}

static MCSymbol *GetLazyPtr(MCSymbol *Sym, MCContext &Ctx) {
  // Remove $stub suffix, add $lazy_ptr.
  StringRef NoStub = Sym->getName().substr(0, Sym->getName().size()-5);
  return Ctx.GetOrCreateSymbol(NoStub + "$lazy_ptr");
}

static MCSymbol *GetAnonSym(MCSymbol *Sym, MCContext &Ctx) {
  // Add $tmp suffix to $stub, yielding $stub$tmp.
  return Ctx.GetOrCreateSymbol(Sym->getName() + "$tmp");
}

void PPCDarwinAsmPrinter::
EmitFunctionStubs(const MachineModuleInfoMachO::SymbolListTy &Stubs) {
  bool isPPC64 = TM.getDataLayout()->getPointerSizeInBits() == 64;
  bool isDarwin = Subtarget.isDarwin();
  
  const TargetLoweringObjectFileMachO &TLOFMacho = 
    static_cast<const TargetLoweringObjectFileMachO &>(getObjFileLowering());

  // .lazy_symbol_pointer
  const MCSection *LSPSection = TLOFMacho.getLazySymbolPointerSection();
  
  // Output stubs for dynamically-linked functions
  if (TM.getRelocationModel() == Reloc::PIC_) {
    const MCSection *StubSection = 
    OutContext.getMachOSection("__TEXT", "__picsymbolstub1",
                               MCSectionMachO::S_SYMBOL_STUBS |
                               MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS,
                               32, SectionKind::getText());
    for (unsigned i = 0, e = Stubs.size(); i != e; ++i) {
      OutStreamer.SwitchSection(StubSection);
      EmitAlignment(4);
      
      MCSymbol *Stub = Stubs[i].first;
      MCSymbol *RawSym = Stubs[i].second.getPointer();
      MCSymbol *LazyPtr = GetLazyPtr(Stub, OutContext);
      MCSymbol *AnonSymbol = GetAnonSym(Stub, OutContext);
                                           
      OutStreamer.EmitLabel(Stub);
      OutStreamer.EmitSymbolAttribute(RawSym, MCSA_IndirectSymbol);

      const MCExpr *Anon = MCSymbolRefExpr::Create(AnonSymbol, OutContext);
      const MCExpr *LazyPtrExpr = MCSymbolRefExpr::Create(LazyPtr, OutContext);
      const MCExpr *Sub =
        MCBinaryExpr::CreateSub(LazyPtrExpr, Anon, OutContext);

      // mflr r0
      OutStreamer.EmitInstruction(MCInstBuilder(PPC::MFLR).addReg(PPC::R0));
      // bcl 20, 31, AnonSymbol
      OutStreamer.EmitInstruction(MCInstBuilder(PPC::BCLalways).addExpr(Anon));
      OutStreamer.EmitLabel(AnonSymbol);
      // mflr r11
      OutStreamer.EmitInstruction(MCInstBuilder(PPC::MFLR).addReg(PPC::R11));
      // addis r11, r11, ha16(LazyPtr - AnonSymbol)
      const MCExpr *SubHa16 = PPCMCExpr::CreateHa(Sub, isDarwin, OutContext);
      OutStreamer.EmitInstruction(MCInstBuilder(PPC::ADDIS)
        .addReg(PPC::R11)
        .addReg(PPC::R11)
        .addExpr(SubHa16));
      // mtlr r0
      OutStreamer.EmitInstruction(MCInstBuilder(PPC::MTLR).addReg(PPC::R0));

      // ldu r12, lo16(LazyPtr - AnonSymbol)(r11)
      // lwzu r12, lo16(LazyPtr - AnonSymbol)(r11)
      const MCExpr *SubLo16 = PPCMCExpr::CreateLo(Sub, isDarwin, OutContext);
      OutStreamer.EmitInstruction(MCInstBuilder(isPPC64 ? PPC::LDU : PPC::LWZU)
        .addReg(PPC::R12)
        .addExpr(SubLo16).addExpr(SubLo16)
        .addReg(PPC::R11));
      // mtctr r12
      OutStreamer.EmitInstruction(MCInstBuilder(PPC::MTCTR).addReg(PPC::R12));
      // bctr
      OutStreamer.EmitInstruction(MCInstBuilder(PPC::BCTR));

      OutStreamer.SwitchSection(LSPSection);
      OutStreamer.EmitLabel(LazyPtr);
      OutStreamer.EmitSymbolAttribute(RawSym, MCSA_IndirectSymbol);

      MCSymbol *DyldStubBindingHelper =
        OutContext.GetOrCreateSymbol(StringRef("dyld_stub_binding_helper"));
      if (isPPC64) {
        // .quad dyld_stub_binding_helper
        OutStreamer.EmitSymbolValue(DyldStubBindingHelper, 8);
      } else {
        // .long dyld_stub_binding_helper
        OutStreamer.EmitSymbolValue(DyldStubBindingHelper, 4);
      }
    }
    OutStreamer.AddBlankLine();
    return;
  }
  
  const MCSection *StubSection =
    OutContext.getMachOSection("__TEXT","__symbol_stub1",
                               MCSectionMachO::S_SYMBOL_STUBS |
                               MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS,
                               16, SectionKind::getText());
  for (unsigned i = 0, e = Stubs.size(); i != e; ++i) {
    MCSymbol *Stub = Stubs[i].first;
    MCSymbol *RawSym = Stubs[i].second.getPointer();
    MCSymbol *LazyPtr = GetLazyPtr(Stub, OutContext);
    const MCExpr *LazyPtrExpr = MCSymbolRefExpr::Create(LazyPtr, OutContext);

    OutStreamer.SwitchSection(StubSection);
    EmitAlignment(4);
    OutStreamer.EmitLabel(Stub);
    OutStreamer.EmitSymbolAttribute(RawSym, MCSA_IndirectSymbol);

    // lis r11, ha16(LazyPtr)
    const MCExpr *LazyPtrHa16 =
      PPCMCExpr::CreateHa(LazyPtrExpr, isDarwin, OutContext);
    OutStreamer.EmitInstruction(MCInstBuilder(PPC::LIS)
      .addReg(PPC::R11)
      .addExpr(LazyPtrHa16));

    // ldu r12, lo16(LazyPtr)(r11)
    // lwzu r12, lo16(LazyPtr)(r11)
    const MCExpr *LazyPtrLo16 =
      PPCMCExpr::CreateLo(LazyPtrExpr, isDarwin, OutContext);
    OutStreamer.EmitInstruction(MCInstBuilder(isPPC64 ? PPC::LDU : PPC::LWZU)
      .addReg(PPC::R12)
      .addExpr(LazyPtrLo16).addExpr(LazyPtrLo16)
      .addReg(PPC::R11));

    // mtctr r12
    OutStreamer.EmitInstruction(MCInstBuilder(PPC::MTCTR).addReg(PPC::R12));
    // bctr
    OutStreamer.EmitInstruction(MCInstBuilder(PPC::BCTR));

    OutStreamer.SwitchSection(LSPSection);
    OutStreamer.EmitLabel(LazyPtr);
    OutStreamer.EmitSymbolAttribute(RawSym, MCSA_IndirectSymbol);

    MCSymbol *DyldStubBindingHelper =
      OutContext.GetOrCreateSymbol(StringRef("dyld_stub_binding_helper"));
    if (isPPC64) {
      // .quad dyld_stub_binding_helper
      OutStreamer.EmitSymbolValue(DyldStubBindingHelper, 8);
    } else {
      // .long dyld_stub_binding_helper
      OutStreamer.EmitSymbolValue(DyldStubBindingHelper, 4);
    }
  }
  
  OutStreamer.AddBlankLine();
}


bool PPCDarwinAsmPrinter::doFinalization(Module &M) {
  bool isPPC64 = TM.getDataLayout()->getPointerSizeInBits() == 64;

  // Darwin/PPC always uses mach-o.
  const TargetLoweringObjectFileMachO &TLOFMacho = 
    static_cast<const TargetLoweringObjectFileMachO &>(getObjFileLowering());
  MachineModuleInfoMachO &MMIMacho =
    MMI->getObjFileInfo<MachineModuleInfoMachO>();
  
  MachineModuleInfoMachO::SymbolListTy Stubs = MMIMacho.GetFnStubList();
  if (!Stubs.empty())
    EmitFunctionStubs(Stubs);

  if (MAI->doesSupportExceptionHandling() && MMI) {
    // Add the (possibly multiple) personalities to the set of global values.
    // Only referenced functions get into the Personalities list.
    const std::vector<const Function*> &Personalities = MMI->getPersonalities();
    for (std::vector<const Function*>::const_iterator I = Personalities.begin(),
         E = Personalities.end(); I != E; ++I) {
      if (*I) {
        MCSymbol *NLPSym = GetSymbolWithGlobalValueBase(*I, "$non_lazy_ptr");
        MachineModuleInfoImpl::StubValueTy &StubSym =
          MMIMacho.getGVStubEntry(NLPSym);
        StubSym = MachineModuleInfoImpl::StubValueTy(getSymbol(*I), true);
      }
    }
  }

  // Output stubs for dynamically-linked functions.
  Stubs = MMIMacho.GetGVStubList();
  
  // Output macho stubs for external and common global variables.
  if (!Stubs.empty()) {
    // Switch with ".non_lazy_symbol_pointer" directive.
    OutStreamer.SwitchSection(TLOFMacho.getNonLazySymbolPointerSection());
    EmitAlignment(isPPC64 ? 3 : 2);
    
    for (unsigned i = 0, e = Stubs.size(); i != e; ++i) {
      // L_foo$stub:
      OutStreamer.EmitLabel(Stubs[i].first);
      //   .indirect_symbol _foo
      MachineModuleInfoImpl::StubValueTy &MCSym = Stubs[i].second;
      OutStreamer.EmitSymbolAttribute(MCSym.getPointer(), MCSA_IndirectSymbol);

      if (MCSym.getInt())
        // External to current translation unit.
        OutStreamer.EmitIntValue(0, isPPC64 ? 8 : 4/*size*/);
      else
        // Internal to current translation unit.
        //
        // When we place the LSDA into the TEXT section, the type info pointers
        // need to be indirect and pc-rel. We accomplish this by using NLPs.
        // However, sometimes the types are local to the file. So we need to
        // fill in the value for the NLP in those cases.
        OutStreamer.EmitValue(MCSymbolRefExpr::Create(MCSym.getPointer(),
                                                      OutContext),
                              isPPC64 ? 8 : 4/*size*/);
    }

    Stubs.clear();
    OutStreamer.AddBlankLine();
  }

  Stubs = MMIMacho.GetHiddenGVStubList();
  if (!Stubs.empty()) {
    OutStreamer.SwitchSection(getObjFileLowering().getDataSection());
    EmitAlignment(isPPC64 ? 3 : 2);
    
    for (unsigned i = 0, e = Stubs.size(); i != e; ++i) {
      // L_foo$stub:
      OutStreamer.EmitLabel(Stubs[i].first);
      //   .long _foo
      OutStreamer.EmitValue(MCSymbolRefExpr::
                            Create(Stubs[i].second.getPointer(),
                                   OutContext),
                            isPPC64 ? 8 : 4/*size*/);
    }

    Stubs.clear();
    OutStreamer.AddBlankLine();
  }

  // Funny Darwin hack: This flag tells the linker that no global symbols
  // contain code that falls through to other global symbols (e.g. the obvious
  // implementation of multiple entry points).  If this doesn't occur, the
  // linker can safely perform dead code stripping.  Since LLVM never generates
  // code that does this, it is always safe to set.
  OutStreamer.EmitAssemblerFlag(MCAF_SubsectionsViaSymbols);

  return AsmPrinter::doFinalization(M);
}

/// createPPCAsmPrinterPass - Returns a pass that prints the PPC assembly code
/// for a MachineFunction to the given output stream, in a format that the
/// Darwin assembler can deal with.
///
static AsmPrinter *createPPCAsmPrinterPass(TargetMachine &tm,
                                           MCStreamer &Streamer) {
  const PPCSubtarget *Subtarget = &tm.getSubtarget<PPCSubtarget>();

  if (Subtarget->isDarwin())
    return new PPCDarwinAsmPrinter(tm, Streamer);
  return new PPCLinuxAsmPrinter(tm, Streamer);
}

// Force static initialization.
extern "C" void LLVMInitializePowerPCAsmPrinter() { 
  TargetRegistry::RegisterAsmPrinter(ThePPC32Target, createPPCAsmPrinterPass);
  TargetRegistry::RegisterAsmPrinter(ThePPC64Target, createPPCAsmPrinterPass);
}
