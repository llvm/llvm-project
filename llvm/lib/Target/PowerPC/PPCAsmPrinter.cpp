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
#include "PPCTargetMachine.h"
#include "PPCSubtarget.h"
#include "InstPrinter/PPCInstPrinter.h"
#include "MCTargetDesc/PPCPredicates.h"
#include "llvm/Constants.h"
#include "llvm/DebugInfo.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/Target/Mangler.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ELF.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/SmallString.h"
using namespace llvm;

namespace {
  class PPCAsmPrinter : public AsmPrinter {
  protected:
    DenseMap<MCSymbol*, MCSymbol*> TOC;
    const PPCSubtarget &Subtarget;
    uint64_t TOCLabelID;
  public:
    explicit PPCAsmPrinter(TargetMachine &TM, MCStreamer &Streamer)
      : AsmPrinter(TM, Streamer),
        Subtarget(TM.getSubtarget<PPCSubtarget>()), TOCLabelID(0) {}

    virtual const char *getPassName() const {
      return "PowerPC Assembly Printer";
    }


    virtual void EmitInstruction(const MachineInstr *MI);

    void printOperand(const MachineInstr *MI, unsigned OpNo, raw_ostream &O);

    bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                         unsigned AsmVariant, const char *ExtraCode,
                         raw_ostream &O);
    bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                               unsigned AsmVariant, const char *ExtraCode,
                               raw_ostream &O);

    MachineLocation getDebugValueLocation(const MachineInstr *MI) const {
      MachineLocation Location;
      assert(MI->getNumOperands() == 4 && "Invalid no. of machine operands!");
      // Frame address.  Currently handles register +- offset only.
      if (MI->getOperand(0).isReg() && MI->getOperand(2).isImm())
        Location.set(MI->getOperand(0).getReg(), MI->getOperand(2).getImm());
      else {
        DEBUG(dbgs() << "DBG_VALUE instruction ignored! " << *MI << "\n");
      }
      return Location;
    }
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
            StubValueTy(Mang->getSymbol(GV), !GV->hasInternalLinkage());
      } else if (GV->isDeclaration() || GV->hasCommonLinkage() ||
                 GV->hasAvailableExternallyLinkage()) {
        SymToPrint = GetSymbolWithGlobalValueBase(GV, "$non_lazy_ptr");
        
        MachineModuleInfoImpl::StubValueTy &StubSym = 
          MMI->getObjFileInfo<MachineModuleInfoMachO>().
                    getHiddenGVStubEntry(SymToPrint);
        if (StubSym.getPointer() == 0)
          StubSym = MachineModuleInfoImpl::
            StubValueTy(Mang->getSymbol(GV), !GV->hasInternalLinkage());
      } else {
        SymToPrint = Mang->getSymbol(GV);
      }
    } else {
      SymToPrint = Mang->getSymbol(GV);
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


/// EmitInstruction -- Print out a single PowerPC MI in Darwin syntax to
/// the current output stream.
///
void PPCAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  MCInst TmpInst;
  
  // Lower multi-instruction pseudo operations.
  switch (MI->getOpcode()) {
  default: break;
  case TargetOpcode::DBG_VALUE: {
    if (!isVerbose() || !OutStreamer.hasRawTextSupport()) return;
      
    SmallString<32> Str;
    raw_svector_ostream O(Str);
    unsigned NOps = MI->getNumOperands();
    assert(NOps==4);
    O << '\t' << MAI->getCommentString() << "DEBUG_VALUE: ";
    // cast away const; DIetc do not take const operands for some reason.
    DIVariable V(const_cast<MDNode *>(MI->getOperand(NOps-1).getMetadata()));
    O << V.getName();
    O << " <- ";
    // Frame address.  Currently handles register +- offset only.
    assert(MI->getOperand(0).isReg() && MI->getOperand(1).isImm());
    O << '['; printOperand(MI, 0, O); O << '+'; printOperand(MI, 1, O);
    O << ']';
    O << "+";
    printOperand(MI, NOps-2, O);
    OutStreamer.EmitRawText(O.str());
    return;
  }
      
  case PPC::MovePCtoLR:
  case PPC::MovePCtoLR8: {
    // Transform %LR = MovePCtoLR
    // Into this, where the label is the PIC base: 
    //     bl L1$pb
    // L1$pb:
    MCSymbol *PICBase = MF->getPICBaseSymbol();
    
    // Emit the 'bl'.
    TmpInst.setOpcode(PPC::BL_Darwin); // Darwin vs SVR4 doesn't matter here.
    
    
    // FIXME: We would like an efficient form for this, so we don't have to do
    // a lot of extra uniquing.
    TmpInst.addOperand(MCOperand::CreateExpr(MCSymbolRefExpr::
                                             Create(PICBase, OutContext)));
    OutStreamer.EmitInstruction(TmpInst);
    
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
      MOSymbol = Mang->getSymbol(MO.getGlobal());
    else if (MO.isCPI())
      MOSymbol = GetCPISymbol(MO.getIndex());
    else if (MO.isJTI())
      MOSymbol = GetJTISymbol(MO.getIndex());
    MCSymbol *&TOCEntry = TOC[MOSymbol];
    // To avoid name clash check if the name already exists.
    while (TOCEntry == 0) {
      if (OutContext.LookupSymbol(Twine(MAI->getPrivateGlobalPrefix()) +
                                  "C" + Twine(TOCLabelID++)) == 0) {
        TOCEntry = GetTempSymbol("C", TOCLabelID);
      }
    }

    const MCExpr *Exp =
      MCSymbolRefExpr::Create(TOCEntry, MCSymbolRefExpr::VK_PPC_TOC_ENTRY,
                              OutContext);
    TmpInst.getOperand(1) = MCOperand::CreateExpr(Exp);
    OutStreamer.EmitInstruction(TmpInst);
    return;
  }
      
  case PPC::MFCRpseud:
  case PPC::MFCR8pseud:
    // Transform: %R3 = MFCRpseud %CR7
    // Into:      %R3 = MFCR      ;; cr7
    OutStreamer.AddComment(PPCInstPrinter::
                           getRegisterName(MI->getOperand(1).getReg()));
    TmpInst.setOpcode(Subtarget.isPPC64() ? PPC::MFCR8 : PPC::MFCR);
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    OutStreamer.EmitInstruction(TmpInst);
    return;
  case PPC::SYNC:
    // In Book E sync is called msync, handle this special case here...
    if (Subtarget.isBookE()) {
      OutStreamer.EmitRawText(StringRef("\tmsync"));
      return;
    }
  }

  LowerPPCMachineInstrToMCInst(MI, TmpInst, *this, Subtarget.isDarwin());
  OutStreamer.EmitInstruction(TmpInst);
}

void PPCLinuxAsmPrinter::EmitFunctionEntryLabel() {
  if (!Subtarget.isPPC64())  // linux/ppc32 - Normal entry label.
    return AsmPrinter::EmitFunctionEntryLabel();
    
  // Emit an official procedure descriptor.
  const MCSection *Current = OutStreamer.getCurrentSection();
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
                        8/*size*/, 0/*addrspace*/);
  MCSymbol *Symbol2 = OutContext.GetOrCreateSymbol(StringRef(".TOC."));
  // Generates a R_PPC64_TOC relocation for TOC base insertion.
  OutStreamer.EmitValue(MCSymbolRefExpr::Create(Symbol2,
                        MCSymbolRefExpr::VK_PPC_TOC, OutContext),
                        8/*size*/, 0/*addrspace*/);
  // Emit a null environment pointer.
  OutStreamer.EmitIntValue(0, 8 /* size */, 0 /* addrspace */);
  OutStreamer.SwitchSection(Current);

  MCSymbol *RealFnSym = OutContext.GetOrCreateSymbol(
                          ".L." + Twine(CurrentFnSym->getName()));
  OutStreamer.EmitLabel(RealFnSym);
  CurrentFnSymForSize = RealFnSym;
}


bool PPCLinuxAsmPrinter::doFinalization(Module &M) {
  const DataLayout *TD = TM.getDataLayout();

  bool isPPC64 = TD->getPointerSizeInBits() == 64;

  if (isPPC64 && !TOC.empty()) {
    const MCSectionELF *Section = OutStreamer.getContext().getELFSection(".toc",
        ELF::SHT_PROGBITS, ELF::SHF_WRITE | ELF::SHF_ALLOC,
        SectionKind::getReadOnly());
    OutStreamer.SwitchSection(Section);

    // FIXME: This is nondeterminstic!
    for (DenseMap<MCSymbol*, MCSymbol*>::iterator I = TOC.begin(),
         E = TOC.end(); I != E; ++I) {
      OutStreamer.EmitLabel(I->second);
      MCSymbol *S = OutContext.GetOrCreateSymbol(I->first->getName());
      OutStreamer.EmitTCEntry(*S);
    }
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
    "power6",
    "power7",
    "ppc64"
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
  if (OutStreamer.hasRawTextSupport())
    OutStreamer.EmitRawText("\t.machine " + Twine(CPUDirectives[Directive]));

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
  SmallString<128> TmpStr(Sym->getName().begin(), Sym->getName().end()-5);
  TmpStr += "$lazy_ptr";
  return Ctx.GetOrCreateSymbol(TmpStr.str());
}

static MCSymbol *GetAnonSym(MCSymbol *Sym, MCContext &Ctx) {
  // Add $tmp suffix to $stub, yielding $stub$tmp.
  SmallString<128> TmpStr(Sym->getName().begin(), Sym->getName().end());
  TmpStr += "$tmp";
  return Ctx.GetOrCreateSymbol(TmpStr.str());
}

void PPCDarwinAsmPrinter::
EmitFunctionStubs(const MachineModuleInfoMachO::SymbolListTy &Stubs) {
  bool isPPC64 = TM.getDataLayout()->getPointerSizeInBits() == 64;
  
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
      // FIXME: MCize this.
      OutStreamer.EmitRawText(StringRef("\tmflr r0"));
      OutStreamer.EmitRawText("\tbcl 20,31," + Twine(AnonSymbol->getName()));
      OutStreamer.EmitLabel(AnonSymbol);
      OutStreamer.EmitRawText(StringRef("\tmflr r11"));
      OutStreamer.EmitRawText("\taddis r11,r11,ha16("+Twine(LazyPtr->getName())+
                              "-" + AnonSymbol->getName() + ")");
      OutStreamer.EmitRawText(StringRef("\tmtlr r0"));
      
      if (isPPC64)
        OutStreamer.EmitRawText("\tldu r12,lo16(" + Twine(LazyPtr->getName()) +
                                "-" + AnonSymbol->getName() + ")(r11)");
      else
        OutStreamer.EmitRawText("\tlwzu r12,lo16(" + Twine(LazyPtr->getName()) +
                                "-" + AnonSymbol->getName() + ")(r11)");
      OutStreamer.EmitRawText(StringRef("\tmtctr r12"));
      OutStreamer.EmitRawText(StringRef("\tbctr"));
      
      OutStreamer.SwitchSection(LSPSection);
      OutStreamer.EmitLabel(LazyPtr);
      OutStreamer.EmitSymbolAttribute(RawSym, MCSA_IndirectSymbol);
      
      if (isPPC64)
        OutStreamer.EmitRawText(StringRef("\t.quad dyld_stub_binding_helper"));
      else
        OutStreamer.EmitRawText(StringRef("\t.long dyld_stub_binding_helper"));
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

    OutStreamer.SwitchSection(StubSection);
    EmitAlignment(4);
    OutStreamer.EmitLabel(Stub);
    OutStreamer.EmitSymbolAttribute(RawSym, MCSA_IndirectSymbol);
    OutStreamer.EmitRawText("\tlis r11,ha16(" + Twine(LazyPtr->getName()) +")");
    if (isPPC64)
      OutStreamer.EmitRawText("\tldu r12,lo16(" + Twine(LazyPtr->getName()) +
                              ")(r11)");
    else
      OutStreamer.EmitRawText("\tlwzu r12,lo16(" + Twine(LazyPtr->getName()) +
                              ")(r11)");
    OutStreamer.EmitRawText(StringRef("\tmtctr r12"));
    OutStreamer.EmitRawText(StringRef("\tbctr"));
    OutStreamer.SwitchSection(LSPSection);
    OutStreamer.EmitLabel(LazyPtr);
    OutStreamer.EmitSymbolAttribute(RawSym, MCSA_IndirectSymbol);
    
    if (isPPC64)
      OutStreamer.EmitRawText(StringRef("\t.quad dyld_stub_binding_helper"));
    else
      OutStreamer.EmitRawText(StringRef("\t.long dyld_stub_binding_helper"));
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
        StubSym = MachineModuleInfoImpl::StubValueTy(Mang->getSymbol(*I), true);
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
        OutStreamer.EmitIntValue(0, isPPC64 ? 8 : 4/*size*/, 0/*addrspace*/);
      else
        // Internal to current translation unit.
        //
        // When we place the LSDA into the TEXT section, the type info pointers
        // need to be indirect and pc-rel. We accomplish this by using NLPs.
        // However, sometimes the types are local to the file. So we need to
        // fill in the value for the NLP in those cases.
        OutStreamer.EmitValue(MCSymbolRefExpr::Create(MCSym.getPointer(),
                                                      OutContext),
                              isPPC64 ? 8 : 4/*size*/, 0/*addrspace*/);
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
                            isPPC64 ? 8 : 4/*size*/, 0/*addrspace*/);
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
