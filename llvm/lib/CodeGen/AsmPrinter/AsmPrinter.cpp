//===-- AsmPrinter.cpp - Common AsmPrinter code ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the AsmPrinter class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "llvm/CodeGen/AsmPrinter.h"
#include "DwarfDebug.h"
#include "DwarfException.h"
#include "llvm/DebugInfo.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/GCMetadataPrinter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/Mangler.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Timer.h"
using namespace llvm;

static const char *DWARFGroupName = "DWARF Emission";
static const char *DbgTimerName = "DWARF Debug Writer";
static const char *EHTimerName = "DWARF Exception Writer";

STATISTIC(EmittedInsts, "Number of machine instrs printed");

char AsmPrinter::ID = 0;

typedef DenseMap<GCStrategy*,GCMetadataPrinter*> gcp_map_type;
static gcp_map_type &getGCMap(void *&P) {
  if (P == 0)
    P = new gcp_map_type();
  return *(gcp_map_type*)P;
}


/// getGVAlignmentLog2 - Return the alignment to use for the specified global
/// value in log2 form.  This rounds up to the preferred alignment if possible
/// and legal.
static unsigned getGVAlignmentLog2(const GlobalValue *GV, const DataLayout &TD,
                                   unsigned InBits = 0) {
  unsigned NumBits = 0;
  if (const GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV))
    NumBits = TD.getPreferredAlignmentLog(GVar);

  // If InBits is specified, round it to it.
  if (InBits > NumBits)
    NumBits = InBits;

  // If the GV has a specified alignment, take it into account.
  if (GV->getAlignment() == 0)
    return NumBits;

  unsigned GVAlign = Log2_32(GV->getAlignment());

  // If the GVAlign is larger than NumBits, or if we are required to obey
  // NumBits because the GV has an assigned section, obey it.
  if (GVAlign > NumBits || GV->hasSection())
    NumBits = GVAlign;
  return NumBits;
}




AsmPrinter::AsmPrinter(TargetMachine &tm, MCStreamer &Streamer)
  : MachineFunctionPass(ID),
    TM(tm), MAI(tm.getMCAsmInfo()),
    OutContext(Streamer.getContext()),
    OutStreamer(Streamer),
    LastMI(0), LastFn(0), Counter(~0U), SetCounter(0) {
  DD = 0; DE = 0; MMI = 0; LI = 0;
  CurrentFnSym = CurrentFnSymForSize = 0;
  GCMetadataPrinters = 0;
  VerboseAsm = Streamer.isVerboseAsm();
}

AsmPrinter::~AsmPrinter() {
  assert(DD == 0 && DE == 0 && "Debug/EH info didn't get finalized");

  if (GCMetadataPrinters != 0) {
    gcp_map_type &GCMap = getGCMap(GCMetadataPrinters);

    for (gcp_map_type::iterator I = GCMap.begin(), E = GCMap.end(); I != E; ++I)
      delete I->second;
    delete &GCMap;
    GCMetadataPrinters = 0;
  }

  delete &OutStreamer;
}

/// getFunctionNumber - Return a unique ID for the current function.
///
unsigned AsmPrinter::getFunctionNumber() const {
  return MF->getFunctionNumber();
}

const TargetLoweringObjectFile &AsmPrinter::getObjFileLowering() const {
  return TM.getTargetLowering()->getObjFileLowering();
}


/// getDataLayout - Return information about data layout.
const DataLayout &AsmPrinter::getDataLayout() const {
  return *TM.getDataLayout();
}

/// getCurrentSection() - Return the current section we are emitting to.
const MCSection *AsmPrinter::getCurrentSection() const {
  return OutStreamer.getCurrentSection();
}



void AsmPrinter::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
  AU.addRequired<MachineModuleInfo>();
  AU.addRequired<GCModuleInfo>();
  if (isVerbose())
    AU.addRequired<MachineLoopInfo>();
}

bool AsmPrinter::doInitialization(Module &M) {
  MMI = getAnalysisIfAvailable<MachineModuleInfo>();
  MMI->AnalyzeModule(M);

  // Initialize TargetLoweringObjectFile.
  const_cast<TargetLoweringObjectFile&>(getObjFileLowering())
    .Initialize(OutContext, TM);

  Mang = new Mangler(OutContext, *TM.getDataLayout());

  // Allow the target to emit any magic that it wants at the start of the file.
  EmitStartOfAsmFile(M);

  // Very minimal debug info. It is ignored if we emit actual debug info. If we
  // don't, this at least helps the user find where a global came from.
  if (MAI->hasSingleParameterDotFile()) {
    // .file "foo.c"
    OutStreamer.EmitFileDirective(M.getModuleIdentifier());
  }

  GCModuleInfo *MI = getAnalysisIfAvailable<GCModuleInfo>();
  assert(MI && "AsmPrinter didn't require GCModuleInfo?");
  for (GCModuleInfo::iterator I = MI->begin(), E = MI->end(); I != E; ++I)
    if (GCMetadataPrinter *MP = GetOrCreateGCPrinter(*I))
      MP->beginAssembly(*this);

  // Emit module-level inline asm if it exists.
  if (!M.getModuleInlineAsm().empty()) {
    OutStreamer.AddComment("Start of file scope inline assembly");
    OutStreamer.AddBlankLine();
    EmitInlineAsm(M.getModuleInlineAsm()+"\n");
    OutStreamer.AddComment("End of file scope inline assembly");
    OutStreamer.AddBlankLine();
  }

  if (MAI->doesSupportDebugInformation())
    DD = new DwarfDebug(this, &M);

  switch (MAI->getExceptionHandlingType()) {
  case ExceptionHandling::None:
    return false;
  case ExceptionHandling::SjLj:
  case ExceptionHandling::DwarfCFI:
    DE = new DwarfCFIException(this);
    return false;
  case ExceptionHandling::ARM:
    DE = new ARMException(this);
    return false;
  case ExceptionHandling::Win64:
    DE = new Win64Exception(this);
    return false;
  }

  llvm_unreachable("Unknown exception type.");
}

void AsmPrinter::EmitLinkage(unsigned Linkage, MCSymbol *GVSym) const {
  switch ((GlobalValue::LinkageTypes)Linkage) {
  case GlobalValue::CommonLinkage:
  case GlobalValue::LinkOnceAnyLinkage:
  case GlobalValue::LinkOnceODRLinkage:
  case GlobalValue::LinkOnceODRAutoHideLinkage:
  case GlobalValue::WeakAnyLinkage:
  case GlobalValue::WeakODRLinkage:
  case GlobalValue::LinkerPrivateWeakLinkage:
    if (MAI->getWeakDefDirective() != 0) {
      // .globl _foo
      OutStreamer.EmitSymbolAttribute(GVSym, MCSA_Global);

      if ((GlobalValue::LinkageTypes)Linkage !=
          GlobalValue::LinkOnceODRAutoHideLinkage)
        // .weak_definition _foo
        OutStreamer.EmitSymbolAttribute(GVSym, MCSA_WeakDefinition);
      else
        OutStreamer.EmitSymbolAttribute(GVSym, MCSA_WeakDefAutoPrivate);
    } else if (MAI->getLinkOnceDirective() != 0) {
      // .globl _foo
      OutStreamer.EmitSymbolAttribute(GVSym, MCSA_Global);
      //NOTE: linkonce is handled by the section the symbol was assigned to.
    } else {
      // .weak _foo
      OutStreamer.EmitSymbolAttribute(GVSym, MCSA_Weak);
    }
    break;
  case GlobalValue::DLLExportLinkage:
  case GlobalValue::AppendingLinkage:
    // FIXME: appending linkage variables should go into a section of
    // their name or something.  For now, just emit them as external.
  case GlobalValue::ExternalLinkage:
    // If external or appending, declare as a global symbol.
    // .globl _foo
    OutStreamer.EmitSymbolAttribute(GVSym, MCSA_Global);
    break;
  case GlobalValue::PrivateLinkage:
  case GlobalValue::InternalLinkage:
  case GlobalValue::LinkerPrivateLinkage:
    break;
  default:
    llvm_unreachable("Unknown linkage type!");
  }
}


/// EmitGlobalVariable - Emit the specified global variable to the .s file.
void AsmPrinter::EmitGlobalVariable(const GlobalVariable *GV) {
  if (GV->hasInitializer()) {
    // Check to see if this is a special global used by LLVM, if so, emit it.
    if (EmitSpecialLLVMGlobal(GV))
      return;

    if (isVerbose()) {
      WriteAsOperand(OutStreamer.GetCommentOS(), GV,
                     /*PrintType=*/false, GV->getParent());
      OutStreamer.GetCommentOS() << '\n';
    }
  }

  MCSymbol *GVSym = Mang->getSymbol(GV);
  EmitVisibility(GVSym, GV->getVisibility(), !GV->isDeclaration());

  if (!GV->hasInitializer())   // External globals require no extra code.
    return;

  if (MAI->hasDotTypeDotSizeDirective())
    OutStreamer.EmitSymbolAttribute(GVSym, MCSA_ELF_TypeObject);

  SectionKind GVKind = TargetLoweringObjectFile::getKindForGlobal(GV, TM);

  const DataLayout *TD = TM.getDataLayout();
  uint64_t Size = TD->getTypeAllocSize(GV->getType()->getElementType());

  // If the alignment is specified, we *must* obey it.  Overaligning a global
  // with a specified alignment is a prompt way to break globals emitted to
  // sections and expected to be contiguous (e.g. ObjC metadata).
  unsigned AlignLog = getGVAlignmentLog2(GV, *TD);

  // Handle common and BSS local symbols (.lcomm).
  if (GVKind.isCommon() || GVKind.isBSSLocal()) {
    if (Size == 0) Size = 1;   // .comm Foo, 0 is undefined, avoid it.
    unsigned Align = 1 << AlignLog;

    // Handle common symbols.
    if (GVKind.isCommon()) {
      if (!getObjFileLowering().getCommDirectiveSupportsAlignment())
        Align = 0;

      // .comm _foo, 42, 4
      OutStreamer.EmitCommonSymbol(GVSym, Size, Align);
      return;
    }

    // Handle local BSS symbols.
    if (MAI->hasMachoZeroFillDirective()) {
      const MCSection *TheSection =
        getObjFileLowering().SectionForGlobal(GV, GVKind, Mang, TM);
      // .zerofill __DATA, __bss, _foo, 400, 5
      OutStreamer.EmitZerofill(TheSection, GVSym, Size, Align);
      return;
    }

    if (Align == 1 ||
        MAI->getLCOMMDirectiveAlignmentType() != LCOMM::NoAlignment) {
      // .lcomm _foo, 42
      OutStreamer.EmitLocalCommonSymbol(GVSym, Size, Align);
      return;
    }

    if (!getObjFileLowering().getCommDirectiveSupportsAlignment())
      Align = 0;

    // .local _foo
    OutStreamer.EmitSymbolAttribute(GVSym, MCSA_Local);
    // .comm _foo, 42, 4
    OutStreamer.EmitCommonSymbol(GVSym, Size, Align);
    return;
  }

  const MCSection *TheSection =
    getObjFileLowering().SectionForGlobal(GV, GVKind, Mang, TM);

  // Handle the zerofill directive on darwin, which is a special form of BSS
  // emission.
  if (GVKind.isBSSExtern() && MAI->hasMachoZeroFillDirective()) {
    if (Size == 0) Size = 1;  // zerofill of 0 bytes is undefined.

    // .globl _foo
    OutStreamer.EmitSymbolAttribute(GVSym, MCSA_Global);
    // .zerofill __DATA, __common, _foo, 400, 5
    OutStreamer.EmitZerofill(TheSection, GVSym, Size, 1 << AlignLog);
    return;
  }

  // Handle thread local data for mach-o which requires us to output an
  // additional structure of data and mangle the original symbol so that we
  // can reference it later.
  //
  // TODO: This should become an "emit thread local global" method on TLOF.
  // All of this macho specific stuff should be sunk down into TLOFMachO and
  // stuff like "TLSExtraDataSection" should no longer be part of the parent
  // TLOF class.  This will also make it more obvious that stuff like
  // MCStreamer::EmitTBSSSymbol is macho specific and only called from macho
  // specific code.
  if (GVKind.isThreadLocal() && MAI->hasMachoTBSSDirective()) {
    // Emit the .tbss symbol
    MCSymbol *MangSym =
      OutContext.GetOrCreateSymbol(GVSym->getName() + Twine("$tlv$init"));

    if (GVKind.isThreadBSS())
      OutStreamer.EmitTBSSSymbol(TheSection, MangSym, Size, 1 << AlignLog);
    else if (GVKind.isThreadData()) {
      OutStreamer.SwitchSection(TheSection);

      EmitAlignment(AlignLog, GV);
      OutStreamer.EmitLabel(MangSym);

      EmitGlobalConstant(GV->getInitializer());
    }

    OutStreamer.AddBlankLine();

    // Emit the variable struct for the runtime.
    const MCSection *TLVSect
      = getObjFileLowering().getTLSExtraDataSection();

    OutStreamer.SwitchSection(TLVSect);
    // Emit the linkage here.
    EmitLinkage(GV->getLinkage(), GVSym);
    OutStreamer.EmitLabel(GVSym);

    // Three pointers in size:
    //   - __tlv_bootstrap - used to make sure support exists
    //   - spare pointer, used when mapped by the runtime
    //   - pointer to mangled symbol above with initializer
    unsigned PtrSize = TD->getPointerSizeInBits()/8;
    OutStreamer.EmitSymbolValue(GetExternalSymbolSymbol("_tlv_bootstrap"),
                          PtrSize, 0);
    OutStreamer.EmitIntValue(0, PtrSize, 0);
    OutStreamer.EmitSymbolValue(MangSym, PtrSize, 0);

    OutStreamer.AddBlankLine();
    return;
  }

  OutStreamer.SwitchSection(TheSection);

  EmitLinkage(GV->getLinkage(), GVSym);
  EmitAlignment(AlignLog, GV);

  OutStreamer.EmitLabel(GVSym);

  EmitGlobalConstant(GV->getInitializer());

  if (MAI->hasDotTypeDotSizeDirective())
    // .size foo, 42
    OutStreamer.EmitELFSize(GVSym, MCConstantExpr::Create(Size, OutContext));

  OutStreamer.AddBlankLine();
}

/// EmitFunctionHeader - This method emits the header for the current
/// function.
void AsmPrinter::EmitFunctionHeader() {
  // Print out constants referenced by the function
  EmitConstantPool();

  // Print the 'header' of function.
  const Function *F = MF->getFunction();

  OutStreamer.SwitchSection(getObjFileLowering().SectionForGlobal(F, Mang, TM));
  EmitVisibility(CurrentFnSym, F->getVisibility());

  EmitLinkage(F->getLinkage(), CurrentFnSym);
  EmitAlignment(MF->getAlignment(), F);

  if (MAI->hasDotTypeDotSizeDirective())
    OutStreamer.EmitSymbolAttribute(CurrentFnSym, MCSA_ELF_TypeFunction);

  if (isVerbose()) {
    WriteAsOperand(OutStreamer.GetCommentOS(), F,
                   /*PrintType=*/false, F->getParent());
    OutStreamer.GetCommentOS() << '\n';
  }

  // Emit the CurrentFnSym.  This is a virtual function to allow targets to
  // do their wild and crazy things as required.
  EmitFunctionEntryLabel();

  // If the function had address-taken blocks that got deleted, then we have
  // references to the dangling symbols.  Emit them at the start of the function
  // so that we don't get references to undefined symbols.
  std::vector<MCSymbol*> DeadBlockSyms;
  MMI->takeDeletedSymbolsForFunction(F, DeadBlockSyms);
  for (unsigned i = 0, e = DeadBlockSyms.size(); i != e; ++i) {
    OutStreamer.AddComment("Address taken block that was later removed");
    OutStreamer.EmitLabel(DeadBlockSyms[i]);
  }

  // Add some workaround for linkonce linkage on Cygwin\MinGW.
  if (MAI->getLinkOnceDirective() != 0 &&
      (F->hasLinkOnceLinkage() || F->hasWeakLinkage())) {
    // FIXME: What is this?
    MCSymbol *FakeStub =
      OutContext.GetOrCreateSymbol(Twine("Lllvm$workaround$fake$stub$")+
                                   CurrentFnSym->getName());
    OutStreamer.EmitLabel(FakeStub);
  }

  // Emit pre-function debug and/or EH information.
  if (DE) {
    NamedRegionTimer T(EHTimerName, DWARFGroupName, TimePassesIsEnabled);
    DE->BeginFunction(MF);
  }
  if (DD) {
    NamedRegionTimer T(DbgTimerName, DWARFGroupName, TimePassesIsEnabled);
    DD->beginFunction(MF);
  }
}

/// EmitFunctionEntryLabel - Emit the label that is the entrypoint for the
/// function.  This can be overridden by targets as required to do custom stuff.
void AsmPrinter::EmitFunctionEntryLabel() {
  // The function label could have already been emitted if two symbols end up
  // conflicting due to asm renaming.  Detect this and emit an error.
  if (CurrentFnSym->isUndefined())
    return OutStreamer.EmitLabel(CurrentFnSym);

  report_fatal_error("'" + Twine(CurrentFnSym->getName()) +
                     "' label emitted multiple times to assembly file");
}

/// emitComments - Pretty-print comments for instructions.
static void emitComments(const MachineInstr &MI, raw_ostream &CommentOS) {
  const MachineFunction *MF = MI.getParent()->getParent();
  const TargetMachine &TM = MF->getTarget();

  // Check for spills and reloads
  int FI;

  const MachineFrameInfo *FrameInfo = MF->getFrameInfo();

  // We assume a single instruction only has a spill or reload, not
  // both.
  const MachineMemOperand *MMO;
  if (TM.getInstrInfo()->isLoadFromStackSlotPostFE(&MI, FI)) {
    if (FrameInfo->isSpillSlotObjectIndex(FI)) {
      MMO = *MI.memoperands_begin();
      CommentOS << MMO->getSize() << "-byte Reload\n";
    }
  } else if (TM.getInstrInfo()->hasLoadFromStackSlot(&MI, MMO, FI)) {
    if (FrameInfo->isSpillSlotObjectIndex(FI))
      CommentOS << MMO->getSize() << "-byte Folded Reload\n";
  } else if (TM.getInstrInfo()->isStoreToStackSlotPostFE(&MI, FI)) {
    if (FrameInfo->isSpillSlotObjectIndex(FI)) {
      MMO = *MI.memoperands_begin();
      CommentOS << MMO->getSize() << "-byte Spill\n";
    }
  } else if (TM.getInstrInfo()->hasStoreToStackSlot(&MI, MMO, FI)) {
    if (FrameInfo->isSpillSlotObjectIndex(FI))
      CommentOS << MMO->getSize() << "-byte Folded Spill\n";
  }

  // Check for spill-induced copies
  if (MI.getAsmPrinterFlag(MachineInstr::ReloadReuse))
    CommentOS << " Reload Reuse\n";
}

/// emitImplicitDef - This method emits the specified machine instruction
/// that is an implicit def.
static void emitImplicitDef(const MachineInstr *MI, AsmPrinter &AP) {
  unsigned RegNo = MI->getOperand(0).getReg();
  AP.OutStreamer.AddComment(Twine("implicit-def: ") +
                            AP.TM.getRegisterInfo()->getName(RegNo));
  AP.OutStreamer.AddBlankLine();
}

static void emitKill(const MachineInstr *MI, AsmPrinter &AP) {
  std::string Str = "kill:";
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &Op = MI->getOperand(i);
    assert(Op.isReg() && "KILL instruction must have only register operands");
    Str += ' ';
    Str += AP.TM.getRegisterInfo()->getName(Op.getReg());
    Str += (Op.isDef() ? "<def>" : "<kill>");
  }
  AP.OutStreamer.AddComment(Str);
  AP.OutStreamer.AddBlankLine();
}

/// emitDebugValueComment - This method handles the target-independent form
/// of DBG_VALUE, returning true if it was able to do so.  A false return
/// means the target will need to handle MI in EmitInstruction.
static bool emitDebugValueComment(const MachineInstr *MI, AsmPrinter &AP) {
  // This code handles only the 3-operand target-independent form.
  if (MI->getNumOperands() != 3)
    return false;

  SmallString<128> Str;
  raw_svector_ostream OS(Str);
  OS << '\t' << AP.MAI->getCommentString() << "DEBUG_VALUE: ";

  // cast away const; DIetc do not take const operands for some reason.
  DIVariable V(const_cast<MDNode*>(MI->getOperand(2).getMetadata()));
  if (V.getContext().isSubprogram())
    OS << DISubprogram(V.getContext()).getDisplayName() << ":";
  OS << V.getName() << " <- ";

  // Register or immediate value. Register 0 means undef.
  if (MI->getOperand(0).isFPImm()) {
    APFloat APF = APFloat(MI->getOperand(0).getFPImm()->getValueAPF());
    if (MI->getOperand(0).getFPImm()->getType()->isFloatTy()) {
      OS << (double)APF.convertToFloat();
    } else if (MI->getOperand(0).getFPImm()->getType()->isDoubleTy()) {
      OS << APF.convertToDouble();
    } else {
      // There is no good way to print long double.  Convert a copy to
      // double.  Ah well, it's only a comment.
      bool ignored;
      APF.convert(APFloat::IEEEdouble, APFloat::rmNearestTiesToEven,
                  &ignored);
      OS << "(long double) " << APF.convertToDouble();
    }
  } else if (MI->getOperand(0).isImm()) {
    OS << MI->getOperand(0).getImm();
  } else if (MI->getOperand(0).isCImm()) {
    MI->getOperand(0).getCImm()->getValue().print(OS, false /*isSigned*/);
  } else {
    assert(MI->getOperand(0).isReg() && "Unknown operand type");
    if (MI->getOperand(0).getReg() == 0) {
      // Suppress offset, it is not meaningful here.
      OS << "undef";
      // NOTE: Want this comment at start of line, don't emit with AddComment.
      AP.OutStreamer.EmitRawText(OS.str());
      return true;
    }
    OS << AP.TM.getRegisterInfo()->getName(MI->getOperand(0).getReg());
  }

  OS << '+' << MI->getOperand(1).getImm();
  // NOTE: Want this comment at start of line, don't emit with AddComment.
  AP.OutStreamer.EmitRawText(OS.str());
  return true;
}

AsmPrinter::CFIMoveType AsmPrinter::needsCFIMoves() {
  if (MAI->getExceptionHandlingType() == ExceptionHandling::DwarfCFI &&
      MF->getFunction()->needsUnwindTableEntry())
    return CFI_M_EH;

  if (MMI->hasDebugInfo())
    return CFI_M_Debug;

  return CFI_M_None;
}

bool AsmPrinter::needsSEHMoves() {
  return MAI->getExceptionHandlingType() == ExceptionHandling::Win64 &&
    MF->getFunction()->needsUnwindTableEntry();
}

bool AsmPrinter::needsRelocationsForDwarfStringPool() const {
  return MAI->doesDwarfUseRelocationsAcrossSections();
}

void AsmPrinter::emitPrologLabel(const MachineInstr &MI) {
  MCSymbol *Label = MI.getOperand(0).getMCSymbol();

  if (MAI->getExceptionHandlingType() != ExceptionHandling::DwarfCFI)
    return;

  if (needsCFIMoves() == CFI_M_None)
    return;

  if (MMI->getCompactUnwindEncoding() != 0)
    OutStreamer.EmitCompactUnwindEncoding(MMI->getCompactUnwindEncoding());

  MachineModuleInfo &MMI = MF->getMMI();
  std::vector<MachineMove> &Moves = MMI.getFrameMoves();
  bool FoundOne = false;
  (void)FoundOne;
  for (std::vector<MachineMove>::iterator I = Moves.begin(),
         E = Moves.end(); I != E; ++I) {
    if (I->getLabel() == Label) {
      EmitCFIFrameMove(*I);
      FoundOne = true;
    }
  }
  assert(FoundOne);
}

/// EmitFunctionBody - This method emits the body and trailer for a
/// function.
void AsmPrinter::EmitFunctionBody() {
  // Emit target-specific gunk before the function body.
  EmitFunctionBodyStart();

  bool ShouldPrintDebugScopes = DD && MMI->hasDebugInfo();

  // Print out code for the function.
  bool HasAnyRealCode = false;
  const MachineInstr *LastMI = 0;
  for (MachineFunction::const_iterator I = MF->begin(), E = MF->end();
       I != E; ++I) {
    // Print a label for the basic block.
    EmitBasicBlockStart(I);
    for (MachineBasicBlock::const_iterator II = I->begin(), IE = I->end();
         II != IE; ++II) {
      LastMI = II;

      // Print the assembly for the instruction.
      if (!II->isLabel() && !II->isImplicitDef() && !II->isKill() &&
          !II->isDebugValue()) {
        HasAnyRealCode = true;
        ++EmittedInsts;
      }

      if (ShouldPrintDebugScopes) {
        NamedRegionTimer T(DbgTimerName, DWARFGroupName, TimePassesIsEnabled);
        DD->beginInstruction(II);
      }

      if (isVerbose())
        emitComments(*II, OutStreamer.GetCommentOS());

      switch (II->getOpcode()) {
      case TargetOpcode::PROLOG_LABEL:
        emitPrologLabel(*II);
        break;

      case TargetOpcode::EH_LABEL:
      case TargetOpcode::GC_LABEL:
        OutStreamer.EmitLabel(II->getOperand(0).getMCSymbol());
        break;
      case TargetOpcode::INLINEASM:
        EmitInlineAsm(II);
        break;
      case TargetOpcode::DBG_VALUE:
        if (isVerbose()) {
          if (!emitDebugValueComment(II, *this))
            EmitInstruction(II);
        }
        break;
      case TargetOpcode::IMPLICIT_DEF:
        if (isVerbose()) emitImplicitDef(II, *this);
        break;
      case TargetOpcode::KILL:
        if (isVerbose()) emitKill(II, *this);
        break;
      default:
        if (!TM.hasMCUseLoc())
          MCLineEntry::Make(&OutStreamer, getCurrentSection());

        EmitInstruction(II);
        break;
      }

      if (ShouldPrintDebugScopes) {
        NamedRegionTimer T(DbgTimerName, DWARFGroupName, TimePassesIsEnabled);
        DD->endInstruction(II);
      }
    }
  }

  // If the last instruction was a prolog label, then we have a situation where
  // we emitted a prolog but no function body. This results in the ending prolog
  // label equaling the end of function label and an invalid "row" in the
  // FDE. We need to emit a noop in this situation so that the FDE's rows are
  // valid.
  bool RequiresNoop = LastMI && LastMI->isPrologLabel();

  // If the function is empty and the object file uses .subsections_via_symbols,
  // then we need to emit *something* to the function body to prevent the
  // labels from collapsing together.  Just emit a noop.
  if ((MAI->hasSubsectionsViaSymbols() && !HasAnyRealCode) || RequiresNoop) {
    MCInst Noop;
    TM.getInstrInfo()->getNoopForMachoTarget(Noop);
    if (Noop.getOpcode()) {
      OutStreamer.AddComment("avoids zero-length function");
      OutStreamer.EmitInstruction(Noop);
    } else  // Target not mc-ized yet.
      OutStreamer.EmitRawText(StringRef("\tnop\n"));
  }

  const Function *F = MF->getFunction();
  for (Function::const_iterator i = F->begin(), e = F->end(); i != e; ++i) {
    const BasicBlock *BB = i;
    if (!BB->hasAddressTaken())
      continue;
    MCSymbol *Sym = GetBlockAddressSymbol(BB);
    if (Sym->isDefined())
      continue;
    OutStreamer.AddComment("Address of block that was removed by CodeGen");
    OutStreamer.EmitLabel(Sym);
  }

  // Emit target-specific gunk after the function body.
  EmitFunctionBodyEnd();

  // If the target wants a .size directive for the size of the function, emit
  // it.
  if (MAI->hasDotTypeDotSizeDirective()) {
    // Create a symbol for the end of function, so we can get the size as
    // difference between the function label and the temp label.
    MCSymbol *FnEndLabel = OutContext.CreateTempSymbol();
    OutStreamer.EmitLabel(FnEndLabel);

    const MCExpr *SizeExp =
      MCBinaryExpr::CreateSub(MCSymbolRefExpr::Create(FnEndLabel, OutContext),
                              MCSymbolRefExpr::Create(CurrentFnSymForSize,
                                                      OutContext),
                              OutContext);
    OutStreamer.EmitELFSize(CurrentFnSym, SizeExp);
  }

  // Emit post-function debug information.
  if (DD) {
    NamedRegionTimer T(DbgTimerName, DWARFGroupName, TimePassesIsEnabled);
    DD->endFunction(MF);
  }
  if (DE) {
    NamedRegionTimer T(EHTimerName, DWARFGroupName, TimePassesIsEnabled);
    DE->EndFunction();
  }
  MMI->EndFunction();

  // Print out jump tables referenced by the function.
  EmitJumpTableInfo();

  OutStreamer.AddBlankLine();
}

/// getDebugValueLocation - Get location information encoded by DBG_VALUE
/// operands.
MachineLocation AsmPrinter::
getDebugValueLocation(const MachineInstr *MI) const {
  // Target specific DBG_VALUE instructions are handled by each target.
  return MachineLocation();
}

/// EmitDwarfRegOp - Emit dwarf register operation.
void AsmPrinter::EmitDwarfRegOp(const MachineLocation &MLoc) const {
  const TargetRegisterInfo *TRI = TM.getRegisterInfo();
  int Reg = TRI->getDwarfRegNum(MLoc.getReg(), false);

  for (MCSuperRegIterator SR(MLoc.getReg(), TRI); SR.isValid() && Reg < 0;
       ++SR) {
    Reg = TRI->getDwarfRegNum(*SR, false);
    // FIXME: Get the bit range this register uses of the superregister
    // so that we can produce a DW_OP_bit_piece
  }

  // FIXME: Handle cases like a super register being encoded as
  // DW_OP_reg 32 DW_OP_piece 4 DW_OP_reg 33

  // FIXME: We have no reasonable way of handling errors in here. The
  // caller might be in the middle of an dwarf expression. We should
  // probably assert that Reg >= 0 once debug info generation is more mature.

  if (int Offset =  MLoc.getOffset()) {
    if (Reg < 32) {
      OutStreamer.AddComment(
        dwarf::OperationEncodingString(dwarf::DW_OP_breg0 + Reg));
      EmitInt8(dwarf::DW_OP_breg0 + Reg);
    } else {
      OutStreamer.AddComment("DW_OP_bregx");
      EmitInt8(dwarf::DW_OP_bregx);
      OutStreamer.AddComment(Twine(Reg));
      EmitULEB128(Reg);
    }
    EmitSLEB128(Offset);
  } else {
    if (Reg < 32) {
      OutStreamer.AddComment(
        dwarf::OperationEncodingString(dwarf::DW_OP_reg0 + Reg));
      EmitInt8(dwarf::DW_OP_reg0 + Reg);
    } else {
      OutStreamer.AddComment("DW_OP_regx");
      EmitInt8(dwarf::DW_OP_regx);
      OutStreamer.AddComment(Twine(Reg));
      EmitULEB128(Reg);
    }
  }

  // FIXME: Produce a DW_OP_bit_piece if we used a superregister
}

bool AsmPrinter::doFinalization(Module &M) {
  // Emit global variables.
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I)
    EmitGlobalVariable(I);

  // Emit visibility info for declarations
  for (Module::const_iterator I = M.begin(), E = M.end(); I != E; ++I) {
    const Function &F = *I;
    if (!F.isDeclaration())
      continue;
    GlobalValue::VisibilityTypes V = F.getVisibility();
    if (V == GlobalValue::DefaultVisibility)
      continue;

    MCSymbol *Name = Mang->getSymbol(&F);
    EmitVisibility(Name, V, false);
  }

  // Emit module flags.
  SmallVector<Module::ModuleFlagEntry, 8> ModuleFlags;
  M.getModuleFlagsMetadata(ModuleFlags);
  if (!ModuleFlags.empty())
    getObjFileLowering().emitModuleFlags(OutStreamer, ModuleFlags, Mang, TM);

  // Finalize debug and EH information.
  if (DE) {
    {
      NamedRegionTimer T(EHTimerName, DWARFGroupName, TimePassesIsEnabled);
      DE->EndModule();
    }
    delete DE; DE = 0;
  }
  if (DD) {
    {
      NamedRegionTimer T(DbgTimerName, DWARFGroupName, TimePassesIsEnabled);
      DD->endModule();
    }
    delete DD; DD = 0;
  }

  // If the target wants to know about weak references, print them all.
  if (MAI->getWeakRefDirective()) {
    // FIXME: This is not lazy, it would be nice to only print weak references
    // to stuff that is actually used.  Note that doing so would require targets
    // to notice uses in operands (due to constant exprs etc).  This should
    // happen with the MC stuff eventually.

    // Print out module-level global variables here.
    for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
         I != E; ++I) {
      if (!I->hasExternalWeakLinkage()) continue;
      OutStreamer.EmitSymbolAttribute(Mang->getSymbol(I), MCSA_WeakReference);
    }

    for (Module::const_iterator I = M.begin(), E = M.end(); I != E; ++I) {
      if (!I->hasExternalWeakLinkage()) continue;
      OutStreamer.EmitSymbolAttribute(Mang->getSymbol(I), MCSA_WeakReference);
    }
  }

  if (MAI->hasSetDirective()) {
    OutStreamer.AddBlankLine();
    for (Module::const_alias_iterator I = M.alias_begin(), E = M.alias_end();
         I != E; ++I) {
      MCSymbol *Name = Mang->getSymbol(I);

      const GlobalValue *GV = I->getAliasedGlobal();
      MCSymbol *Target = Mang->getSymbol(GV);

      if (I->hasExternalLinkage() || !MAI->getWeakRefDirective())
        OutStreamer.EmitSymbolAttribute(Name, MCSA_Global);
      else if (I->hasWeakLinkage())
        OutStreamer.EmitSymbolAttribute(Name, MCSA_WeakReference);
      else
        assert(I->hasLocalLinkage() && "Invalid alias linkage");

      EmitVisibility(Name, I->getVisibility());

      // Emit the directives as assignments aka .set:
      OutStreamer.EmitAssignment(Name,
                                 MCSymbolRefExpr::Create(Target, OutContext));
    }
  }

  GCModuleInfo *MI = getAnalysisIfAvailable<GCModuleInfo>();
  assert(MI && "AsmPrinter didn't require GCModuleInfo?");
  for (GCModuleInfo::iterator I = MI->end(), E = MI->begin(); I != E; )
    if (GCMetadataPrinter *MP = GetOrCreateGCPrinter(*--I))
      MP->finishAssembly(*this);

  // If we don't have any trampolines, then we don't require stack memory
  // to be executable. Some targets have a directive to declare this.
  Function *InitTrampolineIntrinsic = M.getFunction("llvm.init.trampoline");
  if (!InitTrampolineIntrinsic || InitTrampolineIntrinsic->use_empty())
    if (const MCSection *S = MAI->getNonexecutableStackSection(OutContext))
      OutStreamer.SwitchSection(S);

  // Allow the target to emit any magic that it wants at the end of the file,
  // after everything else has gone out.
  EmitEndOfAsmFile(M);

  delete Mang; Mang = 0;
  MMI = 0;

  OutStreamer.Finish();
  return false;
}

void AsmPrinter::SetupMachineFunction(MachineFunction &MF) {
  this->MF = &MF;
  // Get the function symbol.
  CurrentFnSym = Mang->getSymbol(MF.getFunction());
  CurrentFnSymForSize = CurrentFnSym;

  if (isVerbose())
    LI = &getAnalysis<MachineLoopInfo>();
}

namespace {
  // SectionCPs - Keep track the alignment, constpool entries per Section.
  struct SectionCPs {
    const MCSection *S;
    unsigned Alignment;
    SmallVector<unsigned, 4> CPEs;
    SectionCPs(const MCSection *s, unsigned a) : S(s), Alignment(a) {}
  };
}

/// EmitConstantPool - Print to the current output stream assembly
/// representations of the constants in the constant pool MCP. This is
/// used to print out constants which have been "spilled to memory" by
/// the code generator.
///
void AsmPrinter::EmitConstantPool() {
  const MachineConstantPool *MCP = MF->getConstantPool();
  const std::vector<MachineConstantPoolEntry> &CP = MCP->getConstants();
  if (CP.empty()) return;

  // Calculate sections for constant pool entries. We collect entries to go into
  // the same section together to reduce amount of section switch statements.
  SmallVector<SectionCPs, 4> CPSections;
  for (unsigned i = 0, e = CP.size(); i != e; ++i) {
    const MachineConstantPoolEntry &CPE = CP[i];
    unsigned Align = CPE.getAlignment();

    SectionKind Kind;
    switch (CPE.getRelocationInfo()) {
    default: llvm_unreachable("Unknown section kind");
    case 2: Kind = SectionKind::getReadOnlyWithRel(); break;
    case 1:
      Kind = SectionKind::getReadOnlyWithRelLocal();
      break;
    case 0:
    switch (TM.getDataLayout()->getTypeAllocSize(CPE.getType())) {
    case 4:  Kind = SectionKind::getMergeableConst4(); break;
    case 8:  Kind = SectionKind::getMergeableConst8(); break;
    case 16: Kind = SectionKind::getMergeableConst16();break;
    default: Kind = SectionKind::getMergeableConst(); break;
    }
    }

    const MCSection *S = getObjFileLowering().getSectionForConstant(Kind);

    // The number of sections are small, just do a linear search from the
    // last section to the first.
    bool Found = false;
    unsigned SecIdx = CPSections.size();
    while (SecIdx != 0) {
      if (CPSections[--SecIdx].S == S) {
        Found = true;
        break;
      }
    }
    if (!Found) {
      SecIdx = CPSections.size();
      CPSections.push_back(SectionCPs(S, Align));
    }

    if (Align > CPSections[SecIdx].Alignment)
      CPSections[SecIdx].Alignment = Align;
    CPSections[SecIdx].CPEs.push_back(i);
  }

  // Now print stuff into the calculated sections.
  for (unsigned i = 0, e = CPSections.size(); i != e; ++i) {
    OutStreamer.SwitchSection(CPSections[i].S);
    EmitAlignment(Log2_32(CPSections[i].Alignment));

    unsigned Offset = 0;
    for (unsigned j = 0, ee = CPSections[i].CPEs.size(); j != ee; ++j) {
      unsigned CPI = CPSections[i].CPEs[j];
      MachineConstantPoolEntry CPE = CP[CPI];

      // Emit inter-object padding for alignment.
      unsigned AlignMask = CPE.getAlignment() - 1;
      unsigned NewOffset = (Offset + AlignMask) & ~AlignMask;
      OutStreamer.EmitFill(NewOffset - Offset, 0/*fillval*/, 0/*addrspace*/);

      Type *Ty = CPE.getType();
      Offset = NewOffset + TM.getDataLayout()->getTypeAllocSize(Ty);
      OutStreamer.EmitLabel(GetCPISymbol(CPI));

      if (CPE.isMachineConstantPoolEntry())
        EmitMachineConstantPoolValue(CPE.Val.MachineCPVal);
      else
        EmitGlobalConstant(CPE.Val.ConstVal);
    }
  }
}

/// EmitJumpTableInfo - Print assembly representations of the jump tables used
/// by the current function to the current output stream.
///
void AsmPrinter::EmitJumpTableInfo() {
  const MachineJumpTableInfo *MJTI = MF->getJumpTableInfo();
  if (MJTI == 0) return;
  if (MJTI->getEntryKind() == MachineJumpTableInfo::EK_Inline) return;
  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  if (JT.empty()) return;

  // Pick the directive to use to print the jump table entries, and switch to
  // the appropriate section.
  const Function *F = MF->getFunction();
  bool JTInDiffSection = false;
  if (// In PIC mode, we need to emit the jump table to the same section as the
      // function body itself, otherwise the label differences won't make sense.
      // FIXME: Need a better predicate for this: what about custom entries?
      MJTI->getEntryKind() == MachineJumpTableInfo::EK_LabelDifference32 ||
      // We should also do if the section name is NULL or function is declared
      // in discardable section
      // FIXME: this isn't the right predicate, should be based on the MCSection
      // for the function.
      F->isWeakForLinker()) {
    OutStreamer.SwitchSection(getObjFileLowering().SectionForGlobal(F,Mang,TM));
  } else {
    // Otherwise, drop it in the readonly section.
    const MCSection *ReadOnlySection =
      getObjFileLowering().getSectionForConstant(SectionKind::getReadOnly());
    OutStreamer.SwitchSection(ReadOnlySection);
    JTInDiffSection = true;
  }

  EmitAlignment(Log2_32(MJTI->getEntryAlignment(*TM.getDataLayout())));

  // Jump tables in code sections are marked with a data_region directive
  // where that's supported.
  if (!JTInDiffSection)
    OutStreamer.EmitDataRegion(MCDR_DataRegionJT32);

  for (unsigned JTI = 0, e = JT.size(); JTI != e; ++JTI) {
    const std::vector<MachineBasicBlock*> &JTBBs = JT[JTI].MBBs;

    // If this jump table was deleted, ignore it.
    if (JTBBs.empty()) continue;

    // For the EK_LabelDifference32 entry, if the target supports .set, emit a
    // .set directive for each unique entry.  This reduces the number of
    // relocations the assembler will generate for the jump table.
    if (MJTI->getEntryKind() == MachineJumpTableInfo::EK_LabelDifference32 &&
        MAI->hasSetDirective()) {
      SmallPtrSet<const MachineBasicBlock*, 16> EmittedSets;
      const TargetLowering *TLI = TM.getTargetLowering();
      const MCExpr *Base = TLI->getPICJumpTableRelocBaseExpr(MF,JTI,OutContext);
      for (unsigned ii = 0, ee = JTBBs.size(); ii != ee; ++ii) {
        const MachineBasicBlock *MBB = JTBBs[ii];
        if (!EmittedSets.insert(MBB)) continue;

        // .set LJTSet, LBB32-base
        const MCExpr *LHS =
          MCSymbolRefExpr::Create(MBB->getSymbol(), OutContext);
        OutStreamer.EmitAssignment(GetJTSetSymbol(JTI, MBB->getNumber()),
                                MCBinaryExpr::CreateSub(LHS, Base, OutContext));
      }
    }

    // On some targets (e.g. Darwin) we want to emit two consecutive labels
    // before each jump table.  The first label is never referenced, but tells
    // the assembler and linker the extents of the jump table object.  The
    // second label is actually referenced by the code.
    if (JTInDiffSection && MAI->getLinkerPrivateGlobalPrefix()[0])
      // FIXME: This doesn't have to have any specific name, just any randomly
      // named and numbered 'l' label would work.  Simplify GetJTISymbol.
      OutStreamer.EmitLabel(GetJTISymbol(JTI, true));

    OutStreamer.EmitLabel(GetJTISymbol(JTI));

    for (unsigned ii = 0, ee = JTBBs.size(); ii != ee; ++ii)
      EmitJumpTableEntry(MJTI, JTBBs[ii], JTI);
  }
  if (!JTInDiffSection)
    OutStreamer.EmitDataRegion(MCDR_DataRegionEnd);
}

/// EmitJumpTableEntry - Emit a jump table entry for the specified MBB to the
/// current stream.
void AsmPrinter::EmitJumpTableEntry(const MachineJumpTableInfo *MJTI,
                                    const MachineBasicBlock *MBB,
                                    unsigned UID) const {
  assert(MBB && MBB->getNumber() >= 0 && "Invalid basic block");
  const MCExpr *Value = 0;
  switch (MJTI->getEntryKind()) {
  case MachineJumpTableInfo::EK_Inline:
    llvm_unreachable("Cannot emit EK_Inline jump table entry");
  case MachineJumpTableInfo::EK_Custom32:
    Value = TM.getTargetLowering()->LowerCustomJumpTableEntry(MJTI, MBB, UID,
                                                              OutContext);
    break;
  case MachineJumpTableInfo::EK_BlockAddress:
    // EK_BlockAddress - Each entry is a plain address of block, e.g.:
    //     .word LBB123
    Value = MCSymbolRefExpr::Create(MBB->getSymbol(), OutContext);
    break;
  case MachineJumpTableInfo::EK_GPRel32BlockAddress: {
    // EK_GPRel32BlockAddress - Each entry is an address of block, encoded
    // with a relocation as gp-relative, e.g.:
    //     .gprel32 LBB123
    MCSymbol *MBBSym = MBB->getSymbol();
    OutStreamer.EmitGPRel32Value(MCSymbolRefExpr::Create(MBBSym, OutContext));
    return;
  }

  case MachineJumpTableInfo::EK_GPRel64BlockAddress: {
    // EK_GPRel64BlockAddress - Each entry is an address of block, encoded
    // with a relocation as gp-relative, e.g.:
    //     .gpdword LBB123
    MCSymbol *MBBSym = MBB->getSymbol();
    OutStreamer.EmitGPRel64Value(MCSymbolRefExpr::Create(MBBSym, OutContext));
    return;
  }

  case MachineJumpTableInfo::EK_LabelDifference32: {
    // EK_LabelDifference32 - Each entry is the address of the block minus
    // the address of the jump table.  This is used for PIC jump tables where
    // gprel32 is not supported.  e.g.:
    //      .word LBB123 - LJTI1_2
    // If the .set directive is supported, this is emitted as:
    //      .set L4_5_set_123, LBB123 - LJTI1_2
    //      .word L4_5_set_123

    // If we have emitted set directives for the jump table entries, print
    // them rather than the entries themselves.  If we're emitting PIC, then
    // emit the table entries as differences between two text section labels.
    if (MAI->hasSetDirective()) {
      // If we used .set, reference the .set's symbol.
      Value = MCSymbolRefExpr::Create(GetJTSetSymbol(UID, MBB->getNumber()),
                                      OutContext);
      break;
    }
    // Otherwise, use the difference as the jump table entry.
    Value = MCSymbolRefExpr::Create(MBB->getSymbol(), OutContext);
    const MCExpr *JTI = MCSymbolRefExpr::Create(GetJTISymbol(UID), OutContext);
    Value = MCBinaryExpr::CreateSub(Value, JTI, OutContext);
    break;
  }
  }

  assert(Value && "Unknown entry kind!");

  unsigned EntrySize = MJTI->getEntrySize(*TM.getDataLayout());
  OutStreamer.EmitValue(Value, EntrySize, /*addrspace*/0);
}


/// EmitSpecialLLVMGlobal - Check to see if the specified global is a
/// special global used by LLVM.  If so, emit it and return true, otherwise
/// do nothing and return false.
bool AsmPrinter::EmitSpecialLLVMGlobal(const GlobalVariable *GV) {
  if (GV->getName() == "llvm.used") {
    if (MAI->hasNoDeadStrip())    // No need to emit this at all.
      EmitLLVMUsedList(GV->getInitializer());
    return true;
  }

  // Ignore debug and non-emitted data.  This handles llvm.compiler.used.
  if (GV->getSection() == "llvm.metadata" ||
      GV->hasAvailableExternallyLinkage())
    return true;

  if (!GV->hasAppendingLinkage()) return false;

  assert(GV->hasInitializer() && "Not a special LLVM global!");

  if (GV->getName() == "llvm.global_ctors") {
    EmitXXStructorList(GV->getInitializer(), /* isCtor */ true);

    if (TM.getRelocationModel() == Reloc::Static &&
        MAI->hasStaticCtorDtorReferenceInStaticMode()) {
      StringRef Sym(".constructors_used");
      OutStreamer.EmitSymbolAttribute(OutContext.GetOrCreateSymbol(Sym),
                                      MCSA_Reference);
    }
    return true;
  }

  if (GV->getName() == "llvm.global_dtors") {
    EmitXXStructorList(GV->getInitializer(), /* isCtor */ false);

    if (TM.getRelocationModel() == Reloc::Static &&
        MAI->hasStaticCtorDtorReferenceInStaticMode()) {
      StringRef Sym(".destructors_used");
      OutStreamer.EmitSymbolAttribute(OutContext.GetOrCreateSymbol(Sym),
                                      MCSA_Reference);
    }
    return true;
  }

  return false;
}

/// EmitLLVMUsedList - For targets that define a MAI::UsedDirective, mark each
/// global in the specified llvm.used list for which emitUsedDirectiveFor
/// is true, as being used with this directive.
void AsmPrinter::EmitLLVMUsedList(const Constant *List) {
  // Should be an array of 'i8*'.
  const ConstantArray *InitList = dyn_cast<ConstantArray>(List);
  if (InitList == 0) return;

  for (unsigned i = 0, e = InitList->getNumOperands(); i != e; ++i) {
    const GlobalValue *GV =
      dyn_cast<GlobalValue>(InitList->getOperand(i)->stripPointerCasts());
    if (GV && getObjFileLowering().shouldEmitUsedDirectiveFor(GV, Mang))
      OutStreamer.EmitSymbolAttribute(Mang->getSymbol(GV), MCSA_NoDeadStrip);
  }
}

typedef std::pair<unsigned, Constant*> Structor;

static bool priority_order(const Structor& lhs, const Structor& rhs) {
  return lhs.first < rhs.first;
}

/// EmitXXStructorList - Emit the ctor or dtor list taking into account the init
/// priority.
void AsmPrinter::EmitXXStructorList(const Constant *List, bool isCtor) {
  // Should be an array of '{ int, void ()* }' structs.  The first value is the
  // init priority.
  if (!isa<ConstantArray>(List)) return;

  // Sanity check the structors list.
  const ConstantArray *InitList = dyn_cast<ConstantArray>(List);
  if (!InitList) return; // Not an array!
  StructType *ETy = dyn_cast<StructType>(InitList->getType()->getElementType());
  if (!ETy || ETy->getNumElements() != 2) return; // Not an array of pairs!
  if (!isa<IntegerType>(ETy->getTypeAtIndex(0U)) ||
      !isa<PointerType>(ETy->getTypeAtIndex(1U))) return; // Not (int, ptr).

  // Gather the structors in a form that's convenient for sorting by priority.
  SmallVector<Structor, 8> Structors;
  for (unsigned i = 0, e = InitList->getNumOperands(); i != e; ++i) {
    ConstantStruct *CS = dyn_cast<ConstantStruct>(InitList->getOperand(i));
    if (!CS) continue; // Malformed.
    if (CS->getOperand(1)->isNullValue())
      break;  // Found a null terminator, skip the rest.
    ConstantInt *Priority = dyn_cast<ConstantInt>(CS->getOperand(0));
    if (!Priority) continue; // Malformed.
    Structors.push_back(std::make_pair(Priority->getLimitedValue(65535),
                                       CS->getOperand(1)));
  }

  // Emit the function pointers in the target-specific order
  const DataLayout *TD = TM.getDataLayout();
  unsigned Align = Log2_32(TD->getPointerPrefAlignment());
  std::stable_sort(Structors.begin(), Structors.end(), priority_order);
  for (unsigned i = 0, e = Structors.size(); i != e; ++i) {
    const MCSection *OutputSection =
      (isCtor ?
       getObjFileLowering().getStaticCtorSection(Structors[i].first) :
       getObjFileLowering().getStaticDtorSection(Structors[i].first));
    OutStreamer.SwitchSection(OutputSection);
    if (OutStreamer.getCurrentSection() != OutStreamer.getPreviousSection())
      EmitAlignment(Align);
    EmitXXStructor(Structors[i].second);
  }
}

//===--------------------------------------------------------------------===//
// Emission and print routines
//

/// EmitInt8 - Emit a byte directive and value.
///
void AsmPrinter::EmitInt8(int Value) const {
  OutStreamer.EmitIntValue(Value, 1, 0/*addrspace*/);
}

/// EmitInt16 - Emit a short directive and value.
///
void AsmPrinter::EmitInt16(int Value) const {
  OutStreamer.EmitIntValue(Value, 2, 0/*addrspace*/);
}

/// EmitInt32 - Emit a long directive and value.
///
void AsmPrinter::EmitInt32(int Value) const {
  OutStreamer.EmitIntValue(Value, 4, 0/*addrspace*/);
}

/// EmitLabelDifference - Emit something like ".long Hi-Lo" where the size
/// in bytes of the directive is specified by Size and Hi/Lo specify the
/// labels.  This implicitly uses .set if it is available.
void AsmPrinter::EmitLabelDifference(const MCSymbol *Hi, const MCSymbol *Lo,
                                     unsigned Size) const {
  // Get the Hi-Lo expression.
  const MCExpr *Diff =
    MCBinaryExpr::CreateSub(MCSymbolRefExpr::Create(Hi, OutContext),
                            MCSymbolRefExpr::Create(Lo, OutContext),
                            OutContext);

  if (!MAI->hasSetDirective()) {
    OutStreamer.EmitValue(Diff, Size, 0/*AddrSpace*/);
    return;
  }

  // Otherwise, emit with .set (aka assignment).
  MCSymbol *SetLabel = GetTempSymbol("set", SetCounter++);
  OutStreamer.EmitAssignment(SetLabel, Diff);
  OutStreamer.EmitSymbolValue(SetLabel, Size, 0/*AddrSpace*/);
}

/// EmitLabelOffsetDifference - Emit something like ".long Hi+Offset-Lo"
/// where the size in bytes of the directive is specified by Size and Hi/Lo
/// specify the labels.  This implicitly uses .set if it is available.
void AsmPrinter::EmitLabelOffsetDifference(const MCSymbol *Hi, uint64_t Offset,
                                           const MCSymbol *Lo, unsigned Size)
  const {

  // Emit Hi+Offset - Lo
  // Get the Hi+Offset expression.
  const MCExpr *Plus =
    MCBinaryExpr::CreateAdd(MCSymbolRefExpr::Create(Hi, OutContext),
                            MCConstantExpr::Create(Offset, OutContext),
                            OutContext);

  // Get the Hi+Offset-Lo expression.
  const MCExpr *Diff =
    MCBinaryExpr::CreateSub(Plus,
                            MCSymbolRefExpr::Create(Lo, OutContext),
                            OutContext);

  if (!MAI->hasSetDirective())
    OutStreamer.EmitValue(Diff, 4, 0/*AddrSpace*/);
  else {
    // Otherwise, emit with .set (aka assignment).
    MCSymbol *SetLabel = GetTempSymbol("set", SetCounter++);
    OutStreamer.EmitAssignment(SetLabel, Diff);
    OutStreamer.EmitSymbolValue(SetLabel, 4, 0/*AddrSpace*/);
  }
}

/// EmitLabelPlusOffset - Emit something like ".long Label+Offset"
/// where the size in bytes of the directive is specified by Size and Label
/// specifies the label.  This implicitly uses .set if it is available.
void AsmPrinter::EmitLabelPlusOffset(const MCSymbol *Label, uint64_t Offset,
                                      unsigned Size)
  const {

  // Emit Label+Offset (or just Label if Offset is zero)
  const MCExpr *Expr = MCSymbolRefExpr::Create(Label, OutContext);
  if (Offset)
    Expr = MCBinaryExpr::CreateAdd(Expr,
                                   MCConstantExpr::Create(Offset, OutContext),
                                   OutContext);

  OutStreamer.EmitValue(Expr, Size, 0/*AddrSpace*/);
}


//===----------------------------------------------------------------------===//

// EmitAlignment - Emit an alignment directive to the specified power of
// two boundary.  For example, if you pass in 3 here, you will get an 8
// byte alignment.  If a global value is specified, and if that global has
// an explicit alignment requested, it will override the alignment request
// if required for correctness.
//
void AsmPrinter::EmitAlignment(unsigned NumBits, const GlobalValue *GV) const {
  if (GV) NumBits = getGVAlignmentLog2(GV, *TM.getDataLayout(), NumBits);

  if (NumBits == 0) return;   // 1-byte aligned: no need to emit alignment.

  if (getCurrentSection()->getKind().isText())
    OutStreamer.EmitCodeAlignment(1 << NumBits);
  else
    OutStreamer.EmitValueToAlignment(1 << NumBits, 0, 1, 0);
}

//===----------------------------------------------------------------------===//
// Constant emission.
//===----------------------------------------------------------------------===//

/// lowerConstant - Lower the specified LLVM Constant to an MCExpr.
///
static const MCExpr *lowerConstant(const Constant *CV, AsmPrinter &AP) {
  MCContext &Ctx = AP.OutContext;

  if (CV->isNullValue() || isa<UndefValue>(CV))
    return MCConstantExpr::Create(0, Ctx);

  if (const ConstantInt *CI = dyn_cast<ConstantInt>(CV))
    return MCConstantExpr::Create(CI->getZExtValue(), Ctx);

  if (const GlobalValue *GV = dyn_cast<GlobalValue>(CV))
    return MCSymbolRefExpr::Create(AP.Mang->getSymbol(GV), Ctx);

  if (const BlockAddress *BA = dyn_cast<BlockAddress>(CV))
    return MCSymbolRefExpr::Create(AP.GetBlockAddressSymbol(BA), Ctx);

  const ConstantExpr *CE = dyn_cast<ConstantExpr>(CV);
  if (CE == 0) {
    llvm_unreachable("Unknown constant value to lower!");
  }

  switch (CE->getOpcode()) {
  default:
    // If the code isn't optimized, there may be outstanding folding
    // opportunities. Attempt to fold the expression using DataLayout as a
    // last resort before giving up.
    if (Constant *C =
          ConstantFoldConstantExpression(CE, AP.TM.getDataLayout()))
      if (C != CE)
        return lowerConstant(C, AP);

    // Otherwise report the problem to the user.
    {
      std::string S;
      raw_string_ostream OS(S);
      OS << "Unsupported expression in static initializer: ";
      WriteAsOperand(OS, CE, /*PrintType=*/false,
                     !AP.MF ? 0 : AP.MF->getFunction()->getParent());
      report_fatal_error(OS.str());
    }
  case Instruction::GetElementPtr: {
    const DataLayout &TD = *AP.TM.getDataLayout();
    // Generate a symbolic expression for the byte address
    const Constant *PtrVal = CE->getOperand(0);
    SmallVector<Value*, 8> IdxVec(CE->op_begin()+1, CE->op_end());
    int64_t Offset = TD.getIndexedOffset(PtrVal->getType(), IdxVec);

    const MCExpr *Base = lowerConstant(CE->getOperand(0), AP);
    if (Offset == 0)
      return Base;

    // Truncate/sext the offset to the pointer size.
    unsigned Width = TD.getPointerSizeInBits();
    if (Width < 64)
      Offset = SignExtend64(Offset, Width);

    return MCBinaryExpr::CreateAdd(Base, MCConstantExpr::Create(Offset, Ctx),
                                   Ctx);
  }

  case Instruction::Trunc:
    // We emit the value and depend on the assembler to truncate the generated
    // expression properly.  This is important for differences between
    // blockaddress labels.  Since the two labels are in the same function, it
    // is reasonable to treat their delta as a 32-bit value.
    // FALL THROUGH.
  case Instruction::BitCast:
    return lowerConstant(CE->getOperand(0), AP);

  case Instruction::IntToPtr: {
    const DataLayout &TD = *AP.TM.getDataLayout();
    // Handle casts to pointers by changing them into casts to the appropriate
    // integer type.  This promotes constant folding and simplifies this code.
    Constant *Op = CE->getOperand(0);
    Op = ConstantExpr::getIntegerCast(Op, TD.getIntPtrType(CV->getContext()),
                                      false/*ZExt*/);
    return lowerConstant(Op, AP);
  }

  case Instruction::PtrToInt: {
    const DataLayout &TD = *AP.TM.getDataLayout();
    // Support only foldable casts to/from pointers that can be eliminated by
    // changing the pointer to the appropriately sized integer type.
    Constant *Op = CE->getOperand(0);
    Type *Ty = CE->getType();

    const MCExpr *OpExpr = lowerConstant(Op, AP);

    // We can emit the pointer value into this slot if the slot is an
    // integer slot equal to the size of the pointer.
    if (TD.getTypeAllocSize(Ty) == TD.getTypeAllocSize(Op->getType()))
      return OpExpr;

    // Otherwise the pointer is smaller than the resultant integer, mask off
    // the high bits so we are sure to get a proper truncation if the input is
    // a constant expr.
    unsigned InBits = TD.getTypeAllocSizeInBits(Op->getType());
    const MCExpr *MaskExpr = MCConstantExpr::Create(~0ULL >> (64-InBits), Ctx);
    return MCBinaryExpr::CreateAnd(OpExpr, MaskExpr, Ctx);
  }

  // The MC library also has a right-shift operator, but it isn't consistently
  // signed or unsigned between different targets.
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::SDiv:
  case Instruction::SRem:
  case Instruction::Shl:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor: {
    const MCExpr *LHS = lowerConstant(CE->getOperand(0), AP);
    const MCExpr *RHS = lowerConstant(CE->getOperand(1), AP);
    switch (CE->getOpcode()) {
    default: llvm_unreachable("Unknown binary operator constant cast expr");
    case Instruction::Add: return MCBinaryExpr::CreateAdd(LHS, RHS, Ctx);
    case Instruction::Sub: return MCBinaryExpr::CreateSub(LHS, RHS, Ctx);
    case Instruction::Mul: return MCBinaryExpr::CreateMul(LHS, RHS, Ctx);
    case Instruction::SDiv: return MCBinaryExpr::CreateDiv(LHS, RHS, Ctx);
    case Instruction::SRem: return MCBinaryExpr::CreateMod(LHS, RHS, Ctx);
    case Instruction::Shl: return MCBinaryExpr::CreateShl(LHS, RHS, Ctx);
    case Instruction::And: return MCBinaryExpr::CreateAnd(LHS, RHS, Ctx);
    case Instruction::Or:  return MCBinaryExpr::CreateOr (LHS, RHS, Ctx);
    case Instruction::Xor: return MCBinaryExpr::CreateXor(LHS, RHS, Ctx);
    }
  }
  }
}

static void emitGlobalConstantImpl(const Constant *C, unsigned AddrSpace,
                                   AsmPrinter &AP);

/// isRepeatedByteSequence - Determine whether the given value is
/// composed of a repeated sequence of identical bytes and return the
/// byte value.  If it is not a repeated sequence, return -1.
static int isRepeatedByteSequence(const ConstantDataSequential *V) {
  StringRef Data = V->getRawDataValues();
  assert(!Data.empty() && "Empty aggregates should be CAZ node");
  char C = Data[0];
  for (unsigned i = 1, e = Data.size(); i != e; ++i)
    if (Data[i] != C) return -1;
  return static_cast<uint8_t>(C); // Ensure 255 is not returned as -1.
}


/// isRepeatedByteSequence - Determine whether the given value is
/// composed of a repeated sequence of identical bytes and return the
/// byte value.  If it is not a repeated sequence, return -1.
static int isRepeatedByteSequence(const Value *V, TargetMachine &TM) {

  if (const ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
    if (CI->getBitWidth() > 64) return -1;

    uint64_t Size = TM.getDataLayout()->getTypeAllocSize(V->getType());
    uint64_t Value = CI->getZExtValue();

    // Make sure the constant is at least 8 bits long and has a power
    // of 2 bit width.  This guarantees the constant bit width is
    // always a multiple of 8 bits, avoiding issues with padding out
    // to Size and other such corner cases.
    if (CI->getBitWidth() < 8 || !isPowerOf2_64(CI->getBitWidth())) return -1;

    uint8_t Byte = static_cast<uint8_t>(Value);

    for (unsigned i = 1; i < Size; ++i) {
      Value >>= 8;
      if (static_cast<uint8_t>(Value) != Byte) return -1;
    }
    return Byte;
  }
  if (const ConstantArray *CA = dyn_cast<ConstantArray>(V)) {
    // Make sure all array elements are sequences of the same repeated
    // byte.
    assert(CA->getNumOperands() != 0 && "Should be a CAZ");
    int Byte = isRepeatedByteSequence(CA->getOperand(0), TM);
    if (Byte == -1) return -1;

    for (unsigned i = 1, e = CA->getNumOperands(); i != e; ++i) {
      int ThisByte = isRepeatedByteSequence(CA->getOperand(i), TM);
      if (ThisByte == -1) return -1;
      if (Byte != ThisByte) return -1;
    }
    return Byte;
  }
  
  if (const ConstantDataSequential *CDS = dyn_cast<ConstantDataSequential>(V))
    return isRepeatedByteSequence(CDS);

  return -1;
}

static void emitGlobalConstantDataSequential(const ConstantDataSequential *CDS,
                                             unsigned AddrSpace,AsmPrinter &AP){
  
  // See if we can aggregate this into a .fill, if so, emit it as such.
  int Value = isRepeatedByteSequence(CDS, AP.TM);
  if (Value != -1) {
    uint64_t Bytes = AP.TM.getDataLayout()->getTypeAllocSize(CDS->getType());
    // Don't emit a 1-byte object as a .fill.
    if (Bytes > 1)
      return AP.OutStreamer.EmitFill(Bytes, Value, AddrSpace);
  }
  
  // If this can be emitted with .ascii/.asciz, emit it as such.
  if (CDS->isString())
    return AP.OutStreamer.EmitBytes(CDS->getAsString(), AddrSpace);

  // Otherwise, emit the values in successive locations.
  unsigned ElementByteSize = CDS->getElementByteSize();
  if (isa<IntegerType>(CDS->getElementType())) {
    for (unsigned i = 0, e = CDS->getNumElements(); i != e; ++i) {
      if (AP.isVerbose())
        AP.OutStreamer.GetCommentOS() << format("0x%" PRIx64 "\n",
                                                CDS->getElementAsInteger(i));
      AP.OutStreamer.EmitIntValue(CDS->getElementAsInteger(i),
                                  ElementByteSize, AddrSpace);
    }
  } else if (ElementByteSize == 4) {
    // FP Constants are printed as integer constants to avoid losing
    // precision.
    assert(CDS->getElementType()->isFloatTy());
    for (unsigned i = 0, e = CDS->getNumElements(); i != e; ++i) {
      union {
        float F;
        uint32_t I;
      };
      
      F = CDS->getElementAsFloat(i);
      if (AP.isVerbose())
        AP.OutStreamer.GetCommentOS() << "float " << F << '\n';
      AP.OutStreamer.EmitIntValue(I, 4, AddrSpace);
    }
  } else {
    assert(CDS->getElementType()->isDoubleTy());
    for (unsigned i = 0, e = CDS->getNumElements(); i != e; ++i) {
      union {
        double F;
        uint64_t I;
      };
      
      F = CDS->getElementAsDouble(i);
      if (AP.isVerbose())
        AP.OutStreamer.GetCommentOS() << "double " << F << '\n';
      AP.OutStreamer.EmitIntValue(I, 8, AddrSpace);
    }
  }

  const DataLayout &TD = *AP.TM.getDataLayout();
  unsigned Size = TD.getTypeAllocSize(CDS->getType());
  unsigned EmittedSize = TD.getTypeAllocSize(CDS->getType()->getElementType()) *
                        CDS->getNumElements();
  if (unsigned Padding = Size - EmittedSize)
    AP.OutStreamer.EmitZeros(Padding, AddrSpace);

}

static void emitGlobalConstantArray(const ConstantArray *CA, unsigned AddrSpace,
                                    AsmPrinter &AP) {
  // See if we can aggregate some values.  Make sure it can be
  // represented as a series of bytes of the constant value.
  int Value = isRepeatedByteSequence(CA, AP.TM);

  if (Value != -1) {
    uint64_t Bytes = AP.TM.getDataLayout()->getTypeAllocSize(CA->getType());
    AP.OutStreamer.EmitFill(Bytes, Value, AddrSpace);
  }
  else {
    for (unsigned i = 0, e = CA->getNumOperands(); i != e; ++i)
      emitGlobalConstantImpl(CA->getOperand(i), AddrSpace, AP);
  }
}

static void emitGlobalConstantVector(const ConstantVector *CV,
                                     unsigned AddrSpace, AsmPrinter &AP) {
  for (unsigned i = 0, e = CV->getType()->getNumElements(); i != e; ++i)
    emitGlobalConstantImpl(CV->getOperand(i), AddrSpace, AP);

  const DataLayout &TD = *AP.TM.getDataLayout();
  unsigned Size = TD.getTypeAllocSize(CV->getType());
  unsigned EmittedSize = TD.getTypeAllocSize(CV->getType()->getElementType()) *
                         CV->getType()->getNumElements();
  if (unsigned Padding = Size - EmittedSize)
    AP.OutStreamer.EmitZeros(Padding, AddrSpace);
}

static void emitGlobalConstantStruct(const ConstantStruct *CS,
                                     unsigned AddrSpace, AsmPrinter &AP) {
  // Print the fields in successive locations. Pad to align if needed!
  const DataLayout *TD = AP.TM.getDataLayout();
  unsigned Size = TD->getTypeAllocSize(CS->getType());
  const StructLayout *Layout = TD->getStructLayout(CS->getType());
  uint64_t SizeSoFar = 0;
  for (unsigned i = 0, e = CS->getNumOperands(); i != e; ++i) {
    const Constant *Field = CS->getOperand(i);

    // Check if padding is needed and insert one or more 0s.
    uint64_t FieldSize = TD->getTypeAllocSize(Field->getType());
    uint64_t PadSize = ((i == e-1 ? Size : Layout->getElementOffset(i+1))
                        - Layout->getElementOffset(i)) - FieldSize;
    SizeSoFar += FieldSize + PadSize;

    // Now print the actual field value.
    emitGlobalConstantImpl(Field, AddrSpace, AP);

    // Insert padding - this may include padding to increase the size of the
    // current field up to the ABI size (if the struct is not packed) as well
    // as padding to ensure that the next field starts at the right offset.
    AP.OutStreamer.EmitZeros(PadSize, AddrSpace);
  }
  assert(SizeSoFar == Layout->getSizeInBytes() &&
         "Layout of constant struct may be incorrect!");
}

static void emitGlobalConstantFP(const ConstantFP *CFP, unsigned AddrSpace,
                                 AsmPrinter &AP) {
  if (CFP->getType()->isHalfTy()) {
    if (AP.isVerbose()) {
      SmallString<10> Str;
      CFP->getValueAPF().toString(Str);
      AP.OutStreamer.GetCommentOS() << "half " << Str << '\n';
    }
    uint64_t Val = CFP->getValueAPF().bitcastToAPInt().getZExtValue();
    AP.OutStreamer.EmitIntValue(Val, 2, AddrSpace);
    return;
  }

  if (CFP->getType()->isFloatTy()) {
    if (AP.isVerbose()) {
      float Val = CFP->getValueAPF().convertToFloat();
      uint64_t IntVal = CFP->getValueAPF().bitcastToAPInt().getZExtValue();
      AP.OutStreamer.GetCommentOS() << "float " << Val << '\n'
                                    << " (" << format("0x%x", IntVal) << ")\n";
    }
    uint64_t Val = CFP->getValueAPF().bitcastToAPInt().getZExtValue();
    AP.OutStreamer.EmitIntValue(Val, 4, AddrSpace);
    return;
  }

  // FP Constants are printed as integer constants to avoid losing
  // precision.
  if (CFP->getType()->isDoubleTy()) {
    if (AP.isVerbose()) {
      double Val = CFP->getValueAPF().convertToDouble();
      uint64_t IntVal = CFP->getValueAPF().bitcastToAPInt().getZExtValue();
      AP.OutStreamer.GetCommentOS() << "double " << Val << '\n'
                                    << " (" << format("0x%lx", IntVal) << ")\n";
    }

    uint64_t Val = CFP->getValueAPF().bitcastToAPInt().getZExtValue();
    AP.OutStreamer.EmitIntValue(Val, 8, AddrSpace);
    return;
  }

  if (CFP->getType()->isX86_FP80Ty()) {
    // all long double variants are printed as hex
    // API needed to prevent premature destruction
    APInt API = CFP->getValueAPF().bitcastToAPInt();
    const uint64_t *p = API.getRawData();
    if (AP.isVerbose()) {
      // Convert to double so we can print the approximate val as a comment.
      APFloat DoubleVal = CFP->getValueAPF();
      bool ignored;
      DoubleVal.convert(APFloat::IEEEdouble, APFloat::rmNearestTiesToEven,
                        &ignored);
      AP.OutStreamer.GetCommentOS() << "x86_fp80 ~= "
        << DoubleVal.convertToDouble() << '\n';
    }

    if (AP.TM.getDataLayout()->isBigEndian()) {
      AP.OutStreamer.EmitIntValue(p[1], 2, AddrSpace);
      AP.OutStreamer.EmitIntValue(p[0], 8, AddrSpace);
    } else {
      AP.OutStreamer.EmitIntValue(p[0], 8, AddrSpace);
      AP.OutStreamer.EmitIntValue(p[1], 2, AddrSpace);
    }

    // Emit the tail padding for the long double.
    const DataLayout &TD = *AP.TM.getDataLayout();
    AP.OutStreamer.EmitZeros(TD.getTypeAllocSize(CFP->getType()) -
                             TD.getTypeStoreSize(CFP->getType()), AddrSpace);
    return;
  }

  assert(CFP->getType()->isPPC_FP128Ty() &&
         "Floating point constant type not handled");
  // All long double variants are printed as hex
  // API needed to prevent premature destruction.
  APInt API = CFP->getValueAPF().bitcastToAPInt();
  const uint64_t *p = API.getRawData();
  if (AP.TM.getDataLayout()->isBigEndian()) {
    AP.OutStreamer.EmitIntValue(p[0], 8, AddrSpace);
    AP.OutStreamer.EmitIntValue(p[1], 8, AddrSpace);
  } else {
    AP.OutStreamer.EmitIntValue(p[1], 8, AddrSpace);
    AP.OutStreamer.EmitIntValue(p[0], 8, AddrSpace);
  }
}

static void emitGlobalConstantLargeInt(const ConstantInt *CI,
                                       unsigned AddrSpace, AsmPrinter &AP) {
  const DataLayout *TD = AP.TM.getDataLayout();
  unsigned BitWidth = CI->getBitWidth();
  assert((BitWidth & 63) == 0 && "only support multiples of 64-bits");

  // We don't expect assemblers to support integer data directives
  // for more than 64 bits, so we emit the data in at most 64-bit
  // quantities at a time.
  const uint64_t *RawData = CI->getValue().getRawData();
  for (unsigned i = 0, e = BitWidth / 64; i != e; ++i) {
    uint64_t Val = TD->isBigEndian() ? RawData[e - i - 1] : RawData[i];
    AP.OutStreamer.EmitIntValue(Val, 8, AddrSpace);
  }
}

static void emitGlobalConstantImpl(const Constant *CV, unsigned AddrSpace,
                                   AsmPrinter &AP) {
  const DataLayout *TD = AP.TM.getDataLayout();
  uint64_t Size = TD->getTypeAllocSize(CV->getType());
  if (isa<ConstantAggregateZero>(CV) || isa<UndefValue>(CV))
    return AP.OutStreamer.EmitZeros(Size, AddrSpace);

  if (const ConstantInt *CI = dyn_cast<ConstantInt>(CV)) {
    switch (Size) {
    case 1:
    case 2:
    case 4:
    case 8:
      if (AP.isVerbose())
        AP.OutStreamer.GetCommentOS() << format("0x%" PRIx64 "\n",
                                                CI->getZExtValue());
      AP.OutStreamer.EmitIntValue(CI->getZExtValue(), Size, AddrSpace);
      return;
    default:
      emitGlobalConstantLargeInt(CI, AddrSpace, AP);
      return;
    }
  }

  if (const ConstantFP *CFP = dyn_cast<ConstantFP>(CV))
    return emitGlobalConstantFP(CFP, AddrSpace, AP);

  if (isa<ConstantPointerNull>(CV)) {
    AP.OutStreamer.EmitIntValue(0, Size, AddrSpace);
    return;
  }

  if (const ConstantDataSequential *CDS = dyn_cast<ConstantDataSequential>(CV))
    return emitGlobalConstantDataSequential(CDS, AddrSpace, AP);
  
  if (const ConstantArray *CVA = dyn_cast<ConstantArray>(CV))
    return emitGlobalConstantArray(CVA, AddrSpace, AP);

  if (const ConstantStruct *CVS = dyn_cast<ConstantStruct>(CV))
    return emitGlobalConstantStruct(CVS, AddrSpace, AP);

  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CV)) {
    // Look through bitcasts, which might not be able to be MCExpr'ized (e.g. of
    // vectors).
    if (CE->getOpcode() == Instruction::BitCast)
      return emitGlobalConstantImpl(CE->getOperand(0), AddrSpace, AP);

    if (Size > 8) {
      // If the constant expression's size is greater than 64-bits, then we have
      // to emit the value in chunks. Try to constant fold the value and emit it
      // that way.
      Constant *New = ConstantFoldConstantExpression(CE, TD);
      if (New && New != CE)
        return emitGlobalConstantImpl(New, AddrSpace, AP);
    }
  }
  
  if (const ConstantVector *V = dyn_cast<ConstantVector>(CV))
    return emitGlobalConstantVector(V, AddrSpace, AP);
    
  // Otherwise, it must be a ConstantExpr.  Lower it to an MCExpr, then emit it
  // thread the streamer with EmitValue.
  AP.OutStreamer.EmitValue(lowerConstant(CV, AP), Size, AddrSpace);
}

/// EmitGlobalConstant - Print a general LLVM constant to the .s file.
void AsmPrinter::EmitGlobalConstant(const Constant *CV, unsigned AddrSpace) {
  uint64_t Size = TM.getDataLayout()->getTypeAllocSize(CV->getType());
  if (Size)
    emitGlobalConstantImpl(CV, AddrSpace, *this);
  else if (MAI->hasSubsectionsViaSymbols()) {
    // If the global has zero size, emit a single byte so that two labels don't
    // look like they are at the same location.
    OutStreamer.EmitIntValue(0, 1, AddrSpace);
  }
}

void AsmPrinter::EmitMachineConstantPoolValue(MachineConstantPoolValue *MCPV) {
  // Target doesn't support this yet!
  llvm_unreachable("Target does not support EmitMachineConstantPoolValue");
}

void AsmPrinter::printOffset(int64_t Offset, raw_ostream &OS) const {
  if (Offset > 0)
    OS << '+' << Offset;
  else if (Offset < 0)
    OS << Offset;
}

//===----------------------------------------------------------------------===//
// Symbol Lowering Routines.
//===----------------------------------------------------------------------===//

/// GetTempSymbol - Return the MCSymbol corresponding to the assembler
/// temporary label with the specified stem and unique ID.
MCSymbol *AsmPrinter::GetTempSymbol(StringRef Name, unsigned ID) const {
  return OutContext.GetOrCreateSymbol(Twine(MAI->getPrivateGlobalPrefix()) +
                                      Name + Twine(ID));
}

/// GetTempSymbol - Return an assembler temporary label with the specified
/// stem.
MCSymbol *AsmPrinter::GetTempSymbol(StringRef Name) const {
  return OutContext.GetOrCreateSymbol(Twine(MAI->getPrivateGlobalPrefix())+
                                      Name);
}


MCSymbol *AsmPrinter::GetBlockAddressSymbol(const BlockAddress *BA) const {
  return MMI->getAddrLabelSymbol(BA->getBasicBlock());
}

MCSymbol *AsmPrinter::GetBlockAddressSymbol(const BasicBlock *BB) const {
  return MMI->getAddrLabelSymbol(BB);
}

/// GetCPISymbol - Return the symbol for the specified constant pool entry.
MCSymbol *AsmPrinter::GetCPISymbol(unsigned CPID) const {
  return OutContext.GetOrCreateSymbol
    (Twine(MAI->getPrivateGlobalPrefix()) + "CPI" + Twine(getFunctionNumber())
     + "_" + Twine(CPID));
}

/// GetJTISymbol - Return the symbol for the specified jump table entry.
MCSymbol *AsmPrinter::GetJTISymbol(unsigned JTID, bool isLinkerPrivate) const {
  return MF->getJTISymbol(JTID, OutContext, isLinkerPrivate);
}

/// GetJTSetSymbol - Return the symbol for the specified jump table .set
/// FIXME: privatize to AsmPrinter.
MCSymbol *AsmPrinter::GetJTSetSymbol(unsigned UID, unsigned MBBID) const {
  return OutContext.GetOrCreateSymbol
  (Twine(MAI->getPrivateGlobalPrefix()) + Twine(getFunctionNumber()) + "_" +
   Twine(UID) + "_set_" + Twine(MBBID));
}

/// GetSymbolWithGlobalValueBase - Return the MCSymbol for a symbol with
/// global value name as its base, with the specified suffix, and where the
/// symbol is forced to have private linkage if ForcePrivate is true.
MCSymbol *AsmPrinter::GetSymbolWithGlobalValueBase(const GlobalValue *GV,
                                                   StringRef Suffix,
                                                   bool ForcePrivate) const {
  SmallString<60> NameStr;
  Mang->getNameWithPrefix(NameStr, GV, ForcePrivate);
  NameStr.append(Suffix.begin(), Suffix.end());
  return OutContext.GetOrCreateSymbol(NameStr.str());
}

/// GetExternalSymbolSymbol - Return the MCSymbol for the specified
/// ExternalSymbol.
MCSymbol *AsmPrinter::GetExternalSymbolSymbol(StringRef Sym) const {
  SmallString<60> NameStr;
  Mang->getNameWithPrefix(NameStr, Sym);
  return OutContext.GetOrCreateSymbol(NameStr.str());
}



/// PrintParentLoopComment - Print comments about parent loops of this one.
static void PrintParentLoopComment(raw_ostream &OS, const MachineLoop *Loop,
                                   unsigned FunctionNumber) {
  if (Loop == 0) return;
  PrintParentLoopComment(OS, Loop->getParentLoop(), FunctionNumber);
  OS.indent(Loop->getLoopDepth()*2)
    << "Parent Loop BB" << FunctionNumber << "_"
    << Loop->getHeader()->getNumber()
    << " Depth=" << Loop->getLoopDepth() << '\n';
}


/// PrintChildLoopComment - Print comments about child loops within
/// the loop for this basic block, with nesting.
static void PrintChildLoopComment(raw_ostream &OS, const MachineLoop *Loop,
                                  unsigned FunctionNumber) {
  // Add child loop information
  for (MachineLoop::iterator CL = Loop->begin(), E = Loop->end();CL != E; ++CL){
    OS.indent((*CL)->getLoopDepth()*2)
      << "Child Loop BB" << FunctionNumber << "_"
      << (*CL)->getHeader()->getNumber() << " Depth " << (*CL)->getLoopDepth()
      << '\n';
    PrintChildLoopComment(OS, *CL, FunctionNumber);
  }
}

/// emitBasicBlockLoopComments - Pretty-print comments for basic blocks.
static void emitBasicBlockLoopComments(const MachineBasicBlock &MBB,
                                       const MachineLoopInfo *LI,
                                       const AsmPrinter &AP) {
  // Add loop depth information
  const MachineLoop *Loop = LI->getLoopFor(&MBB);
  if (Loop == 0) return;

  MachineBasicBlock *Header = Loop->getHeader();
  assert(Header && "No header for loop");

  // If this block is not a loop header, just print out what is the loop header
  // and return.
  if (Header != &MBB) {
    AP.OutStreamer.AddComment("  in Loop: Header=BB" +
                              Twine(AP.getFunctionNumber())+"_" +
                              Twine(Loop->getHeader()->getNumber())+
                              " Depth="+Twine(Loop->getLoopDepth()));
    return;
  }

  // Otherwise, it is a loop header.  Print out information about child and
  // parent loops.
  raw_ostream &OS = AP.OutStreamer.GetCommentOS();

  PrintParentLoopComment(OS, Loop->getParentLoop(), AP.getFunctionNumber());

  OS << "=>";
  OS.indent(Loop->getLoopDepth()*2-2);

  OS << "This ";
  if (Loop->empty())
    OS << "Inner ";
  OS << "Loop Header: Depth=" + Twine(Loop->getLoopDepth()) << '\n';

  PrintChildLoopComment(OS, Loop, AP.getFunctionNumber());
}


/// EmitBasicBlockStart - This method prints the label for the specified
/// MachineBasicBlock, an alignment (if present) and a comment describing
/// it if appropriate.
void AsmPrinter::EmitBasicBlockStart(const MachineBasicBlock *MBB) const {
  // Emit an alignment directive for this block, if needed.
  if (unsigned Align = MBB->getAlignment())
    EmitAlignment(Align);

  // If the block has its address taken, emit any labels that were used to
  // reference the block.  It is possible that there is more than one label
  // here, because multiple LLVM BB's may have been RAUW'd to this block after
  // the references were generated.
  if (MBB->hasAddressTaken()) {
    const BasicBlock *BB = MBB->getBasicBlock();
    if (isVerbose())
      OutStreamer.AddComment("Block address taken");

    std::vector<MCSymbol*> Syms = MMI->getAddrLabelSymbolToEmit(BB);

    for (unsigned i = 0, e = Syms.size(); i != e; ++i)
      OutStreamer.EmitLabel(Syms[i]);
  }

  // Print some verbose block comments.
  if (isVerbose()) {
    if (const BasicBlock *BB = MBB->getBasicBlock())
      if (BB->hasName())
        OutStreamer.AddComment("%" + BB->getName());
    emitBasicBlockLoopComments(*MBB, LI, *this);
  }

  // Print the main label for the block.
  if (MBB->pred_empty() || isBlockOnlyReachableByFallthrough(MBB)) {
    if (isVerbose() && OutStreamer.hasRawTextSupport()) {
      // NOTE: Want this comment at start of line, don't emit with AddComment.
      OutStreamer.EmitRawText(Twine(MAI->getCommentString()) + " BB#" +
                              Twine(MBB->getNumber()) + ":");
    }
  } else {
    OutStreamer.EmitLabel(MBB->getSymbol());
  }
}

void AsmPrinter::EmitVisibility(MCSymbol *Sym, unsigned Visibility,
                                bool IsDefinition) const {
  MCSymbolAttr Attr = MCSA_Invalid;

  switch (Visibility) {
  default: break;
  case GlobalValue::HiddenVisibility:
    if (IsDefinition)
      Attr = MAI->getHiddenVisibilityAttr();
    else
      Attr = MAI->getHiddenDeclarationVisibilityAttr();
    break;
  case GlobalValue::ProtectedVisibility:
    Attr = MAI->getProtectedVisibilityAttr();
    break;
  }

  if (Attr != MCSA_Invalid)
    OutStreamer.EmitSymbolAttribute(Sym, Attr);
}

/// isBlockOnlyReachableByFallthough - Return true if the basic block has
/// exactly one predecessor and the control transfer mechanism between
/// the predecessor and this block is a fall-through.
bool AsmPrinter::
isBlockOnlyReachableByFallthrough(const MachineBasicBlock *MBB) const {
  // If this is a landing pad, it isn't a fall through.  If it has no preds,
  // then nothing falls through to it.
  if (MBB->isLandingPad() || MBB->pred_empty())
    return false;

  // If there isn't exactly one predecessor, it can't be a fall through.
  MachineBasicBlock::const_pred_iterator PI = MBB->pred_begin(), PI2 = PI;
  ++PI2;
  if (PI2 != MBB->pred_end())
    return false;

  // The predecessor has to be immediately before this block.
  MachineBasicBlock *Pred = *PI;

  if (!Pred->isLayoutSuccessor(MBB))
    return false;

  // If the block is completely empty, then it definitely does fall through.
  if (Pred->empty())
    return true;

  // Check the terminators in the previous blocks
  for (MachineBasicBlock::iterator II = Pred->getFirstTerminator(),
         IE = Pred->end(); II != IE; ++II) {
    MachineInstr &MI = *II;

    // If it is not a simple branch, we are in a table somewhere.
    if (!MI.isBranch() || MI.isIndirectBranch())
      return false;

    // If we are the operands of one of the branches, this is not
    // a fall through.
    for (MachineInstr::mop_iterator OI = MI.operands_begin(),
           OE = MI.operands_end(); OI != OE; ++OI) {
      const MachineOperand& OP = *OI;
      if (OP.isJTI())
        return false;
      if (OP.isMBB() && OP.getMBB() == MBB)
        return false;
    }
  }

  return true;
}



GCMetadataPrinter *AsmPrinter::GetOrCreateGCPrinter(GCStrategy *S) {
  if (!S->usesMetadata())
    return 0;

  gcp_map_type &GCMap = getGCMap(GCMetadataPrinters);
  gcp_map_type::iterator GCPI = GCMap.find(S);
  if (GCPI != GCMap.end())
    return GCPI->second;

  const char *Name = S->getName().c_str();

  for (GCMetadataPrinterRegistry::iterator
         I = GCMetadataPrinterRegistry::begin(),
         E = GCMetadataPrinterRegistry::end(); I != E; ++I)
    if (strcmp(Name, I->getName()) == 0) {
      GCMetadataPrinter *GMP = I->instantiate();
      GMP->S = S;
      GCMap.insert(std::make_pair(S, GMP));
      return GMP;
    }

  report_fatal_error("no GCMetadataPrinter registered for GC: " + Twine(Name));
}
