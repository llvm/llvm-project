//===-- CodeGen/AsmPrinter/DwarfException.cpp - Dwarf Exception Impl ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing DWARF exception info into asm files.
//
//===----------------------------------------------------------------------===//

#include "DwarfException.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/Mangler.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
using namespace llvm;

DwarfCFIException::DwarfCFIException(AsmPrinter *A)
  : DwarfException(A),
    shouldEmitPersonality(false), shouldEmitLSDA(false), shouldEmitMoves(false),
    moveTypeModule(AsmPrinter::CFI_M_None) {}

DwarfCFIException::~DwarfCFIException() {}

/// EndModule - Emit all exception information that should come after the
/// content.
void DwarfCFIException::EndModule() {
  if (moveTypeModule == AsmPrinter::CFI_M_Debug)
    Asm->OutStreamer.EmitCFISections(false, true);

  if (!Asm->MAI->isExceptionHandlingDwarf())
    return;

  const TargetLoweringObjectFile &TLOF = Asm->getObjFileLowering();

  unsigned PerEncoding = TLOF.getPersonalityEncoding();

  if ((PerEncoding & 0x70) != dwarf::DW_EH_PE_pcrel)
    return;

  // Emit references to all used personality functions
  bool AtLeastOne = false;
  const std::vector<const Function*> &Personalities = MMI->getPersonalities();
  for (size_t i = 0, e = Personalities.size(); i != e; ++i) {
    if (!Personalities[i])
      continue;
    MCSymbol *Sym = Asm->Mang->getSymbol(Personalities[i]);
    TLOF.emitPersonalityValue(Asm->OutStreamer, Asm->TM, Sym);
    AtLeastOne = true;
  }

  if (AtLeastOne && !TLOF.isFunctionEHFrameSymbolPrivate()) {
    // This is a temporary hack to keep sections in the same order they
    // were before. This lets us produce bit identical outputs while
    // transitioning to CFI.
    Asm->OutStreamer.SwitchSection(
               const_cast<TargetLoweringObjectFile&>(TLOF).getEHFrameSection());
  }
}

/// BeginFunction - Gather pre-function exception information. Assumes it's
/// being emitted immediately after the function entry point.
void DwarfCFIException::BeginFunction(const MachineFunction *MF) {
  shouldEmitMoves = shouldEmitPersonality = shouldEmitLSDA = false;

  // If any landing pads survive, we need an EH table.
  bool hasLandingPads = !MMI->getLandingPads().empty();

  // See if we need frame move info.
  AsmPrinter::CFIMoveType MoveType = Asm->needsCFIMoves();
  if (MoveType == AsmPrinter::CFI_M_EH ||
      (MoveType == AsmPrinter::CFI_M_Debug &&
       moveTypeModule == AsmPrinter::CFI_M_None))
    moveTypeModule = MoveType;

  shouldEmitMoves = MoveType != AsmPrinter::CFI_M_None;

  const TargetLoweringObjectFile &TLOF = Asm->getObjFileLowering();
  unsigned PerEncoding = TLOF.getPersonalityEncoding();
  const Function *Per = MMI->getPersonalities()[MMI->getPersonalityIndex()];

  shouldEmitPersonality = hasLandingPads &&
    PerEncoding != dwarf::DW_EH_PE_omit && Per;

  unsigned LSDAEncoding = TLOF.getLSDAEncoding();
  shouldEmitLSDA = shouldEmitPersonality &&
    LSDAEncoding != dwarf::DW_EH_PE_omit;

  if (!shouldEmitPersonality && !shouldEmitMoves)
    return;

  Asm->OutStreamer.EmitCFIStartProc();

  // Indicate personality routine, if any.
  if (!shouldEmitPersonality)
    return;

  const MCSymbol *Sym = TLOF.getCFIPersonalitySymbol(Per, Asm->Mang, MMI);
  Asm->OutStreamer.EmitCFIPersonality(Sym, PerEncoding);

  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("eh_func_begin",
                                                Asm->getFunctionNumber()));

  // Provide LSDA information.
  if (!shouldEmitLSDA)
    return;

  Asm->OutStreamer.EmitCFILsda(Asm->GetTempSymbol("exception",
                                                  Asm->getFunctionNumber()),
                               LSDAEncoding);
}

/// EndFunction - Gather and emit post-function exception information.
///
void DwarfCFIException::EndFunction() {
  if (!shouldEmitPersonality && !shouldEmitMoves)
    return;

  Asm->OutStreamer.EmitCFIEndProc();

  if (!shouldEmitPersonality)
    return;

  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("eh_func_end",
                                                Asm->getFunctionNumber()));

  // Map all labels and get rid of any dead landing pads.
  MMI->TidyLandingPads();

  EmitExceptionTable();
}
