//===-- CodeGen/AsmPrinter/Win64Exception.cpp - Dwarf Exception Impl ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing Win64 exception info into asm files.
//
//===----------------------------------------------------------------------===//

#include "Win64Exception.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
using namespace llvm;

Win64Exception::Win64Exception(AsmPrinter *A)
  : EHStreamer(A), shouldEmitPersonality(false), shouldEmitLSDA(false),
    shouldEmitMoves(false) {}

Win64Exception::~Win64Exception() {}

/// endModule - Emit all exception information that should come after the
/// content.
void Win64Exception::endModule() {
}

/// beginFunction - Gather pre-function exception information. Assumes it's
/// being emitted immediately after the function entry point.
void Win64Exception::beginFunction(const MachineFunction *MF) {
  shouldEmitMoves = shouldEmitPersonality = shouldEmitLSDA = false;

  // If any landing pads survive, we need an EH table.
  bool hasLandingPads = !MMI->getLandingPads().empty();

  shouldEmitMoves = Asm->needsSEHMoves();

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

  Asm->OutStreamer.EmitWinCFIStartProc(Asm->CurrentFnSym);

  if (!shouldEmitPersonality)
    return;

  const MCSymbol *PersHandlerSym =
      TLOF.getCFIPersonalitySymbol(Per, *Asm->Mang, Asm->TM, MMI);
  Asm->OutStreamer.EmitWinEHHandler(PersHandlerSym, true, true);

  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("eh_func_begin",
                                                Asm->getFunctionNumber()));
}

/// endFunction - Gather and emit post-function exception information.
///
void Win64Exception::endFunction(const MachineFunction *) {
  if (!shouldEmitPersonality && !shouldEmitMoves)
    return;

  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("eh_func_end",
                                                Asm->getFunctionNumber()));

  // Map all labels and get rid of any dead landing pads.
  MMI->TidyLandingPads();

  if (shouldEmitPersonality) {
    Asm->OutStreamer.PushSection();
    Asm->OutStreamer.EmitWinEHHandlerData();
    emitExceptionTable();
    Asm->OutStreamer.PopSection();
  }
  Asm->OutStreamer.EmitWinCFIEndProc();
}
