//===-- CodeGen/AsmPrinter/ARMException.cpp - ARM EHABI Exception Impl ----===//
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
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/Mangler.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
using namespace llvm;

cl::opt<bool>
EnableARMEHABIDescriptors("arm-enable-ehabi-descriptors", cl::Hidden,
  cl::desc("Generate ARM EHABI tables with unwinding descriptors"),
  cl::init(false));


ARMException::ARMException(AsmPrinter *A)
  : DwarfException(A) {}

ARMException::~ARMException() {}

void ARMException::EndModule() {
}

/// BeginFunction - Gather pre-function exception information. Assumes it's
/// being emitted immediately after the function entry point.
void ARMException::BeginFunction(const MachineFunction *MF) {
  Asm->OutStreamer.EmitFnStart();
  if (Asm->MF->getFunction()->needsUnwindTableEntry())
    Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("eh_func_begin",
                                                  Asm->getFunctionNumber()));
}

/// EndFunction - Gather and emit post-function exception information.
///
void ARMException::EndFunction() {
  if (!Asm->MF->getFunction()->needsUnwindTableEntry())
    Asm->OutStreamer.EmitCantUnwind();
  else {
    Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("eh_func_end",
                                                  Asm->getFunctionNumber()));

    // Emit references to personality.
    if (const Function * Personality =
        MMI->getPersonalities()[MMI->getPersonalityIndex()]) {
      MCSymbol *PerSym = Asm->Mang->getSymbol(Personality);
      Asm->OutStreamer.EmitSymbolAttribute(PerSym, MCSA_Global);
      Asm->OutStreamer.EmitPersonality(PerSym);
    }

    if (EnableARMEHABIDescriptors) {
      // Map all labels and get rid of any dead landing pads.
      MMI->TidyLandingPads();

      Asm->OutStreamer.EmitHandlerData();

      // Emit actual exception table
      EmitExceptionTable();
    }
  }

  Asm->OutStreamer.EmitFnEnd();
}
