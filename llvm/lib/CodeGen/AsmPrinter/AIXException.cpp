//===-- CodeGen/AsmPrinter/AIXException.cpp - AIX Exception Impl ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing AIX exception info into asm files.
//
//===----------------------------------------------------------------------===//

#include "DwarfException.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/MC/MCSectionXCOFF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

AIXException::AIXException(AsmPrinter *A) : DwarfCFIExceptionBase(A) {}

void AIXException::emitExceptionInfoTable(const MCSymbol *LSDA,
                                          const MCSymbol *PerSym) {
  // Generate EH Info Table.
  // The EH Info Table, aka, 'compat unwind section' on AIX, have the following
  // format: struct eh_info_t {
  //   unsigned version;           /* EH info verion 0 */
  // #if defined(__64BIT__)
  //   char _pad[4];               /* padding */
  // #endif
  //   unsigned long lsda;         /* Pointer to LSDA */
  //   unsigned long personality;  /* Pointer to the personality routine */
  //   }

  Asm->OutStreamer->SwitchSection(
      Asm->getObjFileLowering().getCompactUnwindSection());
  MCSymbol *EHInfoLabel = MMI->getContext().getOrCreateSymbol(
      "__ehinfo." + Twine(Asm->getFunctionNumber()));
  Asm->OutStreamer->emitLabel(EHInfoLabel);

  // Version number.
  Asm->emitInt32(0);

  const DataLayout &DL = MMI->getModule()->getDataLayout();
  const unsigned PointerSize = DL.getPointerSize();

  // Add necessary paddings in 64 bit mode.
  Asm->OutStreamer->emitValueToAlignment(PointerSize);

  // LSDA location.
  Asm->OutStreamer->emitValue(MCSymbolRefExpr::create(LSDA, Asm->OutContext),
                              PointerSize);

  // Personality routine.
  Asm->OutStreamer->emitValue(MCSymbolRefExpr::create(PerSym, Asm->OutContext),
                              PointerSize);
}

void AIXException::endFunction(const MachineFunction *MF) {
  const Function &F = MF->getFunction();
  bool HasLandingPads = !MF->getLandingPads().empty();
  const Function *Per = nullptr;
  if (F.hasPersonalityFn())
    Per = dyn_cast<Function>(F.getPersonalityFn()->stripPointerCasts());
  bool EmitEHBlock =
      HasLandingPads || (F.hasPersonalityFn() &&
                         !isNoOpWithoutInvoke(classifyEHPersonality(Per)) &&
                         F.needsUnwindTableEntry());

  if (!EmitEHBlock)
    return;

  const MCSymbol *LSDALabel = emitExceptionTable();
  const MCSymbol *PerSym = Asm->TM.getSymbol(Per);

  emitExceptionInfoTable(LSDALabel, PerSym);
}

} // End of namespace llvm
