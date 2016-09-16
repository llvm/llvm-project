//===-- llvm/Target/ARMTargetObjectFile.cpp - ARM Object Info Impl --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ARMTargetObjectFile.h"
#include "ARMTargetMachine.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ELF.h"
#include "llvm/Target/TargetLowering.h"
using namespace llvm;
using namespace dwarf;

//===----------------------------------------------------------------------===//
//                               ELF Target
//===----------------------------------------------------------------------===//

void ARMElfTargetObjectFile::Initialize(MCContext &Ctx,
                                        const TargetMachine &TM) {
  bool isAAPCS_ABI = static_cast<const ARMTargetMachine &>(TM).TargetABI ==
                     ARMTargetMachine::ARMABI::ARM_ABI_AAPCS;
  TargetLoweringObjectFileELF::Initialize(Ctx, TM);
  InitializeELF(isAAPCS_ABI);

  if (isAAPCS_ABI) {
    LSDASection = nullptr;
  }

  AttributesSection =
      getContext().getELFSection(".ARM.attributes", ELF::SHT_ARM_ATTRIBUTES, 0);
}

const MCExpr *ARMElfTargetObjectFile::getTTypeGlobalReference(
    const GlobalValue *GV, unsigned Encoding, const TargetMachine &TM,
    MachineModuleInfo *MMI, MCStreamer &Streamer) const {
  if (TM.getMCAsmInfo()->getExceptionHandlingType() != ExceptionHandling::ARM)
    return TargetLoweringObjectFileELF::getTTypeGlobalReference(
        GV, Encoding, TM, MMI, Streamer);

  assert(Encoding == DW_EH_PE_absptr && "Can handle absptr encoding only");

  return MCSymbolRefExpr::create(TM.getSymbol(GV, getMangler()),
                                 MCSymbolRefExpr::VK_ARM_TARGET2, getContext());
}

const MCExpr *ARMElfTargetObjectFile::
getDebugThreadLocalSymbol(const MCSymbol *Sym) const {
  return MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_ARM_TLSLDO,
                                 getContext());
}
