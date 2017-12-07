//===-- Nios2Subtarget.cpp - Nios2 Subtarget Information ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Nios2 specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "Nios2Subtarget.h"
#include "Nios2.h"

using namespace llvm;

#define DEBUG_TYPE "nios2-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "Nios2GenSubtargetInfo.inc"

void Nios2Subtarget::anchor() {}

Nios2Subtarget::Nios2Subtarget(const Triple &TT, const std::string &CPU,
                               const std::string &FS, const TargetMachine &TM)
    :

      // Nios2GenSubtargetInfo will display features by llc -march=nios2
      // -mcpu=help
      Nios2GenSubtargetInfo(TT, CPU, FS), TargetTriple(TT),
      InstrInfo(initializeSubtargetDependencies(CPU, FS)), TLInfo(TM, *this),
      FrameLowering(*this) {}

Nios2Subtarget &Nios2Subtarget::initializeSubtargetDependencies(StringRef CPU,
                                                                StringRef FS) {
  if (TargetTriple.getArch() == Triple::nios2) {
    if (CPU != "nios2r2") {
      CPU = "nios2r1";
      Nios2ArchVersion = Nios2r1;
    } else {
      Nios2ArchVersion = Nios2r2;
    }
  } else {
    errs() << "!!!Error, TargetTriple.getArch() = " << TargetTriple.getArch()
           << "CPU = " << CPU << "\n";
    exit(0);
  }

  // Parse features string.
  ParseSubtargetFeatures(CPU, FS);

  return *this;
}
