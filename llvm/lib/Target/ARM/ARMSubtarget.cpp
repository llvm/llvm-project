//===-- ARMSubtarget.cpp - ARM Subtarget Information ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ARM specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "ARMSubtarget.h"
#include "ARMGenSubtarget.inc"
#include "llvm/GlobalValue.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

static cl::opt<bool>
ReserveR9("arm-reserve-r9", cl::Hidden,
          cl::desc("Reserve R9, making it unavailable as GPR"));

ARMSubtarget::ARMSubtarget(const std::string &TT, const std::string &FS,
                           bool isThumb)
  : ARMArchVersion(V4T)
  , ARMFPUType(None)
  , UseNEONForSinglePrecisionFP(false)
  , IsThumb(isThumb)
  , ThumbMode(Thumb1)
  , IsR9Reserved(ReserveR9)
  , stackAlignment(4)
  , CPUString("generic")
  , TargetType(isELF) // Default to ELF unless otherwise specified.
  , TargetABI(ARM_ABI_APCS) {
  // default to soft float ABI
  if (FloatABIType == FloatABI::Default)
    FloatABIType = FloatABI::Soft;

  // Determine default and user specified characteristics

  // Parse features string.
  CPUString = ParseSubtargetFeatures(FS, CPUString);

  // Set the boolean corresponding to the current target triple, or the default
  // if one cannot be determined, to true.
  unsigned Len = TT.length();
  unsigned Idx = 0;

  if (Len >= 5 && TT.substr(0, 4) == "armv")
    Idx = 4;
  else if (Len >= 6 && TT.substr(0, 5) == "thumb") {
    IsThumb = true;
    if (Len >= 7 && TT[5] == 'v')
      Idx = 6;
  }
  if (Idx) {
    unsigned SubVer = TT[Idx];
    if (SubVer > '4' && SubVer <= '9') {
      if (SubVer >= '7') {
        ARMArchVersion = V7A;
      } else if (SubVer == '6') {
        ARMArchVersion = V6;
        if (Len >= Idx+3 && TT[Idx+1] == 't' && TT[Idx+2] == '2')
          ARMArchVersion = V6T2;
      } else if (SubVer == '5') {
        ARMArchVersion = V5T;
        if (Len >= Idx+3 && TT[Idx+1] == 't' && TT[Idx+2] == 'e')
          ARMArchVersion = V5TE;
      }
      if (ARMArchVersion >= V6T2)
        ThumbMode = Thumb2;
    }
  }

  // Thumb2 implies at least V6T2.
  if (ARMArchVersion < V6T2 && ThumbMode >= Thumb2)
    ARMArchVersion = V6T2;

  if (Len >= 10) {
    if (TT.find("-darwin") != std::string::npos)
      // arm-darwin
      TargetType = isDarwin;
  }

  if (TT.find("eabi") != std::string::npos)
    TargetABI = ARM_ABI_AAPCS;

  if (isAAPCS_ABI())
    stackAlignment = 8;

  if (isTargetDarwin())
    IsR9Reserved = ReserveR9 | (ARMArchVersion < V6);
}

/// GVIsIndirectSymbol - true if the GV will be accessed via an indirect symbol.
bool ARMSubtarget::GVIsIndirectSymbol(GlobalValue *GV, bool isStatic) const {
  // If symbol visibility is hidden, the extra load is not needed if
  // the symbol is definitely defined in the current translation unit.
  bool isDecl = GV->isDeclaration() || GV->hasAvailableExternallyLinkage();
  if (GV->hasHiddenVisibility() && (!isDecl && !GV->hasCommonLinkage()))
    return false;
  return !isStatic && (isDecl || GV->isWeakForLinker());
}
