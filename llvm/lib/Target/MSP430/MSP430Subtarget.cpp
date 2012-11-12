//===-- MSP430Subtarget.cpp - MSP430 Subtarget Information ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MSP430 specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "MSP430Subtarget.h"
#include "MSP430.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "MSP430GenSubtargetInfo.inc"

using namespace llvm;

void MSP430Subtarget::anchor() { }

MSP430Subtarget::MSP430Subtarget(const std::string &TT,
                                 const std::string &CPU,
                                 const std::string &FS) :
  MSP430GenSubtargetInfo(TT, CPU, FS) {
  std::string CPUName = "generic";

  // Parse features string.
  ParseSubtargetFeatures(CPUName, FS);
}
