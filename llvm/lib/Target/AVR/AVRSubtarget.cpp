//===-- AVRSubtarget.cpp - AVR Subtarget Information ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AVR specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "AVRSubtarget.h"

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/TargetRegistry.h"

#include "AVR.h"
#include "AVRTargetMachine.h"
#include "MCTargetDesc/AVRMCTargetDesc.h"

#define DEBUG_TYPE "avr-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "AVRGenSubtargetInfo.inc"

namespace llvm {

AVRSubtarget::AVRSubtarget(const Triple &TT, const std::string &CPU,
                           const std::string &FS, const AVRTargetMachine &TM)
    : AVRGenSubtargetInfo(TT, CPU, /*TuneCPU*/ CPU, FS), ELFArch(0),

      // Subtarget features
      m_hasSRAM(false), m_hasJMPCALL(false), m_hasIJMPCALL(false),
      m_hasEIJMPCALL(false), m_hasADDSUBIW(false), m_hasSmallStack(false),
      m_hasMOVW(false), m_hasLPM(false), m_hasLPMX(false), m_hasELPM(false),
      m_hasELPMX(false), m_hasSPM(false), m_hasSPMX(false), m_hasDES(false),
      m_supportsRMW(false), m_supportsMultiplication(false), m_hasBREAK(false),
      m_hasTinyEncoding(false), m_hasMemMappedGPR(false),
      m_FeatureSetDummy(false),

      InstrInfo(), FrameLowering(),
      TLInfo(TM, initializeSubtargetDependencies(CPU, FS, TM)), TSInfo() {
  // Parse features string.
  ParseSubtargetFeatures(CPU, /*TuneCPU*/ CPU, FS);
}

AVRSubtarget &
AVRSubtarget::initializeSubtargetDependencies(StringRef CPU, StringRef FS,
                                              const TargetMachine &TM) {
  // Parse features string.
  ParseSubtargetFeatures(CPU, /*TuneCPU*/ CPU, FS);
  return *this;
}

} // end of namespace llvm
