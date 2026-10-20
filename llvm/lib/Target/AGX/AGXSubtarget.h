//===-- AGXSubtarget.h - Define subtarget for AGX ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The AGX subtarget: generation, encoding family and capability features.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AGX_AGXSUBTARGET_H
#define LLVM_LIB_TARGET_AGX_AGXSUBTARGET_H

#include "AGXFrameLowering.h"
#include "AGXISelLowering.h"
#include "AGXInstrInfo.h"
#include "AGXSelectionDAGInfo.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"

#define GET_SUBTARGETINFO_HEADER
#include "AGXGenSubtargetInfo.inc"

namespace llvm {

class AGXTargetMachine;

class AGXSubtarget : public AGXGenSubtargetInfo {
public:
  enum Generation { Unknown, G13, G14, G15, G16, G17 };

  /// Instruction encoding families, selecting the real instructions that
  /// pseudo instructions lower to.
  enum EncodingFamily { AGX2, Apple9 };

private:
  Generation Gen = Unknown;
  unsigned MaxGPRs = 0;

#define GET_SUBTARGETINFO_MACRO(ATTRIBUTE, DEFAULT, GETTER)                    \
  bool ATTRIBUTE = DEFAULT;
#include "AGXGenSubtargetInfo.inc"

  AGXInstrInfo InstrInfo;
  AGXFrameLowering FrameLowering;
  AGXTargetLowering TLInfo;
  AGXSelectionDAGInfo TSInfo;

public:
  AGXSubtarget(const Triple &TT, StringRef CPU, StringRef FS,
               const AGXTargetMachine &TM);

  AGXSubtarget &initializeSubtargetDependencies(StringRef CPU, StringRef FS);
  void ParseSubtargetFeatures(StringRef CPU, StringRef TuneCPU, StringRef FS);

#define GET_SUBTARGETINFO_MACRO(ATTRIBUTE, DEFAULT, GETTER)                    \
  bool GETTER() const { return ATTRIBUTE; }
#include "AGXGenSubtargetInfo.inc"

  Generation getGeneration() const { return Gen; }
  EncodingFamily getEncodingFamily() const {
    return HasApple9Encoding ? Apple9 : AGX2;
  }
  /// The number of addressable 32-bit general purpose registers.
  unsigned getMaxGPRs() const { return MaxGPRs ? MaxGPRs : 128; }

  /// The threads of one threadgroup that fit on a core when each thread uses
  /// \p Halves 16-bit registers. G13 and G14 run 2 * floor(3328 / (2 * Halves))
  /// SIMD-groups of 32 threads, at most 1024 threads (Mesa's agx_occupancy
  /// table); later generations are assumed to be the same.
  static unsigned maxThreadsForHalves(unsigned Halves);

  /// The 32-bit GPRs \p F may use: getMaxGPRs(), fewer with the function
  /// attributes "agx-max-gprs"="N" (PTX .maxnreg) or
  /// "agx-max-threads-per-threadgroup"="N" (PTX .maxntid / .reqntid), which
  /// keeps the register use low enough for N threads to fit on a core.
  unsigned getMaxGPRs(const Function &F) const;

  const AGXInstrInfo *getInstrInfo() const override { return &InstrInfo; }
  const AGXFrameLowering *getFrameLowering() const override {
    return &FrameLowering;
  }
  const AGXTargetLowering *getTargetLowering() const override {
    return &TLInfo;
  }
  const AGXRegisterInfo *getRegisterInfo() const override {
    return &InstrInfo.getRegisterInfo();
  }
  const AGXSelectionDAGInfo *getSelectionDAGInfo() const override {
    return &TSInfo;
  }

  bool enableMachineScheduler() const override { return true; }
  bool enablePostRAScheduler() const override { return false; }
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AGX_AGXSUBTARGET_H
