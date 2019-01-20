//===-- RISCVELFStreamer.cpp - RISCV ELF Target Streamer Methods ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides RISCV specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "RISCVELFStreamer.h"
#include "RISCVMCTargetDesc.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCSubtargetInfo.h"

using namespace llvm;

// This part is for ELF object output.
RISCVTargetELFStreamer::RISCVTargetELFStreamer(MCStreamer &S,
                                               const MCSubtargetInfo &STI)
    : RISCVTargetStreamer(S) {
  MCAssembler &MCA = getStreamer().getAssembler();

  const FeatureBitset &Features = STI.getFeatureBits();

  unsigned EFlags = MCA.getELFHeaderEFlags();

  if (Features[RISCV::FeatureStdExtC])
    EFlags |= ELF::EF_RISCV_RVC;

  MCA.setELFHeaderEFlags(EFlags);
}

MCELFStreamer &RISCVTargetELFStreamer::getStreamer() {
  return static_cast<MCELFStreamer &>(Streamer);
}

void RISCVTargetELFStreamer::emitDirectiveOptionPush() {}
void RISCVTargetELFStreamer::emitDirectiveOptionPop() {}
void RISCVTargetELFStreamer::emitDirectiveOptionRVC() {}
void RISCVTargetELFStreamer::emitDirectiveOptionNoRVC() {}
void RISCVTargetELFStreamer::emitDirectiveOptionRelax() {}
void RISCVTargetELFStreamer::emitDirectiveOptionNoRelax() {}
