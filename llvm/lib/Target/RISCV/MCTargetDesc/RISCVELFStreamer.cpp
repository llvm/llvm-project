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
#include "MCTargetDesc/RISCVAsmBackend.h"
#include "RISCVMCTargetDesc.h"
#include "Utils/RISCVBaseInfo.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCSubtargetInfo.h"

using namespace llvm;

// This part is for ELF object output.
RISCVTargetELFStreamer::RISCVTargetELFStreamer(MCStreamer &S,
                                               const MCSubtargetInfo &STI)
    : RISCVTargetStreamer(S) {
  MCAssembler &MCA = getStreamer().getAssembler();
  const FeatureBitset &Features = STI.getFeatureBits();
  auto &MAB = static_cast<RISCVAsmBackend &>(MCA.getBackend());
  RISCVABI::ABI ABI = MAB.getTargetABI();
  assert(ABI != RISCVABI::ABI_Unknown && "Improperly initialised target ABI");

  unsigned EFlags = MCA.getELFHeaderEFlags();

  if (Features[RISCV::FeatureStdExtC])
    EFlags |= ELF::EF_RISCV_RVC;

  switch (ABI) {
  case RISCVABI::ABI_ILP32:
  case RISCVABI::ABI_LP64:
    break;
  case RISCVABI::ABI_ILP32F:
  case RISCVABI::ABI_LP64F:
    EFlags |= ELF::EF_RISCV_FLOAT_ABI_SINGLE;
    break;
  case RISCVABI::ABI_ILP32D:
  case RISCVABI::ABI_LP64D:
    EFlags |= ELF::EF_RISCV_FLOAT_ABI_DOUBLE;
    break;
  case RISCVABI::ABI_ILP32E:
    EFlags |= ELF::EF_RISCV_RVE;
    break;
  case RISCVABI::ABI_Unknown:
    llvm_unreachable("Improperly initialised target ABI");
  }

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
