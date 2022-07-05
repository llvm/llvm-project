//===- M88kTargetStreamer.cpp - M88kTargetStreamer class ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the variois M88kTargetStreamer classes.
//
//===----------------------------------------------------------------------===//

#include "M88kTargetStreamer.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;

M88kTargetStreamer::M88kTargetStreamer(MCStreamer &S) : MCTargetStreamer(S) {}

M88kTargetStreamer::~M88kTargetStreamer() = default;

M88kTargetAsmStreamer::M88kTargetAsmStreamer(MCStreamer &S,
                                             formatted_raw_ostream &OS)
    : M88kTargetStreamer(S), OS(OS) {}

void M88kTargetAsmStreamer::emitDirectiveRequires881100() {
  OS << "\t.requires_88110\n";
}

M88kTargetELFStreamer::M88kTargetELFStreamer(MCStreamer &S,
                                             const MCSubtargetInfo &STI)
    : M88kTargetStreamer(S), /*STI(STI),*/ Streamer(S),
      Requires88110(STI.getCPU() == "mc88110") {}

MCELFStreamer &M88kTargetELFStreamer::getStreamer() {
  return static_cast<MCELFStreamer &>(Streamer);
}

void M88kTargetELFStreamer::emitDirectiveRequires881100() {
  Requires88110 = true;
}

void M88kTargetELFStreamer::finish() {
  MCAssembler &MCA = getStreamer().getAssembler();

  // Update e_header flags.
  // TODO Handle ELF::EF_88K_NABI
  unsigned EFlags = MCA.getELFHeaderEFlags();
  if (Requires88110)
    EFlags |= ELF::EF_88K_M88110;

  MCA.setELFHeaderEFlags(EFlags);
}
