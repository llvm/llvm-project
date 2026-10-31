//===-- SuperHTargetStreamer.cpp - SuperH Target Streamer ------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SuperHTargetStreamer.h"
#include "SuperHInstPrinter.h"
#include "SuperHMCTargetDesc.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;

SuperHTargetStreamer::SuperHTargetStreamer(MCStreamer &S)
    : MCTargetStreamer(S) {}

SuperHTargetAsmStreamer::SuperHTargetAsmStreamer(MCStreamer &S,
                                                 formatted_raw_ostream &OS)
    : SuperHTargetStreamer(S), OS(OS) {}

SuperHTargetELFStreamer::SuperHTargetELFStreamer(MCStreamer &S,
                                                 const MCSubtargetInfo &STI)
    : SuperHTargetStreamer(S), W(getStreamer().getWriter()), STI(STI) {
  setEFlags();      
}

MCELFStreamer &SuperHTargetELFStreamer::getStreamer() {
  return static_cast<MCELFStreamer &>(Streamer);
}

unsigned SuperHTargetELFStreamer::getCPUTypeFlag() const {
  switch(STI.getTargetTriple().getSubArch()) {
  default:                        return ELF::EF_SH_UNKNOWN;
  case Triple::SuperHSubArch_1:   return ELF::EF_SH1;
  case Triple::SuperHSubArch_2:   return ELF::EF_SH2;
  case Triple::SuperHSubArch_2a:  return ELF::EF_SH2A;
  case Triple::SuperHSubArch_2e:  return ELF::EF_SH2E;
  case Triple::SuperHSubArch_3:   return ELF::EF_SH3;
  case Triple::SuperHSubArch_3e:  return ELF::EF_SH3E;
  case Triple::SuperHSubArch_4:   return ELF::EF_SH4;
  case Triple::SuperHSubArch_4a:  return ELF::EF_SH4A;
  }
}

void SuperHTargetELFStreamer::setEFlags() {
  W.setELFHeaderEFlags(
    W.getELFHeaderEFlags() | 
    (getCPUTypeFlag() & ELF::EF_SH_MACH_MASK)
  );
}