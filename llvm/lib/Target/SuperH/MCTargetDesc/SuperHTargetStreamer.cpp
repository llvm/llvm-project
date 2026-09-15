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
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;

SuperHTargetStreamer::SuperHTargetStreamer(MCStreamer &S) : MCTargetStreamer(S) {}

SuperHTargetAsmStreamer::SuperHTargetAsmStreamer(MCStreamer &S,
                                                 formatted_raw_ostream &OS)
    : SuperHTargetStreamer(S), OS(OS) {}

SuperHTargetELFStreamer::SuperHTargetELFStreamer(MCStreamer &S,
                                           		 const MCSubtargetInfo &STI)
    : SuperHTargetStreamer(S) {
  ELFObjectWriter &W = getStreamer().getWriter();
  unsigned EFlags = W.getELFHeaderEFlags();
  W.setELFHeaderEFlags(EFlags);
}

MCELFStreamer &SuperHTargetELFStreamer::getStreamer() {
  return static_cast<MCELFStreamer &>(Streamer);
}