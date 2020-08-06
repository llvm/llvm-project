//===-- P2TargetStreamer.cpp - P2 Target Streamer Methods -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides P2 specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "P2InstPrinter.h"
#include "P2MCTargetDesc.h"
#include "P2TargetObjectFile.h"
#include "P2TargetStreamer.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;

P2TargetStreamer::P2TargetStreamer(MCStreamer &S) : MCTargetStreamer(S) {

}

P2TargetAsmStreamer::P2TargetAsmStreamer(MCStreamer &S) : P2TargetStreamer(S) {

}

