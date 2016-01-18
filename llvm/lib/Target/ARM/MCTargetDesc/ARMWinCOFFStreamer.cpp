//===-- ARMWinCOFFStreamer.cpp - ARM Target WinCOFF Streamer ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ARMMCTargetDesc.h"
#include "llvm/MC/MCWinCOFFStreamer.h"

using namespace llvm;

namespace {
class ARMWinCOFFStreamer : public MCWinCOFFStreamer {
public:
  ARMWinCOFFStreamer(MCContext &C, MCAsmBackend &AB, MCCodeEmitter &CE,
                     raw_pwrite_stream &OS)
      : MCWinCOFFStreamer(C, AB, CE, OS) {}

  void EmitAssemblerFlag(MCAssemblerFlag Flag) override;
  void EmitThumbFunc(MCSymbol *Symbol) override;
};

void ARMWinCOFFStreamer::EmitAssemblerFlag(MCAssemblerFlag Flag) {
  switch (Flag) {
  default: llvm_unreachable("not implemented");
  case MCAF_SyntaxUnified:
  case MCAF_Code16:
    break;
  }
}

void ARMWinCOFFStreamer::EmitThumbFunc(MCSymbol *Symbol) {
  getAssembler().setIsThumbFunc(Symbol);
}
}

MCStreamer *llvm::createARMWinCOFFStreamer(MCContext &Context,
                                           MCAsmBackend &MAB,
                                           raw_pwrite_stream &OS,
                                           MCCodeEmitter *Emitter,
                                           bool RelaxAll) {
  return new ARMWinCOFFStreamer(Context, MAB, *Emitter, OS);
}

