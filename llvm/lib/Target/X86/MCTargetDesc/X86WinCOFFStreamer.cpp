//===-- X86WinCOFFStreamer.cpp - X86 Target WinCOFF Streamer ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86MCTargetDesc.h"
#include "llvm/MC/MCWin64EH.h"
#include "llvm/MC/MCWinCOFFStreamer.h"

using namespace llvm;

namespace {
class X86WinCOFFStreamer : public MCWinCOFFStreamer {
  Win64EH::UnwindEmitter EHStreamer;
public:
  X86WinCOFFStreamer(MCContext &C, MCAsmBackend &AB, MCCodeEmitter *CE,
                     raw_pwrite_stream &OS)
      : MCWinCOFFStreamer(C, AB, *CE, OS) {}

  void EmitWinEHHandlerData() override;
  void EmitWindowsUnwindTables() override;
  void FinishImpl() override;
};

void X86WinCOFFStreamer::EmitWinEHHandlerData() {
  MCStreamer::EmitWinEHHandlerData();

  // We have to emit the unwind info now, because this directive
  // actually switches to the .xdata section!
  EHStreamer.EmitUnwindInfo(*this, getCurrentWinFrameInfo());
}

void X86WinCOFFStreamer::EmitWindowsUnwindTables() {
  if (!getNumWinFrameInfos())
    return;
  EHStreamer.Emit(*this);
}

void X86WinCOFFStreamer::FinishImpl() {
  EmitFrames(nullptr);
  EmitWindowsUnwindTables();

  MCWinCOFFStreamer::FinishImpl();
}
}

MCStreamer *llvm::createX86WinCOFFStreamer(MCContext &C, MCAsmBackend &AB,
                                           raw_pwrite_stream &OS,
                                           MCCodeEmitter *CE, bool RelaxAll,
                                           bool IncrementalLinkerCompatible) {
  X86WinCOFFStreamer *S = new X86WinCOFFStreamer(C, AB, CE, OS);
  S->getAssembler().setRelaxAll(RelaxAll);
  S->getAssembler().setIncrementalLinkerCompatible(IncrementalLinkerCompatible);
  return S;
}

