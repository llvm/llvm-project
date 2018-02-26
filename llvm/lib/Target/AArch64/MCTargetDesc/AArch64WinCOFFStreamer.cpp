//===-- AArch64WinCOFFStreamer.cpp - ARM Target WinCOFF Streamer ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AArch64WinCOFFStreamer.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCCodeEmitter.h"

using namespace llvm;

namespace {

class AArch64WinCOFFStreamer : public MCWinCOFFStreamer {
public:
  friend class AArch64TargetWinCOFFStreamer;

  AArch64WinCOFFStreamer(MCContext &C, std::unique_ptr<MCAsmBackend> AB,
                         std::unique_ptr<MCCodeEmitter> CE,
                         raw_pwrite_stream &OS)
      : MCWinCOFFStreamer(C, std::move(AB), std::move(CE), OS) {}

  void FinishImpl() override;
};

void AArch64WinCOFFStreamer::FinishImpl() {
  EmitFrames(nullptr);

  MCWinCOFFStreamer::FinishImpl();
}
} // end anonymous namespace

namespace llvm {
MCWinCOFFStreamer *createAArch64WinCOFFStreamer(
    MCContext &Context, std::unique_ptr<MCAsmBackend> MAB,
    raw_pwrite_stream &OS, std::unique_ptr<MCCodeEmitter> Emitter,
    bool RelaxAll, bool IncrementalLinkerCompatible) {
  auto *S = new AArch64WinCOFFStreamer(Context, std::move(MAB),
                                       std::move(Emitter), OS);
  S->getAssembler().setIncrementalLinkerCompatible(IncrementalLinkerCompatible);
  return S;
}

} // end llvm namespace
