//===-- AArch64WinCOFFStreamer.cpp - ARM Target WinCOFF Streamer ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AArch64WinCOFFStreamer.h"

using namespace llvm;

namespace {

class AArch64WinCOFFStreamer : public MCWinCOFFStreamer {
public:
  friend class AArch64TargetWinCOFFStreamer;

  AArch64WinCOFFStreamer(MCContext &C, std::unique_ptr<MCAsmBackend> AB,
                         MCCodeEmitter &CE, raw_pwrite_stream &OS)
      : MCWinCOFFStreamer(C, std::move(AB), CE, OS) {}
};
} // end anonymous namespace

namespace llvm {
MCWinCOFFStreamer *
createAArch64WinCOFFStreamer(MCContext &Context,
                             std::unique_ptr<MCAsmBackend> MAB,
                             raw_pwrite_stream &OS, MCCodeEmitter *Emitter,
                             bool RelaxAll, bool IncrementalLinkerCompatible) {
  auto *S = new AArch64WinCOFFStreamer(Context, std::move(MAB), *Emitter, OS);
  S->getAssembler().setIncrementalLinkerCompatible(IncrementalLinkerCompatible);
  return S;
}

} // end llvm namespace
