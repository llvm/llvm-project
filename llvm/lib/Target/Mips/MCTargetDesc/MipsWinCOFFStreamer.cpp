//===- MipsWinCOFFStreamer.cpp-----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "MipsMCTargetDesc.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCWinCOFFStreamer.h"

using namespace llvm;

namespace {
class MipsWinCOFFStreamer : public MCWinCOFFStreamer {
public:
  MipsWinCOFFStreamer(MCContext &C, std::unique_ptr<MCAsmBackend> AB,
                      std::unique_ptr<MCCodeEmitter> CE,
                      std::unique_ptr<MCObjectWriter> OW)
      : MCWinCOFFStreamer(C, std::move(AB), std::move(CE), std::move(OW)) {}
};
} // namespace

MCStreamer *llvm::createMipsWinCOFFStreamer(
    MCContext &C, std::unique_ptr<MCAsmBackend> &&AB,
    std::unique_ptr<MCObjectWriter> &&OW, std::unique_ptr<MCCodeEmitter> &&CE) {
  return new MipsWinCOFFStreamer(C, std::move(AB), std::move(CE),
                                 std::move(OW));
}
