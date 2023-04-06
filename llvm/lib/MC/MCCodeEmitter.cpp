//===- MCCodeEmitter.cpp - Instruction Encoding ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

MCCodeEmitter::MCCodeEmitter() = default;

MCCodeEmitter::~MCCodeEmitter() = default;

void MCCodeEmitter::encodeInstruction(const MCInst &Inst,
                                      SmallVectorImpl<char> &CB,
                                      SmallVectorImpl<MCFixup> &Fixups,
                                      const MCSubtargetInfo &STI) const {
  raw_svector_ostream OS(CB);
  encodeInstruction(Inst, OS, Fixups, STI);
}
