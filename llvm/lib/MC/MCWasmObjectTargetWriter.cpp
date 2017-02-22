//===-- MCWasmObjectTargetWriter.cpp - Wasm Target Writer Subclass --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/MCWasmObjectWriter.h"

using namespace llvm;

MCWasmObjectTargetWriter::MCWasmObjectTargetWriter(bool Is64Bit_)
    : Is64Bit(Is64Bit_) {}

bool MCWasmObjectTargetWriter::needsRelocateWithSymbol(const MCSymbol &Sym,
                                                       unsigned Type) const {
  return false;
}

void MCWasmObjectTargetWriter::sortRelocs(
    const MCAssembler &Asm, std::vector<WasmRelocationEntry> &Relocs) {
}
