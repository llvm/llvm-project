//===- AArch64TargetStreamer.cpp - AArch64TargetStreamer class ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the AArch64TargetStreamer class.
//
//===----------------------------------------------------------------------===//

#include "AArch64TargetStreamer.h"
#include "llvm/MC/ConstantPools.h"
using namespace llvm;

//
// AArch64TargetStreamer Implemenation
//
AArch64TargetStreamer::AArch64TargetStreamer(MCStreamer &S)
    : MCTargetStreamer(S), ConstantPools(new AssemblerConstantPools()) {}

AArch64TargetStreamer::~AArch64TargetStreamer() {}

// The constant pool handling is shared by all AArch64TargetStreamer
// implementations.
const MCExpr *AArch64TargetStreamer::addConstantPoolEntry(const MCExpr *Expr,
                                                          unsigned Size) {
  return ConstantPools->addEntry(Streamer, Expr, Size);
}

void AArch64TargetStreamer::emitCurrentConstantPool() {
  ConstantPools->emitForCurrentSection(Streamer);
}

// finish() - write out any non-empty assembler constant pools.
void AArch64TargetStreamer::finish() { ConstantPools->emitAll(Streamer); }

void AArch64TargetStreamer::emitInst(uint32_t Inst) {}
