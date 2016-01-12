//===- ARMTargetStreamer.cpp - ARMTargetStreamer class --*- C++ -*---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ARMTargetStreamer class.
//
//===----------------------------------------------------------------------===//
#include "llvm/ADT/MapVector.h"
#include "llvm/MC/ConstantPools.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCStreamer.h"

using namespace llvm;
//
// ARMTargetStreamer Implemenation
//
ARMTargetStreamer::ARMTargetStreamer(MCStreamer &S)
    : MCTargetStreamer(S), ConstantPools(new AssemblerConstantPools()) {}

ARMTargetStreamer::~ARMTargetStreamer() {}

// The constant pool handling is shared by all ARMTargetStreamer
// implementations.
const MCExpr *ARMTargetStreamer::addConstantPoolEntry(const MCExpr *Expr, SMLoc Loc) {
  return ConstantPools->addEntry(Streamer, Expr, 4, Loc);
}

void ARMTargetStreamer::emitCurrentConstantPool() {
  ConstantPools->emitForCurrentSection(Streamer);
}

// finish() - write out any non-empty assembler constant pools.
void ARMTargetStreamer::finish() { ConstantPools->emitAll(Streamer); }

// reset() - Reset any state
void ARMTargetStreamer::reset() {}

// The remaining callbacks should be handled separately by each
// streamer.
void ARMTargetStreamer::emitFnStart() {}
void ARMTargetStreamer::emitFnEnd() {}
void ARMTargetStreamer::emitCantUnwind() {}
void ARMTargetStreamer::emitPersonality(const MCSymbol *Personality) {}
void ARMTargetStreamer::emitPersonalityIndex(unsigned Index) {}
void ARMTargetStreamer::emitHandlerData() {}
void ARMTargetStreamer::emitSetFP(unsigned FpReg, unsigned SpReg,
                                  int64_t Offset) {}
void ARMTargetStreamer::emitMovSP(unsigned Reg, int64_t Offset) {}
void ARMTargetStreamer::emitPad(int64_t Offset) {}
void ARMTargetStreamer::emitRegSave(const SmallVectorImpl<unsigned> &RegList,
                                    bool isVector) {}
void ARMTargetStreamer::emitUnwindRaw(int64_t StackOffset,
                                      const SmallVectorImpl<uint8_t> &Opcodes) {
}
void ARMTargetStreamer::switchVendor(StringRef Vendor) {}
void ARMTargetStreamer::emitAttribute(unsigned Attribute, unsigned Value) {}
void ARMTargetStreamer::emitTextAttribute(unsigned Attribute,
                                          StringRef String) {}
void ARMTargetStreamer::emitIntTextAttribute(unsigned Attribute,
                                             unsigned IntValue,
                                             StringRef StringValue) {}
void ARMTargetStreamer::emitArch(unsigned Arch) {}
void ARMTargetStreamer::emitArchExtension(unsigned ArchExt) {}
void ARMTargetStreamer::emitObjectArch(unsigned Arch) {}
void ARMTargetStreamer::emitFPU(unsigned FPU) {}
void ARMTargetStreamer::finishAttributeSection() {}
void ARMTargetStreamer::emitInst(uint32_t Inst, char Suffix) {}
void
ARMTargetStreamer::AnnotateTLSDescriptorSequence(const MCSymbolRefExpr *SRE) {}

void ARMTargetStreamer::emitThumbSet(MCSymbol *Symbol, const MCExpr *Value) {}
