//===-- RISCVTargetStreamer.cpp - RISCV Target Streamer Methods -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides RISCV specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "RISCVTargetStreamer.h"
#include "RISCVMCTargetDesc.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/RISCVAttributes.h"

using namespace llvm;

RISCVTargetStreamer::RISCVTargetStreamer(MCStreamer &S) : MCTargetStreamer(S) {}

void RISCVTargetStreamer::finish() { finishAttributeSection(); }

void RISCVTargetStreamer::emitDirectiveOptionPush() {}
void RISCVTargetStreamer::emitDirectiveOptionPop() {}
void RISCVTargetStreamer::emitDirectiveOptionPIC() {}
void RISCVTargetStreamer::emitDirectiveOptionNoPIC() {}
void RISCVTargetStreamer::emitDirectiveOptionRVC() {}
void RISCVTargetStreamer::emitDirectiveOptionNoRVC() {}
void RISCVTargetStreamer::emitDirectiveOptionRelax() {}
void RISCVTargetStreamer::emitDirectiveOptionNoRelax() {}
void RISCVTargetStreamer::emitAttribute(unsigned Attribute, unsigned Value) {}
void RISCVTargetStreamer::finishAttributeSection() {}
void RISCVTargetStreamer::emitTextAttribute(unsigned Attribute,
                                            StringRef String) {}
void RISCVTargetStreamer::emitIntTextAttribute(unsigned Attribute,
                                               unsigned IntValue,
                                               StringRef StringValue) {}

void RISCVTargetStreamer::emitTargetAttributes(const MCSubtargetInfo &STI) {
  if (STI.hasFeature(RISCV::FeatureRV32E))
    emitAttribute(RISCVAttrs::STACK_ALIGN, RISCVAttrs::ALIGN_4);
  else
    emitAttribute(RISCVAttrs::STACK_ALIGN, RISCVAttrs::ALIGN_16);

  std::string Arch = "rv32";
  if (STI.hasFeature(RISCV::Feature64Bit))
    Arch = "rv64";
  if (STI.hasFeature(RISCV::FeatureRV32E))
    Arch += "e1p9";
  else
    Arch += "i2p0";
  if (STI.hasFeature(RISCV::FeatureStdExtM))
    Arch += "_m2p0";
  if (STI.hasFeature(RISCV::FeatureStdExtA))
    Arch += "_a2p0";
  if (STI.hasFeature(RISCV::FeatureStdExtF))
    Arch += "_f2p0";
  if (STI.hasFeature(RISCV::FeatureStdExtD))
    Arch += "_d2p0";
  if (STI.hasFeature(RISCV::FeatureStdExtC))
    Arch += "_c2p0";
  if (STI.hasFeature(RISCV::FeatureStdExtB))
    Arch += "_b0p93";
  if (STI.hasFeature(RISCV::FeatureStdExtV))
    Arch += "_v0p10";
  if (STI.hasFeature(RISCV::FeatureStdExtZfh))
    Arch += "_zfh0p1";
  if (STI.hasFeature(RISCV::FeatureStdExtZba))
    Arch += "_zba0p93";
  if (STI.hasFeature(RISCV::FeatureStdExtZbb))
    Arch += "_zbb0p93";
  if (STI.hasFeature(RISCV::FeatureStdExtZbc))
    Arch += "_zbc0p93";
  if (STI.hasFeature(RISCV::FeatureStdExtZbe))
    Arch += "_zbe0p93";
  if (STI.hasFeature(RISCV::FeatureStdExtZbf))
    Arch += "_zbf0p93";
  if (STI.hasFeature(RISCV::FeatureStdExtZbm))
    Arch += "_zbm0p93";
  if (STI.hasFeature(RISCV::FeatureStdExtZbp))
    Arch += "_zbp0p93";
  if (STI.hasFeature(RISCV::FeatureStdExtZbr))
    Arch += "_zbr0p93";
  if (STI.hasFeature(RISCV::FeatureStdExtZbs))
    Arch += "_zbs0p93";
  if (STI.hasFeature(RISCV::FeatureStdExtZbt))
    Arch += "_zbt0p93";
  if (STI.hasFeature(RISCV::FeatureStdExtZvamo))
    Arch += "_zvamo0p10";
  if (STI.hasFeature(RISCV::FeatureStdExtZvlsseg))
    Arch += "_zvlsseg0p10";

  emitTextAttribute(RISCVAttrs::ARCH, Arch);
}

// This part is for ascii assembly output
RISCVTargetAsmStreamer::RISCVTargetAsmStreamer(MCStreamer &S,
                                               formatted_raw_ostream &OS)
    : RISCVTargetStreamer(S), OS(OS) {}

void RISCVTargetAsmStreamer::emitDirectiveOptionPush() {
  OS << "\t.option\tpush\n";
}

void RISCVTargetAsmStreamer::emitDirectiveOptionPop() {
  OS << "\t.option\tpop\n";
}

void RISCVTargetAsmStreamer::emitDirectiveOptionPIC() {
  OS << "\t.option\tpic\n";
}

void RISCVTargetAsmStreamer::emitDirectiveOptionNoPIC() {
  OS << "\t.option\tnopic\n";
}

void RISCVTargetAsmStreamer::emitDirectiveOptionRVC() {
  OS << "\t.option\trvc\n";
}

void RISCVTargetAsmStreamer::emitDirectiveOptionNoRVC() {
  OS << "\t.option\tnorvc\n";
}

void RISCVTargetAsmStreamer::emitDirectiveOptionRelax() {
  OS << "\t.option\trelax\n";
}

void RISCVTargetAsmStreamer::emitDirectiveOptionNoRelax() {
  OS << "\t.option\tnorelax\n";
}

void RISCVTargetAsmStreamer::emitAttribute(unsigned Attribute, unsigned Value) {
  OS << "\t.attribute\t" << Attribute << ", " << Twine(Value) << "\n";
}

void RISCVTargetAsmStreamer::emitTextAttribute(unsigned Attribute,
                                               StringRef String) {
  OS << "\t.attribute\t" << Attribute << ", \"" << String << "\"\n";
}

void RISCVTargetAsmStreamer::emitIntTextAttribute(unsigned Attribute,
                                                  unsigned IntValue,
                                                  StringRef StringValue) {}

void RISCVTargetAsmStreamer::finishAttributeSection() {}
