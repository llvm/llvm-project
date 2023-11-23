//===-- RISCVTargetStreamer.cpp - RISC-V Target Streamer Methods ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides RISC-V specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "RISCVTargetStreamer.h"
#include "RISCVBaseInfo.h"
#include "RISCVMCTargetDesc.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/RISCVAttributes.h"
#include "llvm/TargetParser/RISCVISAInfo.h"

using namespace llvm;

RISCVTargetStreamer::RISCVTargetStreamer(MCStreamer &S) : MCTargetStreamer(S) {}

void RISCVTargetStreamer::finish() {
  finishAttributeSection();
  emitGNUProgramProperties();
}

void RISCVTargetStreamer::reset() {}

void RISCVTargetStreamer::emitDirectiveOptionPush() {}
void RISCVTargetStreamer::emitDirectiveOptionPop() {}
void RISCVTargetStreamer::emitDirectiveOptionPIC() {}
void RISCVTargetStreamer::emitDirectiveOptionNoPIC() {}
void RISCVTargetStreamer::emitDirectiveOptionRVC() {}
void RISCVTargetStreamer::emitDirectiveOptionNoRVC() {}
void RISCVTargetStreamer::emitDirectiveOptionRelax() {}
void RISCVTargetStreamer::emitDirectiveOptionNoRelax() {}
void RISCVTargetStreamer::emitDirectiveOptionArch(
    ArrayRef<RISCVOptionArchArg> Args) {}
void RISCVTargetStreamer::emitDirectiveVariantCC(MCSymbol &Symbol) {}
void RISCVTargetStreamer::emitAttribute(unsigned Attribute, unsigned Value) {}
void RISCVTargetStreamer::finishAttributeSection() {}
void RISCVTargetStreamer::emitTextAttribute(unsigned Attribute,
                                            StringRef String) {}
void RISCVTargetStreamer::emitIntTextAttribute(unsigned Attribute,
                                               unsigned IntValue,
                                               StringRef StringValue) {}
void RISCVTargetStreamer::setTargetABI(RISCVABI::ABI ABI) {
  assert(ABI != RISCVABI::ABI_Unknown && "Improperly initialized target ABI");
  TargetABI = ABI;
}

void RISCVTargetStreamer::setFlagsFromFeatures(const MCSubtargetInfo &STI) {
  HasRVC = STI.hasFeature(RISCV::FeatureStdExtC) ||
           STI.hasFeature(RISCV::FeatureStdExtZca);
  HasTSO = STI.hasFeature(RISCV::FeatureStdExtZtso);
  HasZicfilp = STI.hasFeature(RISCV::FeatureStdExtZicfilp);
  HasZicfiss = STI.hasFeature(RISCV::FeatureStdExtZicfiss);
  IsRV64 = STI.hasFeature(RISCV::Feature64Bit);
}

void RISCVTargetStreamer::emitTargetAttributes(const MCSubtargetInfo &STI,
                                               bool EmitStackAlign) {
  if (EmitStackAlign) {
    unsigned StackAlign;
    if (TargetABI == RISCVABI::ABI_ILP32E)
      StackAlign = 4;
    else if (TargetABI == RISCVABI::ABI_LP64E)
      StackAlign = 8;
    else
      StackAlign = 16;
    emitAttribute(RISCVAttrs::STACK_ALIGN, StackAlign);
  }

  auto ParseResult = RISCVFeatures::parseFeatureBits(
      STI.hasFeature(RISCV::Feature64Bit), STI.getFeatureBits());
  if (!ParseResult) {
    report_fatal_error(ParseResult.takeError());
  } else {
    auto &ISAInfo = *ParseResult;
    emitTextAttribute(RISCVAttrs::ARCH, ISAInfo->toString());
  }
}

void RISCVTargetStreamer::emitGNUProgramProperties() {
  unsigned FeatureAndFlags = 0;
  // Check Zicfilp or Zicfiss with features
  // TODO should we check with codegen enable
  // ex. -mllvm -riscv-hardware-shadow-stack=true ?
  if (hasZicfilp())
    FeatureAndFlags |= ELF::GNU_PROPERTY_RISCV_FEATURE_1_CFI_LP_SIMPLE;

  if (hasZicfiss())
    FeatureAndFlags |= ELF::GNU_PROPERTY_RISCV_FEATURE_1_CFI_SS;

  if (FeatureAndFlags == 0)
    return;

  MCStreamer &OutStreamer = getStreamer();
  MCContext &Context = OutStreamer.getContext();
  MCSectionELF *Nt = Context.getELFSection(".note.gnu.property", ELF::SHT_NOTE,
                                           ELF::SHF_ALLOC);
  MCSection *Cur = OutStreamer.getCurrentSectionOnly();
  OutStreamer.switchSection(Nt);

  // Emit the note header.
  uint64_t DataSize = isRV64() ? 4 : 3;
  OutStreamer.emitValueToAlignment(isRV64() ? Align(8) : Align(4));
  OutStreamer.emitIntValue(4, 4);     // data size for note name
  OutStreamer.emitIntValue(4 * DataSize, 4); // data size
  OutStreamer.emitIntValue(ELF::NT_GNU_PROPERTY_TYPE_0, 4); // note type
  OutStreamer.emitBytes(StringRef("GNU", 4));               // note name

  // Emit the CFI(Zicfilp/Zicfiss) properties.
  OutStreamer.emitIntValue(ELF::GNU_PROPERTY_RISCV_FEATURE_1_AND,
                           4);        // and property
  OutStreamer.emitIntValue(4, 4);     // data size
  OutStreamer.emitIntValue(FeatureAndFlags, 4); // data
  if (isRV64())
    OutStreamer.emitIntValue(0, 4);     // Padding

  OutStreamer.endSection(Nt);
  OutStreamer.switchSection(Cur);
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

void RISCVTargetAsmStreamer::emitDirectiveOptionArch(
    ArrayRef<RISCVOptionArchArg> Args) {
  OS << "\t.option\tarch";
  for (const auto &Arg : Args) {
    OS << ", ";
    switch (Arg.Type) {
    case RISCVOptionArchArgType::Full:
      break;
    case RISCVOptionArchArgType::Plus:
      OS << "+";
      break;
    case RISCVOptionArchArgType::Minus:
      OS << "-";
      break;
    }
    OS << Arg.Value;
  }
  OS << "\n";
}

void RISCVTargetAsmStreamer::emitDirectiveVariantCC(MCSymbol &Symbol) {
  OS << "\t.variant_cc\t" << Symbol.getName() << "\n";
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
