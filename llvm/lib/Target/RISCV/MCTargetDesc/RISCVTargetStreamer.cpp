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
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/RISCVAttributes.h"
#include "llvm/TargetParser/RISCVISAInfo.h"

using namespace llvm;

// This option controls whether or not we emit ELF attributes for ABI features,
// like RISC-V atomics or X3 usage.
static cl::opt<bool> RiscvAbiAttr(
    "riscv-abi-attributes",
    cl::desc("Enable emitting RISC-V ELF attributes for ABI features"),
    cl::Hidden);

RISCVTargetStreamer::RISCVTargetStreamer(MCStreamer &S) : MCTargetStreamer(S) {}

void RISCVTargetStreamer::finish() { finishAttributeSection(); }
void RISCVTargetStreamer::reset() {}

void RISCVTargetStreamer::emitDirectiveOptionArch(
    ArrayRef<RISCVOptionArchArg> Args) {}
void RISCVTargetStreamer::emitDirectiveOptionExact() {}
void RISCVTargetStreamer::emitDirectiveOptionNoExact() {}
void RISCVTargetStreamer::emitDirectiveOptionPIC() {}
void RISCVTargetStreamer::emitDirectiveOptionNoPIC() {}
void RISCVTargetStreamer::emitDirectiveOptionPop() {}
void RISCVTargetStreamer::emitDirectiveOptionPush() {}
void RISCVTargetStreamer::emitDirectiveOptionRelax() {}
void RISCVTargetStreamer::emitDirectiveOptionNoRelax() {}
void RISCVTargetStreamer::emitDirectiveOptionRVC() {}
void RISCVTargetStreamer::emitDirectiveOptionNoRVC() {}
void RISCVTargetStreamer::emitDirectiveVariantCC(MCSymbol &Symbol) {}
void RISCVTargetStreamer::emitAttribute(unsigned Attribute, unsigned Value) {}
void RISCVTargetStreamer::finishAttributeSection() {}
void RISCVTargetStreamer::emitTextAttribute(unsigned Attribute,
                                            StringRef String) {}
void RISCVTargetStreamer::emitIntTextAttribute(unsigned Attribute,
                                               unsigned IntValue,
                                               StringRef StringValue) {}

void RISCVTargetStreamer::emitNoteGnuPropertySection(
    const uint32_t Feature1And) {
  MCStreamer &OutStreamer = getStreamer();
  MCContext &Ctx = OutStreamer.getContext();

  const Triple &Triple = Ctx.getTargetTriple();
  Align NoteAlign;
  uint64_t DescSize;
  if (Triple.isArch64Bit()) {
    NoteAlign = Align(8);
    DescSize = 16;
  } else {
    assert(Triple.isArch32Bit());
    NoteAlign = Align(4);
    DescSize = 12;
  }

  assert(Ctx.getObjectFileType() == MCContext::Environment::IsELF);
  MCSection *const NoteSection =
      Ctx.getELFSection(".note.gnu.property", ELF::SHT_NOTE, ELF::SHF_ALLOC);
  OutStreamer.pushSection();
  OutStreamer.switchSection(NoteSection);

  // Emit the note header
  OutStreamer.emitValueToAlignment(NoteAlign);
  OutStreamer.emitIntValue(4, 4);                           // n_namsz
  OutStreamer.emitIntValue(DescSize, 4);                    // n_descsz
  OutStreamer.emitIntValue(ELF::NT_GNU_PROPERTY_TYPE_0, 4); // n_type
  OutStreamer.emitBytes(StringRef("GNU", 4));               // n_name

  // Emit n_desc field

  // Emit the feature_1_and property
  OutStreamer.emitIntValue(ELF::GNU_PROPERTY_RISCV_FEATURE_1_AND, 4); // pr_type
  OutStreamer.emitIntValue(4, 4);              // pr_datasz
  OutStreamer.emitIntValue(Feature1And, 4);    // pr_data
  OutStreamer.emitValueToAlignment(NoteAlign); // pr_padding

  OutStreamer.popSection();
}

void RISCVTargetStreamer::setTargetABI(RISCVABI::ABI ABI) {
  assert(ABI != RISCVABI::ABI_Unknown && "Improperly initialized target ABI");
  TargetABI = ABI;
}

void RISCVTargetStreamer::setFlagsFromFeatures(const MCSubtargetInfo &STI) {
  HasRVC = STI.hasFeature(RISCV::FeatureStdExtZca);
  HasTSO = STI.hasFeature(RISCV::FeatureStdExtZtso);
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

  if (RiscvAbiAttr && STI.hasFeature(RISCV::FeatureStdExtA)) {
    unsigned AtomicABITag;
    if (STI.hasFeature(RISCV::FeatureStdExtZalasr))
      AtomicABITag = static_cast<unsigned>(RISCVAttrs::RISCVAtomicAbiTag::A7);
    else if (STI.hasFeature(RISCV::FeatureNoTrailingSeqCstFence))
      AtomicABITag = static_cast<unsigned>(RISCVAttrs::RISCVAtomicAbiTag::A6C);
    else
      AtomicABITag = static_cast<unsigned>(RISCVAttrs::RISCVAtomicAbiTag::A6S);
    emitAttribute(RISCVAttrs::ATOMIC_ABI, AtomicABITag);
  }
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

void RISCVTargetAsmStreamer::emitDirectiveOptionExact() {
  OS << "\t.option\texact\n";
}

void RISCVTargetAsmStreamer::emitDirectiveOptionNoExact() {
  OS << "\t.option\tnoexact\n";
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
