//===-- RISCVELFStreamer.cpp - RISC-V ELF Target Streamer Methods ---------===//
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

#include "RISCVELFStreamer.h"
#include "RISCVAsmBackend.h"
#include "RISCVBaseInfo.h"
#include "RISCVMCTargetDesc.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCSubtargetInfo.h"

using namespace llvm;

// This part is for ELF object output.
RISCVTargetELFStreamer::RISCVTargetELFStreamer(MCStreamer &S,
                                               const MCSubtargetInfo &STI)
    : RISCVTargetStreamer(S), CurrentVendor("riscv") {
  MCAssembler &MCA = getStreamer().getAssembler();
  const FeatureBitset &Features = STI.getFeatureBits();
  auto &MAB = static_cast<RISCVAsmBackend &>(MCA.getBackend());
  setTargetABI(RISCVABI::computeTargetABI(STI.getTargetTriple(), Features,
                                          MAB.getTargetOptions().getABIName()));
  setFlagsFromFeatures(STI);

  // Compute the initial ISA string.  This serves two purposes:
  //   1. Deduplication: subsequent .option arch/rvc/norvc directives compare
  //      against ArchString to avoid propagating redundant ISA updates.
  //   2. Initial symbol: seed the streamer's active ISA so a "$x<ArchString>"
  //      mapping symbol is emitted before the first instruction, recording
  //      the full ISA in the object even when no .option directive is present.
  if (auto ParseResult = RISCVFeatures::parseFeatureBits(
          STI.hasFeature(RISCV::Feature64Bit), Features)) {
    InitialArchString = (*ParseResult)->toString();
    ArchString = InitialArchString;
    getStreamer().setMappingSymbolArch(ArchString);
  }
}

RISCVELFStreamer::RISCVELFStreamer(MCContext &C,
                                   std::unique_ptr<MCAsmBackend> MAB,
                                   std::unique_ptr<MCObjectWriter> MOW,
                                   std::unique_ptr<MCCodeEmitter> MCE)
    : MCELFStreamer(C, std::move(MAB), std::move(MOW), std::move(MCE)) {}

RISCVELFStreamer &RISCVTargetELFStreamer::getStreamer() {
  return static_cast<RISCVELFStreamer &>(Streamer);
}

void RISCVTargetELFStreamer::setArchString(StringRef Arch) {
  if (Arch == ArchString)
    return;
  ArchString = std::string(Arch);
  getStreamer().setMappingSymbolArch(Arch);
}

void RISCVTargetELFStreamer::emitDirectiveOptionPush() {
  ArchStringStack.push_back(ArchString);
}

void RISCVTargetELFStreamer::emitDirectiveOptionPop() {
  if (!ArchStringStack.empty())
    setArchString(ArchStringStack.pop_back_val());
}

void RISCVTargetELFStreamer::emitDirectiveOptionExact() {}
void RISCVTargetELFStreamer::emitDirectiveOptionNoExact() {}
void RISCVTargetELFStreamer::emitDirectiveOptionPIC() {}
void RISCVTargetELFStreamer::emitDirectiveOptionNoPIC() {}
void RISCVTargetELFStreamer::emitDirectiveOptionRelax() {}
void RISCVTargetELFStreamer::emitDirectiveOptionNoRelax() {}
void RISCVTargetELFStreamer::emitDirectiveOptionRVC() {}
void RISCVTargetELFStreamer::emitDirectiveOptionNoRVC() {}

void RISCVTargetELFStreamer::emitAttribute(unsigned Attribute, unsigned Value) {
  getStreamer().setAttributeItem(Attribute, Value, /*OverwriteExisting=*/true);
}

void RISCVTargetELFStreamer::emitTextAttribute(unsigned Attribute,
                                               StringRef String) {
  getStreamer().setAttributeItem(Attribute, String, /*OverwriteExisting=*/true);
}

void RISCVTargetELFStreamer::emitIntTextAttribute(unsigned Attribute,
                                                  unsigned IntValue,
                                                  StringRef StringValue) {
  getStreamer().setAttributeItems(Attribute, IntValue, StringValue,
                                  /*OverwriteExisting=*/true);
}

void RISCVTargetELFStreamer::finishAttributeSection() {
  RISCVELFStreamer &S = getStreamer();
  if (S.Contents.empty())
    return;

  S.emitAttributesSection(CurrentVendor, ".riscv.attributes",
                          ELF::SHT_RISCV_ATTRIBUTES, AttributeSection);
}

void RISCVTargetELFStreamer::finish() {
  RISCVTargetStreamer::finish();
  ELFObjectWriter &W = getStreamer().getWriter();
  RISCVABI::ABI ABI = getTargetABI();

  unsigned EFlags = W.getELFHeaderEFlags();

  if (hasRVC())
    EFlags |= ELF::EF_RISCV_RVC;
  if (hasTSO())
    EFlags |= ELF::EF_RISCV_TSO;

  switch (ABI) {
  case RISCVABI::ABI_ILP32:
  case RISCVABI::ABI_LP64:
    break;
  case RISCVABI::ABI_ILP32F:
  case RISCVABI::ABI_LP64F:
    EFlags |= ELF::EF_RISCV_FLOAT_ABI_SINGLE;
    break;
  case RISCVABI::ABI_ILP32D:
  case RISCVABI::ABI_LP64D:
    EFlags |= ELF::EF_RISCV_FLOAT_ABI_DOUBLE;
    break;
  case RISCVABI::ABI_ILP32E:
  case RISCVABI::ABI_LP64E:
    EFlags |= ELF::EF_RISCV_RVE;
    break;
  case RISCVABI::ABI_Unknown:
    llvm_unreachable("Improperly initialised target ABI");
  }

  W.setELFHeaderEFlags(EFlags);
}

void RISCVTargetELFStreamer::reset() {
  AttributeSection = nullptr;
  ArchString = InitialArchString;
  ArchStringStack.clear();
  // Re-seed the streamer's active ISA so the first instruction after reset
  // still records the full ISA via "$x<ISA>", matching the behaviour set up
  // in the constructor.
  if (!InitialArchString.empty())
    getStreamer().setMappingSymbolArch(InitialArchString);
}

void RISCVTargetELFStreamer::emitDirectiveVariantCC(MCSymbol &Symbol) {
  getStreamer().getAssembler().registerSymbol(Symbol);
  static_cast<MCSymbolELF &>(Symbol).setOther(ELF::STO_RISCV_VARIANT_CC);
}

void RISCVELFStreamer::reset() {
  MCELFStreamer::reset();
  LastMappingSymbols.clear();
  LastEMS = EMS_None;
  MappingSymbolArch.clear();
  LastEmittedArch.clear();
  LastEmittedArchInSection.clear();
  // Call target streamer reset last: it may call setMappingSymbolArch to
  // re-seed the initial ISA after our state has been cleared.
  static_cast<RISCVTargetStreamer *>(getTargetStreamer())->reset();
}

void RISCVELFStreamer::emitDataMappingSymbol() {
  if (LastEMS == EMS_Data)
    return;
  emitMappingSymbol("$d");
  LastEMS = EMS_Data;
}

void RISCVELFStreamer::emitInstructionsMappingSymbol() {
  // Emit a mapping symbol at the start of each instruction run, and whenever
  // the active ISA has changed since the last one emitted in this section.
  // The symbol takes the form "$x<ISA>" when MappingSymbolArch is known, or
  // plain "$x" as a fallback.  The comparison with LastEmittedArch provides
  // deduplication: repeating .option arch with the same ISA, or re-entering a
  // section whose last mapping symbol already matches the active ISA, emits
  // no redundant symbol.
  bool NeedSymbol =
      LastEMS != EMS_Instructions || LastEmittedArch != MappingSymbolArch;
  if (NeedSymbol) {
    if (MappingSymbolArch.empty())
      emitMappingSymbol("$x");
    else
      emitMappingSymbol("$x" + MappingSymbolArch);
    LastEmittedArch = MappingSymbolArch;
  }
  LastEMS = EMS_Instructions;
}

void RISCVELFStreamer::emitMappingSymbol(StringRef Name) {
  auto *Symbol =
      static_cast<MCSymbolELF *>(getContext().createLocalSymbol(Name));
  emitLabel(Symbol);
  Symbol->setType(ELF::STT_NOTYPE);
  Symbol->setBinding(ELF::STB_LOCAL);
}

void RISCVELFStreamer::setMappingSymbolArch(StringRef Arch) {
  MappingSymbolArch = std::string(Arch);
}

void RISCVELFStreamer::changeSection(MCSection *Section, uint32_t Subsection) {
  // We have to keep track of the mapping symbol state of any sections we
  // use. Each one should start off as EMS_None, which is provided as the
  // default constructor by DenseMap::lookup.  The last ISA suffix emitted in
  // each section is also preserved so that re-entering a section only emits a
  // new "$x<ISA>" symbol when the active ISA has actually changed.
  const MCSection *Prev = getPreviousSection().first;
  LastMappingSymbols[Prev] = LastEMS;
  LastEmittedArchInSection[Prev] = LastEmittedArch;
  LastEMS = LastMappingSymbols.lookup(Section);
  auto It = LastEmittedArchInSection.find(Section);
  LastEmittedArch = It != LastEmittedArchInSection.end() ? It->second : "";

  MCELFStreamer::changeSection(Section, Subsection);
}

void RISCVELFStreamer::emitInstruction(const MCInst &Inst,
                                       const MCSubtargetInfo &STI) {
  emitInstructionsMappingSymbol();
  MCELFStreamer::emitInstruction(Inst, STI);
}

void RISCVELFStreamer::emitBytes(StringRef Data) {
  emitDataMappingSymbol();
  MCELFStreamer::emitBytes(Data);
}

void RISCVELFStreamer::emitFill(const MCExpr &NumBytes, uint64_t FillValue,
                                SMLoc Loc) {
  emitDataMappingSymbol();
  MCELFStreamer::emitFill(NumBytes, FillValue, Loc);
}

void RISCVELFStreamer::emitValueImpl(const MCExpr *Value, unsigned Size,
                                     SMLoc Loc) {
  emitDataMappingSymbol();
  MCELFStreamer::emitValueImpl(Value, Size, Loc);
}

MCStreamer *llvm::createRISCVELFStreamer(const Triple &, MCContext &C,
                                         std::unique_ptr<MCAsmBackend> &&MAB,
                                         std::unique_ptr<MCObjectWriter> &&MOW,
                                         std::unique_ptr<MCCodeEmitter> &&MCE) {
  return new RISCVELFStreamer(C, std::move(MAB), std::move(MOW),
                              std::move(MCE));
}

void RISCVTargetELFStreamer::emitNoteGnuPropertySection(
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
