//===-- RISCVELFStreamer.cpp - RISCV ELF Target Streamer Methods ----------===//
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

#include "RISCVELFStreamer.h"
#include "RISCVAsmBackend.h"
#include "RISCVBaseInfo.h"
#include "RISCVMCTargetDesc.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/RISCVAttributes.h"

using namespace llvm;

// This part is for ELF object output.
RISCVTargetELFStreamer::RISCVTargetELFStreamer(MCStreamer &S,
                                               const MCSubtargetInfo &STI)
    : RISCVTargetStreamer(S), CurrentVendor("riscv"), STI(STI) {
  MCAssembler &MCA = getStreamer().getAssembler();
  const FeatureBitset &Features = STI.getFeatureBits();
  auto &MAB = static_cast<RISCVAsmBackend &>(MCA.getBackend());
  setTargetABI(RISCVABI::computeTargetABI(STI.getTargetTriple(), Features,
                                          MAB.getTargetOptions().getABIName()));
}

RISCVELFStreamer &RISCVTargetELFStreamer::getStreamer() {
  return static_cast<RISCVELFStreamer &>(Streamer);
}

void RISCVTargetELFStreamer::emitDirectiveOptionPush() {}
void RISCVTargetELFStreamer::emitDirectiveOptionPop() {}
void RISCVTargetELFStreamer::emitDirectiveOptionPIC() {}
void RISCVTargetELFStreamer::emitDirectiveOptionNoPIC() {}
void RISCVTargetELFStreamer::emitDirectiveOptionRVC() {}
void RISCVTargetELFStreamer::emitDirectiveOptionNoRVC() {}
void RISCVTargetELFStreamer::emitDirectiveOptionRelax() {}
void RISCVTargetELFStreamer::emitDirectiveOptionNoRelax() {}

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
  MCAssembler &MCA = getStreamer().getAssembler();
  const FeatureBitset &Features = STI.getFeatureBits();
  RISCVABI::ABI ABI = getTargetABI();

  unsigned EFlags = MCA.getELFHeaderEFlags();

  if (Features[RISCV::FeatureStdExtC])
    EFlags |= ELF::EF_RISCV_RVC;
  if (Features[RISCV::FeatureStdExtZtso])
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
    EFlags |= ELF::EF_RISCV_RVE;
    break;
  case RISCVABI::ABI_Unknown:
    llvm_unreachable("Improperly initialised target ABI");
  }

  MCA.setELFHeaderEFlags(EFlags);
}

void RISCVTargetELFStreamer::reset() {
  AttributeSection = nullptr;
}

void RISCVTargetELFStreamer::emitDirectiveVariantCC(MCSymbol &Symbol) {
  getStreamer().getAssembler().registerSymbol(Symbol);
  cast<MCSymbolELF>(Symbol).setOther(ELF::STO_RISCV_VARIANT_CC);
}

bool RISCVELFStreamer::requiresFixups(MCContext &C, const MCExpr *Value,
                                      const MCExpr *&LHS, const MCExpr *&RHS) {
  const auto *MBE = dyn_cast<MCBinaryExpr>(Value);
  if (MBE == nullptr)
    return false;

  MCValue E;
  if (!Value->evaluateAsRelocatable(E, nullptr, nullptr))
    return false;
  if (E.getSymA() == nullptr || E.getSymB() == nullptr)
    return false;

  const auto &A = E.getSymA()->getSymbol();
  const auto &B = E.getSymB()->getSymbol();

  LHS = MCBinaryExpr::create(MCBinaryExpr::Add, MCSymbolRefExpr::create(&A, C),
                             MCConstantExpr::create(E.getConstant(), C), C);
  RHS = E.getSymB();

  // If either symbol is in a text section, we need to delay the relocation
  // evaluation as relaxation may alter the size of the symbol.
  //
  // Unfortunately, we cannot identify if the symbol was built with relaxation
  // as we do not track the state per symbol or section.  However, BFD will
  // always emit the relocation and so we follow suit which avoids the need to
  // track that information.
  if (A.isInSection() && A.getSection().getKind().isText())
    return true;
  if (B.isInSection() && B.getSection().getKind().isText())
    return true;

  // Support cross-section symbolic differences ...
  return A.isInSection() && B.isInSection() &&
         A.getSection().getName() != B.getSection().getName();
}

void RISCVELFStreamer::reset() {
  static_cast<RISCVTargetStreamer *>(getTargetStreamer())->reset();
  MCELFStreamer::reset();
}

void RISCVELFStreamer::emitValueImpl(const MCExpr *Value, unsigned Size,
                                     SMLoc Loc) {
  const MCExpr *A, *B;
  if (!requiresFixups(getContext(), Value, A, B))
    return MCELFStreamer::emitValueImpl(Value, Size, Loc);

  MCStreamer::emitValueImpl(Value, Size, Loc);

  MCDataFragment *DF = getOrCreateDataFragment();
  flushPendingLabels(DF, DF->getContents().size());
  MCDwarfLineEntry::make(this, getCurrentSectionOnly());

  MCFixupKind Add, Sub;
  std::tie(Add, Sub) = RISCV::getRelocPairForSize(Size);

  DF->getFixups().push_back(
      MCFixup::create(DF->getContents().size(), A, Add, Loc));
  DF->getFixups().push_back(
      MCFixup::create(DF->getContents().size(), B, Sub, Loc));

  DF->getContents().resize(DF->getContents().size() + Size, 0);
}

namespace llvm {
MCELFStreamer *createRISCVELFStreamer(MCContext &C,
                                      std::unique_ptr<MCAsmBackend> MAB,
                                      std::unique_ptr<MCObjectWriter> MOW,
                                      std::unique_ptr<MCCodeEmitter> MCE,
                                      bool RelaxAll) {
  RISCVELFStreamer *S =
      new RISCVELFStreamer(C, std::move(MAB), std::move(MOW), std::move(MCE));
  S->getAssembler().setRelaxAll(RelaxAll);
  return S;
}
} // namespace llvm
