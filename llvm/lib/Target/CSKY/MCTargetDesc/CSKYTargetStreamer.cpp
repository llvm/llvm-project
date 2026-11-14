//===-- CSKYTargetStreamer.h - CSKY Target Streamer ----------*- C++ -*----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CSKYTargetStreamer.h"
#include "CSKYMCTargetDesc.h"
#include "MCTargetDesc/CSKYMCAsmInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/CSKYAttributes.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/TargetParser/CSKYTargetParser.h"

using namespace llvm;

//
// ConstantPool implementation
//
// Emit the contents of the constant pool using the provided streamer.
void CSKYConstantPool::emitAll(MCStreamer &Streamer) {
  if (Entries.empty())
    return;

  if (CurrentSection != nullptr)
    Streamer.switchSection(CurrentSection);

  Streamer.emitDataRegion(MCDR_DataRegion);
  for (const ConstantPoolEntry &Entry : Entries) {
    Streamer.emitCodeAlignment(
        Align(Entry.Size),
        *Streamer.getContext().getSubtargetInfo()); // align naturally
    Streamer.emitLabel(Entry.Label);
    Streamer.emitValue(Entry.Value, Entry.Size, Entry.Loc);
  }
  Streamer.emitDataRegion(MCDR_DataRegionEnd);
  Entries.clear();
}

const MCExpr *CSKYConstantPool::addEntry(MCStreamer &Streamer,
                                         const MCExpr *Value, unsigned Size,
                                         SMLoc Loc, const MCExpr *AdjustExpr) {
  if (CurrentSection == nullptr)
    CurrentSection = Streamer.getCurrentSectionOnly();

  auto &Context = Streamer.getContext();

  const MCConstantExpr *C = dyn_cast<MCConstantExpr>(Value);

  // Check if there is existing entry for the same constant. If so, reuse it.
  auto Itr = C ? CachedEntries.find(C->getValue()) : CachedEntries.end();
  if (Itr != CachedEntries.end())
    return Itr->second;

  MCSymbol *CPEntryLabel = Context.createTempSymbol();
  const auto SymRef = MCSymbolRefExpr::create(CPEntryLabel, Context);

  if (AdjustExpr) {
    auto *CSKYExpr = cast<MCSpecifierExpr>(Value);

    Value = MCBinaryExpr::createSub(AdjustExpr, SymRef, Context);
    Value = MCBinaryExpr::createSub(CSKYExpr->getSubExpr(), Value, Context);
    Value = MCSpecifierExpr::create(Value, CSKYExpr->getSpecifier(), Context);
  }

  Entries.push_back(ConstantPoolEntry(CPEntryLabel, Value, Size, Loc));

  if (C)
    CachedEntries[C->getValue()] = SymRef;
  return SymRef;
}

bool CSKYConstantPool::empty() { return Entries.empty(); }

void CSKYConstantPool::clearCache() {
  CurrentSection = nullptr;
  CachedEntries.clear();
}

CSKYTargetStreamer::CSKYTargetStreamer(MCStreamer &S)
    : MCTargetStreamer(S), ConstantPool(new CSKYConstantPool()) {}

const MCExpr *
CSKYTargetStreamer::addConstantPoolEntry(const MCExpr *Expr, SMLoc Loc,
                                         const MCExpr *AdjustExpr) {
  uint8_t ELFRefKind = CSKY::S_Invalid;
  ConstantCounter++;

  const MCExpr *OrigExpr = Expr;

  if (auto *CE = dyn_cast<MCSpecifierExpr>(Expr)) {
    Expr = CE->getSubExpr();
    ELFRefKind = CE->getSpecifier();
  }

  if (const MCSymbolRefExpr *SymExpr = dyn_cast<MCSymbolRefExpr>(Expr)) {
    const MCSymbol *Sym = &SymExpr->getSymbol();

    SymbolIndex Index = {Sym, ELFRefKind};

    if (ConstantMap.find(Index) == ConstantMap.end()) {
      ConstantMap[Index] =
          ConstantPool->addEntry(getStreamer(), OrigExpr, 4, Loc, AdjustExpr);
    }
    return ConstantMap[Index];
  }

  return ConstantPool->addEntry(getStreamer(), Expr, 4, Loc, AdjustExpr);
}

void CSKYTargetStreamer::emitCurrentConstantPool() {
  ConstantPool->emitAll(Streamer);
  ConstantPool->clearCache();
}

// finish() - write out any non-empty assembler constant pools.
void CSKYTargetStreamer::finish() {
  if (ConstantCounter != 0) {
    ConstantPool->emitAll(Streamer);
  }

  finishAttributeSection();
}

void CSKYTargetStreamer::emitTargetAttributes(const MCSubtargetInfo &STI,
                                              bool HardFloatABI) {
  // AsmPrinter emits the attributes once for the module. Ignore the repeated
  // requests from the parser instantiated for each inline asm block, which
  // knows neither the CPU nor the float ABI selected for the module.
  if (EmittedTargetAttributes)
    return;
  EmittedTargetAttributes = true;

  StringRef CPU = STI.getCPU();
  CSKY::ArchKind ArchID = CSKY::parseCPUArch(CPU);

  if (ArchID == CSKY::ArchKind::CK804)
    ArchID = CSKY::ArchKind::CK803;

  StringRef CPU_ARCH = CSKY::getArchName(ArchID);

  if (ArchID == CSKY::ArchKind::INVALID) {
    CPU = "ck810";
    CPU_ARCH = "ck810";
  }
  emitTextAttribute(CSKYAttrs::CSKY_ARCH_NAME, CPU_ARCH);
  emitTextAttribute(CSKYAttrs::CSKY_CPU_NAME, CPU);

  unsigned ISAFlag = 0;
  if (STI.hasFeature(CSKY::HasE1))
    ISAFlag |= CSKYAttrs::V2_ISA_E1;

  if (STI.hasFeature(CSKY::HasE2))
    ISAFlag |= CSKYAttrs::V2_ISA_1E2;

  if (STI.hasFeature(CSKY::Has2E3))
    ISAFlag |= CSKYAttrs::V2_ISA_2E3;

  if (STI.hasFeature(CSKY::HasMP))
    ISAFlag |= CSKYAttrs::ISA_MP;

  if (STI.hasFeature(CSKY::Has3E3r1))
    ISAFlag |= CSKYAttrs::V2_ISA_3E3R1;

  if (STI.hasFeature(CSKY::Has3r1E3r2))
    ISAFlag |= CSKYAttrs::V2_ISA_3E3R2;

  if (STI.hasFeature(CSKY::Has3r2E3r3))
    ISAFlag |= CSKYAttrs::V2_ISA_3E3R3;

  if (STI.hasFeature(CSKY::Has3E7))
    ISAFlag |= CSKYAttrs::V2_ISA_3E7;

  if (STI.hasFeature(CSKY::HasMP1E2))
    ISAFlag |= CSKYAttrs::ISA_MP_1E2;

  if (STI.hasFeature(CSKY::Has7E10))
    ISAFlag |= CSKYAttrs::V2_ISA_7E10;

  if (STI.hasFeature(CSKY::Has10E60))
    ISAFlag |= CSKYAttrs::V2_ISA_10E60;

  if (STI.hasFeature(CSKY::FeatureTrust))
    ISAFlag |= CSKYAttrs::ISA_TRUST;

  if (STI.hasFeature(CSKY::FeatureJAVA))
    ISAFlag |= CSKYAttrs::ISA_JAVA;

  if (STI.hasFeature(CSKY::FeatureCache))
    ISAFlag |= CSKYAttrs::ISA_CACHE;

  if (STI.hasFeature(CSKY::FeatureNVIC))
    ISAFlag |= CSKYAttrs::ISA_NVIC;

  if (STI.hasFeature(CSKY::FeatureDSP))
    ISAFlag |= CSKYAttrs::ISA_DSP;

  if (STI.hasFeature(CSKY::HasDSP1E2))
    ISAFlag |= CSKYAttrs::ISA_DSP_1E2;

  if (STI.hasFeature(CSKY::HasDSPE60))
    ISAFlag |= CSKYAttrs::V2_ISA_DSPE60;

  if (STI.hasFeature(CSKY::FeatureDSPV2))
    ISAFlag |= CSKYAttrs::ISA_DSP_ENHANCE;

  if (STI.hasFeature(CSKY::FeatureDSP_Silan))
    ISAFlag |= CSKYAttrs::ISA_DSP_SILAN;

  if (STI.hasFeature(CSKY::FeatureVDSPV1_128))
    ISAFlag |= CSKYAttrs::ISA_VDSP;

  if (STI.hasFeature(CSKY::FeatureVDSPV2))
    ISAFlag |= CSKYAttrs::ISA_VDSP_2;

  if (STI.hasFeature(CSKY::HasVDSP2E3))
    ISAFlag |= CSKYAttrs::ISA_VDSP_2E3;

  if (STI.hasFeature(CSKY::HasVDSP2E60F))
    ISAFlag |= CSKYAttrs::ISA_VDSP_2E60F;

  emitAttribute(CSKYAttrs::CSKY_ISA_FLAGS, ISAFlag);

  unsigned ISAExtFlag = 0;
  if (STI.hasFeature(CSKY::HasFLOATE1))
    ISAExtFlag |= CSKYAttrs::ISA_FLOAT_E1;

  if (STI.hasFeature(CSKY::HasFLOAT1E2))
    ISAExtFlag |= CSKYAttrs::ISA_FLOAT_1E2;

  if (STI.hasFeature(CSKY::HasFLOAT1E3))
    ISAExtFlag |= CSKYAttrs::ISA_FLOAT_1E3;

  if (STI.hasFeature(CSKY::HasFLOAT3E4))
    ISAExtFlag |= CSKYAttrs::ISA_FLOAT_3E4;

  if (STI.hasFeature(CSKY::HasFLOAT7E60))
    ISAExtFlag |= CSKYAttrs::ISA_FLOAT_7E60;

  emitAttribute(CSKYAttrs::CSKY_ISA_EXT_FLAGS, ISAExtFlag);

  if (STI.hasFeature(CSKY::FeatureDSP))
    emitAttribute(CSKYAttrs::CSKY_DSP_VERSION,
                  CSKYAttrs::DSP_VERSION_EXTENSION);
  if (STI.hasFeature(CSKY::FeatureDSPV2))
    emitAttribute(CSKYAttrs::CSKY_DSP_VERSION, CSKYAttrs::DSP_VERSION_2);

  if (STI.hasFeature(CSKY::FeatureVDSPV2))
    emitAttribute(CSKYAttrs::CSKY_VDSP_VERSION, CSKYAttrs::VDSP_VERSION_2);

  if (STI.hasFeature(CSKY::FeatureFPUV2_SF) ||
      STI.hasFeature(CSKY::FeatureFPUV2_DF))
    emitAttribute(CSKYAttrs::CSKY_FPU_VERSION, CSKYAttrs::FPU_VERSION_2);
  else if (STI.hasFeature(CSKY::FeatureFPUV3_HF) ||
           STI.hasFeature(CSKY::FeatureFPUV3_SF) ||
           STI.hasFeature(CSKY::FeatureFPUV3_DF))
    emitAttribute(CSKYAttrs::CSKY_FPU_VERSION, CSKYAttrs::FPU_VERSION_3);

  bool hasAnyFloatExt = STI.hasFeature(CSKY::FeatureFPUV2_SF) ||
                        STI.hasFeature(CSKY::FeatureFPUV2_DF) ||
                        STI.hasFeature(CSKY::FeatureFPUV3_HF) ||
                        STI.hasFeature(CSKY::FeatureFPUV3_SF) ||
                        STI.hasFeature(CSKY::FeatureFPUV3_DF);

  // The hard-float *ABI* (FP values in FP registers at call boundaries) is
  // selected by the "float-abi" module flag, resolved by the caller; it does
  // not depend on ModeHardFloat, which only distinguishes the soft-float ABI
  // that still uses hard-float instructions (SOFTFP) from pure soft float.
  if (hasAnyFloatExt && HardFloatABI)
    emitAttribute(CSKYAttrs::CSKY_FPU_ABI, CSKYAttrs::FPU_ABI_HARD);
  else if (hasAnyFloatExt && STI.hasFeature(CSKY::ModeHardFloat))
    emitAttribute(CSKYAttrs::CSKY_FPU_ABI, CSKYAttrs::FPU_ABI_SOFTFP);
  else
    emitAttribute(CSKYAttrs::CSKY_FPU_ABI, CSKYAttrs::FPU_ABI_SOFT);

  unsigned HardFPFlag = 0;
  if (STI.hasFeature(CSKY::FeatureFPUV3_HF))
    HardFPFlag |= CSKYAttrs::FPU_HARDFP_HALF;
  if (STI.hasFeature(CSKY::FeatureFPUV2_SF) ||
      STI.hasFeature(CSKY::FeatureFPUV3_SF))
    HardFPFlag |= CSKYAttrs::FPU_HARDFP_SINGLE;
  if (STI.hasFeature(CSKY::FeatureFPUV2_DF) ||
      STI.hasFeature(CSKY::FeatureFPUV3_DF))
    HardFPFlag |= CSKYAttrs::FPU_HARDFP_DOUBLE;

  if (HardFPFlag != 0) {
    emitAttribute(CSKYAttrs::CSKY_FPU_DENORMAL, CSKYAttrs::NEEDED);
    emitAttribute(CSKYAttrs::CSKY_FPU_EXCEPTION, CSKYAttrs::NEEDED);
    emitTextAttribute(CSKYAttrs::CSKY_FPU_NUMBER_MODULE, "IEEE 754");
    emitAttribute(CSKYAttrs::CSKY_FPU_HARDFP, HardFPFlag);
  }
}

void CSKYTargetStreamer::emitAttribute(unsigned Attribute, unsigned Value) {}
void CSKYTargetStreamer::emitTextAttribute(unsigned Attribute,
                                           StringRef String) {}
void CSKYTargetStreamer::finishAttributeSection() {}

void CSKYTargetAsmStreamer::emitAttribute(unsigned Attribute, unsigned Value) {
  OS << "\t.csky_attribute\t" << Attribute << ", " << Twine(Value) << "\n";
}

void CSKYTargetAsmStreamer::emitTextAttribute(unsigned Attribute,
                                              StringRef String) {
  OS << "\t.csky_attribute\t" << Attribute << ", \"" << String << "\"\n";
}

void CSKYTargetAsmStreamer::finishAttributeSection() {}
