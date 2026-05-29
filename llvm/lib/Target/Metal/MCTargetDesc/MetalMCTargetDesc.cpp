//===- MetalMCTargetDesc.cpp - Metal Target Implementation ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the Metal target initializer for the MC layer.
///
//===----------------------------------------------------------------------===//

#include "MetalMCTargetDesc.h"
#include "MetalContainerObjectWriter.h"
#include "TargetInfo/MetalTargetInfo.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/Triple.h"
#include <memory>

using namespace llvm;

#define GET_INSTRINFO_MC_DESC
#define GET_INSTRINFO_MC_HELPERS
#include "MetalGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "MetalGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "MetalGenRegisterInfo.inc"

namespace {

// AIR instructions are never printed; this is a null stub.
class MetalInstPrinter : public MCInstPrinter {
public:
  MetalInstPrinter(const MCAsmInfo &MAI, const MCInstrInfo &MII,
                   const MCRegisterInfo &MRI)
      : MCInstPrinter(MAI, MII, MRI) {}

  void printInst(const MCInst *, uint64_t, StringRef, const MCSubtargetInfo &,
                 raw_ostream &) override {
    llvm_unreachable(
        "Metal target is bitcode-only; AIR has no textual instructions");
  }

  std::pair<const char *, uint64_t> getMnemonic(const MCInst &) const override {
    return std::make_pair<const char *, uint64_t>("", 0ull);
  }
};

class MetalMCCodeEmitter : public MCCodeEmitter {
public:
  MetalMCCodeEmitter() {}

  void encodeInstruction(const MCInst &, SmallVectorImpl<char> &,
                         SmallVectorImpl<MCFixup> &,
                         const MCSubtargetInfo &) const override {
    llvm_unreachable("Metal target is bitcode-only; the .metallib container is "
                     "produced by MetalEmbedderPass, not via MC encoding");
  }
};

class MetalAsmBackend : public MCAsmBackend {
public:
  MetalAsmBackend(const MCSubtargetInfo &)
      : MCAsmBackend(llvm::endianness::little) {}
  ~MetalAsmBackend() override = default;

  void applyFixup(const MCFragment &, const MCFixup &, const MCValue &,
                  uint8_t *, uint64_t, bool) override {
    llvm_unreachable("Metal target is bitcode-only; no MC fixups are emitted");
  }

  std::unique_ptr<MCObjectTargetWriter>
  createObjectTargetWriter() const override {
    return createMetalContainerTargetObjectWriter();
  }

  bool writeNopData(raw_ostream &, uint64_t,
                    const MCSubtargetInfo *) const override {
    // No instruction stream means no padding is ever required.
    return true;
  }
};

class MetalMCAsmInfo : public MCAsmInfo {
public:
  explicit MetalMCAsmInfo(const Triple &, const MCTargetOptions &Options)
      : MCAsmInfo(Options) {
    // AIR has no textual assembly form; disable every assembly-language
    // feature so MC never tries to print or parse one.
    HasSingleParameterDotFile = false;
    SupportsDebugInformation = false;
    HasIdentDirective = false;
    HasDotTypeDotSizeDirective = false;
    UsesELFSectionDirectiveForBSS = false;
    WeakDirective = nullptr;
  }
};

} // namespace

static MCInstPrinter *createMetalMCInstPrinter(const Triple &,
                                               unsigned SyntaxVariant,
                                               const MCAsmInfo &MAI,
                                               const MCInstrInfo &MII,
                                               const MCRegisterInfo &MRI) {
  if (SyntaxVariant == 0)
    return new MetalInstPrinter(MAI, MII, MRI);
  return nullptr;
}

static MCCodeEmitter *createMetalMCCodeEmitter(const MCInstrInfo &,
                                               MCContext &) {
  return new MetalMCCodeEmitter();
}

static MCAsmBackend *createMetalMCAsmBackend(const Target &,
                                             const MCSubtargetInfo &STI,
                                             const MCRegisterInfo &,
                                             const MCTargetOptions &) {
  return new MetalAsmBackend(STI);
}

static MCSubtargetInfo *
createMetalMCSubtargetInfo(const Triple &TT, StringRef CPU, StringRef FS) {
  return createMetalMCSubtargetInfoImpl(TT, CPU, /*TuneCPU*/ CPU, FS);
}

static MCRegisterInfo *createMetalMCRegisterInfo(const Triple &) {
  return new MCRegisterInfo();
}

static MCInstrInfo *createMetalMCInstrInfo() { return new MCInstrInfo(); }

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeMetalTargetMC() {
  Target &T = getTheMetalTarget();
  RegisterMCAsmInfo<MetalMCAsmInfo> X(T);
  TargetRegistry::RegisterMCInstrInfo(T, createMetalMCInstrInfo);
  TargetRegistry::RegisterMCInstPrinter(T, createMetalMCInstPrinter);
  TargetRegistry::RegisterMCRegInfo(T, createMetalMCRegisterInfo);
  TargetRegistry::RegisterMCSubtargetInfo(T, createMetalMCSubtargetInfo);
  TargetRegistry::RegisterMCCodeEmitter(T, createMetalMCCodeEmitter);
  TargetRegistry::RegisterMCAsmBackend(T, createMetalMCAsmBackend);
}
