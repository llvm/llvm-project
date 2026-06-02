//===-- EZHMCTargetDesc.cpp - EZH Target Descriptions -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides EZH specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "EZHMCTargetDesc.h"
#include "EZHInstPrinter.h"
#include "EZHMCAsmInfo.h"
#include "TargetInfo/EZHTargetInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TargetParser/Triple.h"
#include <cstdint>
#include <string>

#define GET_INSTRINFO_MC_DESC
#define ENABLE_INSTR_PREDICATE_VERIFIER
#include "EZHGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "EZHGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "EZHGenRegisterInfo.inc"

using namespace llvm;

static MCInstrInfo *createEZHMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitEZHMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createEZHMCRegisterInfo(const Triple & /*TT*/) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitEZHMCRegisterInfo(X, EZH::RA, 0, 0, EZH::PC);
  return X;
}

static MCSubtargetInfo *createEZHMCSubtargetInfo(const Triple &TT,
                                                 StringRef CPU, StringRef FS) {
  StringRef CPUName = CPU.empty() ? "generic" : CPU;
  return createEZHMCSubtargetInfoImpl(TT, CPUName, /*TuneCPU*/ CPUName, FS);
}

static MCStreamer *createMCStreamer(const Triple &T, MCContext &Context,
                                    std::unique_ptr<MCAsmBackend> &&MAB,
                                    std::unique_ptr<MCObjectWriter> &&OW,
                                    std::unique_ptr<MCCodeEmitter> &&Emitter) {
  if (!T.isOSBinFormatELF())
    llvm_unreachable("OS not supported");

  Context.setUseNamesOnTempLabels(true);

  return createELFStreamer(Context, std::move(MAB), std::move(OW),
                           std::move(Emitter));
}

static MCInstPrinter *createEZHMCInstPrinter(const Triple & /*T*/,
                                             unsigned SyntaxVariant,
                                             const MCAsmInfo &MAI,
                                             const MCInstrInfo &MII,
                                             const MCRegisterInfo &MRI) {
  if (SyntaxVariant == 0)
    return new EZHInstPrinter(MAI, MII, MRI);
  return nullptr;
}

static MCRelocationInfo *createEZHElfRelocation(const Triple &TheTriple,
                                                MCContext &Ctx) {
  return createMCRelocationInfo(TheTriple, Ctx);
}

namespace {

class EZHMCInstrAnalysis : public MCInstrAnalysis {
public:
  explicit EZHMCInstrAnalysis(const MCInstrInfo *Info)
      : MCInstrAnalysis(Info) {}

  bool evaluateBranch(const MCInst &Inst, uint64_t Addr, uint64_t Size,
                      uint64_t &Target) const override {
    if (Inst.getNumOperands() == 0)
      return false;
    if (!isConditionalBranch(Inst) && !isUnconditionalBranch(Inst) &&
        !isCall(Inst))
      return false;

    if (Info->get(Inst.getOpcode()).operands()[0].OperandType ==
        MCOI::OPERAND_PCREL) {
      if (!Inst.getOperand(0).isImm())
        return false;
      int64_t Imm = Inst.getOperand(0).getImm();
      Target = Addr + Size + Imm;
      return true;
    } else {
      if (!Inst.getOperand(0).isImm())
        return false;
      int64_t Imm = Inst.getOperand(0).getImm();

      // Skip case where immediate is 0 as that occurs in file that isn't linked
      // and the branch target inferred would be wrong.
      if (Imm == 0)
        return false;

      Target = Imm;
      return true;
    }
  }
};

} // end anonymous namespace

static MCInstrAnalysis *createEZHInstrAnalysis(const MCInstrInfo *Info) {
  return new EZHMCInstrAnalysis(Info);
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeEZHTargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfo<EZHMCAsmInfo> X(getTheEZHTarget());

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(getTheEZHTarget(), createEZHMCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(getTheEZHTarget(), createEZHMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(getTheEZHTarget(),
                                          createEZHMCSubtargetInfo);

  // Register the MC code emitter
  TargetRegistry::RegisterMCCodeEmitter(getTheEZHTarget(),
                                        createEZHMCCodeEmitter);

  // Register the ASM Backend
  TargetRegistry::RegisterMCAsmBackend(getTheEZHTarget(), createEZHAsmBackend);

  // Register the MCInstPrinter.
  TargetRegistry::RegisterMCInstPrinter(getTheEZHTarget(),
                                        createEZHMCInstPrinter);

  // Register the ELF streamer.
  TargetRegistry::RegisterELFStreamer(getTheEZHTarget(), createMCStreamer);

  // Register the MC relocation info.
  TargetRegistry::RegisterMCRelocationInfo(getTheEZHTarget(),
                                           createEZHElfRelocation);

  // Register the MC instruction analyzer.
  TargetRegistry::RegisterMCInstrAnalysis(getTheEZHTarget(),
                                          createEZHInstrAnalysis);
}
