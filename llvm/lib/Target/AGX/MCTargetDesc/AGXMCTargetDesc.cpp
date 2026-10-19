//===-- AGXMCTargetDesc.cpp - AGX target descriptions ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AGXMCTargetDesc.h"
#include "AGXInstPrinter.h"
#include "AGXMCAsmInfo.h"
#include "TargetInfo/AGXTargetInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

#define GET_INSTRINFO_MC_DESC
#define GET_INSTRINFO_NAMED_OPS
#define ENABLE_INSTR_PREDICATE_VERIFIER
#include "AGXGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "AGXGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "AGXGenRegisterInfo.inc"

static MCInstrInfo *createAGXMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitAGXMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createAGXMCRegisterInfo(const Triple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitAGXMCRegisterInfo(X, AGX::R1);
  return X;
}

static MCSubtargetInfo *createAGXMCSubtargetInfo(const Triple &TT,
                                                 StringRef CPU, StringRef FS) {
  if (CPU.empty())
    CPU = "generic";
  return createAGXMCSubtargetInfoImpl(TT, CPU, /*TuneCPU=*/CPU, FS);
}

static MCInstPrinter *createAGXMCInstPrinter(const Triple &T,
                                             unsigned SyntaxVariant,
                                             const MCAsmInfo &MAI,
                                             const MCInstrInfo &MII,
                                             const MCRegisterInfo &MRI) {
  if (SyntaxVariant != 0)
    return nullptr;
  return new AGXInstPrinter(MAI, MII, MRI);
}

static MCStreamer *createAGXELFStreamer(const Triple &T, MCContext &Ctx,
                                        std::unique_ptr<MCAsmBackend> &&MAB,
                                        std::unique_ptr<MCObjectWriter> &&OW,
                                        std::unique_ptr<MCCodeEmitter> &&CE) {
  return createELFStreamer(Ctx, std::move(MAB), std::move(OW), std::move(CE));
}

namespace {

class AGXMCInstrAnalysis : public MCInstrAnalysis {
public:
  explicit AGXMCInstrAnalysis(const MCInstrInfo *Info)
      : MCInstrAnalysis(Info) {}

  bool evaluateBranch(const MCInst &Inst, uint64_t Addr, uint64_t Size,
                      uint64_t &Target) const override {
    const MCInstrDesc &Desc = Info->get(Inst.getOpcode());
    for (unsigned I = 0, E = Inst.getNumOperands(); I != E; ++I) {
      if (I >= Desc.getNumOperands() ||
          Desc.operands()[I].OperandType != MCOI::OPERAND_PCREL)
        continue;
      if (!Inst.getOperand(I).isImm())
        return false;
      // Branch offsets are relative to the start of the branch.
      Target = Addr + int32_t(Inst.getOperand(I).getImm());
      return true;
    }
    return false;
  }
};

} // namespace

static MCInstrAnalysis *createAGXMCInstrAnalysis(const MCInstrInfo *Info) {
  return new AGXMCInstrAnalysis(Info);
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeAGXTargetMC() {
  Target &T = getTheAGXTarget();
  RegisterMCAsmInfo<AGXMCAsmInfo> X(T);
  TargetRegistry::RegisterMCInstrInfo(T, createAGXMCInstrInfo);
  TargetRegistry::RegisterMCRegInfo(T, createAGXMCRegisterInfo);
  TargetRegistry::RegisterMCSubtargetInfo(T, createAGXMCSubtargetInfo);
  TargetRegistry::RegisterMCInstPrinter(T, createAGXMCInstPrinter);
  TargetRegistry::RegisterMCCodeEmitter(T, createAGXMCCodeEmitter);
  TargetRegistry::RegisterMCAsmBackend(T, createAGXAsmBackend);
  TargetRegistry::RegisterELFStreamer(T, createAGXELFStreamer);
  TargetRegistry::RegisterMCInstrAnalysis(T, createAGXMCInstrAnalysis);
}
