//===-- LanaiMCTargetDesc.cpp - Lanai Target Descriptions -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Lanai specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "LanaiMCTargetDesc.h"

#include "InstPrinter/LanaiInstPrinter.h"
#include "LanaiMCAsmInfo.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#include "LanaiGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "LanaiGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "LanaiGenRegisterInfo.inc"

using namespace llvm;

static MCInstrInfo *createLanaiMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitLanaiMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createLanaiMCRegisterInfo(const Triple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitLanaiMCRegisterInfo(X, Lanai::RCA, 0, 0, Lanai::PC);
  return X;
}

static MCSubtargetInfo *
createLanaiMCSubtargetInfo(const Triple &TT, StringRef CPU, StringRef FS) {
  std::string CPUName = CPU;
  if (CPUName.empty())
    CPUName = "generic";

  return createLanaiMCSubtargetInfoImpl(TT, CPUName, FS);
}

static MCCodeGenInfo *createLanaiMCCodeGenInfo(const Triple &TT,
                                               Reloc::Model RM,
                                               CodeModel::Model CM,
                                               CodeGenOpt::Level OL) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  X->initMCCodeGenInfo(RM, CM, OL);
  return X;
}

static MCStreamer *createMCStreamer(const Triple &T, MCContext &Context,
                                    MCAsmBackend &MAB, raw_pwrite_stream &OS,
                                    MCCodeEmitter *Emitter, bool RelaxAll) {
  if (!T.isOSBinFormatELF())
    llvm_unreachable("OS not supported");

  return createELFStreamer(Context, MAB, OS, Emitter, RelaxAll);
}

static MCInstPrinter *createLanaiMCInstPrinter(const Triple &T,
                                               unsigned SyntaxVariant,
                                               const MCAsmInfo &MAI,
                                               const MCInstrInfo &MII,
                                               const MCRegisterInfo &MRI) {
  if (SyntaxVariant == 0)
    return new LanaiInstPrinter(MAI, MII, MRI);
  return 0;
}

MCRelocationInfo *createLanaiElfRelocation(const Triple &TheTriple,
                                           MCContext &Ctx) {
  return createMCRelocationInfo(TheTriple, Ctx);
}

class LanaiMCInstrAnalysis : public MCInstrAnalysis {
public:
  explicit LanaiMCInstrAnalysis(const MCInstrInfo *Info)
      : MCInstrAnalysis(Info) {}

  bool evaluateBranch(const MCInst &Inst, uint64_t Addr, uint64_t Size,
                      uint64_t &Target) const override {
    if (Inst.getNumOperands() == 0)
      return false;

    if (Info->get(Inst.getOpcode()).OpInfo[0].OperandType ==
        MCOI::OPERAND_PCREL) {
      int64_t Imm = Inst.getOperand(0).getImm();
      Target = Addr + Size + Imm;
      return true;
    } else {
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

static MCInstrAnalysis *createLanaiInstrAnalysis(const MCInstrInfo *Info) {
  return new LanaiMCInstrAnalysis(Info);
}

extern "C" void LLVMInitializeLanaiTargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfo<LanaiMCAsmInfo> X(TheLanaiTarget);

  // Register the MC codegen info.
  TargetRegistry::RegisterMCCodeGenInfo(TheLanaiTarget,
                                        createLanaiMCCodeGenInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheLanaiTarget, createLanaiMCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheLanaiTarget, createLanaiMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheLanaiTarget,
                                          createLanaiMCSubtargetInfo);

  // Register the MC code emitter
  TargetRegistry::RegisterMCCodeEmitter(TheLanaiTarget,
                                        llvm::createLanaiMCCodeEmitter);

  // Register the ASM Backend
  TargetRegistry::RegisterMCAsmBackend(TheLanaiTarget, createLanaiAsmBackend);

  // Register the MCInstPrinter.
  TargetRegistry::RegisterMCInstPrinter(TheLanaiTarget,
                                        createLanaiMCInstPrinter);

  // Register the ELF streamer.
  TargetRegistry::RegisterELFStreamer(TheLanaiTarget, createMCStreamer);

  // Register the MC relocation info.
  TargetRegistry::RegisterMCRelocationInfo(TheLanaiTarget,
                                           createLanaiElfRelocation);

  // Register the MC instruction analyzer.
  TargetRegistry::RegisterMCInstrAnalysis(TheLanaiTarget,
                                          createLanaiInstrAnalysis);
}
