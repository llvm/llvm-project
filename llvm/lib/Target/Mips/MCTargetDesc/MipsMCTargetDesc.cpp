//===-- MipsMCTargetDesc.cpp - Mips Target Descriptions -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides Mips specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "MipsMCTargetDesc.h"
#include "MipsAsmBackend.h"
#include "MipsBaseInfo.h"
#include "MipsELFStreamer.h"
#include "MipsInstPrinter.h"
#include "MipsMCAsmInfo.h"
#include "MipsMCNaCl.h"
#include "MipsTargetStreamer.h"
#include "TargetInfo/MipsTargetInfo.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

#define GET_INSTRINFO_MC_DESC
#define ENABLE_INSTR_PREDICATE_VERIFIER
#include "MipsGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "MipsGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "MipsGenRegisterInfo.inc"

void MIPS_MC::initLLVMToCVRegMapping(MCRegisterInfo *MRI) {
  // Mapping from CodeView to MC register id.
  static const struct {
    codeview::RegisterId CVReg;
    MCPhysReg Reg;
  } RegMap[] = {
      {codeview::RegisterId::MIPS_ZERO, Mips::ZERO},
      {codeview::RegisterId::MIPS_AT, Mips::AT},
      {codeview::RegisterId::MIPS_V0, Mips::V0},
      {codeview::RegisterId::MIPS_V1, Mips::V1},
      {codeview::RegisterId::MIPS_A0, Mips::A0},
      {codeview::RegisterId::MIPS_A1, Mips::A1},
      {codeview::RegisterId::MIPS_A2, Mips::A2},
      {codeview::RegisterId::MIPS_A3, Mips::A3},
      {codeview::RegisterId::MIPS_T0, Mips::T0},
      {codeview::RegisterId::MIPS_T1, Mips::T1},
      {codeview::RegisterId::MIPS_T2, Mips::T2},
      {codeview::RegisterId::MIPS_T3, Mips::T3},
      {codeview::RegisterId::MIPS_T4, Mips::T4},
      {codeview::RegisterId::MIPS_T5, Mips::T5},
      {codeview::RegisterId::MIPS_T6, Mips::T6},
      {codeview::RegisterId::MIPS_T7, Mips::T7},
      {codeview::RegisterId::MIPS_S0, Mips::S0},
      {codeview::RegisterId::MIPS_S1, Mips::S1},
      {codeview::RegisterId::MIPS_S2, Mips::S2},
      {codeview::RegisterId::MIPS_S3, Mips::S3},
      {codeview::RegisterId::MIPS_S4, Mips::S4},
      {codeview::RegisterId::MIPS_S5, Mips::S5},
      {codeview::RegisterId::MIPS_S6, Mips::S6},
      {codeview::RegisterId::MIPS_S7, Mips::S7},
      {codeview::RegisterId::MIPS_T8, Mips::T8},
      {codeview::RegisterId::MIPS_T9, Mips::T9},
      {codeview::RegisterId::MIPS_K0, Mips::K0},
      {codeview::RegisterId::MIPS_K1, Mips::K1},
      {codeview::RegisterId::MIPS_GP, Mips::GP},
      {codeview::RegisterId::MIPS_SP, Mips::SP},
      {codeview::RegisterId::MIPS_S8, Mips::FP},
      {codeview::RegisterId::MIPS_RA, Mips::RA},
      {codeview::RegisterId::MIPS_LO, Mips::HI0},
      {codeview::RegisterId::MIPS_HI, Mips::LO0},
      {codeview::RegisterId::MIPS_Fir, Mips::FCR0},
      {codeview::RegisterId::MIPS_Psr, Mips::COP012}, // CP0.Status
      {codeview::RegisterId::MIPS_F0, Mips::F0},
      {codeview::RegisterId::MIPS_F1, Mips::F1},
      {codeview::RegisterId::MIPS_F2, Mips::F2},
      {codeview::RegisterId::MIPS_F3, Mips::F3},
      {codeview::RegisterId::MIPS_F4, Mips::F4},
      {codeview::RegisterId::MIPS_F5, Mips::F5},
      {codeview::RegisterId::MIPS_F6, Mips::F6},
      {codeview::RegisterId::MIPS_F7, Mips::F7},
      {codeview::RegisterId::MIPS_F8, Mips::F8},
      {codeview::RegisterId::MIPS_F9, Mips::F9},
      {codeview::RegisterId::MIPS_F10, Mips::F10},
      {codeview::RegisterId::MIPS_F11, Mips::F11},
      {codeview::RegisterId::MIPS_F12, Mips::F12},
      {codeview::RegisterId::MIPS_F13, Mips::F13},
      {codeview::RegisterId::MIPS_F14, Mips::F14},
      {codeview::RegisterId::MIPS_F15, Mips::F15},
      {codeview::RegisterId::MIPS_F16, Mips::F16},
      {codeview::RegisterId::MIPS_F17, Mips::F17},
      {codeview::RegisterId::MIPS_F18, Mips::F18},
      {codeview::RegisterId::MIPS_F19, Mips::F19},
      {codeview::RegisterId::MIPS_F20, Mips::F20},
      {codeview::RegisterId::MIPS_F21, Mips::F21},
      {codeview::RegisterId::MIPS_F22, Mips::F22},
      {codeview::RegisterId::MIPS_F23, Mips::F23},
      {codeview::RegisterId::MIPS_F24, Mips::F24},
      {codeview::RegisterId::MIPS_F25, Mips::F25},
      {codeview::RegisterId::MIPS_F26, Mips::F26},
      {codeview::RegisterId::MIPS_F27, Mips::F27},
      {codeview::RegisterId::MIPS_F28, Mips::F28},
      {codeview::RegisterId::MIPS_F29, Mips::F29},
      {codeview::RegisterId::MIPS_F30, Mips::F30},
      {codeview::RegisterId::MIPS_F31, Mips::F31},
      {codeview::RegisterId::MIPS_Fsr, Mips::FCR31},
  };
  for (const auto &I : RegMap)
    MRI->mapLLVMRegToCVReg(I.Reg, static_cast<int>(I.CVReg));
}

namespace {
class MipsWinCOFFTargetStreamer : public MipsTargetStreamer {
public:
  MipsWinCOFFTargetStreamer(MCStreamer &S) : MipsTargetStreamer(S) {}
};
} // end namespace

/// Select the Mips CPU for the given triple and cpu name.
StringRef MIPS_MC::selectMipsCPU(const Triple &TT, StringRef CPU) {
  if (CPU.empty() || CPU == "generic") {
    if (TT.getSubArch() == llvm::Triple::MipsSubArch_r6) {
      if (TT.isMIPS32())
        CPU = "mips32r6";
      else
        CPU = "mips64r6";
    } else {
      if (TT.isMIPS32())
        CPU = "mips32";
      else
        CPU = "mips64";
    }
  }
  return CPU;
}

static MCInstrInfo *createMipsMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitMipsMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createMipsMCRegisterInfo(const Triple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitMipsMCRegisterInfo(X, Mips::RA);
  return X;
}

static MCSubtargetInfo *createMipsMCSubtargetInfo(const Triple &TT,
                                                  StringRef CPU, StringRef FS) {
  CPU = MIPS_MC::selectMipsCPU(TT, CPU);
  return createMipsMCSubtargetInfoImpl(TT, CPU, /*TuneCPU*/ CPU, FS);
}

static MCAsmInfo *createMipsMCAsmInfo(const MCRegisterInfo &MRI,
                                      const Triple &TT,
                                      const MCTargetOptions &Options) {
  MCAsmInfo *MAI;

  if (TT.isOSWindows())
    MAI = new MipsCOFFMCAsmInfo();
  else
    MAI = new MipsELFMCAsmInfo(TT, Options);

  unsigned SP = MRI.getDwarfRegNum(Mips::SP, true);
  MCCFIInstruction Inst = MCCFIInstruction::createDefCfaRegister(nullptr, SP);
  MAI->addInitialFrameState(Inst);

  return MAI;
}

static MCInstPrinter *createMipsMCInstPrinter(const Triple &T,
                                              unsigned SyntaxVariant,
                                              const MCAsmInfo &MAI,
                                              const MCInstrInfo &MII,
                                              const MCRegisterInfo &MRI) {
  return new MipsInstPrinter(MAI, MII, MRI);
}

static MCStreamer *createMCStreamer(const Triple &T, MCContext &Context,
                                    std::unique_ptr<MCAsmBackend> &&MAB,
                                    std::unique_ptr<MCObjectWriter> &&OW,
                                    std::unique_ptr<MCCodeEmitter> &&Emitter) {
  MCStreamer *S;
  if (!T.isOSNaCl())
    S = createMipsELFStreamer(Context, std::move(MAB), std::move(OW),
                              std::move(Emitter));
  else
    S = createMipsNaClELFStreamer(Context, std::move(MAB), std::move(OW),
                                  std::move(Emitter));
  return S;
}

static MCTargetStreamer *createMipsAsmTargetStreamer(MCStreamer &S,
                                                     formatted_raw_ostream &OS,
                                                     MCInstPrinter *InstPrint) {
  return new MipsTargetAsmStreamer(S, OS);
}

static MCTargetStreamer *createMipsNullTargetStreamer(MCStreamer &S) {
  return new MipsTargetStreamer(S);
}

static MCTargetStreamer *
createMipsObjectTargetStreamer(MCStreamer &S, const MCSubtargetInfo &STI) {
  if (STI.getTargetTriple().isOSBinFormatCOFF())
    return new MipsWinCOFFTargetStreamer(S);
  return new MipsTargetELFStreamer(S, STI);
}

namespace {

class MipsMCInstrAnalysis : public MCInstrAnalysis {
public:
  MipsMCInstrAnalysis(const MCInstrInfo *Info) : MCInstrAnalysis(Info) {}

  bool evaluateBranch(const MCInst &Inst, uint64_t Addr, uint64_t Size,
                      uint64_t &Target) const override {
    unsigned NumOps = Inst.getNumOperands();
    if (NumOps == 0)
      return false;
    switch (Info->get(Inst.getOpcode()).operands()[NumOps - 1].OperandType) {
    case MCOI::OPERAND_UNKNOWN:
    case MCOI::OPERAND_IMMEDIATE: {
      // j, jal, jalx, jals
      // Absolute branch within the current 256 MB-aligned region
      uint64_t Region = Addr & ~uint64_t(0xfffffff);
      Target = Region + Inst.getOperand(NumOps - 1).getImm();
      return true;
    }
    case MCOI::OPERAND_PCREL:
      // b, beq ...
      Target = Addr + Inst.getOperand(NumOps - 1).getImm();
      return true;
    default:
      return false;
    }
  }
};
}

static MCInstrAnalysis *createMipsMCInstrAnalysis(const MCInstrInfo *Info) {
  return new MipsMCInstrAnalysis(Info);
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeMipsTargetMC() {
  for (Target *T : {&getTheMipsTarget(), &getTheMipselTarget(),
                    &getTheMips64Target(), &getTheMips64elTarget()}) {
    // Register the MC asm info.
    RegisterMCAsmInfoFn X(*T, createMipsMCAsmInfo);

    // Register the MC instruction info.
    TargetRegistry::RegisterMCInstrInfo(*T, createMipsMCInstrInfo);

    // Register the MC register info.
    TargetRegistry::RegisterMCRegInfo(*T, createMipsMCRegisterInfo);

    // Register the elf streamer.
    TargetRegistry::RegisterELFStreamer(*T, createMCStreamer);

    // Register the asm target streamer.
    TargetRegistry::RegisterAsmTargetStreamer(*T, createMipsAsmTargetStreamer);

    TargetRegistry::RegisterNullTargetStreamer(*T,
                                               createMipsNullTargetStreamer);

    TargetRegistry::RegisterCOFFStreamer(*T, createMipsWinCOFFStreamer);

    // Register the MC subtarget info.
    TargetRegistry::RegisterMCSubtargetInfo(*T, createMipsMCSubtargetInfo);

    // Register the MC instruction analyzer.
    TargetRegistry::RegisterMCInstrAnalysis(*T, createMipsMCInstrAnalysis);

    // Register the MCInstPrinter.
    TargetRegistry::RegisterMCInstPrinter(*T, createMipsMCInstPrinter);

    TargetRegistry::RegisterObjectTargetStreamer(
        *T, createMipsObjectTargetStreamer);

    // Register the asm backend.
    TargetRegistry::RegisterMCAsmBackend(*T, createMipsAsmBackend);
  }

  // Register the MC Code Emitter
  for (Target *T : {&getTheMipsTarget(), &getTheMips64Target()})
    TargetRegistry::RegisterMCCodeEmitter(*T, createMipsMCCodeEmitterEB);

  for (Target *T : {&getTheMipselTarget(), &getTheMips64elTarget()})
    TargetRegistry::RegisterMCCodeEmitter(*T, createMipsMCCodeEmitterEL);
}
