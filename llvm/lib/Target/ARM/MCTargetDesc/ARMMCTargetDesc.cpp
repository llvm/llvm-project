//===-- ARMMCTargetDesc.cpp - ARM Target Descriptions ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides ARM specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "ARMMCTargetDesc.h"
#include "ARMAddressingModes.h"
#include "ARMBaseInfo.h"
#include "ARMInstPrinter.h"
#include "ARMMCAsmInfo.h"
#include "TargetInfo/ARMTargetInfo.h"
#include "llvm/ADT/Triple.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetParser.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define GET_REGINFO_MC_DESC
#include "ARMGenRegisterInfo.inc"

static bool getMCRDeprecationInfo(MCInst &MI, const MCSubtargetInfo &STI,
                                  std::string &Info) {
  if (STI.getFeatureBits()[llvm::ARM::HasV7Ops] &&
      (MI.getOperand(0).isImm() && MI.getOperand(0).getImm() == 15) &&
      (MI.getOperand(1).isImm() && MI.getOperand(1).getImm() == 0) &&
      // Checks for the deprecated CP15ISB encoding:
      // mcr p15, #0, rX, c7, c5, #4
      (MI.getOperand(3).isImm() && MI.getOperand(3).getImm() == 7)) {
    if ((MI.getOperand(5).isImm() && MI.getOperand(5).getImm() == 4)) {
      if (MI.getOperand(4).isImm() && MI.getOperand(4).getImm() == 5) {
        Info = "deprecated since v7, use 'isb'";
        return true;
      }

      // Checks for the deprecated CP15DSB encoding:
      // mcr p15, #0, rX, c7, c10, #4
      if (MI.getOperand(4).isImm() && MI.getOperand(4).getImm() == 10) {
        Info = "deprecated since v7, use 'dsb'";
        return true;
      }
    }
    // Checks for the deprecated CP15DMB encoding:
    // mcr p15, #0, rX, c7, c10, #5
    if (MI.getOperand(4).isImm() && MI.getOperand(4).getImm() == 10 &&
        (MI.getOperand(5).isImm() && MI.getOperand(5).getImm() == 5)) {
      Info = "deprecated since v7, use 'dmb'";
      return true;
    }
  }
  if (STI.getFeatureBits()[llvm::ARM::HasV7Ops] &&
      ((MI.getOperand(0).isImm() && MI.getOperand(0).getImm() == 10) ||
       (MI.getOperand(0).isImm() && MI.getOperand(0).getImm() == 11))) {
    Info = "since v7, cp10 and cp11 are reserved for advanced SIMD or floating "
           "point instructions";
    return true;
  }
  return false;
}

static bool getMRCDeprecationInfo(MCInst &MI, const MCSubtargetInfo &STI,
                                  std::string &Info) {
  if (STI.getFeatureBits()[llvm::ARM::HasV7Ops] &&
      ((MI.getOperand(0).isImm() && MI.getOperand(0).getImm() == 10) ||
       (MI.getOperand(0).isImm() && MI.getOperand(0).getImm() == 11))) {
    Info = "since v7, cp10 and cp11 are reserved for advanced SIMD or floating "
           "point instructions";
    return true;
  }
  return false;
}

static bool getITDeprecationInfo(MCInst &MI, const MCSubtargetInfo &STI,
                                 std::string &Info) {
  if (STI.getFeatureBits()[llvm::ARM::HasV8Ops] && MI.getOperand(1).isImm() &&
      MI.getOperand(1).getImm() != 8) {
    Info = "applying IT instruction to more than one subsequent instruction is "
           "deprecated";
    return true;
  }

  return false;
}

static bool getARMStoreDeprecationInfo(MCInst &MI, const MCSubtargetInfo &STI,
                                       std::string &Info) {
  assert(!STI.getFeatureBits()[llvm::ARM::ModeThumb] &&
         "cannot predicate thumb instructions");

  assert(MI.getNumOperands() >= 4 && "expected >= 4 arguments");
  for (unsigned OI = 4, OE = MI.getNumOperands(); OI < OE; ++OI) {
    assert(MI.getOperand(OI).isReg() && "expected register");
    if (MI.getOperand(OI).getReg() == ARM::PC) {
      Info = "use of PC in the list is deprecated";
      return true;
    }
  }
  return false;
}

static bool getARMLoadDeprecationInfo(MCInst &MI, const MCSubtargetInfo &STI,
                                      std::string &Info) {
  assert(!STI.getFeatureBits()[llvm::ARM::ModeThumb] &&
         "cannot predicate thumb instructions");

  assert(MI.getNumOperands() >= 4 && "expected >= 4 arguments");
  bool ListContainsPC = false, ListContainsLR = false;
  for (unsigned OI = 4, OE = MI.getNumOperands(); OI < OE; ++OI) {
    assert(MI.getOperand(OI).isReg() && "expected register");
    switch (MI.getOperand(OI).getReg()) {
    default:
      break;
    case ARM::LR:
      ListContainsLR = true;
      break;
    case ARM::PC:
      ListContainsPC = true;
      break;
    }
  }

  if (ListContainsPC && ListContainsLR) {
    Info = "use of LR and PC simultaneously in the list is deprecated";
    return true;
  }

  return false;
}

#define GET_INSTRINFO_MC_DESC
#include "ARMGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "ARMGenSubtargetInfo.inc"

std::string ARM_MC::ParseARMTriple(const Triple &TT, StringRef CPU) {
  std::string ARMArchFeature;

  ARM::ArchKind ArchID = ARM::parseArch(TT.getArchName());
  if (ArchID != ARM::ArchKind::INVALID &&  (CPU.empty() || CPU == "generic"))
    ARMArchFeature = (ARMArchFeature + "+" + ARM::getArchName(ArchID)).str();

  if (TT.isThumb()) {
    if (!ARMArchFeature.empty())
      ARMArchFeature += ",";
    ARMArchFeature += "+thumb-mode,+v4t";
  }

  if (TT.isOSNaCl()) {
    if (!ARMArchFeature.empty())
      ARMArchFeature += ",";
    ARMArchFeature += "+nacl-trap";
  }

  if (TT.isOSWindows()) {
    if (!ARMArchFeature.empty())
      ARMArchFeature += ",";
    ARMArchFeature += "+noarm";
  }

  return ARMArchFeature;
}

bool ARM_MC::isPredicated(const MCInst &MI, const MCInstrInfo *MCII) {
  const MCInstrDesc &Desc = MCII->get(MI.getOpcode());
  int PredOpIdx = Desc.findFirstPredOperandIdx();
  return PredOpIdx != -1 && MI.getOperand(PredOpIdx).getImm() != ARMCC::AL;
}

bool ARM_MC::isCPSRDefined(const MCInst &MI, const MCInstrInfo *MCII) {
  const MCInstrDesc &Desc = MCII->get(MI.getOpcode());
  for (unsigned I = 0; I < MI.getNumOperands(); ++I) {
    const MCOperand &MO = MI.getOperand(I);
    if (MO.isReg() && MO.getReg() == ARM::CPSR &&
        Desc.OpInfo[I].isOptionalDef())
      return true;
  }
  return false;
}

uint64_t ARM_MC::evaluateBranchTarget(const MCInstrDesc &InstDesc,
                                      uint64_t Addr, int64_t Imm) {
  // For ARM instructions the PC offset is 8 bytes, for Thumb instructions it
  // is 4 bytes.
  uint64_t Offset =
      ((InstDesc.TSFlags & ARMII::FormMask) == ARMII::ThumbFrm) ? 4 : 8;

  // A Thumb instruction BLX(i) can be 16-bit aligned while targets Arm code
  // which is 32-bit aligned. The target address for the case is calculated as
  //   targetAddress = Align(PC,4) + imm32;
  // where
  //   Align(x, y) = y * (x DIV y);
  if (InstDesc.getOpcode() == ARM::tBLXi)
    Addr &= ~0x3;

  return Addr + Imm + Offset;
}

MCSubtargetInfo *ARM_MC::createARMMCSubtargetInfo(const Triple &TT,
                                                  StringRef CPU, StringRef FS) {
  std::string ArchFS = ARM_MC::ParseARMTriple(TT, CPU);
  if (!FS.empty()) {
    if (!ArchFS.empty())
      ArchFS = (Twine(ArchFS) + "," + FS).str();
    else
      ArchFS = std::string(FS);
  }

  return createARMMCSubtargetInfoImpl(TT, CPU, /*TuneCPU*/ CPU, ArchFS);
}

static MCInstrInfo *createARMMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitARMMCInstrInfo(X);
  return X;
}

void ARM_MC::initLLVMToCVRegMapping(MCRegisterInfo *MRI) {
  // Mapping from CodeView to MC register id.
  static const struct {
    codeview::RegisterId CVReg;
    MCPhysReg Reg;
  } RegMap[] = {
      {codeview::RegisterId::ARM_R0, ARM::R0},
      {codeview::RegisterId::ARM_R1, ARM::R1},
      {codeview::RegisterId::ARM_R2, ARM::R2},
      {codeview::RegisterId::ARM_R3, ARM::R3},
      {codeview::RegisterId::ARM_R4, ARM::R4},
      {codeview::RegisterId::ARM_R5, ARM::R5},
      {codeview::RegisterId::ARM_R6, ARM::R6},
      {codeview::RegisterId::ARM_R7, ARM::R7},
      {codeview::RegisterId::ARM_R8, ARM::R8},
      {codeview::RegisterId::ARM_R9, ARM::R9},
      {codeview::RegisterId::ARM_R10, ARM::R10},
      {codeview::RegisterId::ARM_R11, ARM::R11},
      {codeview::RegisterId::ARM_R12, ARM::R12},
      {codeview::RegisterId::ARM_SP, ARM::SP},
      {codeview::RegisterId::ARM_LR, ARM::LR},
      {codeview::RegisterId::ARM_PC, ARM::PC},
      {codeview::RegisterId::ARM_CPSR, ARM::CPSR},
      {codeview::RegisterId::ARM_FPSCR, ARM::FPSCR},
      {codeview::RegisterId::ARM_FPEXC, ARM::FPEXC},
      {codeview::RegisterId::ARM_FS0, ARM::S0},
      {codeview::RegisterId::ARM_FS1, ARM::S1},
      {codeview::RegisterId::ARM_FS2, ARM::S2},
      {codeview::RegisterId::ARM_FS3, ARM::S3},
      {codeview::RegisterId::ARM_FS4, ARM::S4},
      {codeview::RegisterId::ARM_FS5, ARM::S5},
      {codeview::RegisterId::ARM_FS6, ARM::S6},
      {codeview::RegisterId::ARM_FS7, ARM::S7},
      {codeview::RegisterId::ARM_FS8, ARM::S8},
      {codeview::RegisterId::ARM_FS9, ARM::S9},
      {codeview::RegisterId::ARM_FS10, ARM::S10},
      {codeview::RegisterId::ARM_FS11, ARM::S11},
      {codeview::RegisterId::ARM_FS12, ARM::S12},
      {codeview::RegisterId::ARM_FS13, ARM::S13},
      {codeview::RegisterId::ARM_FS14, ARM::S14},
      {codeview::RegisterId::ARM_FS15, ARM::S15},
      {codeview::RegisterId::ARM_FS16, ARM::S16},
      {codeview::RegisterId::ARM_FS17, ARM::S17},
      {codeview::RegisterId::ARM_FS18, ARM::S18},
      {codeview::RegisterId::ARM_FS19, ARM::S19},
      {codeview::RegisterId::ARM_FS20, ARM::S20},
      {codeview::RegisterId::ARM_FS21, ARM::S21},
      {codeview::RegisterId::ARM_FS22, ARM::S22},
      {codeview::RegisterId::ARM_FS23, ARM::S23},
      {codeview::RegisterId::ARM_FS24, ARM::S24},
      {codeview::RegisterId::ARM_FS25, ARM::S25},
      {codeview::RegisterId::ARM_FS26, ARM::S26},
      {codeview::RegisterId::ARM_FS27, ARM::S27},
      {codeview::RegisterId::ARM_FS28, ARM::S28},
      {codeview::RegisterId::ARM_FS29, ARM::S29},
      {codeview::RegisterId::ARM_FS30, ARM::S30},
      {codeview::RegisterId::ARM_FS31, ARM::S31},
      {codeview::RegisterId::ARM_ND0, ARM::D0},
      {codeview::RegisterId::ARM_ND1, ARM::D1},
      {codeview::RegisterId::ARM_ND2, ARM::D2},
      {codeview::RegisterId::ARM_ND3, ARM::D3},
      {codeview::RegisterId::ARM_ND4, ARM::D4},
      {codeview::RegisterId::ARM_ND5, ARM::D5},
      {codeview::RegisterId::ARM_ND6, ARM::D6},
      {codeview::RegisterId::ARM_ND7, ARM::D7},
      {codeview::RegisterId::ARM_ND8, ARM::D8},
      {codeview::RegisterId::ARM_ND9, ARM::D9},
      {codeview::RegisterId::ARM_ND10, ARM::D10},
      {codeview::RegisterId::ARM_ND11, ARM::D11},
      {codeview::RegisterId::ARM_ND12, ARM::D12},
      {codeview::RegisterId::ARM_ND13, ARM::D13},
      {codeview::RegisterId::ARM_ND14, ARM::D14},
      {codeview::RegisterId::ARM_ND15, ARM::D15},
      {codeview::RegisterId::ARM_ND16, ARM::D16},
      {codeview::RegisterId::ARM_ND17, ARM::D17},
      {codeview::RegisterId::ARM_ND18, ARM::D18},
      {codeview::RegisterId::ARM_ND19, ARM::D19},
      {codeview::RegisterId::ARM_ND20, ARM::D20},
      {codeview::RegisterId::ARM_ND21, ARM::D21},
      {codeview::RegisterId::ARM_ND22, ARM::D22},
      {codeview::RegisterId::ARM_ND23, ARM::D23},
      {codeview::RegisterId::ARM_ND24, ARM::D24},
      {codeview::RegisterId::ARM_ND25, ARM::D25},
      {codeview::RegisterId::ARM_ND26, ARM::D26},
      {codeview::RegisterId::ARM_ND27, ARM::D27},
      {codeview::RegisterId::ARM_ND28, ARM::D28},
      {codeview::RegisterId::ARM_ND29, ARM::D29},
      {codeview::RegisterId::ARM_ND30, ARM::D30},
      {codeview::RegisterId::ARM_ND31, ARM::D31},
      {codeview::RegisterId::ARM_NQ0, ARM::Q0},
      {codeview::RegisterId::ARM_NQ1, ARM::Q1},
      {codeview::RegisterId::ARM_NQ2, ARM::Q2},
      {codeview::RegisterId::ARM_NQ3, ARM::Q3},
      {codeview::RegisterId::ARM_NQ4, ARM::Q4},
      {codeview::RegisterId::ARM_NQ5, ARM::Q5},
      {codeview::RegisterId::ARM_NQ6, ARM::Q6},
      {codeview::RegisterId::ARM_NQ7, ARM::Q7},
      {codeview::RegisterId::ARM_NQ8, ARM::Q8},
      {codeview::RegisterId::ARM_NQ9, ARM::Q9},
      {codeview::RegisterId::ARM_NQ10, ARM::Q10},
      {codeview::RegisterId::ARM_NQ11, ARM::Q11},
      {codeview::RegisterId::ARM_NQ12, ARM::Q12},
      {codeview::RegisterId::ARM_NQ13, ARM::Q13},
      {codeview::RegisterId::ARM_NQ14, ARM::Q14},
      {codeview::RegisterId::ARM_NQ15, ARM::Q15},
  };
  for (unsigned I = 0; I < array_lengthof(RegMap); ++I)
    MRI->mapLLVMRegToCVReg(RegMap[I].Reg, static_cast<int>(RegMap[I].CVReg));
}

static MCRegisterInfo *createARMMCRegisterInfo(const Triple &Triple) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitARMMCRegisterInfo(X, ARM::LR, 0, 0, ARM::PC);
  ARM_MC::initLLVMToCVRegMapping(X);
  return X;
}

static MCAsmInfo *createARMMCAsmInfo(const MCRegisterInfo &MRI,
                                     const Triple &TheTriple,
                                     const MCTargetOptions &Options) {
  MCAsmInfo *MAI;
  if (TheTriple.isOSDarwin() || TheTriple.isOSBinFormatMachO())
    MAI = new ARMMCAsmInfoDarwin(TheTriple);
  else if (TheTriple.isWindowsMSVCEnvironment())
    MAI = new ARMCOFFMCAsmInfoMicrosoft();
  else if (TheTriple.isOSWindows())
    MAI = new ARMCOFFMCAsmInfoGNU();
  else
    MAI = new ARMELFMCAsmInfo(TheTriple);

  unsigned Reg = MRI.getDwarfRegNum(ARM::SP, true);
  MAI->addInitialFrameState(MCCFIInstruction::cfiDefCfa(nullptr, Reg, 0));

  return MAI;
}

static MCStreamer *createELFStreamer(const Triple &T, MCContext &Ctx,
                                     std::unique_ptr<MCAsmBackend> &&MAB,
                                     std::unique_ptr<MCObjectWriter> &&OW,
                                     std::unique_ptr<MCCodeEmitter> &&Emitter,
                                     bool RelaxAll) {
  return createARMELFStreamer(
      Ctx, std::move(MAB), std::move(OW), std::move(Emitter), false,
      (T.getArch() == Triple::thumb || T.getArch() == Triple::thumbeb),
      T.isAndroid());
}

static MCStreamer *
createARMMachOStreamer(MCContext &Ctx, std::unique_ptr<MCAsmBackend> &&MAB,
                       std::unique_ptr<MCObjectWriter> &&OW,
                       std::unique_ptr<MCCodeEmitter> &&Emitter, bool RelaxAll,
                       bool DWARFMustBeAtTheEnd) {
  return createMachOStreamer(Ctx, std::move(MAB), std::move(OW),
                             std::move(Emitter), false, DWARFMustBeAtTheEnd);
}

static MCInstPrinter *createARMMCInstPrinter(const Triple &T,
                                             unsigned SyntaxVariant,
                                             const MCAsmInfo &MAI,
                                             const MCInstrInfo &MII,
                                             const MCRegisterInfo &MRI) {
  if (SyntaxVariant == 0)
    return new ARMInstPrinter(MAI, MII, MRI);
  return nullptr;
}

static MCRelocationInfo *createARMMCRelocationInfo(const Triple &TT,
                                                   MCContext &Ctx) {
  if (TT.isOSBinFormatMachO())
    return createARMMachORelocationInfo(Ctx);
  // Default to the stock relocation info.
  return llvm::createMCRelocationInfo(TT, Ctx);
}

namespace {

class ARMMCInstrAnalysis : public MCInstrAnalysis {
public:
  ARMMCInstrAnalysis(const MCInstrInfo *Info) : MCInstrAnalysis(Info) {}

  bool isUnconditionalBranch(const MCInst &Inst) const override {
    // BCCs with the "always" predicate are unconditional branches.
    if (Inst.getOpcode() == ARM::Bcc && Inst.getOperand(1).getImm()==ARMCC::AL)
      return true;
    return MCInstrAnalysis::isUnconditionalBranch(Inst);
  }

  bool isConditionalBranch(const MCInst &Inst) const override {
    // BCCs with the "always" predicate are unconditional branches.
    if (Inst.getOpcode() == ARM::Bcc && Inst.getOperand(1).getImm()==ARMCC::AL)
      return false;
    return MCInstrAnalysis::isConditionalBranch(Inst);
  }

  bool evaluateBranch(const MCInst &Inst, uint64_t Addr, uint64_t Size,
                      uint64_t &Target) const override {
    const MCInstrDesc &Desc = Info->get(Inst.getOpcode());

    // Find the PC-relative immediate operand in the instruction.
    for (unsigned OpNum = 0; OpNum < Desc.getNumOperands(); ++OpNum) {
      if (Inst.getOperand(OpNum).isImm() &&
          Desc.OpInfo[OpNum].OperandType == MCOI::OPERAND_PCREL) {
        int64_t Imm = Inst.getOperand(OpNum).getImm();
        Target = ARM_MC::evaluateBranchTarget(Desc, Addr, Imm);
        return true;
      }
    }
    return false;
  }
};

}

static MCInstrAnalysis *createARMMCInstrAnalysis(const MCInstrInfo *Info) {
  return new ARMMCInstrAnalysis(Info);
}

bool ARM::isCDECoproc(size_t Coproc, const MCSubtargetInfo &STI) {
  // Unfortunately we don't have ARMTargetInfo in the disassembler, so we have
  // to rely on feature bits.
  if (Coproc >= 8)
    return false;
  return STI.getFeatureBits()[ARM::FeatureCoprocCDE0 + Coproc];
}

// Force static initialization.
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeARMTargetMC() {
  for (Target *T : {&getTheARMLETarget(), &getTheARMBETarget(),
                    &getTheThumbLETarget(), &getTheThumbBETarget()}) {
    // Register the MC asm info.
    RegisterMCAsmInfoFn X(*T, createARMMCAsmInfo);

    // Register the MC instruction info.
    TargetRegistry::RegisterMCInstrInfo(*T, createARMMCInstrInfo);

    // Register the MC register info.
    TargetRegistry::RegisterMCRegInfo(*T, createARMMCRegisterInfo);

    // Register the MC subtarget info.
    TargetRegistry::RegisterMCSubtargetInfo(*T,
                                            ARM_MC::createARMMCSubtargetInfo);

    TargetRegistry::RegisterELFStreamer(*T, createELFStreamer);
    TargetRegistry::RegisterCOFFStreamer(*T, createARMWinCOFFStreamer);
    TargetRegistry::RegisterMachOStreamer(*T, createARMMachOStreamer);

    // Register the obj target streamer.
    TargetRegistry::RegisterObjectTargetStreamer(*T,
                                                 createARMObjectTargetStreamer);

    // Register the asm streamer.
    TargetRegistry::RegisterAsmTargetStreamer(*T, createARMTargetAsmStreamer);

    // Register the null TargetStreamer.
    TargetRegistry::RegisterNullTargetStreamer(*T, createARMNullTargetStreamer);

    // Register the MCInstPrinter.
    TargetRegistry::RegisterMCInstPrinter(*T, createARMMCInstPrinter);

    // Register the MC relocation info.
    TargetRegistry::RegisterMCRelocationInfo(*T, createARMMCRelocationInfo);
  }

  // Register the MC instruction analyzer.
  for (Target *T : {&getTheARMLETarget(), &getTheARMBETarget(),
                    &getTheThumbLETarget(), &getTheThumbBETarget()})
    TargetRegistry::RegisterMCInstrAnalysis(*T, createARMMCInstrAnalysis);

  for (Target *T : {&getTheARMLETarget(), &getTheThumbLETarget()}) {
    TargetRegistry::RegisterMCCodeEmitter(*T, createARMLEMCCodeEmitter);
    TargetRegistry::RegisterMCAsmBackend(*T, createARMLEAsmBackend);
  }
  for (Target *T : {&getTheARMBETarget(), &getTheThumbBETarget()}) {
    TargetRegistry::RegisterMCCodeEmitter(*T, createARMBEMCCodeEmitter);
    TargetRegistry::RegisterMCAsmBackend(*T, createARMBEAsmBackend);
  }
}
