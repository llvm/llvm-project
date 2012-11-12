//===-- ARMMCTargetDesc.cpp - ARM Target Descriptions ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides ARM specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "ARMMCTargetDesc.h"
#include "ARMMCAsmInfo.h"
#include "ARMBaseInfo.h"
#include "InstPrinter/ARMInstPrinter.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_REGINFO_MC_DESC
#include "ARMGenRegisterInfo.inc"

#define GET_INSTRINFO_MC_DESC
#include "ARMGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "ARMGenSubtargetInfo.inc"

using namespace llvm;

std::string ARM_MC::ParseARMTriple(StringRef TT, StringRef CPU) {
  // Set the boolean corresponding to the current target triple, or the default
  // if one cannot be determined, to true.
  unsigned Len = TT.size();
  unsigned Idx = 0;

  // FIXME: Enhance Triple helper class to extract ARM version.
  bool isThumb = false;
  if (Len >= 5 && TT.substr(0, 4) == "armv")
    Idx = 4;
  else if (Len >= 6 && TT.substr(0, 5) == "thumb") {
    isThumb = true;
    if (Len >= 7 && TT[5] == 'v')
      Idx = 6;
  }

  bool NoCPU = CPU == "generic" || CPU.empty();
  std::string ARMArchFeature;
  if (Idx) {
    unsigned SubVer = TT[Idx];
    if (SubVer >= '7' && SubVer <= '9') {
      if (Len >= Idx+2 && TT[Idx+1] == 'm') {
        if (NoCPU)
          // v7m: FeatureNoARM, FeatureDB, FeatureHWDiv, FeatureMClass
          ARMArchFeature = "+v7,+noarm,+db,+hwdiv,+mclass";
        else
          // Use CPU to figure out the exact features.
          ARMArchFeature = "+v7";
      } else if (Len >= Idx+3 && TT[Idx+1] == 'e'&& TT[Idx+2] == 'm') {
        if (NoCPU)
          // v7em: FeatureNoARM, FeatureDB, FeatureHWDiv, FeatureDSPThumb2,
          //       FeatureT2XtPk, FeatureMClass
          ARMArchFeature = "+v7,+noarm,+db,+hwdiv,+t2dsp,t2xtpk,+mclass";
        else
          // Use CPU to figure out the exact features.
          ARMArchFeature = "+v7";
      } else if (Len >= Idx+2 && TT[Idx+1] == 's') {
        if (NoCPU)
          // v7s: FeatureNEON, FeatureDB, FeatureDSPThumb2, FeatureT2XtPk
          //      Swift
          ARMArchFeature = "+v7,+swift,+neon,+db,+t2dsp,+t2xtpk";
        else
          // Use CPU to figure out the exact features.
          ARMArchFeature = "+v7";
      } else {
        // v7 CPUs have lots of different feature sets. If no CPU is specified,
        // then assume v7a (e.g. cortex-a8) feature set. Otherwise, return
        // the "minimum" feature set and use CPU string to figure out the exact
        // features.
        if (NoCPU)
          // v7a: FeatureNEON, FeatureDB, FeatureDSPThumb2, FeatureT2XtPk
          ARMArchFeature = "+v7,+neon,+db,+t2dsp,+t2xtpk";
        else
          // Use CPU to figure out the exact features.
          ARMArchFeature = "+v7";
      }
    } else if (SubVer == '6') {
      if (Len >= Idx+3 && TT[Idx+1] == 't' && TT[Idx+2] == '2')
        ARMArchFeature = "+v6t2";
      else if (Len >= Idx+2 && TT[Idx+1] == 'm') {
        if (NoCPU)
          // v6m: FeatureNoARM, FeatureMClass
          ARMArchFeature = "+v6,+noarm,+mclass";
        else
          ARMArchFeature = "+v6";
      } else
        ARMArchFeature = "+v6";
    } else if (SubVer == '5') {
      if (Len >= Idx+3 && TT[Idx+1] == 't' && TT[Idx+2] == 'e')
        ARMArchFeature = "+v5te";
      else
        ARMArchFeature = "+v5t";
    } else if (SubVer == '4' && Len >= Idx+2 && TT[Idx+1] == 't')
      ARMArchFeature = "+v4t";
  }

  if (isThumb) {
    if (ARMArchFeature.empty())
      ARMArchFeature = "+thumb-mode";
    else
      ARMArchFeature += ",+thumb-mode";
  }

  return ARMArchFeature;
}

MCSubtargetInfo *ARM_MC::createARMMCSubtargetInfo(StringRef TT, StringRef CPU,
                                                  StringRef FS) {
  std::string ArchFS = ARM_MC::ParseARMTriple(TT, CPU);
  if (!FS.empty()) {
    if (!ArchFS.empty())
      ArchFS = ArchFS + "," + FS.str();
    else
      ArchFS = FS;
  }

  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitARMMCSubtargetInfo(X, TT, CPU, ArchFS);
  return X;
}

static MCInstrInfo *createARMMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitARMMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createARMMCRegisterInfo(StringRef Triple) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitARMMCRegisterInfo(X, ARM::LR);
  return X;
}

static MCAsmInfo *createARMMCAsmInfo(const Target &T, StringRef TT) {
  Triple TheTriple(TT);

  if (TheTriple.isOSDarwin())
    return new ARMMCAsmInfoDarwin();

  return new ARMELFMCAsmInfo();
}

static MCCodeGenInfo *createARMMCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                             CodeModel::Model CM,
                                             CodeGenOpt::Level OL) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  if (RM == Reloc::Default) {
    Triple TheTriple(TT);
    // Default relocation model on Darwin is PIC, not DynamicNoPIC.
    RM = TheTriple.isOSDarwin() ? Reloc::PIC_ : Reloc::DynamicNoPIC;
  }
  X->InitMCCodeGenInfo(RM, CM, OL);
  return X;
}

// This is duplicated code. Refactor this.
static MCStreamer *createMCStreamer(const Target &T, StringRef TT,
                                    MCContext &Ctx, MCAsmBackend &MAB,
                                    raw_ostream &OS,
                                    MCCodeEmitter *Emitter,
                                    bool RelaxAll,
                                    bool NoExecStack) {
  Triple TheTriple(TT);

  if (TheTriple.isOSDarwin())
    return createMachOStreamer(Ctx, MAB, OS, Emitter, false);

  if (TheTriple.isOSWindows()) {
    llvm_unreachable("ARM does not support Windows COFF format");
  }

  return createELFStreamer(Ctx, MAB, OS, Emitter, false, NoExecStack);
}

static MCInstPrinter *createARMMCInstPrinter(const Target &T,
                                             unsigned SyntaxVariant,
                                             const MCAsmInfo &MAI,
                                             const MCInstrInfo &MII,
                                             const MCRegisterInfo &MRI,
                                             const MCSubtargetInfo &STI) {
  if (SyntaxVariant == 0)
    return new ARMInstPrinter(MAI, MII, MRI, STI);
  return 0;
}

namespace {

class ARMMCInstrAnalysis : public MCInstrAnalysis {
public:
  ARMMCInstrAnalysis(const MCInstrInfo *Info) : MCInstrAnalysis(Info) {}

  virtual bool isUnconditionalBranch(const MCInst &Inst) const {
    // BCCs with the "always" predicate are unconditional branches.
    if (Inst.getOpcode() == ARM::Bcc && Inst.getOperand(1).getImm()==ARMCC::AL)
      return true;
    return MCInstrAnalysis::isUnconditionalBranch(Inst);
  }

  virtual bool isConditionalBranch(const MCInst &Inst) const {
    // BCCs with the "always" predicate are unconditional branches.
    if (Inst.getOpcode() == ARM::Bcc && Inst.getOperand(1).getImm()==ARMCC::AL)
      return false;
    return MCInstrAnalysis::isConditionalBranch(Inst);
  }

  uint64_t evaluateBranch(const MCInst &Inst, uint64_t Addr,
                          uint64_t Size) const {
    // We only handle PCRel branches for now.
    if (Info->get(Inst.getOpcode()).OpInfo[0].OperandType!=MCOI::OPERAND_PCREL)
      return -1ULL;

    int64_t Imm = Inst.getOperand(0).getImm();
    // FIXME: This is not right for thumb.
    return Addr+Imm+8; // In ARM mode the PC is always off by 8 bytes.
  }
};

}

static MCInstrAnalysis *createARMMCInstrAnalysis(const MCInstrInfo *Info) {
  return new ARMMCInstrAnalysis(Info);
}

// Force static initialization.
extern "C" void LLVMInitializeARMTargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfoFn A(TheARMTarget, createARMMCAsmInfo);
  RegisterMCAsmInfoFn B(TheThumbTarget, createARMMCAsmInfo);

  // Register the MC codegen info.
  TargetRegistry::RegisterMCCodeGenInfo(TheARMTarget, createARMMCCodeGenInfo);
  TargetRegistry::RegisterMCCodeGenInfo(TheThumbTarget, createARMMCCodeGenInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheARMTarget, createARMMCInstrInfo);
  TargetRegistry::RegisterMCInstrInfo(TheThumbTarget, createARMMCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheARMTarget, createARMMCRegisterInfo);
  TargetRegistry::RegisterMCRegInfo(TheThumbTarget, createARMMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheARMTarget,
                                          ARM_MC::createARMMCSubtargetInfo);
  TargetRegistry::RegisterMCSubtargetInfo(TheThumbTarget,
                                          ARM_MC::createARMMCSubtargetInfo);

  // Register the MC instruction analyzer.
  TargetRegistry::RegisterMCInstrAnalysis(TheARMTarget,
                                          createARMMCInstrAnalysis);
  TargetRegistry::RegisterMCInstrAnalysis(TheThumbTarget,
                                          createARMMCInstrAnalysis);

  // Register the MC Code Emitter
  TargetRegistry::RegisterMCCodeEmitter(TheARMTarget, createARMMCCodeEmitter);
  TargetRegistry::RegisterMCCodeEmitter(TheThumbTarget, createARMMCCodeEmitter);

  // Register the asm backend.
  TargetRegistry::RegisterMCAsmBackend(TheARMTarget, createARMAsmBackend);
  TargetRegistry::RegisterMCAsmBackend(TheThumbTarget, createARMAsmBackend);

  // Register the object streamer.
  TargetRegistry::RegisterMCObjectStreamer(TheARMTarget, createMCStreamer);
  TargetRegistry::RegisterMCObjectStreamer(TheThumbTarget, createMCStreamer);

  // Register the MCInstPrinter.
  TargetRegistry::RegisterMCInstPrinter(TheARMTarget, createARMMCInstPrinter);
  TargetRegistry::RegisterMCInstPrinter(TheThumbTarget, createARMMCInstPrinter);
}
