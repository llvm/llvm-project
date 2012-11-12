//===-- MipsMCTargetDesc.cpp - Mips Target Descriptions -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Mips specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "MipsMCAsmInfo.h"
#include "MipsMCTargetDesc.h"
#include "InstPrinter/MipsInstPrinter.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#include "MipsGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "MipsGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "MipsGenRegisterInfo.inc"

using namespace llvm;

static std::string ParseMipsTriple(StringRef TT, StringRef CPU) {
  std::string MipsArchFeature;
  size_t DashPosition = 0;
  StringRef TheTriple;

  // Let's see if there is a dash, like mips-unknown-linux.
  DashPosition = TT.find('-');

  if (DashPosition == StringRef::npos) {
    // No dash, we check the string size.
    TheTriple = TT.substr(0);
  } else {
    // We are only interested in substring before dash.
    TheTriple = TT.substr(0,DashPosition);
  }

  if (TheTriple == "mips" || TheTriple == "mipsel") {
    if (CPU.empty() || CPU == "mips32") {
      MipsArchFeature = "+mips32";
    } else if (CPU == "mips32r2") {
      MipsArchFeature = "+mips32r2";
    }
  } else {
      if (CPU.empty() || CPU == "mips64") {
        MipsArchFeature = "+mips64";
      } else if (CPU == "mips64r2") {
        MipsArchFeature = "+mips64r2";
      }
  }
  return MipsArchFeature;
}

static MCInstrInfo *createMipsMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitMipsMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createMipsMCRegisterInfo(StringRef TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitMipsMCRegisterInfo(X, Mips::RA);
  return X;
}

static MCSubtargetInfo *createMipsMCSubtargetInfo(StringRef TT, StringRef CPU,
                                                  StringRef FS) {
  std::string ArchFS = ParseMipsTriple(TT,CPU);
  if (!FS.empty()) {
    if (!ArchFS.empty())
      ArchFS = ArchFS + "," + FS.str();
    else
      ArchFS = FS;
  }
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitMipsMCSubtargetInfo(X, TT, CPU, ArchFS);
  return X;
}

static MCAsmInfo *createMipsMCAsmInfo(const Target &T, StringRef TT) {
  MCAsmInfo *MAI = new MipsMCAsmInfo(T, TT);

  MachineLocation Dst(MachineLocation::VirtualFP);
  MachineLocation Src(Mips::SP, 0);
  MAI->addInitialFrameState(0, Dst, Src);

  return MAI;
}

static MCCodeGenInfo *createMipsMCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                              CodeModel::Model CM,
                                              CodeGenOpt::Level OL) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  if (CM == CodeModel::JITDefault)
    RM = Reloc::Static;
  else if (RM == Reloc::Default)
    RM = Reloc::PIC_;
  X->InitMCCodeGenInfo(RM, CM, OL);
  return X;
}

static MCInstPrinter *createMipsMCInstPrinter(const Target &T,
                                              unsigned SyntaxVariant,
                                              const MCAsmInfo &MAI,
                                              const MCInstrInfo &MII,
                                              const MCRegisterInfo &MRI,
                                              const MCSubtargetInfo &STI) {
  return new MipsInstPrinter(MAI, MII, MRI);
}

static MCStreamer *createMCStreamer(const Target &T, StringRef TT,
                                    MCContext &Ctx, MCAsmBackend &MAB,
                                    raw_ostream &_OS,
                                    MCCodeEmitter *_Emitter,
                                    bool RelaxAll,
                                    bool NoExecStack) {
  Triple TheTriple(TT);

  return createELFStreamer(Ctx, MAB, _OS, _Emitter, RelaxAll, NoExecStack);
}

extern "C" void LLVMInitializeMipsTargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfoFn X(TheMipsTarget, createMipsMCAsmInfo);
  RegisterMCAsmInfoFn Y(TheMipselTarget, createMipsMCAsmInfo);
  RegisterMCAsmInfoFn A(TheMips64Target, createMipsMCAsmInfo);
  RegisterMCAsmInfoFn B(TheMips64elTarget, createMipsMCAsmInfo);

  // Register the MC codegen info.
  TargetRegistry::RegisterMCCodeGenInfo(TheMipsTarget,
                                        createMipsMCCodeGenInfo);
  TargetRegistry::RegisterMCCodeGenInfo(TheMipselTarget,
                                        createMipsMCCodeGenInfo);
  TargetRegistry::RegisterMCCodeGenInfo(TheMips64Target,
                                        createMipsMCCodeGenInfo);
  TargetRegistry::RegisterMCCodeGenInfo(TheMips64elTarget,
                                        createMipsMCCodeGenInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheMipsTarget, createMipsMCInstrInfo);
  TargetRegistry::RegisterMCInstrInfo(TheMipselTarget, createMipsMCInstrInfo);
  TargetRegistry::RegisterMCInstrInfo(TheMips64Target, createMipsMCInstrInfo);
  TargetRegistry::RegisterMCInstrInfo(TheMips64elTarget,
                                      createMipsMCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheMipsTarget, createMipsMCRegisterInfo);
  TargetRegistry::RegisterMCRegInfo(TheMipselTarget, createMipsMCRegisterInfo);
  TargetRegistry::RegisterMCRegInfo(TheMips64Target, createMipsMCRegisterInfo);
  TargetRegistry::RegisterMCRegInfo(TheMips64elTarget,
                                    createMipsMCRegisterInfo);

  // Register the MC Code Emitter
  TargetRegistry::RegisterMCCodeEmitter(TheMipsTarget,
                                        createMipsMCCodeEmitterEB);
  TargetRegistry::RegisterMCCodeEmitter(TheMipselTarget,
                                        createMipsMCCodeEmitterEL);
  TargetRegistry::RegisterMCCodeEmitter(TheMips64Target,
                                        createMipsMCCodeEmitterEB);
  TargetRegistry::RegisterMCCodeEmitter(TheMips64elTarget,
                                        createMipsMCCodeEmitterEL);

  // Register the object streamer.
  TargetRegistry::RegisterMCObjectStreamer(TheMipsTarget, createMCStreamer);
  TargetRegistry::RegisterMCObjectStreamer(TheMipselTarget, createMCStreamer);
  TargetRegistry::RegisterMCObjectStreamer(TheMips64Target, createMCStreamer);
  TargetRegistry::RegisterMCObjectStreamer(TheMips64elTarget,
                                           createMCStreamer);

  // Register the asm backend.
  TargetRegistry::RegisterMCAsmBackend(TheMipsTarget,
                                       createMipsAsmBackendEB32);
  TargetRegistry::RegisterMCAsmBackend(TheMipselTarget,
                                       createMipsAsmBackendEL32);
  TargetRegistry::RegisterMCAsmBackend(TheMips64Target,
                                       createMipsAsmBackendEB64);
  TargetRegistry::RegisterMCAsmBackend(TheMips64elTarget,
                                       createMipsAsmBackendEL64);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheMipsTarget,
                                          createMipsMCSubtargetInfo);
  TargetRegistry::RegisterMCSubtargetInfo(TheMipselTarget,
                                          createMipsMCSubtargetInfo);
  TargetRegistry::RegisterMCSubtargetInfo(TheMips64Target,
                                          createMipsMCSubtargetInfo);
  TargetRegistry::RegisterMCSubtargetInfo(TheMips64elTarget,
                                          createMipsMCSubtargetInfo);

  // Register the MCInstPrinter.
  TargetRegistry::RegisterMCInstPrinter(TheMipsTarget,
                                        createMipsMCInstPrinter);
  TargetRegistry::RegisterMCInstPrinter(TheMipselTarget,
                                        createMipsMCInstPrinter);
  TargetRegistry::RegisterMCInstPrinter(TheMips64Target,
                                        createMipsMCInstPrinter);
  TargetRegistry::RegisterMCInstPrinter(TheMips64elTarget,
                                        createMipsMCInstPrinter);
}
