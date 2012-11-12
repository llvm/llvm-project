//===-- PPCMCTargetDesc.cpp - PowerPC Target Descriptions -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides PowerPC specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "PPCMCTargetDesc.h"
#include "PPCMCAsmInfo.h"
#include "InstPrinter/PPCInstPrinter.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#include "PPCGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "PPCGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "PPCGenRegisterInfo.inc"

using namespace llvm;

static MCInstrInfo *createPPCMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitPPCMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createPPCMCRegisterInfo(StringRef TT) {
  Triple TheTriple(TT);
  bool isPPC64 = (TheTriple.getArch() == Triple::ppc64);
  unsigned Flavour = isPPC64 ? 0 : 1;
  unsigned RA = isPPC64 ? PPC::LR8 : PPC::LR;

  MCRegisterInfo *X = new MCRegisterInfo();
  InitPPCMCRegisterInfo(X, RA, Flavour, Flavour);
  return X;
}

static MCSubtargetInfo *createPPCMCSubtargetInfo(StringRef TT, StringRef CPU,
                                                 StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitPPCMCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

static MCAsmInfo *createPPCMCAsmInfo(const Target &T, StringRef TT) {
  Triple TheTriple(TT);
  bool isPPC64 = TheTriple.getArch() == Triple::ppc64;

  MCAsmInfo *MAI;
  if (TheTriple.isOSDarwin())
    MAI = new PPCMCAsmInfoDarwin(isPPC64);
  else
    MAI = new PPCLinuxMCAsmInfo(isPPC64);

  // Initial state of the frame pointer is R1.
  MachineLocation Dst(MachineLocation::VirtualFP);
  MachineLocation Src(PPC::R1, 0);
  MAI->addInitialFrameState(0, Dst, Src);

  return MAI;
}

static MCCodeGenInfo *createPPCMCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                             CodeModel::Model CM,
                                             CodeGenOpt::Level OL) {
  MCCodeGenInfo *X = new MCCodeGenInfo();

  if (RM == Reloc::Default) {
    Triple T(TT);
    if (T.isOSDarwin())
      RM = Reloc::DynamicNoPIC;
    else
      RM = Reloc::Static;
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
  if (Triple(TT).isOSDarwin())
    return createMachOStreamer(Ctx, MAB, OS, Emitter, RelaxAll);

  return createELFStreamer(Ctx, MAB, OS, Emitter, RelaxAll, NoExecStack);
}

static MCInstPrinter *createPPCMCInstPrinter(const Target &T,
                                             unsigned SyntaxVariant,
                                             const MCAsmInfo &MAI,
                                             const MCInstrInfo &MII,
                                             const MCRegisterInfo &MRI,
                                             const MCSubtargetInfo &STI) {
  return new PPCInstPrinter(MAI, MII, MRI, SyntaxVariant);
}

extern "C" void LLVMInitializePowerPCTargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfoFn C(ThePPC32Target, createPPCMCAsmInfo);
  RegisterMCAsmInfoFn D(ThePPC64Target, createPPCMCAsmInfo);  

  // Register the MC codegen info.
  TargetRegistry::RegisterMCCodeGenInfo(ThePPC32Target, createPPCMCCodeGenInfo);
  TargetRegistry::RegisterMCCodeGenInfo(ThePPC64Target, createPPCMCCodeGenInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(ThePPC32Target, createPPCMCInstrInfo);
  TargetRegistry::RegisterMCInstrInfo(ThePPC64Target, createPPCMCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(ThePPC32Target, createPPCMCRegisterInfo);
  TargetRegistry::RegisterMCRegInfo(ThePPC64Target, createPPCMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(ThePPC32Target,
                                          createPPCMCSubtargetInfo);
  TargetRegistry::RegisterMCSubtargetInfo(ThePPC64Target,
                                          createPPCMCSubtargetInfo);

  // Register the MC Code Emitter
  TargetRegistry::RegisterMCCodeEmitter(ThePPC32Target, createPPCMCCodeEmitter);
  TargetRegistry::RegisterMCCodeEmitter(ThePPC64Target, createPPCMCCodeEmitter);
  
    // Register the asm backend.
  TargetRegistry::RegisterMCAsmBackend(ThePPC32Target, createPPCAsmBackend);
  TargetRegistry::RegisterMCAsmBackend(ThePPC64Target, createPPCAsmBackend);
  
  // Register the object streamer.
  TargetRegistry::RegisterMCObjectStreamer(ThePPC32Target, createMCStreamer);
  TargetRegistry::RegisterMCObjectStreamer(ThePPC64Target, createMCStreamer);

  // Register the MCInstPrinter.
  TargetRegistry::RegisterMCInstPrinter(ThePPC32Target, createPPCMCInstPrinter);
  TargetRegistry::RegisterMCInstPrinter(ThePPC64Target, createPPCMCInstPrinter);
}
