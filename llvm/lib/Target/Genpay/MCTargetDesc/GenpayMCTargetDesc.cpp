#include "GenpayMCTargetDesc.h"
#include "GenpayInstPrinter.h"
#include "GenpayMCAsmInfo.h"
#include "TargetInfo/GenpayTargetInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

#define GET_INSTRINFO_MC_DESC
#include "GenpayGenInstrInfo.inc"

#define GET_REGINFO_MC_DESC
#include "GenpayGenRegisterInfo.inc"

static MCInstrInfo *createGenpayMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitGenpayMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createGenpayMCRegisterInfo(const Triple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitGenpayMCRegisterInfo(X, Genpay::SP);
  return X;
}

static MCSubtargetInfo *createGenpayMCSubtargetInfo(const Triple &TT,
                                                    StringRef CPU,
                                                    StringRef FS) {
  return new MCSubtargetInfo(TT, CPU, /*TuneCPU=*/CPU, FS,
                             /*PN=*/ArrayRef<StringRef>(),
                             /*PF=*/ArrayRef<SubtargetFeatureKV>(),
                             /*PD=*/ArrayRef<SubtargetSubTypeKV>(),
                             /*WPR=*/nullptr, /*WL=*/nullptr, /*RA=*/nullptr,
                             /*IS=*/nullptr, /*OC=*/nullptr, /*FP=*/nullptr);
}

static MCAsmInfo *createGenpayMCAsmInfo(const MCRegisterInfo &MRI,
                                        const Triple &TT,
                                        const MCTargetOptions &Options) {
  return new GenpayMCAsmInfo(TT);
}

static MCInstPrinter *createGenpayMCInstPrinter(const Triple &T,
                                                unsigned SyntaxVariant,
                                                const MCAsmInfo &MAI,
                                                const MCInstrInfo &MII,
                                                const MCRegisterInfo &MRI) {
  return new GenpayInstPrinter(MAI, MII, MRI);
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeGenpayTargetMC() {
  TargetRegistry::RegisterMCInstrInfo(getTheGenpayTarget(),
                                      createGenpayMCInstrInfo);
  TargetRegistry::RegisterMCRegInfo(getTheGenpayTarget(),
                                    createGenpayMCRegisterInfo);
  TargetRegistry::RegisterMCSubtargetInfo(getTheGenpayTarget(),
                                          createGenpayMCSubtargetInfo);
  TargetRegistry::RegisterMCAsmInfo(getTheGenpayTarget(),
                                    createGenpayMCAsmInfo);
  TargetRegistry::RegisterMCInstPrinter(getTheGenpayTarget(),
                                        createGenpayMCInstPrinter);
}
