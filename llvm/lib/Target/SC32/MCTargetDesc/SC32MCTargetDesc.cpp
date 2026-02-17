#include "SC32MCTargetDesc.h"
#include "SC32InstPrinter.h"
#include "SC32MCAsmInfo.h"
#include "TargetInfo/SC32TargetInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

#define GET_SUBTARGETINFO_MC_DESC
#include "SC32GenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "SC32GenRegisterInfo.inc"

#define GET_INSTRINFO_MC_DESC
#include "SC32GenInstrInfo.inc"

static MCAsmInfo *createSC32MCAsmInfo(const MCRegisterInfo &MRI,
                                      const Triple &TT,
                                      const MCTargetOptions &Options) {
  return new SC32MCAsmInfo;
}

static MCRegisterInfo *createSC32MCRegisterInfo(const Triple &TT) {
  return new MCRegisterInfo;
}

static MCInstrInfo *createSC32MCInstrInfo() { return new MCInstrInfo; }

static MCSubtargetInfo *createSC32MCSubtargetInfo(const Triple &TT,
                                                  StringRef CPU, StringRef FS) {
  return createSC32MCSubtargetInfoImpl(TT, CPU, CPU, FS);
}

static MCInstPrinter *createSC32MCInstPrinter(const Triple &T,
                                              unsigned SyntaxVariant,
                                              const MCAsmInfo &MAI,
                                              const MCInstrInfo &MII,
                                              const MCRegisterInfo &MRI) {
  return new SC32InstPrinter(MAI, MII, MRI);
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeSC32TargetMC() {
  Target &T = getTheSC32Target();

  TargetRegistry::RegisterMCAsmInfo(T, createSC32MCAsmInfo);
  TargetRegistry::RegisterMCRegInfo(T, createSC32MCRegisterInfo);
  TargetRegistry::RegisterMCInstrInfo(T, createSC32MCInstrInfo);
  TargetRegistry::RegisterMCSubtargetInfo(T, createSC32MCSubtargetInfo);
  TargetRegistry::RegisterMCInstPrinter(T, createSC32MCInstPrinter);
}
