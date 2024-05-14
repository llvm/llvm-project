#include "InArchMCTargetDesc.h"
#include "InArch.h"
#include "InArchInfo.h"
#include "InArchInstPrinter.h"
#include "InArchMCAsmInfo.h"
#include "InArchTargetStreamer.h"
#include "TargetInfo/InArchTargetInfo.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;

#define GET_REGINFO_MC_DESC
#include "InArchGenRegisterInfo.inc"

#define GET_INSTRINFO_MC_DESC
#include "InArchGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "InArchGenSubtargetInfo.inc"

static MCRegisterInfo *createInArchMCRegisterInfo(const Triple &TT) {
  INARCH_DUMP_MAGENTA
  MCRegisterInfo *X = new MCRegisterInfo();
  InitInArchMCRegisterInfo(X, InArch::R0);
  return X;
}

static MCInstrInfo *createInArchMCInstrInfo() {
  INARCH_DUMP_MAGENTA
  MCInstrInfo *X = new MCInstrInfo();
  InitInArchMCInstrInfo(X);
  return X;
}

static MCSubtargetInfo *createInArchMCSubtargetInfo(const Triple &TT,
                                                 StringRef CPU, StringRef FS) {
  INARCH_DUMP_MAGENTA
  return createInArchMCSubtargetInfoImpl(TT, CPU, /*TuneCPU*/ CPU, FS);
}

static MCAsmInfo *createInArchMCAsmInfo(const MCRegisterInfo &MRI,
                                     const Triple &TT,
                                     const MCTargetOptions &Options) {
  INARCH_DUMP_MAGENTA
  MCAsmInfo *MAI = new InArchELFMCAsmInfo(TT);
  unsigned SP = MRI.getDwarfRegNum(InArch::R1, true);
  MCCFIInstruction Inst = MCCFIInstruction::cfiDefCfa(nullptr, SP, 0);
  MAI->addInitialFrameState(Inst);
  return MAI;
}

static MCInstPrinter *createInArchMCInstPrinter(const Triple &T,
                                              unsigned SyntaxVariant,
                                              const MCAsmInfo &MAI,
                                              const MCInstrInfo &MII,
                                              const MCRegisterInfo &MRI) {
  return new InArchInstPrinter(MAI, MII, MRI);
}

InArchTargetStreamer::InArchTargetStreamer(MCStreamer &S) : MCTargetStreamer(S) {}
InArchTargetStreamer::~InArchTargetStreamer() = default;

static MCTargetStreamer *createTargetAsmStreamer(MCStreamer &S,
                                                 formatted_raw_ostream &OS,
                                                 MCInstPrinter *InstPrint,
                                                 bool isVerboseAsm) {
  return new InArchTargetStreamer(S);
}

// We need to define this function for linking succeed
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeInArchTargetMC() {
  INARCH_DUMP_MAGENTA
  Target &TheInArchTarget = getTheInArchTarget();
  RegisterMCAsmInfoFn X(TheInArchTarget, createInArchMCAsmInfo);
  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheInArchTarget, createInArchMCRegisterInfo);
    // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheInArchTarget, createInArchMCInstrInfo);
    // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheInArchTarget,
                                          createInArchMCSubtargetInfo);
    // Register the MCInstPrinter
  TargetRegistry::RegisterMCInstPrinter(TheInArchTarget, createInArchMCInstPrinter);
  TargetRegistry::RegisterAsmTargetStreamer(TheInArchTarget,
                                            createTargetAsmStreamer);
  // Register the MC Code Emitter.
  TargetRegistry::RegisterMCCodeEmitter(TheInArchTarget, createInArchMCCodeEmitter);
  // Register the asm backend.
  TargetRegistry::RegisterMCAsmBackend(TheInArchTarget, createInArchAsmBackend);
}