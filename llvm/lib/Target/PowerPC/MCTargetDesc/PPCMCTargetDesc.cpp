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
#include "InstPrinter/PPCInstPrinter.h"
#include "PPCMCAsmInfo.h"
#include "PPCTargetStreamer.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#include "PPCGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "PPCGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "PPCGenRegisterInfo.inc"

using namespace llvm;

// Pin the vtable to this file.
PPCTargetStreamer::~PPCTargetStreamer() {}

static MCInstrInfo *createPPCMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitPPCMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createPPCMCRegisterInfo(StringRef TT) {
  Triple TheTriple(TT);
  bool isPPC64 = (TheTriple.getArch() == Triple::ppc64 ||
                  TheTriple.getArch() == Triple::ppc64le);
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

static MCAsmInfo *createPPCMCAsmInfo(const MCRegisterInfo &MRI, StringRef TT) {
  Triple TheTriple(TT);
  bool isPPC64 = (TheTriple.getArch() == Triple::ppc64 ||
                  TheTriple.getArch() == Triple::ppc64le);

  MCAsmInfo *MAI;
  if (TheTriple.isOSDarwin())
    MAI = new PPCMCAsmInfoDarwin(isPPC64, TheTriple);
  else
    MAI = new PPCLinuxMCAsmInfo(isPPC64);

  // Initial state of the frame pointer is R1.
  unsigned Reg = isPPC64 ? PPC::X1 : PPC::R1;
  MCCFIInstruction Inst =
      MCCFIInstruction::createDefCfa(0, MRI.getDwarfRegNum(Reg, true), 0);
  MAI->addInitialFrameState(Inst);

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
  if (CM == CodeModel::Default) {
    Triple T(TT);
    if (!T.isOSDarwin() &&
        (T.getArch() == Triple::ppc64 || T.getArch() == Triple::ppc64le))
      CM = CodeModel::Medium;
  }
  X->InitMCCodeGenInfo(RM, CM, OL);
  return X;
}

namespace {
class PPCTargetAsmStreamer : public PPCTargetStreamer {
  formatted_raw_ostream &OS;

public:
  PPCTargetAsmStreamer(formatted_raw_ostream &OS) : OS(OS) {}
  virtual void emitTCEntry(const MCSymbol &S) {
    OS << "\t.tc ";
    OS << S.getName();
    OS << "[TC],";
    OS << S.getName();
    OS << '\n';
  }
};

class PPCTargetELFStreamer : public PPCTargetStreamer {
  virtual void emitTCEntry(const MCSymbol &S) {
    // Creates a R_PPC64_TOC relocation
    Streamer->EmitSymbolValue(&S, 8);
  }
};
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

  PPCTargetStreamer *S = new PPCTargetELFStreamer();
  return createELFStreamer(Ctx, S, MAB, OS, Emitter, RelaxAll, NoExecStack);
}

static MCStreamer *
createMCAsmStreamer(MCContext &Ctx, formatted_raw_ostream &OS,
                    bool isVerboseAsm, bool useLoc, bool useCFI,
                    bool useDwarfDirectory, MCInstPrinter *InstPrint,
                    MCCodeEmitter *CE, MCAsmBackend *TAB, bool ShowInst) {
  PPCTargetStreamer *S = new PPCTargetAsmStreamer(OS);

  return llvm::createAsmStreamer(Ctx, S, OS, isVerboseAsm, useLoc, useCFI,
                                 useDwarfDirectory, InstPrint, CE, TAB,
                                 ShowInst);
}

static MCInstPrinter *createPPCMCInstPrinter(const Target &T,
                                             unsigned SyntaxVariant,
                                             const MCAsmInfo &MAI,
                                             const MCInstrInfo &MII,
                                             const MCRegisterInfo &MRI,
                                             const MCSubtargetInfo &STI) {
  bool isDarwin = Triple(STI.getTargetTriple()).isOSDarwin();
  return new PPCInstPrinter(MAI, MII, MRI, isDarwin);
}

extern "C" void LLVMInitializePowerPCTargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfoFn C(ThePPC32Target, createPPCMCAsmInfo);
  RegisterMCAsmInfoFn D(ThePPC64Target, createPPCMCAsmInfo);  
  RegisterMCAsmInfoFn E(ThePPC64LETarget, createPPCMCAsmInfo);  

  // Register the MC codegen info.
  TargetRegistry::RegisterMCCodeGenInfo(ThePPC32Target, createPPCMCCodeGenInfo);
  TargetRegistry::RegisterMCCodeGenInfo(ThePPC64Target, createPPCMCCodeGenInfo);
  TargetRegistry::RegisterMCCodeGenInfo(ThePPC64LETarget,
                                        createPPCMCCodeGenInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(ThePPC32Target, createPPCMCInstrInfo);
  TargetRegistry::RegisterMCInstrInfo(ThePPC64Target, createPPCMCInstrInfo);
  TargetRegistry::RegisterMCInstrInfo(ThePPC64LETarget,
                                      createPPCMCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(ThePPC32Target, createPPCMCRegisterInfo);
  TargetRegistry::RegisterMCRegInfo(ThePPC64Target, createPPCMCRegisterInfo);
  TargetRegistry::RegisterMCRegInfo(ThePPC64LETarget, createPPCMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(ThePPC32Target,
                                          createPPCMCSubtargetInfo);
  TargetRegistry::RegisterMCSubtargetInfo(ThePPC64Target,
                                          createPPCMCSubtargetInfo);
  TargetRegistry::RegisterMCSubtargetInfo(ThePPC64LETarget,
                                          createPPCMCSubtargetInfo);

  // Register the MC Code Emitter
  TargetRegistry::RegisterMCCodeEmitter(ThePPC32Target, createPPCMCCodeEmitter);
  TargetRegistry::RegisterMCCodeEmitter(ThePPC64Target, createPPCMCCodeEmitter);
  TargetRegistry::RegisterMCCodeEmitter(ThePPC64LETarget,
                                        createPPCMCCodeEmitter);
  
    // Register the asm backend.
  TargetRegistry::RegisterMCAsmBackend(ThePPC32Target, createPPCAsmBackend);
  TargetRegistry::RegisterMCAsmBackend(ThePPC64Target, createPPCAsmBackend);
  TargetRegistry::RegisterMCAsmBackend(ThePPC64LETarget, createPPCAsmBackend);
  
  // Register the object streamer.
  TargetRegistry::RegisterMCObjectStreamer(ThePPC32Target, createMCStreamer);
  TargetRegistry::RegisterMCObjectStreamer(ThePPC64Target, createMCStreamer);
  TargetRegistry::RegisterMCObjectStreamer(ThePPC64LETarget, createMCStreamer);

  // Register the asm streamer.
  TargetRegistry::RegisterAsmStreamer(ThePPC32Target, createMCAsmStreamer);
  TargetRegistry::RegisterAsmStreamer(ThePPC64Target, createMCAsmStreamer);
  TargetRegistry::RegisterAsmStreamer(ThePPC64LETarget, createMCAsmStreamer);

  // Register the MCInstPrinter.
  TargetRegistry::RegisterMCInstPrinter(ThePPC32Target, createPPCMCInstPrinter);
  TargetRegistry::RegisterMCInstPrinter(ThePPC64Target, createPPCMCInstPrinter);
  TargetRegistry::RegisterMCInstPrinter(ThePPC64LETarget,
                                        createPPCMCInstPrinter);
}
