//===-- AArch64MCTargetDesc.cpp - AArch64 Target Descriptions ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides AArch64 specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "AArch64MCTargetDesc.h"
#include "AArch64ELFStreamer.h"
#include "AArch64MCAsmInfo.h"
#include "InstPrinter/AArch64InstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define GET_INSTRINFO_MC_DESC
#include "AArch64GenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "AArch64GenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "AArch64GenRegisterInfo.inc"

static MCInstrInfo *createAArch64MCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitAArch64MCInstrInfo(X);
  return X;
}

static MCSubtargetInfo *
createAArch64MCSubtargetInfo(const Triple &TT, StringRef CPU, StringRef FS) {
  if (CPU.empty())
    CPU = "generic";

  return createAArch64MCSubtargetInfoImpl(TT, CPU, FS);
}

static MCRegisterInfo *createAArch64MCRegisterInfo(const Triple &Triple) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitAArch64MCRegisterInfo(X, AArch64::LR);
  return X;
}

static MCAsmInfo *createAArch64MCAsmInfo(const MCRegisterInfo &MRI,
                                         const Triple &TheTriple) {
  MCAsmInfo *MAI;
  if (TheTriple.isOSBinFormatMachO())
    MAI = new AArch64MCAsmInfoDarwin();
  else {
    assert(TheTriple.isOSBinFormatELF() && "Only expect Darwin or ELF");
    MAI = new AArch64MCAsmInfoELF(TheTriple);
  }

  // Initial state of the frame pointer is SP.
  unsigned Reg = MRI.getDwarfRegNum(AArch64::SP, true);
  MCCFIInstruction Inst = MCCFIInstruction::createDefCfa(nullptr, Reg, 0);
  MAI->addInitialFrameState(Inst);

  return MAI;
}

static void adjustCodeGenOpts(const Triple &TT, Reloc::Model RM,
                              CodeModel::Model &CM) {
  assert((TT.isOSBinFormatELF() || TT.isOSBinFormatMachO()) &&
         "Only expect Darwin and ELF targets");

  if (CM == CodeModel::Default)
    CM = CodeModel::Small;
  // The default MCJIT memory managers make no guarantees about where they can
  // find an executable page; JITed code needs to be able to refer to globals
  // no matter how far away they are.
  else if (CM == CodeModel::JITDefault)
    CM = CodeModel::Large;
  else if (CM != CodeModel::Small && CM != CodeModel::Large)
    report_fatal_error(
        "Only small and large code models are allowed on AArch64");
}

static MCInstPrinter *createAArch64MCInstPrinter(const Triple &T,
                                                 unsigned SyntaxVariant,
                                                 const MCAsmInfo &MAI,
                                                 const MCInstrInfo &MII,
                                                 const MCRegisterInfo &MRI) {
  if (SyntaxVariant == 0)
    return new AArch64InstPrinter(MAI, MII, MRI);
  if (SyntaxVariant == 1)
    return new AArch64AppleInstPrinter(MAI, MII, MRI);

  return nullptr;
}

static MCStreamer *createELFStreamer(const Triple &T, MCContext &Ctx,
                                     MCAsmBackend &TAB, raw_pwrite_stream &OS,
                                     MCCodeEmitter *Emitter, bool RelaxAll) {
  return createAArch64ELFStreamer(Ctx, TAB, OS, Emitter, RelaxAll);
}

static MCStreamer *createMachOStreamer(MCContext &Ctx, MCAsmBackend &TAB,
                                       raw_pwrite_stream &OS,
                                       MCCodeEmitter *Emitter, bool RelaxAll,
                                       bool DWARFMustBeAtTheEnd) {
  return createMachOStreamer(Ctx, TAB, OS, Emitter, RelaxAll,
                             DWARFMustBeAtTheEnd,
                             /*LabelSections*/ true);
}

// Force static initialization.
extern "C" void LLVMInitializeAArch64TargetMC() {
  for (Target *T :
       {&TheAArch64leTarget, &TheAArch64beTarget, &TheARM64Target}) {
    // Register the MC asm info.
    RegisterMCAsmInfoFn X(*T, createAArch64MCAsmInfo);

    // Register the MC codegen info.
    TargetRegistry::registerMCAdjustCodeGenOpts(*T, adjustCodeGenOpts);

    // Register the MC instruction info.
    TargetRegistry::RegisterMCInstrInfo(*T, createAArch64MCInstrInfo);

    // Register the MC register info.
    TargetRegistry::RegisterMCRegInfo(*T, createAArch64MCRegisterInfo);

    // Register the MC subtarget info.
    TargetRegistry::RegisterMCSubtargetInfo(*T, createAArch64MCSubtargetInfo);

    // Register the MC Code Emitter
    TargetRegistry::RegisterMCCodeEmitter(*T, createAArch64MCCodeEmitter);

    // Register the obj streamers.
    TargetRegistry::RegisterELFStreamer(*T, createELFStreamer);
    TargetRegistry::RegisterMachOStreamer(*T, createMachOStreamer);

    // Register the obj target streamer.
    TargetRegistry::RegisterObjectTargetStreamer(
        *T, createAArch64ObjectTargetStreamer);

    // Register the asm streamer.
    TargetRegistry::RegisterAsmTargetStreamer(*T,
                                              createAArch64AsmTargetStreamer);
    // Register the MCInstPrinter.
    TargetRegistry::RegisterMCInstPrinter(*T, createAArch64MCInstPrinter);
  }

  // Register the asm backend.
  for (Target *T : {&TheAArch64leTarget, &TheARM64Target})
    TargetRegistry::RegisterMCAsmBackend(*T, createAArch64leAsmBackend);
  TargetRegistry::RegisterMCAsmBackend(TheAArch64beTarget,
                                       createAArch64beAsmBackend);
}
