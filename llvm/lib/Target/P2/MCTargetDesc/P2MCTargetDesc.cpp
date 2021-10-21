//===-- P2MCTargetDesc.cpp - P2 Target Descriptions -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides P2 specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "P2MCTargetDesc.h"

#include "P2InstPrinter.h"
#include "P2MCAsmInfo.h"
#include "P2TargetStreamer.h"
#include "P2ELFStreamer.h"
#include "TargetInfo/P2TargetInfo.h"

#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;

#define GET_INSTRINFO_MC_DESC
#include "P2GenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "P2GenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "P2GenRegisterInfo.inc"

static MCInstrInfo *createP2MCInstrInfo() {
    MCInstrInfo *X = new MCInstrInfo();
    InitP2MCInstrInfo(X); // defined in P2GenInstrInfo.inc
    return X;
}

static MCRegisterInfo *createP2MCRegisterInfo(const Triple &TT) {
    MCRegisterInfo *X = new MCRegisterInfo();
    InitP2MCRegisterInfo(X, 0); // defined in P2GenRegisterInfo.inc
    return X;
}

static MCSubtargetInfo *createP2MCSubtargetInfo(const Triple &TT, StringRef CPU, StringRef FS) {
    return createP2MCSubtargetInfoImpl(TT, CPU, CPU, FS);
    // createP2MCSubtargetInfoImpl defined in P2GenSubtargetInfo.inc
}

static MCInstPrinter *createP2MCInstPrinter(const Triple &T,
                                                unsigned SyntaxVariant,
                                                const MCAsmInfo &MAI,
                                                const MCInstrInfo &MII,
                                                const MCRegisterInfo &MRI) {
    return new P2InstPrinter(MAI, MII, MRI);
}

static MCStreamer *createMCStreamer(const Triple &TT, MCContext &Context,
                                    std::unique_ptr<MCAsmBackend> &&MAB,
                                    std::unique_ptr<MCObjectWriter> &&OW,
                                    std::unique_ptr<MCCodeEmitter> &&Emitter,
                                    bool RelaxAll) {
    return createELFStreamer(Context, std::move(MAB), std::move(OW), std::move(Emitter), RelaxAll);
}

static MCTargetStreamer *createP2ObjectTargetStreamer(MCStreamer &S, const MCSubtargetInfo &STI) {
    return new P2ELFStreamer(S, STI);
}

static MCTargetStreamer *createP2AsmTargetStreamer(MCStreamer &S,
                                                     formatted_raw_ostream &OS,
                                                     MCInstPrinter *InstPrint,
                                                     bool isVerboseAsm) {
    return new P2TargetAsmStreamer(S);
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeP2TargetMC() {
    // Register the MC asm info.
    RegisterMCAsmInfo<P2MCAsmInfo> X(getTheP2Target());
    // Register the MC instruction info.
    TargetRegistry::RegisterMCInstrInfo(getTheP2Target(), createP2MCInstrInfo);
    // Register the MC register info.
    TargetRegistry::RegisterMCRegInfo(getTheP2Target(), createP2MCRegisterInfo);
    // Register the MC subtarget info.
    TargetRegistry::RegisterMCSubtargetInfo(getTheP2Target(), createP2MCSubtargetInfo);
    // Register the MC instruction analyzer.
    // TargetRegistry::RegisterMCInstrAnalysis(getTheP2Target(), createP2MCInstrAnalysis);
    // Register the MCInstPrinter.
    TargetRegistry::RegisterMCInstPrinter(getTheP2Target(), createP2MCInstPrinter);
    // Register the elf streamer.
    TargetRegistry::RegisterELFStreamer(getTheP2Target(), createMCStreamer);
    // Register the asm target streamer.
    TargetRegistry::RegisterAsmTargetStreamer(getTheP2Target(), createP2AsmTargetStreamer);
    // Register the MC Code Emitter
    TargetRegistry::RegisterMCCodeEmitter(getTheP2Target(), createP2MCCodeEmitter);
    // Register the asm backend.
    TargetRegistry::RegisterMCAsmBackend(getTheP2Target(), createP2AsmBackend);
    // register the object taret streamer
    TargetRegistry::RegisterObjectTargetStreamer(getTheP2Target(), createP2ObjectTargetStreamer);
}