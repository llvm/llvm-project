//===-- MCTargetPlugin.cpp - Example llvm-mc target plugin ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A minimal MC-only target, built as a shared module instead of being linked
// into the tools. Loading it with llvm-mc's --load option registers the
// "mcplugin" target, which knows exactly one instruction:
//
//   nop     # encoding: [0x2a]
//
// The point of the example is the registration path rather than the
// instruction set: everything here is registered from a static initializer
// that runs at dlopen time, which is early enough for the tool to find the
// target in the TargetRegistry afterwards.
//
// It has no register file, no relocations and no TableGen'd tables, so the
// hand-written MC components below are as small as the interfaces allow.
//
// Only textual output is supported: -filetype=obj additionally needs an
// object streamer and an asm backend, which a one-instruction target with no
// object format of its own has nothing useful to say about.
//
//===----------------------------------------------------------------------===//

#include "MCTargetPlugin.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

namespace {

//===----------------------------------------------------------------------===//
// Target description
//===----------------------------------------------------------------------===//

/// The instruction table this target would have had TableGen generate.
enum { NOP = 0, NumOpcodes };

const MCInstrDesc Instrs[NumOpcodes] = {
    // Opcode, NumOperands, NumDefs, Size, SchedClass, NumImplicitUses,
    // NumImplicitDefs, OpInfoOffset, ImplicitOffset, Flags, TSFlags
    {NOP, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
};
const unsigned InstrNameIndices[NumOpcodes] = {0};
const char InstrNameData[] = "NOP";

/// MCSubtargetInfo insists on a processor table; this target has no CPUs and
/// no features, so every table below is empty.
constexpr char EmptyProcNames[] = "";

class MCPluginSubtargetInfo : public MCSubtargetInfo {
public:
  MCPluginSubtargetInfo(const Triple &TT, StringRef CPU, StringRef FS)
      : MCSubtargetInfo(TT, CPU, /*TuneCPU=*/"", FS, StringTable(EmptyProcNames),
                        /*PF=*/{}, /*PD=*/{}, /*PSM=*/nullptr, /*WPR=*/nullptr,
                        /*WL=*/nullptr, /*RA=*/nullptr, /*IS=*/nullptr,
                        /*OC=*/nullptr, /*FP=*/nullptr) {}
};

class MCPluginInstPrinter : public MCInstPrinter {
public:
  using MCInstPrinter::MCInstPrinter;

  std::pair<const char *, uint64_t> getMnemonic(const MCInst &MI) const override {
    return {"nop", 0};
  }

  void printInst(const MCInst *MI, uint64_t Address, StringRef Annot,
                 const MCSubtargetInfo &STI, raw_ostream &OS) override {
    OS << "\tnop";
    printAnnotation(OS, Annot);
  }
};

class MCPluginCodeEmitter : public MCCodeEmitter {
public:
  void encodeInstruction(const MCInst &Inst, SmallVectorImpl<char> &CB,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const override {
    assert(Inst.getOpcode() == NOP && "mcplugin has one instruction");
    CB.push_back(0x2a);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

Target &llvm::getTheMCPluginTarget() {
  static Target TheMCPluginTarget;
  return TheMCPluginTarget;
}

static MCAsmInfo *createMCPluginAsmInfo(const MCRegisterInfo &MRI,
                                        const Triple &TT,
                                        const MCTargetOptions &Options) {
  return new MCAsmInfo(Options);
}

static MCRegisterInfo *createMCPluginRegInfo(const Triple &TT) {
  // A target with no registers still needs an (empty) MCRegisterInfo.
  return new MCRegisterInfo();
}

static MCInstrInfo *createMCPluginInstrInfo() {
  auto *II = new MCInstrInfo();
  II->InitMCInstrInfo(Instrs, InstrNameIndices, InstrNameData,
                      /*DF=*/nullptr, /*CDI=*/nullptr, NumOpcodes);
  return II;
}

static MCSubtargetInfo *createMCPluginSubtargetInfo(const Triple &TT,
                                                    StringRef CPU,
                                                    StringRef FS) {
  return new MCPluginSubtargetInfo(TT, CPU, FS);
}

static MCInstPrinter *createMCPluginInstPrinter(const Triple &T,
                                                unsigned SyntaxVariant,
                                                const MCAsmInfo &MAI,
                                                const MCInstrInfo &MII,
                                                const MCRegisterInfo &MRI) {
  return new MCPluginInstPrinter(MAI, MII, MRI);
}

static MCCodeEmitter *createMCPluginCodeEmitter(const MCInstrInfo &II,
                                                MCContext &Ctx) {
  return new MCPluginCodeEmitter();
}

/// Registering from a static initializer is what makes this usable through
/// --load: by the time the loading tool looks a target up, dlopen has already
/// run this.
static struct RegisterMCPluginTarget {
  RegisterMCPluginTarget() {
    Target &T = getTheMCPluginTarget();
    // The target has no Triple::ArchType of its own, so it never matches a
    // triple and has to be selected by name, with llvm-mc's --arch option.
    TargetRegistry::RegisterTarget(
        T, "mcplugin", "Example MC target plugin", "MCPlugin",
        [](Triple::ArchType) { return false; }, /*HasJIT=*/false);
    TargetRegistry::RegisterMCAsmInfo(T, createMCPluginAsmInfo);
    TargetRegistry::RegisterMCRegInfo(T, createMCPluginRegInfo);
    TargetRegistry::RegisterMCInstrInfo(T, createMCPluginInstrInfo);
    TargetRegistry::RegisterMCSubtargetInfo(T, createMCPluginSubtargetInfo);
    TargetRegistry::RegisterMCInstPrinter(T, createMCPluginInstPrinter);
    TargetRegistry::RegisterMCCodeEmitter(T, createMCPluginCodeEmitter);
  }
} Registration;
