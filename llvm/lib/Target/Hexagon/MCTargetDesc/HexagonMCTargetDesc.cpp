//===-- HexagonMCTargetDesc.cpp - Hexagon Target Descriptions -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Hexagon specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "HexagonMCTargetDesc.h"
#include "Hexagon.h"
#include "HexagonMCAsmInfo.h"
#include "HexagonMCELFStreamer.h"
#include "MCTargetDesc/HexagonInstPrinter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define GET_INSTRINFO_MC_DESC
#include "HexagonGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "HexagonGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "HexagonGenRegisterInfo.inc"

cl::opt<bool> llvm::HexagonDisableCompound
  ("mno-compound",
   cl::desc("Disable looking for compound instructions for Hexagon"));

cl::opt<bool> llvm::HexagonDisableDuplex
  ("mno-pairing",
   cl::desc("Disable looking for duplex instructions for Hexagon"));

static cl::opt<bool> HexagonV4ArchVariant("mv4", cl::Hidden, cl::init(false),
  cl::desc("Build for Hexagon V4"));

static cl::opt<bool> HexagonV5ArchVariant("mv5", cl::Hidden, cl::init(false),
  cl::desc("Build for Hexagon V5"));

static cl::opt<bool> HexagonV55ArchVariant("mv55", cl::Hidden, cl::init(false),
  cl::desc("Build for Hexagon V55"));

static cl::opt<bool> HexagonV60ArchVariant("mv60", cl::Hidden, cl::init(false),
  cl::desc("Build for Hexagon V60"));


static StringRef DefaultArch = "hexagonv60";

static StringRef HexagonGetArchVariant() {
  if (HexagonV4ArchVariant)
    return "hexagonv4";
  if (HexagonV5ArchVariant)
    return "hexagonv5";
  if (HexagonV55ArchVariant)
    return "hexagonv55";
  if (HexagonV60ArchVariant)
    return "hexagonv60";
  return "";
}

StringRef Hexagon_MC::selectHexagonCPU(const Triple &TT, StringRef CPU) {
  StringRef ArchV = HexagonGetArchVariant();
  if (!ArchV.empty() && !CPU.empty()) {
    if (ArchV != CPU)
      report_fatal_error("conflicting architectures specified.");
    return CPU;
  }
  if (ArchV.empty()) {
    if (CPU.empty())
      CPU = DefaultArch;
    return CPU;
  }
  return ArchV;
}

MCInstrInfo *llvm::createHexagonMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitHexagonMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createHexagonMCRegisterInfo(const Triple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitHexagonMCRegisterInfo(X, Hexagon::R31);
  return X;
}

static MCSubtargetInfo *
createHexagonMCSubtargetInfo(const Triple &TT, StringRef CPU, StringRef FS) {
  CPU = Hexagon_MC::selectHexagonCPU(TT, CPU);
  return createHexagonMCSubtargetInfoImpl(TT, CPU, FS);
}

namespace {
class HexagonTargetAsmStreamer : public HexagonTargetStreamer {
public:
  HexagonTargetAsmStreamer(MCStreamer &S,
                           formatted_raw_ostream &, bool,
                           MCInstPrinter &)
      : HexagonTargetStreamer(S) {}
  void prettyPrintAsm(MCInstPrinter &InstPrinter, raw_ostream &OS,
                      const MCInst &Inst, const MCSubtargetInfo &STI) override {
    assert(HexagonMCInstrInfo::isBundle(Inst));
    assert(HexagonMCInstrInfo::bundleSize(Inst) <= HEXAGON_PACKET_SIZE);
    std::string Buffer;
    {
      raw_string_ostream TempStream(Buffer);
      InstPrinter.printInst(&Inst, TempStream, "", STI);
    }
    StringRef Contents(Buffer);
    auto PacketBundle = Contents.rsplit('\n');
    auto HeadTail = PacketBundle.first.split('\n');
    StringRef Separator = "\n";
    StringRef Indent = "\t\t";
    OS << "\t{\n";
    while (!HeadTail.first.empty()) {
      StringRef InstTxt;
      auto Duplex = HeadTail.first.split('\v');
      if (!Duplex.second.empty()) {
        OS << Indent << Duplex.first << Separator;
        InstTxt = Duplex.second;
      } else if (!HeadTail.first.trim().startswith("immext")) {
        InstTxt = Duplex.first;
      }
      if (!InstTxt.empty())
        OS << Indent << InstTxt << Separator;
      HeadTail = HeadTail.second.split('\n');
    }
    OS << "\t}" << PacketBundle.second;
  }
};
}

namespace {
class HexagonTargetELFStreamer : public HexagonTargetStreamer {
public:
  MCELFStreamer &getStreamer() {
    return static_cast<MCELFStreamer &>(Streamer);
  }
  HexagonTargetELFStreamer(MCStreamer &S, MCSubtargetInfo const &STI)
      : HexagonTargetStreamer(S) {
    auto Bits = STI.getFeatureBits();
    unsigned Flags = 0;
    if (Bits[Hexagon::ArchV60])
      Flags = ELF::EF_HEXAGON_MACH_V60;
    else if (Bits[Hexagon::ArchV55])
      Flags = ELF::EF_HEXAGON_MACH_V55;
    else if (Bits[Hexagon::ArchV5])
      Flags = ELF::EF_HEXAGON_MACH_V5;
    else if (Bits[Hexagon::ArchV4])
      Flags = ELF::EF_HEXAGON_MACH_V4;
    getStreamer().getAssembler().setELFHeaderEFlags(Flags);
  }
  void EmitCommonSymbolSorted(MCSymbol *Symbol, uint64_t Size,
                              unsigned ByteAlignment,
                              unsigned AccessSize) override {
    HexagonMCELFStreamer &HexagonELFStreamer =
        static_cast<HexagonMCELFStreamer &>(getStreamer());
    HexagonELFStreamer.HexagonMCEmitCommonSymbol(Symbol, Size, ByteAlignment,
                                                 AccessSize);
  }
  void EmitLocalCommonSymbolSorted(MCSymbol *Symbol, uint64_t Size,
                                   unsigned ByteAlignment,
                                   unsigned AccessSize) override {
    HexagonMCELFStreamer &HexagonELFStreamer =
        static_cast<HexagonMCELFStreamer &>(getStreamer());
    HexagonELFStreamer.HexagonMCEmitLocalCommonSymbol(
        Symbol, Size, ByteAlignment, AccessSize);
  }
};
}

static MCAsmInfo *createHexagonMCAsmInfo(const MCRegisterInfo &MRI,
                                         const Triple &TT) {
  MCAsmInfo *MAI = new HexagonMCAsmInfo(TT);

  // VirtualFP = (R30 + #0).
  MCCFIInstruction Inst =
      MCCFIInstruction::createDefCfa(nullptr, Hexagon::R30, 0);
  MAI->addInitialFrameState(Inst);

  return MAI;
}

static MCInstPrinter *createHexagonMCInstPrinter(const Triple &T,
                                                 unsigned SyntaxVariant,
                                                 const MCAsmInfo &MAI,
                                                 const MCInstrInfo &MII,
                                                 const MCRegisterInfo &MRI) {
  if (SyntaxVariant == 0)
    return (new HexagonInstPrinter(MAI, MII, MRI));
  else
    return nullptr;
}

static MCTargetStreamer *createMCAsmTargetStreamer(MCStreamer &S,
                                                   formatted_raw_ostream &OS,
                                                   MCInstPrinter *InstPrint,
                                                   bool IsVerboseAsm) {
  return new HexagonTargetAsmStreamer(S,  OS, IsVerboseAsm, *InstPrint);
}

static MCStreamer *createMCStreamer(Triple const &T, MCContext &Context,
                                    MCAsmBackend &MAB, raw_pwrite_stream &OS,
                                    MCCodeEmitter *Emitter, bool RelaxAll) {
  return createHexagonELFStreamer(Context, MAB, OS, Emitter);
}

static MCTargetStreamer *
createHexagonObjectTargetStreamer(MCStreamer &S, MCSubtargetInfo const &STI) {
  return new HexagonTargetELFStreamer(S, STI);
}

// Force static initialization.
extern "C" void LLVMInitializeHexagonTargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfoFn X(TheHexagonTarget, createHexagonMCAsmInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheHexagonTarget,
                                      createHexagonMCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheHexagonTarget,
                                    createHexagonMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheHexagonTarget,
                                          createHexagonMCSubtargetInfo);

  // Register the MC Code Emitter
  TargetRegistry::RegisterMCCodeEmitter(TheHexagonTarget,
                                        createHexagonMCCodeEmitter);

  // Register the asm backend
  TargetRegistry::RegisterMCAsmBackend(TheHexagonTarget,
                                       createHexagonAsmBackend);

  // Register the obj streamer
  TargetRegistry::RegisterELFStreamer(TheHexagonTarget, createMCStreamer);

  // Register the asm streamer
  TargetRegistry::RegisterAsmTargetStreamer(TheHexagonTarget,
                                            createMCAsmTargetStreamer);

  // Register the MC Inst Printer
  TargetRegistry::RegisterMCInstPrinter(TheHexagonTarget,
                                        createHexagonMCInstPrinter);

  TargetRegistry::RegisterObjectTargetStreamer(
      TheHexagonTarget, createHexagonObjectTargetStreamer);
}
