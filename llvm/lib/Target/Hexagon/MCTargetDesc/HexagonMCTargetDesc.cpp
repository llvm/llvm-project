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

#include "Hexagon.h"
#include "HexagonTargetStreamer.h"
#include "MCTargetDesc/HexagonInstPrinter.h"
#include "MCTargetDesc/HexagonMCAsmInfo.h"
#include "MCTargetDesc/HexagonMCELFStreamer.h"
#include "MCTargetDesc/HexagonMCInstrInfo.h"
#include "MCTargetDesc/HexagonMCTargetDesc.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetRegistry.h"
#include <cassert>
#include <cstdint>
#include <new>
#include <string>

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

static cl::opt<bool> HexagonV62ArchVariant("mv62", cl::Hidden, cl::init(false),
  cl::desc("Build for Hexagon V62"));

static cl::opt<bool> EnableHVX("mhvx", cl::Hidden, cl::init(false),
  cl::desc("Enable Hexagon Vector Extension (HVX)"));

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
  if (HexagonV62ArchVariant)
    return "hexagonv62";
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

unsigned llvm::HexagonGetLastSlot() { return HexagonItinerariesV4FU::SLOT3; }

namespace {

class HexagonTargetAsmStreamer : public HexagonTargetStreamer {
public:
  HexagonTargetAsmStreamer(MCStreamer &S,
                           formatted_raw_ostream &OS,
                           bool isVerboseAsm,
                           MCInstPrinter &IP)
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

class HexagonTargetELFStreamer : public HexagonTargetStreamer {
public:
  MCELFStreamer &getStreamer() {
    return static_cast<MCELFStreamer &>(Streamer);
  }
  HexagonTargetELFStreamer(MCStreamer &S, MCSubtargetInfo const &STI)
      : HexagonTargetStreamer(S) {
    MCAssembler &MCA = getStreamer().getAssembler();
    MCA.setELFHeaderEFlags(Hexagon_MC::GetELFFlags(STI));
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

} // end anonymous namespace

llvm::MCInstrInfo *llvm::createHexagonMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitHexagonMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createHexagonMCRegisterInfo(const Triple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitHexagonMCRegisterInfo(X, Hexagon::R31);
  return X;
}

static MCAsmInfo *createHexagonMCAsmInfo(const MCRegisterInfo &MRI,
                                         const Triple &TT) {
  MCAsmInfo *MAI = new HexagonMCAsmInfo(TT);

  // VirtualFP = (R30 + #0).
  MCCFIInstruction Inst =
      MCCFIInstruction::createDefCfa(nullptr,
          MRI.getDwarfRegNum(Hexagon::R30, true), 0);
  MAI->addInitialFrameState(Inst);

  return MAI;
}

static MCInstPrinter *createHexagonMCInstPrinter(const Triple &T,
                                                 unsigned SyntaxVariant,
                                                 const MCAsmInfo &MAI,
                                                 const MCInstrInfo &MII,
                                                 const MCRegisterInfo &MRI)
{
  if (SyntaxVariant == 0)
    return new HexagonInstPrinter(MAI, MII, MRI);
  else
    return nullptr;
}

static MCTargetStreamer *
createMCAsmTargetStreamer(MCStreamer &S, formatted_raw_ostream &OS,
                          MCInstPrinter *IP, bool IsVerboseAsm) {
  return new HexagonTargetAsmStreamer(S, OS, IsVerboseAsm, *IP);
}

static MCStreamer *createMCStreamer(Triple const &T,
                                    MCContext &Context,
                                    MCAsmBackend &MAB,
                                    raw_pwrite_stream &OS,
                                    MCCodeEmitter *Emitter,
                                    bool RelaxAll) {
  return createHexagonELFStreamer(T, Context, MAB, OS, Emitter);
}

static MCTargetStreamer *
createHexagonObjectTargetStreamer(MCStreamer &S, const MCSubtargetInfo &STI) {
  return new HexagonTargetELFStreamer(S, STI);
}

static void LLVM_ATTRIBUTE_UNUSED clearFeature(MCSubtargetInfo* STI, uint64_t F) {
  uint64_t FB = STI->getFeatureBits().to_ullong();
  if (FB & (1ULL << F))
    STI->ToggleFeature(F);
}

static bool LLVM_ATTRIBUTE_UNUSED checkFeature(MCSubtargetInfo* STI, uint64_t F) {
  uint64_t FB = STI->getFeatureBits().to_ullong();
  return (FB & (1ULL << F)) != 0;
}

StringRef Hexagon_MC::ParseHexagonTriple(const Triple &TT, StringRef CPU) {
  StringRef CPUName = Hexagon_MC::selectHexagonCPU(TT, CPU);
  StringRef FS = "";
  if (EnableHVX) {
    if (CPUName.equals_lower("hexagonv60") ||
        CPUName.equals_lower("hexagonv62"))
      FS = "+hvx";
  }
  return FS;
}

static bool isCPUValid(std::string CPU)
{
  std::vector<std::string> table
  {
    "hexagonv4",
    "hexagonv5",
    "hexagonv55",
    "hexagonv60",
    "hexagonv62",
  };

  return std::find(table.begin(), table.end(), CPU) != table.end();
}

MCSubtargetInfo *Hexagon_MC::createHexagonMCSubtargetInfo(const Triple &TT,
                                                          StringRef CPU,
                                                          StringRef FS) {
  StringRef ArchFS = (FS.size()) ? FS : Hexagon_MC::ParseHexagonTriple(TT, CPU);
  StringRef CPUName = Hexagon_MC::selectHexagonCPU(TT, CPU);
  if (!isCPUValid(CPUName.str())) {
    errs() << "error: invalid CPU \"" << CPUName.str().c_str()
           << "\" specified\n";
    return nullptr;
  }

  MCSubtargetInfo *X = createHexagonMCSubtargetInfoImpl(TT, CPUName, ArchFS);
  if (X->getFeatureBits()[Hexagon::ExtensionHVXDbl]) {
    llvm::FeatureBitset Features = X->getFeatureBits();
    X->setFeatureBits(Features.set(Hexagon::ExtensionHVX));
  }
  return X;
}

unsigned Hexagon_MC::GetELFFlags(const MCSubtargetInfo &STI) {
  static std::map<StringRef,unsigned> ElfFlags = {
    {"hexagonv4",  ELF::EF_HEXAGON_MACH_V4},
    {"hexagonv5",  ELF::EF_HEXAGON_MACH_V5},
    {"hexagonv55", ELF::EF_HEXAGON_MACH_V55},
    {"hexagonv60", ELF::EF_HEXAGON_MACH_V60},
    {"hexagonv62", ELF::EF_HEXAGON_MACH_V62},
  };

  auto F = ElfFlags.find(STI.getCPU());
  assert(F != ElfFlags.end() && "Unrecognized Architecture");
  return F->second;
}

namespace {
class HexagonMCInstrAnalysis : public MCInstrAnalysis {
public:
  HexagonMCInstrAnalysis(MCInstrInfo const *Info) : MCInstrAnalysis(Info) {}

  bool isUnconditionalBranch(MCInst const &Inst) const override {
    //assert(!HexagonMCInstrInfo::isBundle(Inst));
    return MCInstrAnalysis::isUnconditionalBranch(Inst);
  }

  bool isConditionalBranch(MCInst const &Inst) const override {
    //assert(!HexagonMCInstrInfo::isBundle(Inst));
    return MCInstrAnalysis::isConditionalBranch(Inst);
  }

  bool evaluateBranch(MCInst const &Inst, uint64_t Addr,
                      uint64_t Size, uint64_t &Target) const override {
    //assert(!HexagonMCInstrInfo::isBundle(Inst));
    if(!HexagonMCInstrInfo::isExtendable(*Info, Inst))
      return false;
    auto const &Extended(HexagonMCInstrInfo::getExtendableOperand(*Info, Inst));
    assert(Extended.isExpr());
    int64_t Value;
    if(!Extended.getExpr()->evaluateAsAbsolute(Value))
      return false;
    Target = Value;
    return true;
  }
};
}

static MCInstrAnalysis *createHexagonMCInstrAnalysis(const MCInstrInfo *Info) {
  return new HexagonMCInstrAnalysis(Info);
}

// Force static initialization.
extern "C" void LLVMInitializeHexagonTargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfoFn X(getTheHexagonTarget(), createHexagonMCAsmInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(getTheHexagonTarget(),
                                      createHexagonMCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(getTheHexagonTarget(),
                                    createHexagonMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(getTheHexagonTarget(),
    Hexagon_MC::createHexagonMCSubtargetInfo);

  // Register the MC Code Emitter
  TargetRegistry::RegisterMCCodeEmitter(getTheHexagonTarget(),
                                        createHexagonMCCodeEmitter);

  // Register the asm backend
  TargetRegistry::RegisterMCAsmBackend(getTheHexagonTarget(),
                                       createHexagonAsmBackend);


  // Register the MC instruction analyzer.
  TargetRegistry::RegisterMCInstrAnalysis(getTheHexagonTarget(),
                                          createHexagonMCInstrAnalysis);

  // Register the obj streamer
  TargetRegistry::RegisterELFStreamer(getTheHexagonTarget(),
                                      createMCStreamer);

  // Register the obj target streamer
  TargetRegistry::RegisterObjectTargetStreamer(getTheHexagonTarget(),
                                      createHexagonObjectTargetStreamer);

  // Register the asm streamer
  TargetRegistry::RegisterAsmTargetStreamer(getTheHexagonTarget(),
                                            createMCAsmTargetStreamer);

  // Register the MC Inst Printer
  TargetRegistry::RegisterMCInstPrinter(getTheHexagonTarget(),
                                        createHexagonMCInstPrinter);
}
