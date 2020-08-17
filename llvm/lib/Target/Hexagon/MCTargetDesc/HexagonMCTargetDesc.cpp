//===-- HexagonMCTargetDesc.cpp - Hexagon Target Descriptions -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides Hexagon specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "HexagonArch.h"
#include "HexagonTargetStreamer.h"
#include "MCTargetDesc/HexagonInstPrinter.h"
#include "MCTargetDesc/HexagonMCAsmInfo.h"
#include "MCTargetDesc/HexagonMCELFStreamer.h"
#include "MCTargetDesc/HexagonMCInstrInfo.h"
#include "MCTargetDesc/HexagonMCTargetDesc.h"
#include "TargetInfo/HexagonTargetInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>

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

namespace { // These flags are to be deprecated
cl::opt<bool> MV5("mv5", cl::Hidden, cl::desc("Build for Hexagon V5"),
                  cl::init(false));
cl::opt<bool> MV55("mv55", cl::Hidden, cl::desc("Build for Hexagon V55"),
                   cl::init(false));
cl::opt<bool> MV60("mv60", cl::Hidden, cl::desc("Build for Hexagon V60"),
                   cl::init(false));
cl::opt<bool> MV62("mv62", cl::Hidden, cl::desc("Build for Hexagon V62"),
                   cl::init(false));
cl::opt<bool> MV65("mv65", cl::Hidden, cl::desc("Build for Hexagon V65"),
                   cl::init(false));
cl::opt<bool> MV66("mv66", cl::Hidden, cl::desc("Build for Hexagon V66"),
                   cl::init(false));
cl::opt<bool> MV67("mv67", cl::Hidden, cl::desc("Build for Hexagon V67"),
                   cl::init(false));
cl::opt<bool> MV67T("mv67t", cl::Hidden, cl::desc("Build for Hexagon V67T"),
                    cl::init(false));

cl::opt<Hexagon::ArchEnum>
    EnableHVX("mhvx",
      cl::desc("Enable Hexagon Vector eXtensions"),
      cl::values(
        clEnumValN(Hexagon::ArchEnum::V60, "v60", "Build for HVX v60"),
        clEnumValN(Hexagon::ArchEnum::V62, "v62", "Build for HVX v62"),
        clEnumValN(Hexagon::ArchEnum::V65, "v65", "Build for HVX v65"),
        clEnumValN(Hexagon::ArchEnum::V66, "v66", "Build for HVX v66"),
        clEnumValN(Hexagon::ArchEnum::V67, "v67", "Build for HVX v67"),
        // Sentinel for no value specified.
        clEnumValN(Hexagon::ArchEnum::Generic, "", "")),
      // Sentinel for flag not present.
      cl::init(Hexagon::ArchEnum::NoArch), cl::ValueOptional);
} // namespace

static cl::opt<bool>
  DisableHVX("mno-hvx", cl::Hidden,
             cl::desc("Disable Hexagon Vector eXtensions"));


static StringRef DefaultArch = "hexagonv60";

static StringRef HexagonGetArchVariant() {
  if (MV5)
    return "hexagonv5";
  if (MV55)
    return "hexagonv55";
  if (MV60)
    return "hexagonv60";
  if (MV62)
    return "hexagonv62";
  if (MV65)
    return "hexagonv65";
  if (MV66)
    return "hexagonv66";
  if (MV67)
    return "hexagonv67";
  if (MV67T)
    return "hexagonv67t";
  return "";
}

StringRef Hexagon_MC::selectHexagonCPU(StringRef CPU) {
  StringRef ArchV = HexagonGetArchVariant();
  if (!ArchV.empty() && !CPU.empty()) {
    // Tiny cores have a "t" suffix that is discarded when creating a secondary
    // non-tiny subtarget.  See: addArchSubtarget
    std::pair<StringRef,StringRef> ArchP = ArchV.split('t');
    std::pair<StringRef,StringRef> CPUP = CPU.split('t');
    if (!ArchP.first.equals(CPUP.first))
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

unsigned llvm::HexagonGetLastSlot() { return HexagonItinerariesV5FU::SLOT3; }

unsigned llvm::HexagonConvertUnits(unsigned ItinUnits, unsigned *Lanes) {
  enum {
    CVI_NONE = 0,
    CVI_XLANE = 1 << 0,
    CVI_SHIFT = 1 << 1,
    CVI_MPY0 = 1 << 2,
    CVI_MPY1 = 1 << 3,
    CVI_ZW = 1 << 4
  };

  if (ItinUnits == HexagonItinerariesV62FU::CVI_ALL ||
      ItinUnits == HexagonItinerariesV62FU::CVI_ALL_NOMEM)
    return (*Lanes = 4, CVI_XLANE);
  else if (ItinUnits & HexagonItinerariesV62FU::CVI_MPY01 &&
           ItinUnits & HexagonItinerariesV62FU::CVI_XLSHF)
    return (*Lanes = 2, CVI_XLANE | CVI_MPY0);
  else if (ItinUnits & HexagonItinerariesV62FU::CVI_MPY01)
    return (*Lanes = 2, CVI_MPY0);
  else if (ItinUnits & HexagonItinerariesV62FU::CVI_XLSHF)
    return (*Lanes = 2, CVI_XLANE);
  else if (ItinUnits & HexagonItinerariesV62FU::CVI_XLANE &&
           ItinUnits & HexagonItinerariesV62FU::CVI_SHIFT &&
           ItinUnits & HexagonItinerariesV62FU::CVI_MPY0 &&
           ItinUnits & HexagonItinerariesV62FU::CVI_MPY1)
    return (*Lanes = 1, CVI_XLANE | CVI_SHIFT | CVI_MPY0 | CVI_MPY1);
  else if (ItinUnits & HexagonItinerariesV62FU::CVI_XLANE &&
           ItinUnits & HexagonItinerariesV62FU::CVI_SHIFT)
    return (*Lanes = 1, CVI_XLANE | CVI_SHIFT);
  else if (ItinUnits & HexagonItinerariesV62FU::CVI_MPY0 &&
           ItinUnits & HexagonItinerariesV62FU::CVI_MPY1)
    return (*Lanes = 1, CVI_MPY0 | CVI_MPY1);
  else if (ItinUnits == HexagonItinerariesV62FU::CVI_ZW)
    return (*Lanes = 1, CVI_ZW);
  else if (ItinUnits == HexagonItinerariesV62FU::CVI_XLANE)
    return (*Lanes = 1, CVI_XLANE);
  else if (ItinUnits == HexagonItinerariesV62FU::CVI_SHIFT)
    return (*Lanes = 1, CVI_SHIFT);

  return (*Lanes = 0, CVI_NONE);
}


namespace llvm {
namespace HexagonFUnits {
bool isSlot0Only(unsigned units) {
  return HexagonItinerariesV62FU::SLOT0 == units;
}
} // namespace HexagonFUnits
} // namespace llvm

namespace {

class HexagonTargetAsmStreamer : public HexagonTargetStreamer {
public:
  HexagonTargetAsmStreamer(MCStreamer &S,
                           formatted_raw_ostream &OS,
                           bool isVerboseAsm,
                           MCInstPrinter &IP)
      : HexagonTargetStreamer(S) {}

  void prettyPrintAsm(MCInstPrinter &InstPrinter, uint64_t Address,
                      const MCInst &Inst, const MCSubtargetInfo &STI,
                      raw_ostream &OS) override {
    assert(HexagonMCInstrInfo::isBundle(Inst));
    assert(HexagonMCInstrInfo::bundleSize(Inst) <= HEXAGON_PACKET_SIZE);
    std::string Buffer;
    {
      raw_string_ostream TempStream(Buffer);
      InstPrinter.printInst(&Inst, Address, "", STI, TempStream);
    }
    StringRef Contents(Buffer);
    auto PacketBundle = Contents.rsplit('\n');
    auto HeadTail = PacketBundle.first.split('\n');
    StringRef Separator = "\n";
    StringRef Indent = "\t";
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

    if (HexagonMCInstrInfo::isMemReorderDisabled(Inst))
      OS << "\n\t} :mem_noshuf" << PacketBundle.second;
    else
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


  void emitCommonSymbolSorted(MCSymbol *Symbol, uint64_t Size,
                              unsigned ByteAlignment,
                              unsigned AccessSize) override {
    HexagonMCELFStreamer &HexagonELFStreamer =
        static_cast<HexagonMCELFStreamer &>(getStreamer());
    HexagonELFStreamer.HexagonMCEmitCommonSymbol(Symbol, Size, ByteAlignment,
                                                 AccessSize);
  }

  void emitLocalCommonSymbolSorted(MCSymbol *Symbol, uint64_t Size,
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
                                         const Triple &TT,
                                         const MCTargetOptions &Options) {
  MCAsmInfo *MAI = new HexagonMCAsmInfo(TT);

  // VirtualFP = (R30 + #0).
  MCCFIInstruction Inst = MCCFIInstruction::cfiDefCfa(
      nullptr, MRI.getDwarfRegNum(Hexagon::R30, true), 0);
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

static MCStreamer *createMCStreamer(Triple const &T, MCContext &Context,
                                    std::unique_ptr<MCAsmBackend> &&MAB,
                                    std::unique_ptr<MCObjectWriter> &&OW,
                                    std::unique_ptr<MCCodeEmitter> &&Emitter,
                                    bool RelaxAll) {
  return createHexagonELFStreamer(T, Context, std::move(MAB), std::move(OW),
                                  std::move(Emitter));
}

static MCTargetStreamer *
createHexagonObjectTargetStreamer(MCStreamer &S, const MCSubtargetInfo &STI) {
  return new HexagonTargetELFStreamer(S, STI);
}

static void LLVM_ATTRIBUTE_UNUSED clearFeature(MCSubtargetInfo* STI, uint64_t F) {
  if (STI->getFeatureBits()[F])
    STI->ToggleFeature(F);
}

static bool LLVM_ATTRIBUTE_UNUSED checkFeature(MCSubtargetInfo* STI, uint64_t F) {
  return STI->getFeatureBits()[F];
}

namespace {
std::string selectHexagonFS(StringRef CPU, StringRef FS) {
  SmallVector<StringRef, 3> Result;
  if (!FS.empty())
    Result.push_back(FS);

  switch (EnableHVX) {
  case Hexagon::ArchEnum::V5:
  case Hexagon::ArchEnum::V55:
    break;
  case Hexagon::ArchEnum::V60:
    Result.push_back("+hvxv60");
    break;
  case Hexagon::ArchEnum::V62:
    Result.push_back("+hvxv62");
    break;
  case Hexagon::ArchEnum::V65:
    Result.push_back("+hvxv65");
    break;
  case Hexagon::ArchEnum::V66:
    Result.push_back("+hvxv66");
    break;
  case Hexagon::ArchEnum::V67:
    Result.push_back("+hvxv67");
    break;
  case Hexagon::ArchEnum::Generic:{
    Result.push_back(StringSwitch<StringRef>(CPU)
             .Case("hexagonv60", "+hvxv60")
             .Case("hexagonv62", "+hvxv62")
             .Case("hexagonv65", "+hvxv65")
             .Case("hexagonv66", "+hvxv66")
             .Case("hexagonv67", "+hvxv67")
             .Case("hexagonv67t", "+hvxv67"));
    break;
  }
  case Hexagon::ArchEnum::NoArch:
    // Sentinel if -mhvx isn't specified
    break;
  }
  return join(Result.begin(), Result.end(), ",");
}
}

static bool isCPUValid(const std::string &CPU) {
  return Hexagon::CpuTable.find(CPU) != Hexagon::CpuTable.cend();
}

namespace {
std::pair<std::string, std::string> selectCPUAndFS(StringRef CPU,
                                                   StringRef FS) {
  std::pair<std::string, std::string> Result;
  Result.first = std::string(Hexagon_MC::selectHexagonCPU(CPU));
  Result.second = selectHexagonFS(Result.first, FS);
  return Result;
}
std::mutex ArchSubtargetMutex;
std::unordered_map<std::string, std::unique_ptr<MCSubtargetInfo const>>
    ArchSubtarget;
} // namespace

MCSubtargetInfo const *
Hexagon_MC::getArchSubtarget(MCSubtargetInfo const *STI) {
  std::lock_guard<std::mutex> Lock(ArchSubtargetMutex);
  auto Existing = ArchSubtarget.find(std::string(STI->getCPU()));
  if (Existing == ArchSubtarget.end())
    return nullptr;
  return Existing->second.get();
}

FeatureBitset Hexagon_MC::completeHVXFeatures(const FeatureBitset &S) {
  using namespace Hexagon;
  // Make sure that +hvx-length turns hvx on, and that "hvx" alone
  // turns on hvxvNN, corresponding to the existing ArchVNN.
  FeatureBitset FB = S;
  unsigned CpuArch = ArchV5;
  for (unsigned F : {ArchV67, ArchV66, ArchV65, ArchV62, ArchV60, ArchV55,
                     ArchV5}) {
    if (!FB.test(F))
      continue;
    CpuArch = F;
    break;
  }
  bool UseHvx = false;
  for (unsigned F : {ExtensionHVX, ExtensionHVX64B, ExtensionHVX128B}) {
    if (!FB.test(F))
      continue;
    UseHvx = true;
    break;
  }
  bool HasHvxVer = false;
  for (unsigned F : {ExtensionHVXV60, ExtensionHVXV62, ExtensionHVXV65,
                     ExtensionHVXV66, ExtensionHVXV67}) {
    if (!FB.test(F))
      continue;
    HasHvxVer = true;
    UseHvx = true;
    break;
  }

  if (!UseHvx || HasHvxVer)
    return FB;

  // HasHvxVer is false, and UseHvx is true.
  switch (CpuArch) {
    case ArchV67:
      FB.set(ExtensionHVXV67);
      LLVM_FALLTHROUGH;
    case ArchV66:
      FB.set(ExtensionHVXV66);
      LLVM_FALLTHROUGH;
    case ArchV65:
      FB.set(ExtensionHVXV65);
      LLVM_FALLTHROUGH;
    case ArchV62:
      FB.set(ExtensionHVXV62);
      LLVM_FALLTHROUGH;
    case ArchV60:
      FB.set(ExtensionHVXV60);
      break;
  }
  return FB;
}

MCSubtargetInfo *Hexagon_MC::createHexagonMCSubtargetInfo(const Triple &TT,
                                                          StringRef CPU,
                                                          StringRef FS) {
  std::pair<std::string, std::string> Features = selectCPUAndFS(CPU, FS);
  StringRef CPUName = Features.first;
  StringRef ArchFS = Features.second;

  MCSubtargetInfo *X = createHexagonMCSubtargetInfoImpl(
      TT, CPUName, /*TuneCPU*/ CPUName, ArchFS);
  if (X != nullptr && (CPUName == "hexagonv67t"))
    addArchSubtarget(X, ArchFS);

  if (CPU.equals("help"))
      exit(0);

  if (!isCPUValid(CPUName.str())) {
    errs() << "error: invalid CPU \"" << CPUName.str().c_str()
           << "\" specified\n";
    return nullptr;
  }

  if (HexagonDisableDuplex) {
    llvm::FeatureBitset Features = X->getFeatureBits();
    X->setFeatureBits(Features.reset(Hexagon::FeatureDuplex));
  }

  X->setFeatureBits(completeHVXFeatures(X->getFeatureBits()));

  // The Z-buffer instructions are grandfathered in for current
  // architectures but omitted for new ones.  Future instruction
  // sets may introduce new/conflicting z-buffer instructions.
  const bool ZRegOnDefault =
      (CPUName == "hexagonv67") || (CPUName == "hexagonv66");
  if (ZRegOnDefault) {
    llvm::FeatureBitset Features = X->getFeatureBits();
    X->setFeatureBits(Features.set(Hexagon::ExtensionZReg));
  }

  return X;
}

void Hexagon_MC::addArchSubtarget(MCSubtargetInfo const *STI,
                                  StringRef FS) {
  assert(STI != nullptr);
  if (STI->getCPU().contains("t")) {
    auto ArchSTI = createHexagonMCSubtargetInfo(
        STI->getTargetTriple(),
        STI->getCPU().substr(0, STI->getCPU().size() - 1), FS);
    std::lock_guard<std::mutex> Lock(ArchSubtargetMutex);
    ArchSubtarget[std::string(STI->getCPU())] =
        std::unique_ptr<MCSubtargetInfo const>(ArchSTI);
  }
}

unsigned Hexagon_MC::GetELFFlags(const MCSubtargetInfo &STI) {
  static std::map<StringRef,unsigned> ElfFlags = {
    {"hexagonv5",  ELF::EF_HEXAGON_MACH_V5},
    {"hexagonv55", ELF::EF_HEXAGON_MACH_V55},
    {"hexagonv60", ELF::EF_HEXAGON_MACH_V60},
    {"hexagonv62", ELF::EF_HEXAGON_MACH_V62},
    {"hexagonv65", ELF::EF_HEXAGON_MACH_V65},
    {"hexagonv66", ELF::EF_HEXAGON_MACH_V66},
    {"hexagonv67", ELF::EF_HEXAGON_MACH_V67},
    {"hexagonv67t", ELF::EF_HEXAGON_MACH_V67T},
  };

  auto F = ElfFlags.find(STI.getCPU());
  assert(F != ElfFlags.end() && "Unrecognized Architecture");
  return F->second;
}

llvm::ArrayRef<MCPhysReg> Hexagon_MC::GetVectRegRev() {
  return makeArrayRef(VectRegRev);
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
    if (!(isCall(Inst) || isUnconditionalBranch(Inst) ||
          isConditionalBranch(Inst)))
      return false;

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
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeHexagonTargetMC() {
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
