//===-- X86MCTargetDesc.cpp - X86 Target Descriptions ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides X86 specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "X86MCTargetDesc.h"
#include "InstPrinter/X86ATTInstPrinter.h"
#include "InstPrinter/X86IntelInstPrinter.h"
#include "X86BaseInfo.h"
#include "X86MCAsmInfo.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Triple.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"

#if _MSC_VER
#include <intrin.h>
#endif

using namespace llvm;

#define GET_REGINFO_MC_DESC
#include "X86GenRegisterInfo.inc"

#define GET_INSTRINFO_MC_DESC
#define GET_GENINSTRINFO_MC_HELPERS
#include "X86GenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "X86GenSubtargetInfo.inc"

std::string X86_MC::ParseX86Triple(const Triple &TT) {
  std::string FS;
  if (TT.getArch() == Triple::x86_64)
    FS = "+64bit-mode,-32bit-mode,-16bit-mode";
  else if (TT.getEnvironment() != Triple::CODE16)
    FS = "-64bit-mode,+32bit-mode,-16bit-mode";
  else
    FS = "-64bit-mode,-32bit-mode,+16bit-mode";

  return FS;
}

unsigned X86_MC::getDwarfRegFlavour(const Triple &TT, bool isEH) {
  if (TT.getArch() == Triple::x86_64)
    return DWARFFlavour::X86_64;

  if (TT.isOSDarwin())
    return isEH ? DWARFFlavour::X86_32_DarwinEH : DWARFFlavour::X86_32_Generic;
  if (TT.isOSCygMing())
    // Unsupported by now, just quick fallback
    return DWARFFlavour::X86_32_Generic;
  return DWARFFlavour::X86_32_Generic;
}

void X86_MC::initLLVMToSEHAndCVRegMapping(MCRegisterInfo *MRI) {
  // FIXME: TableGen these.
  for (unsigned Reg = X86::NoRegister + 1; Reg < X86::NUM_TARGET_REGS; ++Reg) {
    unsigned SEH = MRI->getEncodingValue(Reg);
    MRI->mapLLVMRegToSEHReg(Reg, SEH);
  }

  // Mapping from CodeView to MC register id.
  static const struct {
    codeview::RegisterId CVReg;
    MCPhysReg Reg;
  } RegMap[] = {
    { codeview::RegisterId::CVRegAL, X86::AL},
    { codeview::RegisterId::CVRegCL, X86::CL},
    { codeview::RegisterId::CVRegDL, X86::DL},
    { codeview::RegisterId::CVRegBL, X86::BL},
    { codeview::RegisterId::CVRegAH, X86::AH},
    { codeview::RegisterId::CVRegCH, X86::CH},
    { codeview::RegisterId::CVRegDH, X86::DH},
    { codeview::RegisterId::CVRegBH, X86::BH},
    { codeview::RegisterId::CVRegAX, X86::AX},
    { codeview::RegisterId::CVRegCX, X86::CX},
    { codeview::RegisterId::CVRegDX, X86::DX},
    { codeview::RegisterId::CVRegBX, X86::BX},
    { codeview::RegisterId::CVRegSP, X86::SP},
    { codeview::RegisterId::CVRegBP, X86::BP},
    { codeview::RegisterId::CVRegSI, X86::SI},
    { codeview::RegisterId::CVRegDI, X86::DI},
    { codeview::RegisterId::CVRegEAX, X86::EAX},
    { codeview::RegisterId::CVRegECX, X86::ECX},
    { codeview::RegisterId::CVRegEDX, X86::EDX},
    { codeview::RegisterId::CVRegEBX, X86::EBX},
    { codeview::RegisterId::CVRegESP, X86::ESP},
    { codeview::RegisterId::CVRegEBP, X86::EBP},
    { codeview::RegisterId::CVRegESI, X86::ESI},
    { codeview::RegisterId::CVRegEDI, X86::EDI},

    { codeview::RegisterId::CVRegEFLAGS, X86::EFLAGS},

    { codeview::RegisterId::CVRegST0, X86::FP0},
    { codeview::RegisterId::CVRegST1, X86::FP1},
    { codeview::RegisterId::CVRegST2, X86::FP2},
    { codeview::RegisterId::CVRegST3, X86::FP3},
    { codeview::RegisterId::CVRegST4, X86::FP4},
    { codeview::RegisterId::CVRegST5, X86::FP5},
    { codeview::RegisterId::CVRegST6, X86::FP6},
    { codeview::RegisterId::CVRegST7, X86::FP7},

    { codeview::RegisterId::CVRegXMM0, X86::XMM0},
    { codeview::RegisterId::CVRegXMM1, X86::XMM1},
    { codeview::RegisterId::CVRegXMM2, X86::XMM2},
    { codeview::RegisterId::CVRegXMM3, X86::XMM3},
    { codeview::RegisterId::CVRegXMM4, X86::XMM4},
    { codeview::RegisterId::CVRegXMM5, X86::XMM5},
    { codeview::RegisterId::CVRegXMM6, X86::XMM6},
    { codeview::RegisterId::CVRegXMM7, X86::XMM7},

    { codeview::RegisterId::CVRegXMM8, X86::XMM8},
    { codeview::RegisterId::CVRegXMM9, X86::XMM9},
    { codeview::RegisterId::CVRegXMM10, X86::XMM10},
    { codeview::RegisterId::CVRegXMM11, X86::XMM11},
    { codeview::RegisterId::CVRegXMM12, X86::XMM12},
    { codeview::RegisterId::CVRegXMM13, X86::XMM13},
    { codeview::RegisterId::CVRegXMM14, X86::XMM14},
    { codeview::RegisterId::CVRegXMM15, X86::XMM15},

    { codeview::RegisterId::CVRegSIL, X86::SIL},
    { codeview::RegisterId::CVRegDIL, X86::DIL},
    { codeview::RegisterId::CVRegBPL, X86::BPL},
    { codeview::RegisterId::CVRegSPL, X86::SPL},
    { codeview::RegisterId::CVRegRAX, X86::RAX},
    { codeview::RegisterId::CVRegRBX, X86::RBX},
    { codeview::RegisterId::CVRegRCX, X86::RCX},
    { codeview::RegisterId::CVRegRDX, X86::RDX},
    { codeview::RegisterId::CVRegRSI, X86::RSI},
    { codeview::RegisterId::CVRegRDI, X86::RDI},
    { codeview::RegisterId::CVRegRBP, X86::RBP},
    { codeview::RegisterId::CVRegRSP, X86::RSP},
    { codeview::RegisterId::CVRegR8, X86::R8},
    { codeview::RegisterId::CVRegR9, X86::R9},
    { codeview::RegisterId::CVRegR10, X86::R10},
    { codeview::RegisterId::CVRegR11, X86::R11},
    { codeview::RegisterId::CVRegR12, X86::R12},
    { codeview::RegisterId::CVRegR13, X86::R13},
    { codeview::RegisterId::CVRegR14, X86::R14},
    { codeview::RegisterId::CVRegR15, X86::R15},
    { codeview::RegisterId::CVRegR8B, X86::R8B},
    { codeview::RegisterId::CVRegR9B, X86::R9B},
    { codeview::RegisterId::CVRegR10B, X86::R10B},
    { codeview::RegisterId::CVRegR11B, X86::R11B},
    { codeview::RegisterId::CVRegR12B, X86::R12B},
    { codeview::RegisterId::CVRegR13B, X86::R13B},
    { codeview::RegisterId::CVRegR14B, X86::R14B},
    { codeview::RegisterId::CVRegR15B, X86::R15B},
    { codeview::RegisterId::CVRegR8W, X86::R8W},
    { codeview::RegisterId::CVRegR9W, X86::R9W},
    { codeview::RegisterId::CVRegR10W, X86::R10W},
    { codeview::RegisterId::CVRegR11W, X86::R11W},
    { codeview::RegisterId::CVRegR12W, X86::R12W},
    { codeview::RegisterId::CVRegR13W, X86::R13W},
    { codeview::RegisterId::CVRegR14W, X86::R14W},
    { codeview::RegisterId::CVRegR15W, X86::R15W},
    { codeview::RegisterId::CVRegR8D, X86::R8D},
    { codeview::RegisterId::CVRegR9D, X86::R9D},
    { codeview::RegisterId::CVRegR10D, X86::R10D},
    { codeview::RegisterId::CVRegR11D, X86::R11D},
    { codeview::RegisterId::CVRegR12D, X86::R12D},
    { codeview::RegisterId::CVRegR13D, X86::R13D},
    { codeview::RegisterId::CVRegR14D, X86::R14D},
    { codeview::RegisterId::CVRegR15D, X86::R15D},
    { codeview::RegisterId::CVRegAMD64_YMM0, X86::YMM0},
    { codeview::RegisterId::CVRegAMD64_YMM1, X86::YMM1},
    { codeview::RegisterId::CVRegAMD64_YMM2, X86::YMM2},
    { codeview::RegisterId::CVRegAMD64_YMM3, X86::YMM3},
    { codeview::RegisterId::CVRegAMD64_YMM4, X86::YMM4},
    { codeview::RegisterId::CVRegAMD64_YMM5, X86::YMM5},
    { codeview::RegisterId::CVRegAMD64_YMM6, X86::YMM6},
    { codeview::RegisterId::CVRegAMD64_YMM7, X86::YMM7},
    { codeview::RegisterId::CVRegAMD64_YMM8, X86::YMM8},
    { codeview::RegisterId::CVRegAMD64_YMM9, X86::YMM9},
    { codeview::RegisterId::CVRegAMD64_YMM10, X86::YMM10},
    { codeview::RegisterId::CVRegAMD64_YMM11, X86::YMM11},
    { codeview::RegisterId::CVRegAMD64_YMM12, X86::YMM12},
    { codeview::RegisterId::CVRegAMD64_YMM13, X86::YMM13},
    { codeview::RegisterId::CVRegAMD64_YMM14, X86::YMM14},
    { codeview::RegisterId::CVRegAMD64_YMM15, X86::YMM15},
  };
  for (unsigned I = 0; I < array_lengthof(RegMap); ++I)
    MRI->mapLLVMRegToCVReg(RegMap[I].Reg, static_cast<int>(RegMap[I].CVReg));
}

MCSubtargetInfo *X86_MC::createX86MCSubtargetInfo(const Triple &TT,
                                                  StringRef CPU, StringRef FS) {
  std::string ArchFS = X86_MC::ParseX86Triple(TT);
  if (!FS.empty()) {
    if (!ArchFS.empty())
      ArchFS = (Twine(ArchFS) + "," + FS).str();
    else
      ArchFS = FS;
  }

  std::string CPUName = CPU;
  if (CPUName.empty())
    CPUName = "generic";

  return createX86MCSubtargetInfoImpl(TT, CPUName, ArchFS);
}

static MCInstrInfo *createX86MCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitX86MCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createX86MCRegisterInfo(const Triple &TT) {
  unsigned RA = (TT.getArch() == Triple::x86_64)
                    ? X86::RIP  // Should have dwarf #16.
                    : X86::EIP; // Should have dwarf #8.

  MCRegisterInfo *X = new MCRegisterInfo();
  InitX86MCRegisterInfo(X, RA, X86_MC::getDwarfRegFlavour(TT, false),
                        X86_MC::getDwarfRegFlavour(TT, true), RA);
  X86_MC::initLLVMToSEHAndCVRegMapping(X);
  return X;
}

static MCAsmInfo *createX86MCAsmInfo(const MCRegisterInfo &MRI,
                                     const Triple &TheTriple) {
  bool is64Bit = TheTriple.getArch() == Triple::x86_64;

  MCAsmInfo *MAI;
  if (TheTriple.isOSBinFormatMachO()) {
    if (is64Bit)
      MAI = new X86_64MCAsmInfoDarwin(TheTriple);
    else
      MAI = new X86MCAsmInfoDarwin(TheTriple);
  } else if (TheTriple.isOSBinFormatELF()) {
    // Force the use of an ELF container.
    MAI = new X86ELFMCAsmInfo(TheTriple);
  } else if (TheTriple.isWindowsMSVCEnvironment() ||
             TheTriple.isWindowsCoreCLREnvironment()) {
    MAI = new X86MCAsmInfoMicrosoft(TheTriple);
  } else if (TheTriple.isOSCygMing() ||
             TheTriple.isWindowsItaniumEnvironment()) {
    MAI = new X86MCAsmInfoGNUCOFF(TheTriple);
  } else {
    // The default is ELF.
    MAI = new X86ELFMCAsmInfo(TheTriple);
  }

  // Initialize initial frame state.
  // Calculate amount of bytes used for return address storing
  int stackGrowth = is64Bit ? -8 : -4;

  // Initial state of the frame pointer is esp+stackGrowth.
  unsigned StackPtr = is64Bit ? X86::RSP : X86::ESP;
  MCCFIInstruction Inst = MCCFIInstruction::createDefCfa(
      nullptr, MRI.getDwarfRegNum(StackPtr, true), -stackGrowth);
  MAI->addInitialFrameState(Inst);

  // Add return address to move list
  unsigned InstPtr = is64Bit ? X86::RIP : X86::EIP;
  MCCFIInstruction Inst2 = MCCFIInstruction::createOffset(
      nullptr, MRI.getDwarfRegNum(InstPtr, true), stackGrowth);
  MAI->addInitialFrameState(Inst2);

  return MAI;
}

static MCInstPrinter *createX86MCInstPrinter(const Triple &T,
                                             unsigned SyntaxVariant,
                                             const MCAsmInfo &MAI,
                                             const MCInstrInfo &MII,
                                             const MCRegisterInfo &MRI) {
  if (SyntaxVariant == 0)
    return new X86ATTInstPrinter(MAI, MII, MRI);
  if (SyntaxVariant == 1)
    return new X86IntelInstPrinter(MAI, MII, MRI);
  return nullptr;
}

static MCRelocationInfo *createX86MCRelocationInfo(const Triple &TheTriple,
                                                   MCContext &Ctx) {
  // Default to the stock relocation info.
  return llvm::createMCRelocationInfo(TheTriple, Ctx);
}

namespace llvm {
namespace X86_MC {

class X86MCInstrAnalysis : public MCInstrAnalysis {
  X86MCInstrAnalysis(const X86MCInstrAnalysis &) = delete;
  X86MCInstrAnalysis &operator=(const X86MCInstrAnalysis &) = delete;
  virtual ~X86MCInstrAnalysis() = default;

public:
  X86MCInstrAnalysis(const MCInstrInfo *MCII) : MCInstrAnalysis(MCII) {}

  bool isDependencyBreaking(const MCSubtargetInfo &STI,
                            const MCInst &Inst) const override;
  bool clearsSuperRegisters(const MCRegisterInfo &MRI, const MCInst &Inst,
                            APInt &Mask) const override;
};

bool X86MCInstrAnalysis::isDependencyBreaking(const MCSubtargetInfo &STI,
                                              const MCInst &Inst) const {
  if (STI.getCPU() == "btver2") {
    // Reference: Agner Fog's microarchitecture.pdf - Section 20 "AMD Bobcat and
    // Jaguar pipeline", subsection 8 "Dependency-breaking instructions".
    switch (Inst.getOpcode()) {
    default:
      return false;
    case X86::SUB32rr:
    case X86::SUB64rr:
    case X86::SBB32rr:
    case X86::SBB64rr:
    case X86::XOR32rr:
    case X86::XOR64rr:
    case X86::XORPSrr:
    case X86::XORPDrr:
    case X86::VXORPSrr:
    case X86::VXORPDrr:
    case X86::ANDNPSrr:
    case X86::VANDNPSrr:
    case X86::ANDNPDrr:
    case X86::VANDNPDrr:
    case X86::PXORrr:
    case X86::VPXORrr:
    case X86::PANDNrr:
    case X86::VPANDNrr:
    case X86::PSUBBrr:
    case X86::PSUBWrr:
    case X86::PSUBDrr:
    case X86::PSUBQrr:
    case X86::VPSUBBrr:
    case X86::VPSUBWrr:
    case X86::VPSUBDrr:
    case X86::VPSUBQrr:
    case X86::PCMPEQBrr:
    case X86::PCMPEQWrr:
    case X86::PCMPEQDrr:
    case X86::PCMPEQQrr:
    case X86::VPCMPEQBrr:
    case X86::VPCMPEQWrr:
    case X86::VPCMPEQDrr:
    case X86::VPCMPEQQrr:
    case X86::PCMPGTBrr:
    case X86::PCMPGTWrr:
    case X86::PCMPGTDrr:
    case X86::PCMPGTQrr:
    case X86::VPCMPGTBrr:
    case X86::VPCMPGTWrr:
    case X86::VPCMPGTDrr:
    case X86::VPCMPGTQrr:
    case X86::MMX_PXORirr:
    case X86::MMX_PANDNirr:
    case X86::MMX_PSUBBirr:
    case X86::MMX_PSUBDirr:
    case X86::MMX_PSUBQirr:
    case X86::MMX_PSUBWirr:
    case X86::MMX_PCMPGTBirr:
    case X86::MMX_PCMPGTDirr:
    case X86::MMX_PCMPGTWirr:
    case X86::MMX_PCMPEQBirr:
    case X86::MMX_PCMPEQDirr:
    case X86::MMX_PCMPEQWirr:
      return Inst.getOperand(1).getReg() == Inst.getOperand(2).getReg();
    case X86::CMP32rr:
    case X86::CMP64rr:
      return Inst.getOperand(0).getReg() == Inst.getOperand(1).getReg();
    }
  }

  return false;
}

bool X86MCInstrAnalysis::clearsSuperRegisters(const MCRegisterInfo &MRI,
                                              const MCInst &Inst,
                                              APInt &Mask) const {
  const MCInstrDesc &Desc = Info->get(Inst.getOpcode());
  unsigned NumDefs = Desc.getNumDefs();
  unsigned NumImplicitDefs = Desc.getNumImplicitDefs();
  assert(Mask.getBitWidth() == NumDefs + NumImplicitDefs &&
         "Unexpected number of bits in the mask!");

  bool HasVEX = (Desc.TSFlags & X86II::EncodingMask) == X86II::VEX;
  bool HasEVEX = (Desc.TSFlags & X86II::EncodingMask) == X86II::EVEX;
  bool HasXOP = (Desc.TSFlags & X86II::EncodingMask) == X86II::XOP;

  const MCRegisterClass &GR32RC = MRI.getRegClass(X86::GR32RegClassID);
  const MCRegisterClass &VR128XRC = MRI.getRegClass(X86::VR128XRegClassID);
  const MCRegisterClass &VR256XRC = MRI.getRegClass(X86::VR256XRegClassID);

  auto ClearsSuperReg = [=](unsigned RegID) {
    // On X86-64, a general purpose integer register is viewed as a 64-bit
    // register internal to the processor.
    // An update to the lower 32 bits of a 64 bit integer register is
    // architecturally defined to zero extend the upper 32 bits.
    if (GR32RC.contains(RegID))
      return true;

    // Early exit if this instruction has no vex/evex/xop prefix.
    if (!HasEVEX && !HasVEX && !HasXOP)
      return false;

    // All VEX and EVEX encoded instructions are defined to zero the high bits
    // of the destination register up to VLMAX (i.e. the maximum vector register
    // width pertaining to the instruction).
    // We assume the same behavior for XOP instructions too.
    return VR128XRC.contains(RegID) || VR256XRC.contains(RegID);
  };

  Mask.clearAllBits();
  for (unsigned I = 0, E = NumDefs; I < E; ++I) {
    const MCOperand &Op = Inst.getOperand(I);
    if (ClearsSuperReg(Op.getReg()))
      Mask.setBit(I);
  }

  for (unsigned I = 0, E = NumImplicitDefs; I < E; ++I) {
    const MCPhysReg Reg = Desc.getImplicitDefs()[I];
    if (ClearsSuperReg(Reg))
      Mask.setBit(NumDefs + I);
  }

  return Mask.getBoolValue();
}

} // end of namespace X86_MC

} // end of namespace llvm

static MCInstrAnalysis *createX86MCInstrAnalysis(const MCInstrInfo *Info) {
  return new X86_MC::X86MCInstrAnalysis(Info);
}

// Force static initialization.
extern "C" void LLVMInitializeX86TargetMC() {
  for (Target *T : {&getTheX86_32Target(), &getTheX86_64Target()}) {
    // Register the MC asm info.
    RegisterMCAsmInfoFn X(*T, createX86MCAsmInfo);

    // Register the MC instruction info.
    TargetRegistry::RegisterMCInstrInfo(*T, createX86MCInstrInfo);

    // Register the MC register info.
    TargetRegistry::RegisterMCRegInfo(*T, createX86MCRegisterInfo);

    // Register the MC subtarget info.
    TargetRegistry::RegisterMCSubtargetInfo(*T,
                                            X86_MC::createX86MCSubtargetInfo);

    // Register the MC instruction analyzer.
    TargetRegistry::RegisterMCInstrAnalysis(*T, createX86MCInstrAnalysis);

    // Register the code emitter.
    TargetRegistry::RegisterMCCodeEmitter(*T, createX86MCCodeEmitter);

    // Register the obj target streamer.
    TargetRegistry::RegisterObjectTargetStreamer(*T,
                                                 createX86ObjectTargetStreamer);

    // Register the asm target streamer.
    TargetRegistry::RegisterAsmTargetStreamer(*T, createX86AsmTargetStreamer);

    TargetRegistry::RegisterCOFFStreamer(*T, createX86WinCOFFStreamer);

    // Register the MCInstPrinter.
    TargetRegistry::RegisterMCInstPrinter(*T, createX86MCInstPrinter);

    // Register the MC relocation info.
    TargetRegistry::RegisterMCRelocationInfo(*T, createX86MCRelocationInfo);
  }

  // Register the asm backend.
  TargetRegistry::RegisterMCAsmBackend(getTheX86_32Target(),
                                       createX86_32AsmBackend);
  TargetRegistry::RegisterMCAsmBackend(getTheX86_64Target(),
                                       createX86_64AsmBackend);
}

unsigned llvm::getX86SubSuperRegisterOrZero(unsigned Reg, unsigned Size,
                                            bool High) {
  switch (Size) {
  default: return 0;
  case 8:
    if (High) {
      switch (Reg) {
      default: return getX86SubSuperRegisterOrZero(Reg, 64);
      case X86::SIL: case X86::SI: case X86::ESI: case X86::RSI:
        return X86::SI;
      case X86::DIL: case X86::DI: case X86::EDI: case X86::RDI:
        return X86::DI;
      case X86::BPL: case X86::BP: case X86::EBP: case X86::RBP:
        return X86::BP;
      case X86::SPL: case X86::SP: case X86::ESP: case X86::RSP:
        return X86::SP;
      case X86::AH: case X86::AL: case X86::AX: case X86::EAX: case X86::RAX:
        return X86::AH;
      case X86::DH: case X86::DL: case X86::DX: case X86::EDX: case X86::RDX:
        return X86::DH;
      case X86::CH: case X86::CL: case X86::CX: case X86::ECX: case X86::RCX:
        return X86::CH;
      case X86::BH: case X86::BL: case X86::BX: case X86::EBX: case X86::RBX:
        return X86::BH;
      }
    } else {
      switch (Reg) {
      default: return 0;
      case X86::AH: case X86::AL: case X86::AX: case X86::EAX: case X86::RAX:
        return X86::AL;
      case X86::DH: case X86::DL: case X86::DX: case X86::EDX: case X86::RDX:
        return X86::DL;
      case X86::CH: case X86::CL: case X86::CX: case X86::ECX: case X86::RCX:
        return X86::CL;
      case X86::BH: case X86::BL: case X86::BX: case X86::EBX: case X86::RBX:
        return X86::BL;
      case X86::SIL: case X86::SI: case X86::ESI: case X86::RSI:
        return X86::SIL;
      case X86::DIL: case X86::DI: case X86::EDI: case X86::RDI:
        return X86::DIL;
      case X86::BPL: case X86::BP: case X86::EBP: case X86::RBP:
        return X86::BPL;
      case X86::SPL: case X86::SP: case X86::ESP: case X86::RSP:
        return X86::SPL;
      case X86::R8B: case X86::R8W: case X86::R8D: case X86::R8:
        return X86::R8B;
      case X86::R9B: case X86::R9W: case X86::R9D: case X86::R9:
        return X86::R9B;
      case X86::R10B: case X86::R10W: case X86::R10D: case X86::R10:
        return X86::R10B;
      case X86::R11B: case X86::R11W: case X86::R11D: case X86::R11:
        return X86::R11B;
      case X86::R12B: case X86::R12W: case X86::R12D: case X86::R12:
        return X86::R12B;
      case X86::R13B: case X86::R13W: case X86::R13D: case X86::R13:
        return X86::R13B;
      case X86::R14B: case X86::R14W: case X86::R14D: case X86::R14:
        return X86::R14B;
      case X86::R15B: case X86::R15W: case X86::R15D: case X86::R15:
        return X86::R15B;
      }
    }
  case 16:
    switch (Reg) {
    default: return 0;
    case X86::AH: case X86::AL: case X86::AX: case X86::EAX: case X86::RAX:
      return X86::AX;
    case X86::DH: case X86::DL: case X86::DX: case X86::EDX: case X86::RDX:
      return X86::DX;
    case X86::CH: case X86::CL: case X86::CX: case X86::ECX: case X86::RCX:
      return X86::CX;
    case X86::BH: case X86::BL: case X86::BX: case X86::EBX: case X86::RBX:
      return X86::BX;
    case X86::SIL: case X86::SI: case X86::ESI: case X86::RSI:
      return X86::SI;
    case X86::DIL: case X86::DI: case X86::EDI: case X86::RDI:
      return X86::DI;
    case X86::BPL: case X86::BP: case X86::EBP: case X86::RBP:
      return X86::BP;
    case X86::SPL: case X86::SP: case X86::ESP: case X86::RSP:
      return X86::SP;
    case X86::R8B: case X86::R8W: case X86::R8D: case X86::R8:
      return X86::R8W;
    case X86::R9B: case X86::R9W: case X86::R9D: case X86::R9:
      return X86::R9W;
    case X86::R10B: case X86::R10W: case X86::R10D: case X86::R10:
      return X86::R10W;
    case X86::R11B: case X86::R11W: case X86::R11D: case X86::R11:
      return X86::R11W;
    case X86::R12B: case X86::R12W: case X86::R12D: case X86::R12:
      return X86::R12W;
    case X86::R13B: case X86::R13W: case X86::R13D: case X86::R13:
      return X86::R13W;
    case X86::R14B: case X86::R14W: case X86::R14D: case X86::R14:
      return X86::R14W;
    case X86::R15B: case X86::R15W: case X86::R15D: case X86::R15:
      return X86::R15W;
    }
  case 32:
    switch (Reg) {
    default: return 0;
    case X86::AH: case X86::AL: case X86::AX: case X86::EAX: case X86::RAX:
      return X86::EAX;
    case X86::DH: case X86::DL: case X86::DX: case X86::EDX: case X86::RDX:
      return X86::EDX;
    case X86::CH: case X86::CL: case X86::CX: case X86::ECX: case X86::RCX:
      return X86::ECX;
    case X86::BH: case X86::BL: case X86::BX: case X86::EBX: case X86::RBX:
      return X86::EBX;
    case X86::SIL: case X86::SI: case X86::ESI: case X86::RSI:
      return X86::ESI;
    case X86::DIL: case X86::DI: case X86::EDI: case X86::RDI:
      return X86::EDI;
    case X86::BPL: case X86::BP: case X86::EBP: case X86::RBP:
      return X86::EBP;
    case X86::SPL: case X86::SP: case X86::ESP: case X86::RSP:
      return X86::ESP;
    case X86::R8B: case X86::R8W: case X86::R8D: case X86::R8:
      return X86::R8D;
    case X86::R9B: case X86::R9W: case X86::R9D: case X86::R9:
      return X86::R9D;
    case X86::R10B: case X86::R10W: case X86::R10D: case X86::R10:
      return X86::R10D;
    case X86::R11B: case X86::R11W: case X86::R11D: case X86::R11:
      return X86::R11D;
    case X86::R12B: case X86::R12W: case X86::R12D: case X86::R12:
      return X86::R12D;
    case X86::R13B: case X86::R13W: case X86::R13D: case X86::R13:
      return X86::R13D;
    case X86::R14B: case X86::R14W: case X86::R14D: case X86::R14:
      return X86::R14D;
    case X86::R15B: case X86::R15W: case X86::R15D: case X86::R15:
      return X86::R15D;
    }
  case 64:
    switch (Reg) {
    default: return 0;
    case X86::AH: case X86::AL: case X86::AX: case X86::EAX: case X86::RAX:
      return X86::RAX;
    case X86::DH: case X86::DL: case X86::DX: case X86::EDX: case X86::RDX:
      return X86::RDX;
    case X86::CH: case X86::CL: case X86::CX: case X86::ECX: case X86::RCX:
      return X86::RCX;
    case X86::BH: case X86::BL: case X86::BX: case X86::EBX: case X86::RBX:
      return X86::RBX;
    case X86::SIL: case X86::SI: case X86::ESI: case X86::RSI:
      return X86::RSI;
    case X86::DIL: case X86::DI: case X86::EDI: case X86::RDI:
      return X86::RDI;
    case X86::BPL: case X86::BP: case X86::EBP: case X86::RBP:
      return X86::RBP;
    case X86::SPL: case X86::SP: case X86::ESP: case X86::RSP:
      return X86::RSP;
    case X86::R8B: case X86::R8W: case X86::R8D: case X86::R8:
      return X86::R8;
    case X86::R9B: case X86::R9W: case X86::R9D: case X86::R9:
      return X86::R9;
    case X86::R10B: case X86::R10W: case X86::R10D: case X86::R10:
      return X86::R10;
    case X86::R11B: case X86::R11W: case X86::R11D: case X86::R11:
      return X86::R11;
    case X86::R12B: case X86::R12W: case X86::R12D: case X86::R12:
      return X86::R12;
    case X86::R13B: case X86::R13W: case X86::R13D: case X86::R13:
      return X86::R13;
    case X86::R14B: case X86::R14W: case X86::R14D: case X86::R14:
      return X86::R14;
    case X86::R15B: case X86::R15W: case X86::R15D: case X86::R15:
      return X86::R15;
    }
  }
}

unsigned llvm::getX86SubSuperRegister(unsigned Reg, unsigned Size, bool High) {
  unsigned Res = getX86SubSuperRegisterOrZero(Reg, Size, High);
  assert(Res != 0 && "Unexpected register or VT");
  return Res;
}


