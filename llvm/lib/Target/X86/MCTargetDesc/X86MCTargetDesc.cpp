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
#include "X86MCAsmInfo.h"
#include "llvm/ADT/Triple.h"
#include "llvm/MC/MCCodeGenInfo.h"
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

void X86_MC::InitLLVM2SEHRegisterMapping(MCRegisterInfo *MRI) {
  // FIXME: TableGen these.
  for (unsigned Reg = X86::NoRegister+1; Reg < X86::NUM_TARGET_REGS; ++Reg) {
    unsigned SEH = MRI->getEncodingValue(Reg);
    MRI->mapLLVMRegToSEHReg(Reg, SEH);
  }
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
  X86_MC::InitLLVM2SEHRegisterMapping(X);
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

static MCCodeGenInfo *createX86MCCodeGenInfo(const Triple &TT, Reloc::Model RM,
                                             CodeModel::Model CM,
                                             CodeGenOpt::Level OL) {
  MCCodeGenInfo *X = new MCCodeGenInfo();

  bool is64Bit = TT.getArch() == Triple::x86_64;

  if (RM == Reloc::Default) {
    // Darwin defaults to PIC in 64 bit mode and dynamic-no-pic in 32 bit mode.
    // Win64 requires rip-rel addressing, thus we force it to PIC. Otherwise we
    // use static relocation model by default.
    if (TT.isOSDarwin()) {
      if (is64Bit)
        RM = Reloc::PIC_;
      else
        RM = Reloc::DynamicNoPIC;
    } else if (TT.isOSWindows() && is64Bit)
      RM = Reloc::PIC_;
    else
      RM = Reloc::Static;
  }

  // ELF and X86-64 don't have a distinct DynamicNoPIC model.  DynamicNoPIC
  // is defined as a model for code which may be used in static or dynamic
  // executables but not necessarily a shared library. On X86-32 we just
  // compile in -static mode, in x86-64 we use PIC.
  if (RM == Reloc::DynamicNoPIC) {
    if (is64Bit)
      RM = Reloc::PIC_;
    else if (!TT.isOSDarwin())
      RM = Reloc::Static;
  }

  // If we are on Darwin, disallow static relocation model in X86-64 mode, since
  // the Mach-O file format doesn't support it.
  if (RM == Reloc::Static && TT.isOSDarwin() && is64Bit)
    RM = Reloc::PIC_;

  // For static codegen, if we're not already set, use Small codegen.
  if (CM == CodeModel::Default)
    CM = CodeModel::Small;
  else if (CM == CodeModel::JITDefault)
    // 64-bit JIT places everything in the same buffer except external funcs.
    CM = is64Bit ? CodeModel::Large : CodeModel::Small;

  X->initMCCodeGenInfo(RM, CM, OL);
  return X;
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
  if (TheTriple.isOSBinFormatMachO() && TheTriple.getArch() == Triple::x86_64)
    return createX86_64MachORelocationInfo(Ctx);
  else if (TheTriple.isOSBinFormatELF())
    return createX86_64ELFRelocationInfo(Ctx);
  // Default to the stock relocation info.
  return llvm::createMCRelocationInfo(TheTriple, Ctx);
}

static MCInstrAnalysis *createX86MCInstrAnalysis(const MCInstrInfo *Info) {
  return new MCInstrAnalysis(Info);
}

// Force static initialization.
extern "C" void LLVMInitializeX86TargetMC() {
  for (Target *T : {&TheX86_32Target, &TheX86_64Target}) {
    // Register the MC asm info.
    RegisterMCAsmInfoFn X(*T, createX86MCAsmInfo);

    // Register the MC codegen info.
    RegisterMCCodeGenInfoFn Y(*T, createX86MCCodeGenInfo);

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

    // Register the object streamer.
    TargetRegistry::RegisterCOFFStreamer(*T, createX86WinCOFFStreamer);

    // Register the MCInstPrinter.
    TargetRegistry::RegisterMCInstPrinter(*T, createX86MCInstPrinter);

    // Register the MC relocation info.
    TargetRegistry::RegisterMCRelocationInfo(*T, createX86MCRelocationInfo);
  }

  // Register the asm backend.
  TargetRegistry::RegisterMCAsmBackend(TheX86_32Target,
                                       createX86_32AsmBackend);
  TargetRegistry::RegisterMCAsmBackend(TheX86_64Target,
                                       createX86_64AsmBackend);
}
