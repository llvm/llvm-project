//===-- MipsSubtarget.cpp - Mips Subtarget Information --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Mips specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "MipsMachineFunction.h"
#include "Mips.h"
#include "MipsRegisterInfo.h"
#include "MipsSubtarget.h"
#include "MipsTargetMachine.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "mips-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "MipsGenSubtargetInfo.inc"

// FIXME: Maybe this should be on by default when Mips16 is specified
//
static cl::opt<bool> Mixed16_32(
  "mips-mixed-16-32",
  cl::init(false),
  cl::desc("Allow for a mixture of Mips16 "
           "and Mips32 code in a single source file"),
  cl::Hidden);

static cl::opt<bool> Mips_Os16(
  "mips-os16",
  cl::init(false),
  cl::desc("Compile all functions that don' use "
           "floating point as Mips 16"),
  cl::Hidden);

static cl::opt<bool>
Mips16HardFloat("mips16-hard-float", cl::NotHidden,
                cl::desc("MIPS: mips16 hard float enable."),
                cl::init(false));

static cl::opt<bool>
Mips16ConstantIslands(
  "mips16-constant-islands", cl::NotHidden,
  cl::desc("MIPS: mips16 constant islands enable."),
  cl::init(true));

/// Select the Mips CPU for the given triple and cpu name.
/// FIXME: Merge with the copy in MipsMCTargetDesc.cpp
static StringRef selectMipsCPU(Triple TT, StringRef CPU) {
  if (CPU.empty() || CPU == "generic") {
    if (TT.getArch() == Triple::mips || TT.getArch() == Triple::mipsel)
      CPU = "mips32";
    else
      CPU = "mips64";
  }
  return CPU;
}

void MipsSubtarget::anchor() { }

static std::string computeDataLayout(const MipsSubtarget &ST) {
  std::string Ret = "";

  // There are both little and big endian mips.
  if (ST.isLittle())
    Ret += "e";
  else
    Ret += "E";

  Ret += "-m:m";

  // Pointers are 32 bit on some ABIs.
  if (!ST.isABI_N64())
    Ret += "-p:32:32";

  // 8 and 16 bit integers only need no have natural alignment, but try to
  // align them to 32 bits. 64 bit integers have natural alignment.
  Ret += "-i8:8:32-i16:16:32-i64:64";

  // 32 bit registers are always available and the stack is at least 64 bit
  // aligned. On N64 64 bit registers are also available and the stack is
  // 128 bit aligned.
  if (ST.isABI_N64() || ST.isABI_N32())
    Ret += "-n32:64-S128";
  else
    Ret += "-n32-S64";

  return Ret;
}

MipsSubtarget::MipsSubtarget(const std::string &TT, const std::string &CPU,
                             const std::string &FS, bool little,
                             MipsTargetMachine *_TM)
    : MipsGenSubtargetInfo(TT, CPU, FS), MipsArchVersion(Mips32),
      MipsABI(UnknownABI), IsLittle(little), IsSingleFloat(false),
      IsFPXX(false), NoABICalls(false), IsFP64bit(false), UseOddSPReg(true),
      IsNaN2008bit(false), IsGP64bit(false), HasVFPU(false), HasCnMips(false),
      IsLinux(true), HasMips3_32(false), HasMips3_32r2(false),
      HasMips4_32(false), HasMips4_32r2(false), HasMips5_32r2(false),
      InMips16Mode(false), InMips16HardFloat(Mips16HardFloat),
      InMicroMipsMode(false), HasDSP(false), HasDSPR2(false),
      AllowMixed16_32(Mixed16_32 | Mips_Os16), Os16(Mips_Os16),
      HasMSA(false), TM(_TM), TargetTriple(TT),
      DL(computeDataLayout(initializeSubtargetDependencies(CPU, FS, TM))),
      TSInfo(DL), JITInfo(), InstrInfo(MipsInstrInfo::create(*this)),
      FrameLowering(MipsFrameLowering::create(*this)),
      TLInfo(MipsTargetLowering::create(*TM, *this)) {

  PreviousInMips16Mode = InMips16Mode;

  // Don't even attempt to generate code for MIPS-I, MIPS-II, MIPS-III, and
  // MIPS-V. They have not been tested and currently exist for the integrated
  // assembler only.
  if (MipsArchVersion == Mips1)
    report_fatal_error("Code generation for MIPS-I is not implemented", false);
  if (MipsArchVersion == Mips2)
    report_fatal_error("Code generation for MIPS-II is not implemented", false);
  if (MipsArchVersion == Mips3)
    report_fatal_error("Code generation for MIPS-III is not implemented",
                       false);
  if (MipsArchVersion == Mips5)
    report_fatal_error("Code generation for MIPS-V is not implemented", false);

  // Assert exactly one ABI was chosen.
  assert(MipsABI != UnknownABI);
  assert((((getFeatureBits() & Mips::FeatureO32) != 0) +
          ((getFeatureBits() & Mips::FeatureEABI) != 0) +
          ((getFeatureBits() & Mips::FeatureN32) != 0) +
          ((getFeatureBits() & Mips::FeatureN64) != 0)) == 1);

  // Check if Architecture and ABI are compatible.
  assert(((!isGP64bit() && (isABI_O32() || isABI_EABI())) ||
          (isGP64bit() && (isABI_N32() || isABI_N64()))) &&
         "Invalid  Arch & ABI pair.");

  if (hasMSA() && !isFP64bit())
    report_fatal_error("MSA requires a 64-bit FPU register file (FR=1 mode). "
                       "See -mattr=+fp64.",
                       false);

  if (!isABI_O32() && !useOddSPReg())
    report_fatal_error("-mattr=+nooddspreg requires the O32 ABI.", false);

  if (IsFPXX && (isABI_N32() || isABI_N64()))
    report_fatal_error("FPXX is not permitted for the N32/N64 ABI's.", false);

  if (hasMips32r6()) {
    StringRef ISA = hasMips64r6() ? "MIPS64r6" : "MIPS32r6";

    assert(isFP64bit());
    assert(isNaN2008());
    if (hasDSP())
      report_fatal_error(ISA + " is not compatible with the DSP ASE", false);
  }

  // Is the target system Linux ?
  if (TT.find("linux") == std::string::npos)
    IsLinux = false;

  // Set UseSmallSection.
  // TODO: Investigate the IsLinux check. I suspect it's really checking for
  //       bare-metal.
  UseSmallSection = !IsLinux && (TM->getRelocationModel() == Reloc::Static);
}

/// This overrides the PostRAScheduler bit in the SchedModel for any CPU.
bool MipsSubtarget::enablePostMachineScheduler() const { return true; }

void MipsSubtarget::getCriticalPathRCs(RegClassVector &CriticalPathRCs) const {
  CriticalPathRCs.clear();
  CriticalPathRCs.push_back(isGP64bit() ?
                            &Mips::GPR64RegClass : &Mips::GPR32RegClass);
}

CodeGenOpt::Level MipsSubtarget::getOptLevelToEnablePostRAScheduler() const {
  return CodeGenOpt::Aggressive;
}

MipsSubtarget &
MipsSubtarget::initializeSubtargetDependencies(StringRef CPU, StringRef FS,
                                               const TargetMachine *TM) {
  std::string CPUName = selectMipsCPU(TargetTriple, CPU);
  
  // Parse features string.
  ParseSubtargetFeatures(CPUName, FS);
  // Initialize scheduling itinerary for the specified CPU.
  InstrItins = getInstrItineraryForCPU(CPUName);

  if (InMips16Mode && !TM->Options.UseSoftFloat)
    InMips16HardFloat = true;

  return *this;
}

bool MipsSubtarget::abiUsesSoftFloat() const {
  return TM->Options.UseSoftFloat && !InMips16HardFloat;
}

bool MipsSubtarget::useConstantIslands() {
  DEBUG(dbgs() << "use constant islands " << Mips16ConstantIslands << "\n");
  return Mips16ConstantIslands;
}

Reloc::Model MipsSubtarget::getRelocationModel() const {
  return TM->getRelocationModel();
}
