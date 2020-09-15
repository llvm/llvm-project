//===-- PowerPCSubtarget.cpp - PPC Subtarget Information ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the PPC specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "PPCSubtarget.h"
#include "GISel/PPCCallLowering.h"
#include "GISel/PPCLegalizerInfo.h"
#include "GISel/PPCRegisterBankInfo.h"
#include "PPC.h"
#include "PPCRegisterInfo.h"
#include "PPCTargetMachine.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelect.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include <cstdlib>

using namespace llvm;

#define DEBUG_TYPE "ppc-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "PPCGenSubtargetInfo.inc"

static cl::opt<bool> UseSubRegLiveness("ppc-track-subreg-liveness",
cl::desc("Enable subregister liveness tracking for PPC"), cl::Hidden);

static cl::opt<bool>
    EnableMachinePipeliner("ppc-enable-pipeliner",
                           cl::desc("Enable Machine Pipeliner for PPC"),
                           cl::init(false), cl::Hidden);

PPCSubtarget &PPCSubtarget::initializeSubtargetDependencies(StringRef CPU,
                                                            StringRef FS) {
  initializeEnvironment();
  initSubtargetFeatures(CPU, FS);
  return *this;
}

PPCSubtarget::PPCSubtarget(const Triple &TT, const std::string &CPU,
                           const std::string &FS, const PPCTargetMachine &TM)
    : PPCGenSubtargetInfo(TT, CPU, /*TuneCPU*/ CPU, FS), TargetTriple(TT),
      IsPPC64(TargetTriple.getArch() == Triple::ppc64 ||
              TargetTriple.getArch() == Triple::ppc64le),
      TM(TM), FrameLowering(initializeSubtargetDependencies(CPU, FS)),
      InstrInfo(*this), TLInfo(TM, *this) {
  CallLoweringInfo.reset(new PPCCallLowering(*getTargetLowering()));
  Legalizer.reset(new PPCLegalizerInfo(*this));
  auto *RBI = new PPCRegisterBankInfo(*getRegisterInfo());
  RegBankInfo.reset(RBI);

  InstSelector.reset(createPPCInstructionSelector(
      *static_cast<const PPCTargetMachine *>(&TM), *this, *RBI));
}

void PPCSubtarget::initializeEnvironment() {
  StackAlignment = Align(16);
  CPUDirective = PPC::DIR_NONE;
  HasMFOCRF = false;
  Has64BitSupport = false;
  Use64BitRegs = false;
  UseCRBits = false;
  HasHardFloat = false;
  HasAltivec = false;
  HasSPE = false;
  HasFPU = false;
  HasVSX = false;
  NeedsTwoConstNR = false;
  HasP8Vector = false;
  HasP8Altivec = false;
  HasP8Crypto = false;
  HasP9Vector = false;
  HasP9Altivec = false;
  HasMMA = false;
  HasP10Vector = false;
  HasPrefixInstrs = false;
  HasPCRelativeMemops = false;
  HasFCPSGN = false;
  HasFSQRT = false;
  HasFRE = false;
  HasFRES = false;
  HasFRSQRTE = false;
  HasFRSQRTES = false;
  HasRecipPrec = false;
  HasSTFIWX = false;
  HasLFIWAX = false;
  HasFPRND = false;
  HasFPCVT = false;
  HasISEL = false;
  HasBPERMD = false;
  HasExtDiv = false;
  HasCMPB = false;
  HasLDBRX = false;
  IsBookE = false;
  HasOnlyMSYNC = false;
  IsPPC4xx = false;
  IsPPC6xx = false;
  IsE500 = false;
  FeatureMFTB = false;
  AllowsUnalignedFPAccess = false;
  DeprecatedDST = false;
  HasICBT = false;
  HasInvariantFunctionDescriptors = false;
  HasPartwordAtomics = false;
  HasDirectMove = false;
  HasHTM = false;
  HasFloat128 = false;
  HasFusion = false;
  HasStoreFusion = false;
  HasAddiLoadFusion = false;
  HasAddisLoadFusion = false;
  IsISA3_0 = false;
  IsISA3_1 = false;
  UseLongCalls = false;
  SecurePlt = false;
  VectorsUseTwoUnits = false;
  UsePPCPreRASchedStrategy = false;
  UsePPCPostRASchedStrategy = false;
  PairedVectorMemops = false;
  PredictableSelectIsExpensive = false;

  HasPOPCNTD = POPCNTD_Unavailable;
}

void PPCSubtarget::initSubtargetFeatures(StringRef CPU, StringRef FS) {
  // Determine default and user specified characteristics
  std::string CPUName = std::string(CPU);
  if (CPUName.empty() || CPU == "generic") {
    // If cross-compiling with -march=ppc64le without -mcpu
    if (TargetTriple.getArch() == Triple::ppc64le)
      CPUName = "ppc64le";
    else if (TargetTriple.getSubArch() == Triple::PPCSubArch_spe)
      CPUName = "e500";
    else
      CPUName = "generic";
  }

  // Initialize scheduling itinerary for the specified CPU.
  InstrItins = getInstrItineraryForCPU(CPUName);

  // Parse features string.
  ParseSubtargetFeatures(CPUName, /*TuneCPU*/ CPUName, FS);

  // If the user requested use of 64-bit regs, but the cpu selected doesn't
  // support it, ignore.
  if (IsPPC64 && has64BitSupport())
    Use64BitRegs = true;

  if ((TargetTriple.isOSFreeBSD() && TargetTriple.getOSMajorVersion() >= 13) ||
      TargetTriple.isOSNetBSD() || TargetTriple.isOSOpenBSD() ||
      TargetTriple.isMusl())
    SecurePlt = true;

  if (HasSPE && IsPPC64)
    report_fatal_error( "SPE is only supported for 32-bit targets.\n", false);
  if (HasSPE && (HasAltivec || HasVSX || HasFPU))
    report_fatal_error(
        "SPE and traditional floating point cannot both be enabled.\n", false);

  // If not SPE, set standard FPU
  if (!HasSPE)
    HasFPU = true;

  StackAlignment = getPlatformStackAlignment();

  // Determine endianness.
  // FIXME: Part of the TargetMachine.
  IsLittleEndian = (TargetTriple.getArch() == Triple::ppc64le);
}

bool PPCSubtarget::enableMachineScheduler() const { return true; }

bool PPCSubtarget::enableMachinePipeliner() const {
  return getSchedModel().hasInstrSchedModel() && EnableMachinePipeliner;
}

bool PPCSubtarget::useDFAforSMS() const { return false; }

// This overrides the PostRAScheduler bit in the SchedModel for each CPU.
bool PPCSubtarget::enablePostRAScheduler() const { return true; }

PPCGenSubtargetInfo::AntiDepBreakMode PPCSubtarget::getAntiDepBreakMode() const {
  return TargetSubtargetInfo::ANTIDEP_ALL;
}

void PPCSubtarget::getCriticalPathRCs(RegClassVector &CriticalPathRCs) const {
  CriticalPathRCs.clear();
  CriticalPathRCs.push_back(isPPC64() ?
                            &PPC::G8RCRegClass : &PPC::GPRCRegClass);
}

void PPCSubtarget::overrideSchedPolicy(MachineSchedPolicy &Policy,
                                       unsigned NumRegionInstrs) const {
  // The GenericScheduler that we use defaults to scheduling bottom up only.
  // We want to schedule from both the top and the bottom and so we set
  // OnlyBottomUp to false.
  // We want to do bi-directional scheduling since it provides a more balanced
  // schedule leading to better performance.
  Policy.OnlyBottomUp = false;
  // Spilling is generally expensive on all PPC cores, so always enable
  // register-pressure tracking.
  Policy.ShouldTrackPressure = true;
}

bool PPCSubtarget::useAA() const {
  return true;
}

bool PPCSubtarget::enableSubRegLiveness() const {
  return UseSubRegLiveness;
}

bool PPCSubtarget::isGVIndirectSymbol(const GlobalValue *GV) const {
  // Large code model always uses the TOC even for local symbols.
  if (TM.getCodeModel() == CodeModel::Large)
    return true;
  if (TM.shouldAssumeDSOLocal(*GV->getParent(), GV))
    return false;
  return true;
}

bool PPCSubtarget::isELFv2ABI() const { return TM.isELFv2ABI(); }
bool PPCSubtarget::isPPC64() const { return TM.isPPC64(); }

bool PPCSubtarget::isUsingPCRelativeCalls() const {
  return isPPC64() && hasPCRelativeMemops() && isELFv2ABI() &&
         CodeModel::Medium == getTargetMachine().getCodeModel();
}

// GlobalISEL
const CallLowering *PPCSubtarget::getCallLowering() const {
  return CallLoweringInfo.get();
}

const RegisterBankInfo *PPCSubtarget::getRegBankInfo() const {
  return RegBankInfo.get();
}

const LegalizerInfo *PPCSubtarget::getLegalizerInfo() const {
  return Legalizer.get();
}

InstructionSelector *PPCSubtarget::getInstructionSelector() const {
  return InstSelector.get();
}
