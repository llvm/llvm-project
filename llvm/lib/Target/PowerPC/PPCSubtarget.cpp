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
#include "PPC.h"
#include "PPCRegisterInfo.h"
#include "PPCTargetMachine.h"
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

static cl::opt<bool> QPXStackUnaligned("qpx-stack-unaligned",
  cl::desc("Even when QPX is enabled the stack is not 32-byte aligned"),
  cl::Hidden);

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
    : PPCGenSubtargetInfo(TT, CPU, FS), TargetTriple(TT),
      IsPPC64(TargetTriple.getArch() == Triple::ppc64 ||
              TargetTriple.getArch() == Triple::ppc64le),
      TM(TM), FrameLowering(initializeSubtargetDependencies(CPU, FS)),
      InstrInfo(*this), TLInfo(TM, *this) {}

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
  HasQPX = false;
  HasVSX = false;
  NeedsTwoConstNR = false;
  HasP8Vector = false;
  HasP8Altivec = false;
  HasP8Crypto = false;
  HasP9Vector = false;
  HasP9Altivec = false;
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
  DeprecatedDST = false;
  HasLazyResolverStubs = false;
  HasICBT = false;
  HasInvariantFunctionDescriptors = false;
  HasPartwordAtomics = false;
  HasDirectMove = false;
  IsQPXStackUnaligned = false;
  HasHTM = false;
  HasFloat128 = false;
  IsISA3_0 = false;
  UseLongCalls = false;
  SecurePlt = false;
  VectorsUseTwoUnits = false;
  UsePPCPreRASchedStrategy = false;
  UsePPCPostRASchedStrategy = false;

  HasPOPCNTD = POPCNTD_Unavailable;
}

void PPCSubtarget::initSubtargetFeatures(StringRef CPU, StringRef FS) {
  // Determine default and user specified characteristics
  std::string CPUName = CPU;
  if (CPUName.empty() || CPU == "generic") {
    // If cross-compiling with -march=ppc64le without -mcpu
    if (TargetTriple.getArch() == Triple::ppc64le)
      CPUName = "ppc64le";
    else
      CPUName = "generic";
  }

  // Initialize scheduling itinerary for the specified CPU.
  InstrItins = getInstrItineraryForCPU(CPUName);

  // Parse features string.
  ParseSubtargetFeatures(CPUName, FS);

  // If the user requested use of 64-bit regs, but the cpu selected doesn't
  // support it, ignore.
  if (IsPPC64 && has64BitSupport())
    Use64BitRegs = true;

  // Set up darwin-specific properties.
  if (isDarwin())
    HasLazyResolverStubs = true;

  if ((TargetTriple.isOSFreeBSD() && TargetTriple.getOSMajorVersion() >= 13) ||
      TargetTriple.isOSNetBSD() || TargetTriple.isOSOpenBSD() ||
      TargetTriple.isMusl())
    SecurePlt = true;

  if (HasSPE && IsPPC64)
    report_fatal_error( "SPE is only supported for 32-bit targets.\n", false);
  if (HasSPE && (HasAltivec || HasQPX || HasVSX || HasFPU))
    report_fatal_error(
        "SPE and traditional floating point cannot both be enabled.\n", false);

  // If not SPE, set standard FPU
  if (!HasSPE)
    HasFPU = true;

  // QPX requires a 32-byte aligned stack. Note that we need to do this if
  // we're compiling for a BG/Q system regardless of whether or not QPX
  // is enabled because external functions will assume this alignment.
  IsQPXStackUnaligned = QPXStackUnaligned;
  StackAlignment = getPlatformStackAlignment();

  // Determine endianness.
  // FIXME: Part of the TargetMachine.
  IsLittleEndian = (TargetTriple.getArch() == Triple::ppc64le);
}

/// Return true if accesses to the specified global have to go through a dyld
/// lazy resolution stub.  This means that an extra load is required to get the
/// address of the global.
bool PPCSubtarget::hasLazyResolverStub(const GlobalValue *GV) const {
  if (!HasLazyResolverStubs)
    return false;
  if (!TM.shouldAssumeDSOLocal(*GV->getParent(), GV))
    return true;
  // 32 bit macho has no relocation for a-b if a is undefined, even if b is in
  // the section that is being relocated. This means we have to use o load even
  // for GVs that are known to be local to the dso.
  if (GV->isDeclarationForLinker() || GV->hasCommonLinkage())
    return true;
  return false;
}

bool PPCSubtarget::enableMachineScheduler() const { return true; }

bool PPCSubtarget::enableMachinePipeliner() const {
  return (CPUDirective == PPC::DIR_PWR9) && EnableMachinePipeliner;
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
