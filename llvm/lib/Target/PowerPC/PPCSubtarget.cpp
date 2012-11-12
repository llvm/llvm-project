//===-- PowerPCSubtarget.cpp - PPC Subtarget Information ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PPC specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "PPCSubtarget.h"
#include "PPCRegisterInfo.h"
#include "PPC.h"
#include "llvm/GlobalValue.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include <cstdlib>

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "PPCGenSubtargetInfo.inc"

using namespace llvm;

PPCSubtarget::PPCSubtarget(const std::string &TT, const std::string &CPU,
                           const std::string &FS, bool is64Bit)
  : PPCGenSubtargetInfo(TT, CPU, FS)
  , StackAlignment(16)
  , DarwinDirective(PPC::DIR_NONE)
  , HasMFOCRF(false)
  , Has64BitSupport(false)
  , Use64BitRegs(false)
  , IsPPC64(is64Bit)
  , HasAltivec(false)
  , HasFSQRT(false)
  , HasSTFIWX(false)
  , HasISEL(false)
  , IsBookE(false)
  , HasLazyResolverStubs(false)
  , IsJITCodeModel(false)
  , TargetTriple(TT) {

  // Determine default and user specified characteristics
  std::string CPUName = CPU;
  if (CPUName.empty())
    CPUName = "generic";
#if (defined(__APPLE__) || defined(__linux__)) && \
    (defined(__ppc__) || defined(__powerpc__))
  if (CPUName == "generic")
    CPUName = sys::getHostCPUName();
#endif

  // Initialize scheduling itinerary for the specified CPU.
  InstrItins = getInstrItineraryForCPU(CPUName);

  // Make sure 64-bit features are available when CPUname is generic
  std::string FullFS = FS;

  // If we are generating code for ppc64, verify that options make sense.
  if (is64Bit) {
    Has64BitSupport = true;
    // Silently force 64-bit register use on ppc64.
    Use64BitRegs = true;
    if (!FullFS.empty())
      FullFS = "+64bit," + FullFS;
    else
      FullFS = "+64bit";
  }

  // Parse features string.
  ParseSubtargetFeatures(CPUName, FullFS);

  // If the user requested use of 64-bit regs, but the cpu selected doesn't
  // support it, ignore.
  if (use64BitRegs() && !has64BitSupport())
    Use64BitRegs = false;

  // Set up darwin-specific properties.
  if (isDarwin())
    HasLazyResolverStubs = true;
}

/// SetJITMode - This is called to inform the subtarget info that we are
/// producing code for the JIT.
void PPCSubtarget::SetJITMode() {
  // JIT mode doesn't want lazy resolver stubs, it knows exactly where
  // everything is.  This matters for PPC64, which codegens in PIC mode without
  // stubs.
  HasLazyResolverStubs = false;

  // Calls to external functions need to use indirect calls
  IsJITCodeModel = true;
}


/// hasLazyResolverStub - Return true if accesses to the specified global have
/// to go through a dyld lazy resolution stub.  This means that an extra load
/// is required to get the address of the global.
bool PPCSubtarget::hasLazyResolverStub(const GlobalValue *GV,
                                       const TargetMachine &TM) const {
  // We never have stubs if HasLazyResolverStubs=false or if in static mode.
  if (!HasLazyResolverStubs || TM.getRelocationModel() == Reloc::Static)
    return false;
  // If symbol visibility is hidden, the extra load is not needed if
  // the symbol is definitely defined in the current translation unit.
  bool isDecl = GV->isDeclaration() && !GV->isMaterializable();
  if (GV->hasHiddenVisibility() && !isDecl && !GV->hasCommonLinkage())
    return false;
  return GV->hasWeakLinkage() || GV->hasLinkOnceLinkage() ||
         GV->hasCommonLinkage() || isDecl;
}

bool PPCSubtarget::enablePostRAScheduler(
           CodeGenOpt::Level OptLevel,
           TargetSubtargetInfo::AntiDepBreakMode& Mode,
           RegClassVector& CriticalPathRCs) const {
  // FIXME: It would be best to use TargetSubtargetInfo::ANTIDEP_ALL here,
  // but we can't because we can't reassign the cr registers. There is a
  // dependence between the cr register and the RLWINM instruction used
  // to extract its value which the anti-dependency breaker can't currently
  // see. Maybe we should make a late-expanded pseudo to encode this dependency.
  // (the relevant code is in PPCDAGToDAGISel::SelectSETCC)

  Mode = TargetSubtargetInfo::ANTIDEP_CRITICAL;

  CriticalPathRCs.clear();

  if (isPPC64())
    CriticalPathRCs.push_back(&PPC::G8RCRegClass);
  else
    CriticalPathRCs.push_back(&PPC::GPRCRegClass);
    
  CriticalPathRCs.push_back(&PPC::F8RCRegClass);
  CriticalPathRCs.push_back(&PPC::VRRCRegClass);

  return OptLevel >= CodeGenOpt::Default;
}

