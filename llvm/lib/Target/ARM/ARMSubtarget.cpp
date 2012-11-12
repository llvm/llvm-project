//===-- ARMSubtarget.cpp - ARM Subtarget Information ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ARM specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "ARMSubtarget.h"
#include "ARMBaseRegisterInfo.h"
#include "ARMBaseInstrInfo.h"
#include "llvm/GlobalValue.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/CommandLine.h"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "ARMGenSubtargetInfo.inc"

using namespace llvm;

static cl::opt<bool>
ReserveR9("arm-reserve-r9", cl::Hidden,
          cl::desc("Reserve R9, making it unavailable as GPR"));

static cl::opt<bool>
DarwinUseMOVT("arm-darwin-use-movt", cl::init(true), cl::Hidden);

static cl::opt<bool>
UseFusedMulOps("arm-use-mulops",
               cl::init(true), cl::Hidden);

static cl::opt<bool>
StrictAlign("arm-strict-align", cl::Hidden,
            cl::desc("Disallow all unaligned memory accesses"));

ARMSubtarget::ARMSubtarget(const std::string &TT, const std::string &CPU,
                           const std::string &FS)
  : ARMGenSubtargetInfo(TT, CPU, FS)
  , ARMProcFamily(Others)
  , HasV4TOps(false)
  , HasV5TOps(false)
  , HasV5TEOps(false)
  , HasV6Ops(false)
  , HasV6T2Ops(false)
  , HasV7Ops(false)
  , HasVFPv2(false)
  , HasVFPv3(false)
  , HasVFPv4(false)
  , HasNEON(false)
  , UseNEONForSinglePrecisionFP(false)
  , UseMulOps(UseFusedMulOps)
  , SlowFPVMLx(false)
  , HasVMLxForwarding(false)
  , SlowFPBrcc(false)
  , InThumbMode(false)
  , HasThumb2(false)
  , IsMClass(false)
  , NoARM(false)
  , PostRAScheduler(false)
  , IsR9Reserved(ReserveR9)
  , UseMovt(false)
  , SupportsTailCall(false)
  , HasFP16(false)
  , HasD16(false)
  , HasHardwareDivide(false)
  , HasHardwareDivideInARM(false)
  , HasT2ExtractPack(false)
  , HasDataBarrier(false)
  , Pref32BitThumb(false)
  , AvoidCPSRPartialUpdate(false)
  , HasRAS(false)
  , HasMPExtension(false)
  , FPOnlySP(false)
  , AllowsUnalignedMem(false)
  , Thumb2DSP(false)
  , stackAlignment(4)
  , CPUString(CPU)
  , TargetTriple(TT)
  , TargetABI(ARM_ABI_APCS) {
  // Determine default and user specified characteristics
  if (CPUString.empty())
    CPUString = "generic";

  // Insert the architecture feature derived from the target triple into the
  // feature string. This is important for setting features that are implied
  // based on the architecture version.
  std::string ArchFS = ARM_MC::ParseARMTriple(TT, CPUString);
  if (!FS.empty()) {
    if (!ArchFS.empty())
      ArchFS = ArchFS + "," + FS;
    else
      ArchFS = FS;
  }
  ParseSubtargetFeatures(CPUString, ArchFS);

  // Thumb2 implies at least V6T2. FIXME: Fix tests to explicitly specify a
  // ARM version or CPU and then remove this.
  if (!HasV6T2Ops && hasThumb2())
    HasV4TOps = HasV5TOps = HasV5TEOps = HasV6Ops = HasV6T2Ops = true;

  // Keep a pointer to static instruction cost data for the specified CPU.
  SchedModel = getSchedModelForCPU(CPUString);

  // Initialize scheduling itinerary for the specified CPU.
  InstrItins = getInstrItineraryForCPU(CPUString);

  if ((TT.find("eabi") != std::string::npos) || (isTargetIOS() && isMClass()))
    // FIXME: We might want to separate AAPCS and EABI. Some systems, e.g.
    // Darwin-EABI conforms to AACPS but not the rest of EABI.
    TargetABI = ARM_ABI_AAPCS;

  if (isAAPCS_ABI())
    stackAlignment = 8;

  if (!isTargetIOS())
    UseMovt = hasV6T2Ops();
  else {
    IsR9Reserved = ReserveR9 | !HasV6Ops;
    UseMovt = DarwinUseMOVT && hasV6T2Ops();
    SupportsTailCall = !getTargetTriple().isOSVersionLT(5, 0);
  }

  if (!isThumb() || hasThumb2())
    PostRAScheduler = true;

  // v6+ may or may not support unaligned mem access depending on the system
  // configuration.
  if (!StrictAlign && hasV6Ops() && isTargetDarwin())
    AllowsUnalignedMem = true;
}

/// GVIsIndirectSymbol - true if the GV will be accessed via an indirect symbol.
bool
ARMSubtarget::GVIsIndirectSymbol(const GlobalValue *GV,
                                 Reloc::Model RelocM) const {
  if (RelocM == Reloc::Static)
    return false;

  // Materializable GVs (in JIT lazy compilation mode) do not require an extra
  // load from stub.
  bool isDecl = GV->hasAvailableExternallyLinkage();
  if (GV->isDeclaration() && !GV->isMaterializable())
    isDecl = true;

  if (!isTargetDarwin()) {
    // Extra load is needed for all externally visible.
    if (GV->hasLocalLinkage() || GV->hasHiddenVisibility())
      return false;
    return true;
  } else {
    if (RelocM == Reloc::PIC_) {
      // If this is a strong reference to a definition, it is definitely not
      // through a stub.
      if (!isDecl && !GV->isWeakForLinker())
        return false;

      // Unless we have a symbol with hidden visibility, we have to go through a
      // normal $non_lazy_ptr stub because this symbol might be resolved late.
      if (!GV->hasHiddenVisibility())  // Non-hidden $non_lazy_ptr reference.
        return true;

      // If symbol visibility is hidden, we have a stub for common symbol
      // references and external declarations.
      if (isDecl || GV->hasCommonLinkage())
        // Hidden $non_lazy_ptr reference.
        return true;

      return false;
    } else {
      // If this is a strong reference to a definition, it is definitely not
      // through a stub.
      if (!isDecl && !GV->isWeakForLinker())
        return false;

      // Unless we have a symbol with hidden visibility, we have to go through a
      // normal $non_lazy_ptr stub because this symbol might be resolved late.
      if (!GV->hasHiddenVisibility())  // Non-hidden $non_lazy_ptr reference.
        return true;
    }
  }

  return false;
}

unsigned ARMSubtarget::getMispredictionPenalty() const {
  return SchedModel->MispredictPenalty;
}

bool ARMSubtarget::enablePostRAScheduler(
           CodeGenOpt::Level OptLevel,
           TargetSubtargetInfo::AntiDepBreakMode& Mode,
           RegClassVector& CriticalPathRCs) const {
  Mode = TargetSubtargetInfo::ANTIDEP_CRITICAL;
  CriticalPathRCs.clear();
  CriticalPathRCs.push_back(&ARM::GPRRegClass);
  return PostRAScheduler && OptLevel >= CodeGenOpt::Default;
}
