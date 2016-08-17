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
#include "ARMFrameLowering.h"
#include "ARMISelLowering.h"
#include "ARMInstrInfo.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMSelectionDAGInfo.h"
#include "ARMSubtarget.h"
#include "ARMTargetMachine.h"
#include "Thumb1FrameLowering.h"
#include "Thumb1InstrInfo.h"
#include "Thumb2InstrInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/TargetParser.h"

using namespace llvm;

#define DEBUG_TYPE "arm-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "ARMGenSubtargetInfo.inc"

static cl::opt<bool>
UseFusedMulOps("arm-use-mulops",
               cl::init(true), cl::Hidden);

enum ITMode {
  DefaultIT,
  RestrictedIT,
  NoRestrictedIT
};

static cl::opt<ITMode>
IT(cl::desc("IT block support"), cl::Hidden, cl::init(DefaultIT),
   cl::ZeroOrMore,
   cl::values(clEnumValN(DefaultIT, "arm-default-it",
                         "Generate IT block based on arch"),
              clEnumValN(RestrictedIT, "arm-restrict-it",
                         "Disallow deprecated IT based on ARMv8"),
              clEnumValN(NoRestrictedIT, "arm-no-restrict-it",
                         "Allow IT blocks based on ARMv7"),
              clEnumValEnd));

/// ForceFastISel - Use the fast-isel, even for subtargets where it is not
/// currently supported (for testing only).
static cl::opt<bool>
ForceFastISel("arm-force-fast-isel",
               cl::init(false), cl::Hidden);

/// initializeSubtargetDependencies - Initializes using a CPU and feature string
/// so that we can use initializer lists for subtarget initialization.
ARMSubtarget &ARMSubtarget::initializeSubtargetDependencies(StringRef CPU,
                                                            StringRef FS) {
  initializeEnvironment();
  initSubtargetFeatures(CPU, FS);
  return *this;
}

ARMFrameLowering *ARMSubtarget::initializeFrameLowering(StringRef CPU,
                                                        StringRef FS) {
  ARMSubtarget &STI = initializeSubtargetDependencies(CPU, FS);
  if (STI.isThumb1Only())
    return (ARMFrameLowering *)new Thumb1FrameLowering(STI);

  return new ARMFrameLowering(STI);
}

ARMSubtarget::ARMSubtarget(const Triple &TT, const std::string &CPU,
                           const std::string &FS,
                           const ARMBaseTargetMachine &TM, bool IsLittle)
    : ARMGenSubtargetInfo(TT, CPU, FS), UseMulOps(UseFusedMulOps),
      CPUString(CPU), IsLittle(IsLittle), TargetTriple(TT), Options(TM.Options),
      TM(TM), FrameLowering(initializeFrameLowering(CPU, FS)),
      // At this point initializeSubtargetDependencies has been called so
      // we can query directly.
      InstrInfo(isThumb1Only()
                    ? (ARMBaseInstrInfo *)new Thumb1InstrInfo(*this)
                    : !isThumb()
                          ? (ARMBaseInstrInfo *)new ARMInstrInfo(*this)
                          : (ARMBaseInstrInfo *)new Thumb2InstrInfo(*this)),
      TLInfo(TM, *this) {}

void ARMSubtarget::initializeEnvironment() {
  // MCAsmInfo isn't always present (e.g. in opt) so we can't initialize this
  // directly from it, but we can try to make sure they're consistent when both
  // available.
  UseSjLjEH = isTargetDarwin() && !isTargetWatchABI();
  assert((!TM.getMCAsmInfo() ||
          (TM.getMCAsmInfo()->getExceptionHandlingType() ==
           ExceptionHandling::SjLj) == UseSjLjEH) &&
         "inconsistent sjlj choice between CodeGen and MC");
}

void ARMSubtarget::initSubtargetFeatures(StringRef CPU, StringRef FS) {
  if (CPUString.empty()) {
    CPUString = "generic";

    if (isTargetDarwin()) {
      StringRef ArchName = TargetTriple.getArchName();
      unsigned ArchKind = llvm::ARM::parseArch(ArchName);
      if (ArchKind == llvm::ARM::AK_ARMV7S)
        // Default to the Swift CPU when targeting armv7s/thumbv7s.
        CPUString = "swift";
      else if (ArchKind == llvm::ARM::AK_ARMV7K)
        // Default to the Cortex-a7 CPU when targeting armv7k/thumbv7k.
        // ARMv7k does not use SjLj exception handling.
        CPUString = "cortex-a7";
    }
  }

  // Insert the architecture feature derived from the target triple into the
  // feature string. This is important for setting features that are implied
  // based on the architecture version.
  std::string ArchFS = ARM_MC::ParseARMTriple(TargetTriple, CPUString);
  if (!FS.empty()) {
    if (!ArchFS.empty())
      ArchFS = (Twine(ArchFS) + "," + FS).str();
    else
      ArchFS = FS;
  }
  ParseSubtargetFeatures(CPUString, ArchFS);

  // FIXME: This used enable V6T2 support implicitly for Thumb2 mode.
  // Assert this for now to make the change obvious.
  assert(hasV6T2Ops() || !hasThumb2());

  // Keep a pointer to static instruction cost data for the specified CPU.
  SchedModel = getSchedModelForCPU(CPUString);

  // Initialize scheduling itinerary for the specified CPU.
  InstrItins = getInstrItineraryForCPU(CPUString);

  // FIXME: this is invalid for WindowsCE
  if (isTargetWindows())
    NoARM = true;

  if (isAAPCS_ABI())
    stackAlignment = 8;
  if (isTargetNaCl() || isAAPCS16_ABI())
    stackAlignment = 16;

  // FIXME: Completely disable sibcall for Thumb1 since ThumbRegisterInfo::
  // emitEpilogue is not ready for them. Thumb tail calls also use t2B, as
  // the Thumb1 16-bit unconditional branch doesn't have sufficient relocation
  // support in the assembler and linker to be used. This would need to be
  // fixed to fully support tail calls in Thumb1.
  //
  // Doing this is tricky, since the LDM/POP instruction on Thumb doesn't take
  // LR.  This means if we need to reload LR, it takes an extra instructions,
  // which outweighs the value of the tail call; but here we don't know yet
  // whether LR is going to be used.  Probably the right approach is to
  // generate the tail call here and turn it back into CALL/RET in
  // emitEpilogue if LR is used.

  // Thumb1 PIC calls to external symbols use BX, so they can be tail calls,
  // but we need to make sure there are enough registers; the only valid
  // registers are the 4 used for parameters.  We don't currently do this
  // case.

  SupportsTailCall = !isThumb() || hasV8MBaselineOps();

  if (isTargetMachO() && isTargetIOS() && getTargetTriple().isOSVersionLT(5, 0))
    SupportsTailCall = false;

  switch (IT) {
  case DefaultIT:
    RestrictIT = hasV8Ops();
    break;
  case RestrictedIT:
    RestrictIT = true;
    break;
  case NoRestrictedIT:
    RestrictIT = false;
    break;
  }

  // NEON f32 ops are non-IEEE 754 compliant. Darwin is ok with it by default.
  const FeatureBitset &Bits = getFeatureBits();
  if ((Bits[ARM::ProcA5] || Bits[ARM::ProcA8]) && // Where this matters
      (Options.UnsafeFPMath || isTargetDarwin()))
    UseNEONForSinglePrecisionFP = true;

  if (isRWPI())
    ReserveR9 = true;

  // FIXME: Teach TableGen to deal with these instead of doing it manually here.
  switch (ARMProcFamily) {
  case Others:
  case CortexA5:
    break;
  case CortexA7:
    LdStMultipleTiming = DoubleIssue;
    break;
  case CortexA8:
    LdStMultipleTiming = DoubleIssue;
    break;
  case CortexA9:
    LdStMultipleTiming = DoubleIssueCheckUnalignedAccess;
    PreISelOperandLatencyAdjustment = 1;
    break;
  case CortexA12:
    break;
  case CortexA15:
    MaxInterleaveFactor = 2;
    PreISelOperandLatencyAdjustment = 1;
    PartialUpdateClearance = 12;
    break;
  case CortexA17:
  case CortexA32:
  case CortexA35:
  case CortexA53:
  case CortexA57:
  case CortexA72:
  case CortexA73:
  case CortexR4:
  case CortexR4F:
  case CortexR5:
  case CortexR7:
  case CortexM3:
  case ExynosM1:
    break;
  case Krait:
    PreISelOperandLatencyAdjustment = 1;
    break;
  case Swift:
    MaxInterleaveFactor = 2;
    LdStMultipleTiming = SingleIssuePlusExtras;
    PreISelOperandLatencyAdjustment = 1;
    PartialUpdateClearance = 12;
    break;
  }
}

bool ARMSubtarget::isAPCS_ABI() const {
  assert(TM.TargetABI != ARMBaseTargetMachine::ARM_ABI_UNKNOWN);
  return TM.TargetABI == ARMBaseTargetMachine::ARM_ABI_APCS;
}
bool ARMSubtarget::isAAPCS_ABI() const {
  assert(TM.TargetABI != ARMBaseTargetMachine::ARM_ABI_UNKNOWN);
  return TM.TargetABI == ARMBaseTargetMachine::ARM_ABI_AAPCS ||
         TM.TargetABI == ARMBaseTargetMachine::ARM_ABI_AAPCS16;
}
bool ARMSubtarget::isAAPCS16_ABI() const {
  assert(TM.TargetABI != ARMBaseTargetMachine::ARM_ABI_UNKNOWN);
  return TM.TargetABI == ARMBaseTargetMachine::ARM_ABI_AAPCS16;
}

bool ARMSubtarget::isROPI() const {
  return TM.getRelocationModel() == Reloc::ROPI ||
         TM.getRelocationModel() == Reloc::ROPI_RWPI;
}
bool ARMSubtarget::isRWPI() const {
  return TM.getRelocationModel() == Reloc::RWPI ||
         TM.getRelocationModel() == Reloc::ROPI_RWPI;
}

bool ARMSubtarget::isGVIndirectSymbol(const GlobalValue *GV) const {
  if (!TM.shouldAssumeDSOLocal(*GV->getParent(), GV))
    return true;

  // 32 bit macho has no relocation for a-b if a is undefined, even if b is in
  // the section that is being relocated. This means we have to use o load even
  // for GVs that are known to be local to the dso.
  if (isTargetDarwin() && TM.isPositionIndependent() &&
      (GV->isDeclarationForLinker() || GV->hasCommonLinkage()))
    return true;

  return false;
}

unsigned ARMSubtarget::getMispredictionPenalty() const {
  return SchedModel.MispredictPenalty;
}

bool ARMSubtarget::hasSinCos() const {
  return isTargetWatchOS() ||
    (isTargetIOS() && !getTargetTriple().isOSVersionLT(7, 0));
}

bool ARMSubtarget::enableMachineScheduler() const {
  // Enable the MachineScheduler before register allocation for out-of-order
  // architectures where we do not use the PostRA scheduler anymore (for now
  // restricted to swift).
  return getSchedModel().isOutOfOrder() && isSwift();
}

// This overrides the PostRAScheduler bit in the SchedModel for any CPU.
bool ARMSubtarget::enablePostRAScheduler() const {
  // No need for PostRA scheduling on out of order CPUs (for now restricted to
  // swift).
  if (getSchedModel().isOutOfOrder() && isSwift())
    return false;
  return (!isThumb() || hasThumb2());
}

bool ARMSubtarget::enableAtomicExpand() const {
  return hasAnyDataBarrier() && (!isThumb() || hasV8MBaselineOps());
}

bool ARMSubtarget::useStride4VFPs(const MachineFunction &MF) const {
  // For general targets, the prologue can grow when VFPs are allocated with
  // stride 4 (more vpush instructions). But WatchOS uses a compact unwind
  // format which it's more important to get right.
  return isTargetWatchABI() || (isSwift() && !MF.getFunction()->optForMinSize());
}

bool ARMSubtarget::useMovt(const MachineFunction &MF) const {
  // NOTE Windows on ARM needs to use mov.w/mov.t pairs to materialise 32-bit
  // immediates as it is inherently position independent, and may be out of
  // range otherwise.
  return !NoMovt && hasV8MBaselineOps() &&
         (isTargetWindows() || !MF.getFunction()->optForMinSize());
}

bool ARMSubtarget::useFastISel() const {
  // Enable fast-isel for any target, for testing only.
  if (ForceFastISel)
    return true;

  // Limit fast-isel to the targets that are or have been tested.
  if (!hasV6Ops())
    return false;

  // Thumb2 support on iOS; ARM support on iOS, Linux and NaCl.
  return TM.Options.EnableFastISel &&
         ((isTargetMachO() && !isThumb1Only()) ||
          (isTargetLinux() && !isThumb()) || (isTargetNaCl() && !isThumb()));
}
