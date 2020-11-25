//===-- AArch64Subtarget.cpp - AArch64 Subtarget Information ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AArch64 specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "AArch64Subtarget.h"

#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "AArch64PBQPRegAlloc.h"
#include "AArch64TargetMachine.h"
#include "GISel/AArch64CallLowering.h"
#include "GISel/AArch64LegalizerInfo.h"
#include "GISel/AArch64RegisterBankInfo.h"
#include "MCTargetDesc/AArch64AddressingModes.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelect.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Support/TargetParser.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-subtarget"

#define GET_SUBTARGETINFO_CTOR
#define GET_SUBTARGETINFO_TARGET_DESC
#include "AArch64GenSubtargetInfo.inc"

static cl::opt<bool>
EnableEarlyIfConvert("aarch64-early-ifcvt", cl::desc("Enable the early if "
                     "converter pass"), cl::init(true), cl::Hidden);

// If OS supports TBI, use this flag to enable it.
static cl::opt<bool>
UseAddressTopByteIgnored("aarch64-use-tbi", cl::desc("Assume that top byte of "
                         "an address is ignored"), cl::init(false), cl::Hidden);

static cl::opt<bool>
    UseNonLazyBind("aarch64-enable-nonlazybind",
                   cl::desc("Call nonlazybind functions via direct GOT load"),
                   cl::init(false), cl::Hidden);

static cl::opt<unsigned> SVEVectorBitsMax(
    "aarch64-sve-vector-bits-max",
    cl::desc("Assume SVE vector registers are at most this big, "
             "with zero meaning no maximum size is assumed."),
    cl::init(0), cl::Hidden);

static cl::opt<unsigned> SVEVectorBitsMin(
    "aarch64-sve-vector-bits-min",
    cl::desc("Assume SVE vector registers are at least this big, "
             "with zero meaning no minimum size is assumed."),
    cl::init(0), cl::Hidden);

AArch64Subtarget &
AArch64Subtarget::initializeSubtargetDependencies(StringRef FS,
                                                  StringRef CPUString) {
  // Determine default and user-specified characteristics

  if (CPUString.empty())
    CPUString = "generic";

  ParseSubtargetFeatures(CPUString, /*TuneCPU*/ CPUString, FS);
  initializeProperties();

  return *this;
}

void AArch64Subtarget::initializeProperties() {
  // Initialize CPU specific properties. We should add a tablegen feature for
  // this in the future so we can specify it together with the subtarget
  // features.
  switch (ARMProcFamily) {
  case Others:
    break;
  case Carmel:
    CacheLineSize = 64;
    break;
  case CortexA35:
    break;
  case CortexA53:
    PrefFunctionLogAlignment = 3;
    break;
  case CortexA55:
    break;
  case CortexA57:
    MaxInterleaveFactor = 4;
    PrefFunctionLogAlignment = 4;
    break;
  case CortexA65:
    PrefFunctionLogAlignment = 3;
    break;
  case CortexA72:
  case CortexA73:
  case CortexA75:
  case CortexA76:
  case CortexA77:
  case CortexA78:
  case CortexR82:
  case CortexX1:
    PrefFunctionLogAlignment = 4;
    break;
  case A64FX:
    CacheLineSize = 256;
    PrefFunctionLogAlignment = 5;
    PrefLoopLogAlignment = 5;
    break;
  case AppleA7:
  case AppleA10:
  case AppleA11:
  case AppleA12:
  case AppleA13:
    CacheLineSize = 64;
    PrefetchDistance = 280;
    MinPrefetchStride = 2048;
    MaxPrefetchIterationsAhead = 3;
    break;
  case ExynosM3:
    MaxInterleaveFactor = 4;
    MaxJumpTableSize = 20;
    PrefFunctionLogAlignment = 5;
    PrefLoopLogAlignment = 4;
    break;
  case Falkor:
    MaxInterleaveFactor = 4;
    // FIXME: remove this to enable 64-bit SLP if performance looks good.
    MinVectorRegisterBitWidth = 128;
    CacheLineSize = 128;
    PrefetchDistance = 820;
    MinPrefetchStride = 2048;
    MaxPrefetchIterationsAhead = 8;
    break;
  case Kryo:
    MaxInterleaveFactor = 4;
    VectorInsertExtractBaseCost = 2;
    CacheLineSize = 128;
    PrefetchDistance = 740;
    MinPrefetchStride = 1024;
    MaxPrefetchIterationsAhead = 11;
    // FIXME: remove this to enable 64-bit SLP if performance looks good.
    MinVectorRegisterBitWidth = 128;
    break;
  case NeoverseE1:
    PrefFunctionLogAlignment = 3;
    break;
  case NeoverseN1:
  case NeoverseN2:
  case NeoverseV1:
    PrefFunctionLogAlignment = 4;
    break;
  case Saphira:
    MaxInterleaveFactor = 4;
    // FIXME: remove this to enable 64-bit SLP if performance looks good.
    MinVectorRegisterBitWidth = 128;
    break;
  case ThunderX2T99:
    CacheLineSize = 64;
    PrefFunctionLogAlignment = 3;
    PrefLoopLogAlignment = 2;
    MaxInterleaveFactor = 4;
    PrefetchDistance = 128;
    MinPrefetchStride = 1024;
    MaxPrefetchIterationsAhead = 4;
    // FIXME: remove this to enable 64-bit SLP if performance looks good.
    MinVectorRegisterBitWidth = 128;
    break;
  case ThunderX:
  case ThunderXT88:
  case ThunderXT81:
  case ThunderXT83:
    CacheLineSize = 128;
    PrefFunctionLogAlignment = 3;
    PrefLoopLogAlignment = 2;
    // FIXME: remove this to enable 64-bit SLP if performance looks good.
    MinVectorRegisterBitWidth = 128;
    break;
  case TSV110:
    CacheLineSize = 64;
    PrefFunctionLogAlignment = 4;
    PrefLoopLogAlignment = 2;
    break;
  case ThunderX3T110:
    CacheLineSize = 64;
    PrefFunctionLogAlignment = 4;
    PrefLoopLogAlignment = 2;
    MaxInterleaveFactor = 4;
    PrefetchDistance = 128;
    MinPrefetchStride = 1024;
    MaxPrefetchIterationsAhead = 4;
    // FIXME: remove this to enable 64-bit SLP if performance looks good.
    MinVectorRegisterBitWidth = 128;
    break;
  }
}

AArch64Subtarget::AArch64Subtarget(const Triple &TT, const std::string &CPU,
                                   const std::string &FS,
                                   const TargetMachine &TM, bool LittleEndian)
    : AArch64GenSubtargetInfo(TT, CPU, /*TuneCPU*/ CPU, FS),
      ReserveXRegister(AArch64::GPR64commonRegClass.getNumRegs()),
      CustomCallSavedXRegs(AArch64::GPR64commonRegClass.getNumRegs()),
      IsLittle(LittleEndian),
      TargetTriple(TT), FrameLowering(),
      InstrInfo(initializeSubtargetDependencies(FS, CPU)), TSInfo(),
      TLInfo(TM, *this) {
  if (AArch64::isX18ReservedByDefault(TT))
    ReserveXRegister.set(18);

  CallLoweringInfo.reset(new AArch64CallLowering(*getTargetLowering()));
  InlineAsmLoweringInfo.reset(new InlineAsmLowering(getTargetLowering()));
  Legalizer.reset(new AArch64LegalizerInfo(*this));

  auto *RBI = new AArch64RegisterBankInfo(*getRegisterInfo());

  // FIXME: At this point, we can't rely on Subtarget having RBI.
  // It's awkward to mix passing RBI and the Subtarget; should we pass
  // TII/TRI as well?
  InstSelector.reset(createAArch64InstructionSelector(
      *static_cast<const AArch64TargetMachine *>(&TM), *this, *RBI));

  RegBankInfo.reset(RBI);
}

const CallLowering *AArch64Subtarget::getCallLowering() const {
  return CallLoweringInfo.get();
}

const InlineAsmLowering *AArch64Subtarget::getInlineAsmLowering() const {
  return InlineAsmLoweringInfo.get();
}

InstructionSelector *AArch64Subtarget::getInstructionSelector() const {
  return InstSelector.get();
}

const LegalizerInfo *AArch64Subtarget::getLegalizerInfo() const {
  return Legalizer.get();
}

const RegisterBankInfo *AArch64Subtarget::getRegBankInfo() const {
  return RegBankInfo.get();
}

/// Find the target operand flags that describe how a global value should be
/// referenced for the current subtarget.
unsigned
AArch64Subtarget::ClassifyGlobalReference(const GlobalValue *GV,
                                          const TargetMachine &TM) const {
  // MachO large model always goes via a GOT, simply to get a single 8-byte
  // absolute relocation on all global addresses.
  if (TM.getCodeModel() == CodeModel::Large && isTargetMachO())
    return AArch64II::MO_GOT;

  if (!TM.shouldAssumeDSOLocal(*GV->getParent(), GV)) {
    if (GV->hasDLLImportStorageClass())
      return AArch64II::MO_GOT | AArch64II::MO_DLLIMPORT;
    if (getTargetTriple().isOSWindows())
      return AArch64II::MO_GOT | AArch64II::MO_COFFSTUB;
    return AArch64II::MO_GOT;
  }

  // The small code model's direct accesses use ADRP, which cannot
  // necessarily produce the value 0 (if the code is above 4GB).
  // Same for the tiny code model, where we have a pc relative LDR.
  if ((useSmallAddressing() || TM.getCodeModel() == CodeModel::Tiny) &&
      GV->hasExternalWeakLinkage())
    return AArch64II::MO_GOT;

  // References to tagged globals are marked with MO_NC | MO_TAGGED to indicate
  // that their nominal addresses are tagged and outside of the code model. In
  // AArch64ExpandPseudo::expandMI we emit an additional instruction to set the
  // tag if necessary based on MO_TAGGED.
  if (AllowTaggedGlobals && !isa<FunctionType>(GV->getValueType()))
    return AArch64II::MO_NC | AArch64II::MO_TAGGED;

  return AArch64II::MO_NO_FLAG;
}

unsigned AArch64Subtarget::classifyGlobalFunctionReference(
    const GlobalValue *GV, const TargetMachine &TM) const {
  // MachO large model always goes via a GOT, because we don't have the
  // relocations available to do anything else..
  if (TM.getCodeModel() == CodeModel::Large && isTargetMachO() &&
      !GV->hasInternalLinkage())
    return AArch64II::MO_GOT;

  // NonLazyBind goes via GOT unless we know it's available locally.
  auto *F = dyn_cast<Function>(GV);
  if (UseNonLazyBind && F && F->hasFnAttribute(Attribute::NonLazyBind) &&
      !TM.shouldAssumeDSOLocal(*GV->getParent(), GV))
    return AArch64II::MO_GOT;

  // Use ClassifyGlobalReference for setting MO_DLLIMPORT/MO_COFFSTUB.
  if (getTargetTriple().isOSWindows())
    return ClassifyGlobalReference(GV, TM);

  return AArch64II::MO_NO_FLAG;
}

void AArch64Subtarget::overrideSchedPolicy(MachineSchedPolicy &Policy,
                                           unsigned NumRegionInstrs) const {
  // LNT run (at least on Cyclone) showed reasonably significant gains for
  // bi-directional scheduling. 253.perlbmk.
  Policy.OnlyTopDown = false;
  Policy.OnlyBottomUp = false;
  // Enabling or Disabling the latency heuristic is a close call: It seems to
  // help nearly no benchmark on out-of-order architectures, on the other hand
  // it regresses register pressure on a few benchmarking.
  Policy.DisableLatencyHeuristic = DisableLatencySchedHeuristic;
}

bool AArch64Subtarget::enableEarlyIfConversion() const {
  return EnableEarlyIfConvert;
}

bool AArch64Subtarget::supportsAddressTopByteIgnored() const {
  if (!UseAddressTopByteIgnored)
    return false;

  if (TargetTriple.isiOS()) {
    unsigned Major, Minor, Micro;
    TargetTriple.getiOSVersion(Major, Minor, Micro);
    return Major >= 8;
  }

  return false;
}

std::unique_ptr<PBQPRAConstraint>
AArch64Subtarget::getCustomPBQPConstraints() const {
  return balanceFPOps() ? std::make_unique<A57ChainingConstraint>() : nullptr;
}

void AArch64Subtarget::mirFileLoaded(MachineFunction &MF) const {
  // We usually compute max call frame size after ISel. Do the computation now
  // if the .mir file didn't specify it. Note that this will probably give you
  // bogus values after PEI has eliminated the callframe setup/destroy pseudo
  // instructions, specify explicitly if you need it to be correct.
  MachineFrameInfo &MFI = MF.getFrameInfo();
  if (!MFI.isMaxCallFrameSizeComputed())
    MFI.computeMaxCallFrameSize(MF);
}

unsigned AArch64Subtarget::getMaxSVEVectorSizeInBits() const {
  assert(HasSVE && "Tried to get SVE vector length without SVE support!");
  assert(SVEVectorBitsMax % 128 == 0 &&
         "SVE requires vector length in multiples of 128!");
  assert((SVEVectorBitsMax >= SVEVectorBitsMin || SVEVectorBitsMax == 0) &&
         "Minimum SVE vector size should not be larger than its maximum!");
  if (SVEVectorBitsMax == 0)
    return 0;
  return (std::max(SVEVectorBitsMin, SVEVectorBitsMax) / 128) * 128;
}

unsigned AArch64Subtarget::getMinSVEVectorSizeInBits() const {
  assert(HasSVE && "Tried to get SVE vector length without SVE support!");
  assert(SVEVectorBitsMin % 128 == 0 &&
         "SVE requires vector length in multiples of 128!");
  assert((SVEVectorBitsMax >= SVEVectorBitsMin || SVEVectorBitsMax == 0) &&
         "Minimum SVE vector size should not be larger than its maximum!");
  if (SVEVectorBitsMax == 0)
    return (SVEVectorBitsMin / 128) * 128;
  return (std::min(SVEVectorBitsMin, SVEVectorBitsMax) / 128) * 128;
}

bool AArch64Subtarget::useSVEForFixedLengthVectors() const {
  // Prefer NEON unless larger SVE registers are available.
  return hasSVE() && getMinSVEVectorSizeInBits() >= 256;
}
