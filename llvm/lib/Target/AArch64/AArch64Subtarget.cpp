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
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/TargetParser/AArch64TargetParser.h"

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

static cl::opt<bool> UseAA("aarch64-use-aa", cl::init(true),
                           cl::desc("Enable the use of AA during codegen."));

static cl::opt<unsigned> OverrideVectorInsertExtractBaseCost(
    "aarch64-insert-extract-base-cost",
    cl::desc("Base cost of vector insert/extract element"), cl::Hidden);

// Reserve a list of X# registers, so they are unavailable for register
// allocator, but can still be used as ABI requests, such as passing arguments
// to function call.
static cl::list<std::string>
ReservedRegsForRA("reserve-regs-for-regalloc", cl::desc("Reserve physical "
                  "registers, so they can't be used by register allocator. "
                  "Should only be used for testing register allocator."),
                  cl::CommaSeparated, cl::Hidden);

static cl::opt<bool> ForceStreamingCompatibleSVE(
    "force-streaming-compatible-sve",
    cl::desc(
        "Force the use of streaming-compatible SVE code for all functions"),
    cl::Hidden);

static cl::opt<AArch64PAuth::AuthCheckMethod>
    AuthenticatedLRCheckMethod("aarch64-authenticated-lr-check-method",
                               cl::Hidden,
                               cl::desc("Override the variant of check applied "
                                        "to authenticated LR during tail call"),
                               cl::values(AUTH_CHECK_METHOD_CL_VALUES_LR));

unsigned AArch64Subtarget::getVectorInsertExtractBaseCost() const {
  if (OverrideVectorInsertExtractBaseCost.getNumOccurrences() > 0)
    return OverrideVectorInsertExtractBaseCost;
  return VectorInsertExtractBaseCost;
}

AArch64Subtarget &AArch64Subtarget::initializeSubtargetDependencies(
    StringRef FS, StringRef CPUString, StringRef TuneCPUString) {
  // Determine default and user-specified characteristics

  if (CPUString.empty())
    CPUString = "generic";

  if (TuneCPUString.empty())
    TuneCPUString = CPUString;

  ParseSubtargetFeatures(CPUString, TuneCPUString, FS);
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
  case CortexA53:
  case CortexA55:
    PrefFunctionAlignment = Align(16);
    PrefLoopAlignment = Align(16);
    MaxBytesForLoopAlignment = 8;
    break;
  case CortexA57:
    MaxInterleaveFactor = 4;
    PrefFunctionAlignment = Align(16);
    PrefLoopAlignment = Align(16);
    MaxBytesForLoopAlignment = 8;
    break;
  case CortexA65:
    PrefFunctionAlignment = Align(8);
    break;
  case CortexA72:
  case CortexA73:
  case CortexA75:
    PrefFunctionAlignment = Align(16);
    PrefLoopAlignment = Align(16);
    MaxBytesForLoopAlignment = 8;
    break;
  case CortexA76:
  case CortexA77:
  case CortexA78:
  case CortexA78C:
  case CortexR82:
  case CortexX1:
  case CortexX1C:
    PrefFunctionAlignment = Align(16);
    PrefLoopAlignment = Align(32);
    MaxBytesForLoopAlignment = 16;
    break;
  case CortexA510:
    PrefFunctionAlignment = Align(16);
    VScaleForTuning = 1;
    PrefLoopAlignment = Align(16);
    MaxBytesForLoopAlignment = 8;
    break;
  case CortexA710:
  case CortexA715:
  case CortexX2:
  case CortexX3:
    PrefFunctionAlignment = Align(16);
    VScaleForTuning = 1;
    PrefLoopAlignment = Align(32);
    MaxBytesForLoopAlignment = 16;
    break;
  case A64FX:
    CacheLineSize = 256;
    PrefFunctionAlignment = Align(8);
    PrefLoopAlignment = Align(4);
    MaxInterleaveFactor = 4;
    PrefetchDistance = 128;
    MinPrefetchStride = 1024;
    MaxPrefetchIterationsAhead = 4;
    VScaleForTuning = 4;
    break;
  case AppleA7:
  case AppleA10:
  case AppleA11:
  case AppleA12:
  case AppleA13:
  case AppleA14:
  case AppleA15:
  case AppleA16:
    CacheLineSize = 64;
    PrefetchDistance = 280;
    MinPrefetchStride = 2048;
    MaxPrefetchIterationsAhead = 3;
    switch (ARMProcFamily) {
    case AppleA14:
    case AppleA15:
    case AppleA16:
      MaxInterleaveFactor = 4;
      break;
    default:
      break;
    }
    break;
  case ExynosM3:
    MaxInterleaveFactor = 4;
    MaxJumpTableSize = 20;
    PrefFunctionAlignment = Align(32);
    PrefLoopAlignment = Align(16);
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
    PrefFunctionAlignment = Align(8);
    break;
  case NeoverseN1:
    PrefFunctionAlignment = Align(16);
    PrefLoopAlignment = Align(32);
    MaxBytesForLoopAlignment = 16;
    break;
  case NeoverseN2:
  case NeoverseV2:
    PrefFunctionAlignment = Align(16);
    PrefLoopAlignment = Align(32);
    MaxBytesForLoopAlignment = 16;
    VScaleForTuning = 1;
    break;
  case NeoverseV1:
    PrefFunctionAlignment = Align(16);
    PrefLoopAlignment = Align(32);
    MaxBytesForLoopAlignment = 16;
    VScaleForTuning = 2;
    DefaultSVETFOpts = TailFoldingOpts::Simple;
    break;
  case Neoverse512TVB:
    PrefFunctionAlignment = Align(16);
    VScaleForTuning = 1;
    MaxInterleaveFactor = 4;
    break;
  case Saphira:
    MaxInterleaveFactor = 4;
    // FIXME: remove this to enable 64-bit SLP if performance looks good.
    MinVectorRegisterBitWidth = 128;
    break;
  case ThunderX2T99:
    CacheLineSize = 64;
    PrefFunctionAlignment = Align(8);
    PrefLoopAlignment = Align(4);
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
    PrefFunctionAlignment = Align(8);
    PrefLoopAlignment = Align(4);
    // FIXME: remove this to enable 64-bit SLP if performance looks good.
    MinVectorRegisterBitWidth = 128;
    break;
  case TSV110:
    CacheLineSize = 64;
    PrefFunctionAlignment = Align(16);
    PrefLoopAlignment = Align(4);
    break;
  case ThunderX3T110:
    CacheLineSize = 64;
    PrefFunctionAlignment = Align(16);
    PrefLoopAlignment = Align(4);
    MaxInterleaveFactor = 4;
    PrefetchDistance = 128;
    MinPrefetchStride = 1024;
    MaxPrefetchIterationsAhead = 4;
    // FIXME: remove this to enable 64-bit SLP if performance looks good.
    MinVectorRegisterBitWidth = 128;
    break;
  case Ampere1:
  case Ampere1A:
    CacheLineSize = 64;
    PrefFunctionAlignment = Align(64);
    PrefLoopAlignment = Align(64);
    MaxInterleaveFactor = 4;
    break;
  }
}

AArch64Subtarget::AArch64Subtarget(const Triple &TT, StringRef CPU,
                                   StringRef TuneCPU, StringRef FS,
                                   const TargetMachine &TM, bool LittleEndian,
                                   unsigned MinSVEVectorSizeInBitsOverride,
                                   unsigned MaxSVEVectorSizeInBitsOverride,
                                   bool StreamingSVEMode,
                                   bool StreamingCompatibleSVEMode)
    : AArch64GenSubtargetInfo(TT, CPU, TuneCPU, FS),
      ReserveXRegister(AArch64::GPR64commonRegClass.getNumRegs()),
      ReserveXRegisterForRA(AArch64::GPR64commonRegClass.getNumRegs()),
      CustomCallSavedXRegs(AArch64::GPR64commonRegClass.getNumRegs()),
      IsLittle(LittleEndian),
      StreamingSVEMode(StreamingSVEMode),
      StreamingCompatibleSVEMode(StreamingCompatibleSVEMode),
      MinSVEVectorSizeInBits(MinSVEVectorSizeInBitsOverride),
      MaxSVEVectorSizeInBits(MaxSVEVectorSizeInBitsOverride), TargetTriple(TT),
      InstrInfo(initializeSubtargetDependencies(FS, CPU, TuneCPU)),
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

  auto TRI = getRegisterInfo();
  StringSet<> ReservedRegNames;
  ReservedRegNames.insert(ReservedRegsForRA.begin(), ReservedRegsForRA.end());
  for (unsigned i = 0; i < 29; ++i) {
    if (ReservedRegNames.count(TRI->getName(AArch64::X0 + i)))
      ReserveXRegisterForRA.set(i);
  }
  // X30 is named LR, so we can't use TRI->getName to check X30.
  if (ReservedRegNames.count("X30") || ReservedRegNames.count("LR"))
    ReserveXRegisterForRA.set(30);
  // X29 is named FP, so we can't use TRI->getName to check X29.
  if (ReservedRegNames.count("X29") || ReservedRegNames.count("FP"))
    ReserveXRegisterForRA.set(29);

  AddressCheckPSV.reset(new AddressCheckPseudoSourceValue(TM));
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

  // All globals dynamically protected by MTE must have their address tags
  // synthesized. This is done by having the loader stash the tag in the GOT
  // entry. Force all tagged globals (even ones with internal linkage) through
  // the GOT.
  if (GV->isTagged())
    return AArch64II::MO_GOT;

  if (!TM.shouldAssumeDSOLocal(*GV->getParent(), GV)) {
    if (GV->hasDLLImportStorageClass()) {
      if (isWindowsArm64EC() && GV->getValueType()->isFunctionTy())
        return AArch64II::MO_GOT | AArch64II::MO_DLLIMPORTAUX;
      return AArch64II::MO_GOT | AArch64II::MO_DLLIMPORT;
    }
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

  if (getTargetTriple().isOSWindows()) {
    if (isWindowsArm64EC() && GV->getValueType()->isFunctionTy() &&
        GV->hasDLLImportStorageClass()) {
      // On Arm64EC, if we're calling a function directly, use MO_DLLIMPORT,
      // not MO_DLLIMPORTAUX.
      return AArch64II::MO_GOT | AArch64II::MO_DLLIMPORT;
    }

    // Use ClassifyGlobalReference for setting MO_DLLIMPORT/MO_COFFSTUB.
    return ClassifyGlobalReference(GV, TM);
  }

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

  if (TargetTriple.isDriverKit())
    return true;
  if (TargetTriple.isiOS()) {
    return TargetTriple.getiOSVersion() >= VersionTuple(8);
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

bool AArch64Subtarget::useAA() const { return UseAA; }

bool AArch64Subtarget::isStreamingCompatible() const {
  return StreamingCompatibleSVEMode || ForceStreamingCompatibleSVE;
}

bool AArch64Subtarget::isNeonAvailable() const {
  return hasNEON() && !isStreaming() && !isStreamingCompatible();
}

bool AArch64Subtarget::isSVEAvailable() const{
  // FIXME: Also return false if FEAT_FA64 is set, but we can't do this yet
  // as we don't yet support the feature in LLVM.
  return hasSVE() && !isStreaming() && !isStreamingCompatible();
}

// If return address signing is enabled, tail calls are emitted as follows:
//
// ```
//   <authenticate LR>
//   <check LR>
//   TCRETURN          ; the callee may sign and spill the LR in its prologue
// ```
//
// LR may require explicit checking because if FEAT_FPAC is not implemented
// and LR was tampered with, then `<authenticate LR>` will not generate an
// exception on its own. Later, if the callee spills the signed LR value and
// neither FEAT_PAuth2 nor FEAT_EPAC are implemented, the valid PAC replaces
// the higher bits of LR thus hiding the authentication failure.
AArch64PAuth::AuthCheckMethod
AArch64Subtarget::getAuthenticatedLRCheckMethod() const {
  if (AuthenticatedLRCheckMethod.getNumOccurrences())
    return AuthenticatedLRCheckMethod;

  // At now, use None by default because checks may introduce an unexpected
  // performance regression or incompatibility with execute-only mappings.
  return AArch64PAuth::AuthCheckMethod::None;
}
