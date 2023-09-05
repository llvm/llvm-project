//===-- RISCVSubtarget.cpp - RISC-V Subtarget Information -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RISC-V specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "RISCVSubtarget.h"
#include "GISel/RISCVCallLowering.h"
#include "GISel/RISCVLegalizerInfo.h"
#include "GISel/RISCVRegisterBankInfo.h"
#include "RISCV.h"
#include "RISCVFrameLowering.h"
#include "RISCVMacroFusion.h"
#include "RISCVTargetMachine.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InstructionCost.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "RISCVGenSubtargetInfo.inc"

static cl::opt<bool> EnableSubRegLiveness("riscv-enable-subreg-liveness",
                                          cl::init(true), cl::Hidden);

static cl::opt<unsigned> RVVVectorLMULMax(
    "riscv-v-fixed-length-vector-lmul-max",
    cl::desc("The maximum LMUL value to use for fixed length vectors. "
             "Fractional LMUL values are not supported."),
    cl::init(8), cl::Hidden);

static cl::opt<bool> RISCVDisableUsingConstantPoolForLargeInts(
    "riscv-disable-using-constant-pool-for-large-ints",
    cl::desc("Disable using constant pool for large integers."),
    cl::init(false), cl::Hidden);

static cl::opt<unsigned> RISCVMaxBuildIntsCost(
    "riscv-max-build-ints-cost",
    cl::desc("The maximum cost used for building integers."), cl::init(0),
    cl::Hidden);

static cl::opt<bool> UseAA("riscv-use-aa", cl::init(true),
                           cl::desc("Enable the use of AA during codegen."));

void RISCVSubtarget::anchor() {}

RISCVSubtarget &
RISCVSubtarget::initializeSubtargetDependencies(const Triple &TT, StringRef CPU,
                                                StringRef TuneCPU, StringRef FS,
                                                StringRef ABIName) {
  // Determine default and user-specified characteristics
  bool Is64Bit = TT.isArch64Bit();
  if (CPU.empty() || CPU == "generic")
    CPU = Is64Bit ? "generic-rv64" : "generic-rv32";

  if (TuneCPU.empty())
    TuneCPU = CPU;

  ParseSubtargetFeatures(CPU, TuneCPU, FS);
  TargetABI = RISCVABI::computeTargetABI(TT, getFeatureBits(), ABIName);
  RISCVFeatures::validate(TT, getFeatureBits());
  return *this;
}

RISCVSubtarget::RISCVSubtarget(const Triple &TT, StringRef CPU,
                               StringRef TuneCPU, StringRef FS,
                               StringRef ABIName, unsigned RVVVectorBitsMin,
                               unsigned RVVVectorBitsMax,
                               const TargetMachine &TM)
    : RISCVGenSubtargetInfo(TT, CPU, TuneCPU, FS),
      RVVVectorBitsMin(RVVVectorBitsMin), RVVVectorBitsMax(RVVVectorBitsMax),
      FrameLowering(
          initializeSubtargetDependencies(TT, CPU, TuneCPU, FS, ABIName)),
      InstrInfo(*this), RegInfo(getHwMode()), TLInfo(TM, *this) {
  CallLoweringInfo.reset(new RISCVCallLowering(*getTargetLowering()));
  Legalizer.reset(new RISCVLegalizerInfo(*this));

  auto *RBI = new RISCVRegisterBankInfo(getHwMode());
  RegBankInfo.reset(RBI);
  InstSelector.reset(createRISCVInstructionSelector(
      *static_cast<const RISCVTargetMachine *>(&TM), *this, *RBI));
}

const CallLowering *RISCVSubtarget::getCallLowering() const {
  return CallLoweringInfo.get();
}

InstructionSelector *RISCVSubtarget::getInstructionSelector() const {
  return InstSelector.get();
}

const LegalizerInfo *RISCVSubtarget::getLegalizerInfo() const {
  return Legalizer.get();
}

const RegisterBankInfo *RISCVSubtarget::getRegBankInfo() const {
  return RegBankInfo.get();
}

bool RISCVSubtarget::useConstantPoolForLargeInts() const {
  return !RISCVDisableUsingConstantPoolForLargeInts;
}

unsigned RISCVSubtarget::getMaxBuildIntsCost() const {
  // Loading integer from constant pool needs two instructions (the reason why
  // the minimum cost is 2): an address calculation instruction and a load
  // instruction. Usually, address calculation and instructions used for
  // building integers (addi, slli, etc.) can be done in one cycle, so here we
  // set the default cost to (LoadLatency + 1) if no threshold is provided.
  return RISCVMaxBuildIntsCost == 0
             ? getSchedModel().LoadLatency + 1
             : std::max<unsigned>(2, RISCVMaxBuildIntsCost);
}

unsigned RISCVSubtarget::getMaxRVVVectorSizeInBits() const {
  assert(hasVInstructions() &&
         "Tried to get vector length without Zve or V extension support!");

  // ZvlLen specifies the minimum required vlen. The upper bound provided by
  // riscv-v-vector-bits-max should be no less than it.
  if (RVVVectorBitsMax != 0 && RVVVectorBitsMax < ZvlLen)
    report_fatal_error("riscv-v-vector-bits-max specified is lower "
                       "than the Zvl*b limitation");

  return RVVVectorBitsMax;
}

unsigned RISCVSubtarget::getMinRVVVectorSizeInBits() const {
  assert(hasVInstructions() &&
         "Tried to get vector length without Zve or V extension support!");

  if (RVVVectorBitsMin == -1U)
    return ZvlLen;

  // ZvlLen specifies the minimum required vlen. The lower bound provided by
  // riscv-v-vector-bits-min should be no less than it.
  if (RVVVectorBitsMin != 0 && RVVVectorBitsMin < ZvlLen)
    report_fatal_error("riscv-v-vector-bits-min specified is lower "
                       "than the Zvl*b limitation");

  return RVVVectorBitsMin;
}

unsigned RISCVSubtarget::getMaxLMULForFixedLengthVectors() const {
  assert(hasVInstructions() &&
         "Tried to get vector length without Zve or V extension support!");
  assert(RVVVectorLMULMax <= 8 &&
         llvm::has_single_bit<uint32_t>(RVVVectorLMULMax) &&
         "V extension requires a LMUL to be at most 8 and a power of 2!");
  return llvm::bit_floor(std::clamp<unsigned>(RVVVectorLMULMax, 1, 8));
}

bool RISCVSubtarget::useRVVForFixedLengthVectors() const {
  return hasVInstructions() && getMinRVVVectorSizeInBits() != 0;
}

InstructionCost RISCVSubtarget::getLMULCost(MVT VT) const {
  // TODO: Here assume reciprocal throughput is 1 for LMUL_1, it is
  // implementation-defined.
  if (!VT.isVector())
    return InstructionCost::getInvalid();
  unsigned DLenFactor = getDLenFactor();
  unsigned Cost;
  if (VT.isScalableVector()) {
    unsigned LMul;
    bool Fractional;
    std::tie(LMul, Fractional) =
        RISCVVType::decodeVLMUL(RISCVTargetLowering::getLMUL(VT));
    if (Fractional)
      Cost = LMul <= DLenFactor ? (DLenFactor / LMul) : 1;
    else
      Cost = (LMul * DLenFactor);
  } else {
    Cost = divideCeil(VT.getSizeInBits(), getRealMinVLen() / DLenFactor);
  }
  return Cost;
}


/// Return the cost of a vrgather.vv instruction for the type VT.  vrgather.vv
/// is generally quadratic in the number of vreg implied by LMUL.  Note that
/// operand (index and possibly mask) are handled separately.
InstructionCost RISCVSubtarget::getVRGatherVVCost(MVT VT) const {
  return getLMULCost(VT) * getLMULCost(VT);
}

/// Return the cost of a vrgather.vi (or vx) instruction for the type VT.
/// vrgather.vi/vx may be linear in the number of vregs implied by LMUL,
/// or may track the vrgather.vv cost. It is implementation-dependent.
InstructionCost RISCVSubtarget::getVRGatherVICost(MVT VT) const {
  return getLMULCost(VT);
}

/// Return the cost of a vslidedown.vi/vx or vslideup.vi/vx instruction
/// for the type VT.  (This does not cover the vslide1up or vslide1down
/// variants.)  Slides may be linear in the number of vregs implied by LMUL,
/// or may track the vrgather.vv cost. It is implementation-dependent.
InstructionCost RISCVSubtarget::getVSlideCost(MVT VT) const {
  return getLMULCost(VT);
}

bool RISCVSubtarget::enableSubRegLiveness() const {
  // FIXME: Enable subregister liveness by default for RVV to better handle
  // LMUL>1 and segment load/store.
  return EnableSubRegLiveness;
}

void RISCVSubtarget::getPostRAMutations(
    std::vector<std::unique_ptr<ScheduleDAGMutation>> &Mutations) const {
  Mutations.push_back(createRISCVMacroFusionDAGMutation());
}

  /// Enable use of alias analysis during code generation (during MI
  /// scheduling, DAGCombine, etc.).
bool RISCVSubtarget::useAA() const { return UseAA; }
