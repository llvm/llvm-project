//===-- RISCVSubtarget.cpp - RISCV Subtarget Information ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RISCV specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "RISCVSubtarget.h"
#include "RISCV.h"
#include "RISCVCallLowering.h"
#include "RISCVFrameLowering.h"
#include "RISCVLegalizerInfo.h"
#include "RISCVRegisterBankInfo.h"
#include "RISCVTargetMachine.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "RISCVGenSubtargetInfo.inc"

static cl::opt<unsigned> RVVVectorBitsMax(
    "riscv-v-vector-bits-max",
    cl::desc("Assume V extension vector registers are at most this big, "
             "with zero meaning no maximum size is assumed."),
    cl::init(0), cl::Hidden);

static cl::opt<unsigned> RVVVectorBitsMin(
    "riscv-v-vector-bits-min",
    cl::desc("Assume V extension vector registers are at least this big, "
             "with zero meaning no minimum size is assumed."),
    cl::init(0), cl::Hidden);

static cl::opt<unsigned> RVVVectorLMULMax(
    "riscv-v-fixed-length-vector-lmul-max",
    cl::desc("The maximum LMUL value to use for fixed length vectors. "
             "Fractional LMUL values are not supported."),
    cl::init(8), cl::Hidden);

static cl::opt<unsigned> RVVVectorELENMax(
    "riscv-v-fixed-length-vector-elen-max",
    cl::desc("The maximum ELEN value to use for fixed length vectors."),
    cl::init(64), cl::Hidden);

static cl::opt<bool> RISCVDisableUsingConstantPoolForLargeInts(
    "riscv-disable-using-constant-pool-for-large-ints",
    cl::desc("Disable using constant pool for large integers."),
    cl::init(false), cl::Hidden);

static cl::opt<unsigned> RISCVMaxBuildIntsCost(
    "riscv-max-build-ints-cost",
    cl::desc("The maximum cost used for building integers."), cl::init(0),
    cl::Hidden);

void RISCVSubtarget::anchor() {}

RISCVSubtarget &
RISCVSubtarget::initializeSubtargetDependencies(const Triple &TT, StringRef CPU,
                                                StringRef TuneCPU, StringRef FS,
                                                StringRef ABIName) {
  // Determine default and user-specified characteristics
  bool Is64Bit = TT.isArch64Bit();
  if (CPU.empty())
    CPU = Is64Bit ? "generic-rv64" : "generic-rv32";
  if (CPU == "generic")
    report_fatal_error(Twine("CPU 'generic' is not supported. Use ") +
                       (Is64Bit ? "generic-rv64" : "generic-rv32"));

  if (TuneCPU.empty())
    TuneCPU = CPU;

  ParseSubtargetFeatures(CPU, TuneCPU, FS);
  if (Is64Bit) {
    XLenVT = MVT::i64;
    XLen = 64;
  }

  TargetABI = RISCVABI::computeTargetABI(TT, getFeatureBits(), ABIName);
  RISCVFeatures::validate(TT, getFeatureBits());
  return *this;
}

RISCVSubtarget::RISCVSubtarget(const Triple &TT, StringRef CPU,
                               StringRef TuneCPU, StringRef FS,
                               StringRef ABIName, const TargetMachine &TM)
    : RISCVGenSubtargetInfo(TT, CPU, TuneCPU, FS),
      UserReservedRegister(RISCV::NUM_TARGET_REGS),
      FrameLowering(initializeSubtargetDependencies(TT, CPU, TuneCPU, FS, ABIName)),
      InstrInfo(*this), RegInfo(getHwMode()), TLInfo(TM, *this) {
  CallLoweringInfo.reset(new RISCVCallLowering(*getTargetLowering()));
  Legalizer.reset(new RISCVLegalizerInfo(*this));

  auto *RBI = new RISCVRegisterBankInfo(*getRegisterInfo());
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
  if (RVVVectorBitsMax == 0)
    return 0;

  // ZvlLen specifies the minimum required vlen. The upper bound provided by
  // riscv-v-vector-bits-max should be no less than it.
  if (RVVVectorBitsMax < ZvlLen)
    report_fatal_error("riscv-v-vector-bits-max specified is lower "
                       "than the Zvl*b limitation");

  // FIXME: Change to >= 32 when VLEN = 32 is supported
  assert(RVVVectorBitsMax >= 64 && RVVVectorBitsMax <= 65536 &&
         isPowerOf2_32(RVVVectorBitsMax) &&
         "V extension requires vector length to be in the range of 128 to "
         "65536 and a power of 2!");
  assert(RVVVectorBitsMax >= RVVVectorBitsMin &&
         "Minimum V extension vector length should not be larger than its "
         "maximum!");
  unsigned Max = std::max(RVVVectorBitsMin, RVVVectorBitsMax);
  return PowerOf2Floor((Max < 128 || Max > 65536) ? 0 : Max);
}

unsigned RISCVSubtarget::getMinRVVVectorSizeInBits() const {
  // ZvlLen specifies the minimum required vlen. The lower bound provided by
  // riscv-v-vector-bits-min should be no less than it.
  if (RVVVectorBitsMin != 0 && RVVVectorBitsMin < ZvlLen)
    report_fatal_error("riscv-v-vector-bits-min specified is lower "
                       "than the Zvl*b limitation");

  assert(hasVInstructions() &&
         "Tried to get vector length without Zve or V extension support!");
  // FIXME: Change to >= 32 when VLEN = 32 is supported
  assert((RVVVectorBitsMin == 0 ||
          (RVVVectorBitsMin >= 64 && RVVVectorBitsMax <= 65536 &&
           isPowerOf2_32(RVVVectorBitsMin))) &&
         "V extension requires vector length to be in the range of 128 to "
         "65536 and a power of 2!");
  assert((RVVVectorBitsMax >= RVVVectorBitsMin || RVVVectorBitsMax == 0) &&
         "Minimum V extension vector length should not be larger than its "
         "maximum!");
  unsigned Min = RVVVectorBitsMin;
  if (RVVVectorBitsMax != 0)
    Min = std::min(RVVVectorBitsMin, RVVVectorBitsMax);
  return PowerOf2Floor((Min < 128 || Min > 65536) ? 0 : Min);
}

unsigned RISCVSubtarget::getMaxLMULForFixedLengthVectors() const {
  assert(hasVInstructions() &&
         "Tried to get vector length without Zve or V extension support!");
  assert(RVVVectorLMULMax <= 8 && isPowerOf2_32(RVVVectorLMULMax) &&
         "V extension requires a LMUL to be at most 8 and a power of 2!");
  return PowerOf2Floor(
      std::max<unsigned>(std::min<unsigned>(RVVVectorLMULMax, 8), 1));
}

unsigned RISCVSubtarget::getMaxELENForFixedLengthVectors() const {
  assert(hasVInstructions() &&
         "Tried to get maximum ELEN without Zve or V extension support!");
  assert(RVVVectorELENMax <= 64 && RVVVectorELENMax >= 8 &&
         isPowerOf2_32(RVVVectorELENMax) &&
         "V extension requires a ELEN to be a power of 2 between 8 and 64!");
  return PowerOf2Floor(
      std::max<unsigned>(std::min<unsigned>(RVVVectorELENMax, 64), 8));
}

bool RISCVSubtarget::useRVVForFixedLengthVectors() const {
  return hasVInstructions() && getMinRVVVectorSizeInBits() != 0;
}
