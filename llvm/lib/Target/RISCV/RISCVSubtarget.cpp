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
#include "llvm/Support/TargetRegistry.h"

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

unsigned RISCVSubtarget::getMaxRVVVectorSizeInBits() const {
  assert(hasStdExtV() && "Tried to get vector length without V support!");
  if (RVVVectorBitsMax == 0)
    return 0;
  assert(RVVVectorBitsMax >= 128 && isPowerOf2_32(RVVVectorBitsMax) &&
         "V extension requires vector length to be at least 128 and a power of "
         "2!");
  assert(RVVVectorBitsMax >= RVVVectorBitsMin &&
         "Minimum V extension vector length should not be larger than its "
         "maximum!");
  unsigned Max = std::max(RVVVectorBitsMin, RVVVectorBitsMax);
  return PowerOf2Floor(Max < 128 ? 0 : Max);
}

unsigned RISCVSubtarget::getMinRVVVectorSizeInBits() const {
  assert(hasStdExtV() &&
         "Tried to get vector length without V extension support!");
  assert((RVVVectorBitsMin == 0 ||
          (RVVVectorBitsMin >= 128 && isPowerOf2_32(RVVVectorBitsMin))) &&
         "V extension requires vector length to be at least 128 and a power of "
         "2!");
  assert((RVVVectorBitsMax >= RVVVectorBitsMin || RVVVectorBitsMax == 0) &&
         "Minimum V extension vector length should not be larger than its "
         "maximum!");
  unsigned Min = RVVVectorBitsMin;
  if (RVVVectorBitsMax != 0)
    Min = std::min(RVVVectorBitsMin, RVVVectorBitsMax);
  return PowerOf2Floor(Min < 128 ? 0 : Min);
}

unsigned RISCVSubtarget::getMaxLMULForFixedLengthVectors() const {
  assert(hasStdExtV() &&
         "Tried to get maximum LMUL without V extension support!");
  assert(RVVVectorLMULMax <= 8 && isPowerOf2_32(RVVVectorLMULMax) &&
         "V extension requires a LMUL to be at most 8 and a power of 2!");
  return PowerOf2Floor(std::max<unsigned>(RVVVectorLMULMax, 1));
}

bool RISCVSubtarget::useRVVForFixedLengthVectors() const {
  return hasStdExtV() && getMinRVVVectorSizeInBits() != 0;
}

unsigned RISCVSubtarget::getLMULForFixedLengthVector(MVT VT) const {
  unsigned MinVLen = getMinRVVVectorSizeInBits();

  // Masks only occupy a single register. An LMUL==1 operation can only use
  // at most 1/8 of the register. Only an LMUL==8 operaton on i8 types can
  // use the whole register.
  if (VT.getVectorElementType() == MVT::i1)
    MinVLen /= 8;

  return divideCeil(VT.getSizeInBits(), MinVLen);
}
