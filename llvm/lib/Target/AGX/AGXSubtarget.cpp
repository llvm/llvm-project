//===-- AGXSubtarget.cpp - AGX subtarget information ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AGXSubtarget.h"
#include "AGXTargetMachine.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "agx-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "AGXGenSubtargetInfo.inc"

AGXSubtarget &AGXSubtarget::initializeSubtargetDependencies(StringRef CPU,
                                                            StringRef FS) {
  if (CPU.empty())
    CPU = "generic";
  ParseSubtargetFeatures(CPU, /*TuneCPU=*/CPU, FS);
  if (HasMesaISA) {
    if (Gen != G13 && Gen != G14)
      report_fatal_error("the agx-strict feature needs a G13 or G14 processor, "
                         "not " + CPU);
    // Instructions Mesa's compiler never emits.
    HasDeviceAtomicFAdd = false;
    HasDeviceAtomicU64MinMax = false;
    HasSIMDMatrix = false;
    HasBlockBarrier = false;
  }
  return *this;
}

AGXSubtarget::AGXSubtarget(const Triple &TT, StringRef CPU, StringRef FS,
                           const AGXTargetMachine &TM)
    : AGXGenSubtargetInfo(TT, CPU.empty() ? "generic" : CPU,
                          /*TuneCPU=*/CPU.empty() ? "generic" : CPU, FS),
      InstrInfo(initializeSubtargetDependencies(CPU, FS)),
      FrameLowering(*this), TLInfo(TM, *this) {}

unsigned AGXSubtarget::maxThreadsForHalves(unsigned Halves) {
  // Registers are allocated in groups of 8 halves.
  Halves = std::max<unsigned>(alignTo(Halves, 8), 104);
  return std::min(64 * (1664 / Halves), 1024u);
}

unsigned AGXSubtarget::getMaxGPRs(const Function &F) const {
  // Enough for any instruction's operands and the reserved r0.
  constexpr unsigned MinGPRs = 16;
  unsigned Max = getMaxGPRs();
  if (unsigned N = F.getFnAttributeAsParsedInteger("agx-max-gprs", 0))
    Max = std::min(Max, std::max(N, MinGPRs));
  if (unsigned T = F.getFnAttributeAsParsedInteger(
          "agx-max-threads-per-threadgroup", 0)) {
    // The largest register count with maxThreadsForHalves(2 * GPRs) >= T.
    unsigned K = std::max<unsigned>(divideCeil(std::min(T, 1024u), 64), 1);
    Max = std::min(Max, std::max<unsigned>(alignDown(1664 / K, 8) / 2, MinGPRs));
  }
  return Max;
}

