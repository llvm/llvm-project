//===- NVPTXSubtarget.cpp - NVPTX Subtarget Information -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the NVPTX specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "NVPTXSubtarget.h"
#include "NVPTXSelectionDAGInfo.h"
#include "NVPTXTargetMachine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"

using namespace llvm;

#define DEBUG_TYPE "nvptx-subtarget"

#define GET_SUBTARGETINFO_ENUM
#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "NVPTXGenSubtargetInfo.inc"

static cl::opt<bool>
    NoF16Math("nvptx-no-f16-math", cl::Hidden,
              cl::desc("NVPTX Specific: Disable generation of f16 math ops."),
              cl::init(false));
// Pin the vtable to this file.
void NVPTXSubtarget::anchor() {}

NVPTXSubtarget &NVPTXSubtarget::initializeSubtargetDependencies(StringRef CPU,
                                                                StringRef FS) {
  TargetName = std::string(CPU);

  ParseSubtargetFeatures(getTargetName(), /*TuneCPU=*/getTargetName(), FS);

  // Re-map SM version numbers, SmVersion carries the regular SMs which do
  // have relative order, while FullSmVersion allows distinguishing sm_90 from
  // sm_90a, which would *not* be a subset of sm_91.
  SmVersion = getSmVersion();

  // Set default to PTX 6.0 (CUDA 9.0)
  if (PTXVersion == 0) {
    PTXVersion = 60;
  }

  return *this;
}

NVPTXSubtarget::NVPTXSubtarget(const Triple &TT, const std::string &CPU,
                               const std::string &FS,
                               const NVPTXTargetMachine &TM)
    : NVPTXGenSubtargetInfo(TT, CPU, /*TuneCPU*/ CPU, FS), PTXVersion(0),
      FullSmVersion(200), SmVersion(getSmVersion()),
      TLInfo(TM, initializeSubtargetDependencies(CPU, FS)) {
  TSInfo = std::make_unique<NVPTXSelectionDAGInfo>();
}

NVPTXSubtarget::~NVPTXSubtarget() = default;

const SelectionDAGTargetInfo *NVPTXSubtarget::getSelectionDAGInfo() const {
  return TSInfo.get();
}

bool NVPTXSubtarget::allowFP16Math() const {
  return hasFP16Math() && NoF16Math == false;
}

bool NVPTXSubtarget::hasNativeBF16Support(int Opcode) const {
  if (!hasBF16Math())
    return false;

  switch (Opcode) {
  // Several BF16 instructions are available on sm_90 only.
  case ISD::FADD:
  case ISD::FMUL:
  case ISD::FSUB:
  case ISD::SELECT:
  case ISD::SELECT_CC:
  case ISD::SETCC:
  case ISD::FEXP2:
  case ISD::FCEIL:
  case ISD::FFLOOR:
  case ISD::FNEARBYINT:
  case ISD::FRINT:
  case ISD::FROUNDEVEN:
  case ISD::FTRUNC:
    return getSmVersion() >= 90 && getPTXVersion() >= 78;
  // Several BF16 instructions are available on sm_80 only.
  case ISD::FMINNUM:
  case ISD::FMAXNUM:
  case ISD::FMAXNUM_IEEE:
  case ISD::FMINNUM_IEEE:
  case ISD::FMAXIMUM:
  case ISD::FMINIMUM:
    return getSmVersion() >= 80 && getPTXVersion() >= 70;
  }
  return true;
}

void NVPTXSubtarget::failIfClustersUnsupported(
    std::string const &FailureMessage) const {
  if (hasClusters())
    return;

  report_fatal_error(formatv(
      "NVPTX SM architecture \"{}\" and PTX version \"{}\" do not support {}. "
      "Requires SM >= 90 and PTX >= 78.",
      getFullSmVersion(), PTXVersion, FailureMessage));
}
