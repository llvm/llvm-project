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

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "NVPTXGenSubtargetInfo.inc"

static cl::opt<bool>
    NoF16Math("nvptx-no-f16-math", cl::Hidden,
              cl::desc("NVPTX Specific: Disable generation of f16 math ops."),
              cl::init(false));

static cl::opt<bool> NoF32x2("nvptx-no-f32x2", cl::Hidden,
                             cl::desc("NVPTX Specific: Disable generation of "
                                      "f32x2 instructions and registers."),
                             cl::init(false));

// Pin the vtable to this file.
void NVPTXSubtarget::anchor() {}

// Returns the minimum PTX version required for a given SM target.
// This must be kept in sync with the "Supported Targets" column of the
// "PTX Release History" table in the PTX ISA documentation:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#release-notes-ptx-release-history
//
// Note: LLVM's minimum supported PTX version is 3.2 (see FeaturePTX in
// NVPTX.td), so older SMs that supported earlier PTX versions instead use 3.2
// as their effective minimum.
static unsigned minPTXVersion(NVPTX::GPUKind Arch) {
  switch (Arch) {
  case NVPTX::GK_NONE:
    llvm_unreachable("architecture is resolved before this is reached");
  case NVPTX::GK_SM_20:
  case NVPTX::GK_SM_21:
  case NVPTX::GK_SM_30:
  case NVPTX::GK_SM_35:
    return 32;
  case NVPTX::GK_SM_32_:
  case NVPTX::GK_SM_50:
    return 40;
  case NVPTX::GK_SM_37:
  case NVPTX::GK_SM_52:
    return 41;
  case NVPTX::GK_SM_53:
    return 42;
  case NVPTX::GK_SM_60:
  case NVPTX::GK_SM_61:
  case NVPTX::GK_SM_62:
    return 50;
  case NVPTX::GK_SM_70:
    return 60;
  case NVPTX::GK_SM_72:
    return 61;
  case NVPTX::GK_SM_75:
    return 63;
  case NVPTX::GK_SM_80:
    return 70;
  case NVPTX::GK_SM_86:
    return 71;
  case NVPTX::GK_SM_87:
    return 74;
  case NVPTX::GK_SM_89:
  case NVPTX::GK_SM_90:
    return 78;
  case NVPTX::GK_SM_90a:
    return 80;
  case NVPTX::GK_SM_100:
  case NVPTX::GK_SM_100a:
  case NVPTX::GK_SM_101:
  case NVPTX::GK_SM_101a:
    return 86;
  case NVPTX::GK_SM_120:
  case NVPTX::GK_SM_120a:
    return 87;
  case NVPTX::GK_SM_100f:
  case NVPTX::GK_SM_101f:
  case NVPTX::GK_SM_103:
  case NVPTX::GK_SM_103f:
  case NVPTX::GK_SM_103a:
  case NVPTX::GK_SM_120f:
  case NVPTX::GK_SM_121:
  case NVPTX::GK_SM_121f:
  case NVPTX::GK_SM_121a:
    return 88;
  case NVPTX::GK_SM_88:
  case NVPTX::GK_SM_110:
  case NVPTX::GK_SM_110f:
  case NVPTX::GK_SM_110a:
    return 90;
  case NVPTX::GK_SM_107:
  case NVPTX::GK_SM_107f:
  case NVPTX::GK_SM_107a:
    return 94;
  }
  llvm_unreachable("invalid NVPTX GPUKind");
}

NVPTXSubtarget &NVPTXSubtarget::initializeSubtargetDependencies(StringRef CPU,
                                                                StringRef FS) {
  // If the user did not provide a target we default to the `sm_75` target.
  StringRef RequestedCPU = CPU.empty() ? StringRef("sm_75") : CPU;
  ParseSubtargetFeatures(RequestedCPU, /*TuneCPU=*/RequestedCPU, FS);

  Arch = NVPTX::parseArch(RequestedCPU);

  // An unrecognized name has already been diagnosed and its features dropped,
  // leaving the subtarget at the oldest architecture, so name it that.
  if (Arch == NVPTX::GK_NONE)
    Arch = NVPTX::GK_SM_20;

  unsigned MinPTX = minPTXVersion(Arch);

  if (PTXVersion == 0) {
    // User didn't request a specific PTX version; use the minimum for this SM.
    PTXVersion = MinPTX;
  } else if (PTXVersion < MinPTX) {
    // User explicitly requested an insufficient PTX version.
    reportFatalUsageError(
        formatv("PTX version {0}.{1} does not support target '{2}'. "
                "Minimum required PTX version is {3}.{4}. "
                "Either remove the PTX version to use the default, "
                "or increase it to at least {3}.{4}.",
                PTXVersion / 10, PTXVersion % 10, RequestedCPU, MinPTX / 10,
                MinPTX % 10));
  }

  return *this;
}

NVPTXSubtarget::NVPTXSubtarget(const Triple &TT, StringRef CPU, StringRef FS,
                               const NVPTXTargetMachine &TM)
    : NVPTXGenSubtargetInfo(TT, CPU, /*TuneCPU*/ CPU, FS), PTXVersion(0),
      InstrInfo(initializeSubtargetDependencies(CPU, FS)), TLInfo(TM, *this),
      TSInfo(std::make_unique<NVPTXSelectionDAGInfo>()) {}

NVPTXSubtarget::~NVPTXSubtarget() = default;

bool NVPTXSubtarget::allowFP16Math() const {
  return hasFP16Math() && NoF16Math == false;
}

bool NVPTXSubtarget::hasF32x2Instructions() const {
  return hasFeature(NVPTX::SM100) && PTXVersion >= 86 && !NoF32x2;
}

bool NVPTXSubtarget::hasNativeBF16Support(unsigned Opcode) const {
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
  case ISD::FTANH:
  case ISD::FCEIL:
  case ISD::FFLOOR:
  case ISD::FNEARBYINT:
  case ISD::FRINT:
  case ISD::FROUNDEVEN:
  case ISD::FTRUNC:
    return hasFeature(NVPTX::SM90) && getPTXVersion() >= 78;
  // Several BF16 instructions are available on sm_80 only.
  case ISD::FMINNUM:
  case ISD::FMAXNUM:
  case ISD::FMAXNUM_IEEE:
  case ISD::FMINNUM_IEEE:
  case ISD::FMAXIMUM:
  case ISD::FMINIMUM:
    return hasFeature(NVPTX::SM80) && getPTXVersion() >= 70;
  }
  return true;
}
