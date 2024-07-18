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
#include "NVPTXTargetMachine.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"

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
static cl::opt<std::string>
    CustomSM("nvptx-custom-sm", cl::Hidden,
             cl::desc("NVPTX Specific: Override SM ID for sm_custom."),
             cl::init(""));
static cl::opt<unsigned>
    CustomPTX("nvptx-custom-ptx", cl::Hidden,
              cl::desc("NVPTX Specific: Override PTX version for sm_custom."),
              cl::init(0));

// Pin the vtable to this file.
void NVPTXSubtarget::anchor() {}

NVPTXSubtarget &NVPTXSubtarget::initializeSubtargetDependencies(StringRef CPU,
                                                                StringRef FS) {
  // Provide the default CPU if we don't have one.
  TargetName = std::string(CPU.empty() ? "sm_30" : CPU);

  ParseSubtargetFeatures(TargetName, /*TuneCPU*/ TargetName, FS);
  if (TargetName == "sm_custom") {
    if (CustomSM.empty() || CustomPTX == 0)
      llvm::report_fatal_error("Target sm_custom requires --nvptx-custom-sm "
                               "and --nvptx-custom-ptx to be specified.",
                               false);
    StringRef IDS(CustomSM);
    if (!IDS.starts_with("sm_"))
      llvm::report_fatal_error("Custom SM name must begin with 'sm_'");
    IDS = IDS.drop_front(3);
    if (IDS.ends_with("a"))
      IDS = IDS.drop_back(1);
    unsigned SMID;
    if (IDS.getAsInteger(10, SMID))
      llvm::report_fatal_error("Invalid custom SM format. Must be sm_NNN[a]");
    TargetName = CustomSM;
    FullSmVersion = SMID * 10;
    PTXVersion = CustomPTX;
  }

  // Re-map SM version numbers, SmVersion carries the regular SMs which do
  // have relative order, while FullSmVersion allows distinguishing sm_90 from
  // sm_90a, which would *not* be a subset of sm_91.
  SmVersion = getSmVersion();

  // Set default to PTX 6.0 (CUDA 9.0)
  if (PTXVersion == 0)
    PTXVersion = 60;

  return *this;
}

NVPTXSubtarget::NVPTXSubtarget(const Triple &TT, const std::string &CPU,
                               const std::string &FS,
                               const NVPTXTargetMachine &TM)
    : NVPTXGenSubtargetInfo(TT, CPU, /*TuneCPU*/ CPU, FS), PTXVersion(0),
      FullSmVersion(200), SmVersion(getSmVersion()), TM(TM),
      TLInfo(TM, initializeSubtargetDependencies(CPU, FS)) {}

bool NVPTXSubtarget::hasImageHandles() const {
  // Enable handles for Kepler+, where CUDA supports indirect surfaces and
  // textures
  if (TM.getDrvInterface() == NVPTX::CUDA)
    return (SmVersion >= 30);

  // Disabled, otherwise
  return false;
}

bool NVPTXSubtarget::allowFP16Math() const {
  return hasFP16Math() && NoF16Math == false;
}
