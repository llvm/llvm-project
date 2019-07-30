//===-- DPUSubtarget.cpp - DPU Subtarget Information ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the DPU specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "DPUSubtarget.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "dpu-subtarget"

#define GET_SUBTARGETINFO_ENUM
#define GET_SUBTARGETINFO_CTOR
#define GET_SUBTARGETINFO_TARGET_DESC

#include "DPUGenSubtargetInfo.inc"

void DPUSubtarget::anchor() {}

DPUSubtarget::DPUSubtarget(const Triple &TT, const std::string &CPU,
                           const std::string &FS, const TargetMachine &TM)
    : DPUGenSubtargetInfo(TT, CPU, FS), InstrInfo(), FrameLowering(*this),
      TargetLowering(TM, *this), TSInfo() {}

bool DPUSubtarget::enableMachineScheduler() const { return true; }
