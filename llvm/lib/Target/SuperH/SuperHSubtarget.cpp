//===-- SuperHSubtarget.h - Define Subtarget for SuperH ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the SuperH specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "SuperHSubtarget.h"
#include "MCTargetDesc/SuperHMCTargetDesc.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "sh-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "SuperHGenSubtargetInfo.inc"

SuperHSubtarget::SuperHSubtarget(const StringRef &CPU, const StringRef &TuneCPU,
                               const StringRef &FS, const TargetMachine &TM)
    : SuperHGenSubtargetInfo(TM.getTargetTriple(), CPU, TuneCPU, FS),
      InstrInfo(initializeSubtargetDependencies(CPU, TuneCPU, FS)), 
      TLInfo(TM, *this), FrameLowering(*this) {
  // TSInfo = std::make_unique<SuperHSelectionDAGInfo>();
}

SuperHSubtarget::~SuperHSubtarget() = default;


SuperHSubtarget &SuperHSubtarget::initializeSubtargetDependencies(
    StringRef CPU, StringRef TuneCPU, StringRef FS) {
  const Triple &TT = getTargetTriple();
  // Determine default and user specified characteristics
  std::string CPUName = std::string(CPU);
  if (TuneCPU.empty())
    TuneCPU = CPUName;

  // Parse features string.
  ParseSubtargetFeatures(CPUName, TuneCPU, FS);
  return *this;
}