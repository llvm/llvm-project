//===-- AIRSubtarget.h - Define Subtarget for AIR -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the AIR specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRSUBTARGET_H
#define LLVM_LIB_TARGET_AIR_AIRSUBTARGET_H

#include "AIRFrameLowering.h"
#include "AIRInstrInfo.h"
#include "AIRTargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetMachine.h"

#define GET_SUBTARGETINFO_HEADER
#include "AIRGenSubtargetInfo.inc"

namespace llvm {

class AIRTargetMachine;

class AIRSubtarget : public AIRGenSubtargetInfo {
  AIRInstrInfo InstrInfo;
  AIRFrameLowering FL;
  AIRTargetLowering TL;

  virtual void anchor();

public:
  AIRSubtarget(const Triple &TT, StringRef CPU, StringRef FS,
                 const AIRTargetMachine &TM);

  void ParseSubtargetFeatures(StringRef CPU, StringRef TuneCPU, StringRef FS);

  const AIRTargetLowering *getTargetLowering() const override { return &TL; }

  const AIRFrameLowering *getFrameLowering() const override { return &FL; }

  const AIRInstrInfo *getInstrInfo() const override { return &InstrInfo; }

  const AIRRegisterInfo *getRegisterInfo() const override {
    return &InstrInfo.getRegisterInfo();
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRSUBTARGET_H
