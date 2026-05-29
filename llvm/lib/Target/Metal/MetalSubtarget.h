//===-- MetalSubtarget.h - Define Subtarget for Metal -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Metal specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_METALSUBTARGET_H
#define LLVM_LIB_TARGET_METAL_METALSUBTARGET_H

#include "MetalFrameLowering.h"
#include "MetalInstrInfo.h"
#include "MetalTargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetMachine.h"

#define GET_SUBTARGETINFO_HEADER
#include "MetalGenSubtargetInfo.inc"

namespace llvm {

class MetalTargetMachine;

class MetalSubtarget : public MetalGenSubtargetInfo {
  MetalInstrInfo InstrInfo;
  MetalFrameLowering FL;
  MetalTargetLowering TL;

  virtual void anchor();

public:
  MetalSubtarget(const Triple &TT, StringRef CPU, StringRef FS,
                 const MetalTargetMachine &TM);

  void ParseSubtargetFeatures(StringRef CPU, StringRef TuneCPU, StringRef FS);

  const MetalTargetLowering *getTargetLowering() const override { return &TL; }

  const MetalFrameLowering *getFrameLowering() const override { return &FL; }

  const MetalInstrInfo *getInstrInfo() const override { return &InstrInfo; }

  const MetalRegisterInfo *getRegisterInfo() const override {
    return &InstrInfo.getRegisterInfo();
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_METAL_METALSUBTARGET_H
