//===-- RISCVMCObjectFileInfo.cpp - RISCV object file properties ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the RISCVMCObjectFileInfo properties.
//
//===----------------------------------------------------------------------===//

#include "RISCVMCObjectFileInfo.h"
#include "RISCVMCTargetDesc.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSubtargetInfo.h"

using namespace llvm;

unsigned RISCVMCObjectFileInfo::getTextSectionAlignment() const {
  const MCSubtargetInfo *STI = getContext().getSubtargetInfo();
  return STI->hasFeature(RISCV::FeatureStdExtC) ? 2 : 4;
}
