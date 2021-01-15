//===- AMDGPUMIRFormatter.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Implementation of AMDGPU overrides of MIRFormatter.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUMIRFormatter.h"
#include "SIMachineFunctionInfo.h"

using namespace llvm;

bool AMDGPUMIRFormatter::parseCustomPseudoSourceValue(
    StringRef Src, MachineFunction &MF, PerFunctionMIParsingState &PFS,
    const PseudoSourceValue *&PSV, ErrorCallbackType ErrorCallback) const {
  if (Src == "BufferResource") {
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    PSV = new AMDGPUBufferPseudoSourceValue(*TII);
    return false;
  }
  if (Src == "ImageResource") {
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    PSV = new AMDGPUImagePseudoSourceValue(*TII);
    return false;
  }
  if (Src == "GWSResource") {
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    PSV = new AMDGPUGWSResourcePseudoSourceValue(*TII);
    return false;
  }
  llvm_unreachable("unknown MIR custom pseudo source value");
}
