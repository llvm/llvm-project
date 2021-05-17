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
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"

using namespace llvm;

bool AMDGPUMIRFormatter::parseCustomPseudoSourceValue(
    StringRef Src, MachineFunction &MF, PerFunctionMIParsingState &PFS,
    const PseudoSourceValue *&PSV, ErrorCallbackType ErrorCallback) const {
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  const SIInstrInfo &TII = *MF.getSubtarget<GCNSubtarget>().getInstrInfo();
  if (Src == "BufferResource") {
    PSV = MFI->getBufferPSV(TII);
    return false;
  }
  if (Src == "ImageResource") {
    PSV = MFI->getImagePSV(TII);
    return false;
  }
  if (Src == "GWSResource") {
    PSV = MFI->getGWSPSV(TII);
    return false;
  }
  llvm_unreachable("unknown MIR custom pseudo source value");
}
