//===- AMDGPUGlobalISelUtils -------------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUGLOBALISELUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUGLOBALISELUTILS_H

#include "AMDGPUInstrInfo.h"
#include "llvm/CodeGen/Register.h"
#include <tuple>

namespace llvm {

class MachineInstr;
class MachineRegisterInfo;

namespace AMDGPU {

/// Returns Base register, constant offset, and offset def point.
std::tuple<Register, unsigned, MachineInstr *>
getBaseWithConstantOffset(MachineRegisterInfo &MRI, Register Reg);

bool isLegalVOP3PShuffleMask(ArrayRef<int> Mask);

/// Return number of address arguments, and the number of gradients for an image
/// intrinsic.
inline std::pair<int, int>
getImageNumVAddr(const AMDGPU::ImageDimIntrinsicInfo *ImageDimIntr,
                 const AMDGPU::MIMGBaseOpcodeInfo *BaseOpcode) {
  const AMDGPU::MIMGDimInfo *DimInfo
    = AMDGPU::getMIMGDimInfo(ImageDimIntr->Dim);

  int NumGradients = BaseOpcode->Gradients ? DimInfo->NumGradients : 0;
  int NumCoords = BaseOpcode->Coordinates ? DimInfo->NumCoords : 0;
  int NumLCM = BaseOpcode->LodOrClampOrMip ? 1 : 0;
  int NumVAddr = BaseOpcode->NumExtraArgs + NumGradients + NumCoords + NumLCM;
  return {NumVAddr, NumGradients};
}

/// Return index of dmask in an gMIR image intrinsic
inline int getDMaskIdx(const AMDGPU::MIMGBaseOpcodeInfo *BaseOpcode,
                       int NumDefs) {
  assert(!BaseOpcode->Atomic);
  return NumDefs + 1 + (BaseOpcode->Store ? 1 : 0);
}

/// Return first address operand index in a gMIR image intrinsic.
inline int getImageVAddrIdxBegin(const AMDGPU::MIMGBaseOpcodeInfo *BaseOpcode,
                                 int NumDefs) {
  if (BaseOpcode->Atomic)
    return NumDefs + 1 + (BaseOpcode->AtomicX2 ? 2 : 1);
  return getDMaskIdx(BaseOpcode, NumDefs) + 1;
}

}
}

#endif
