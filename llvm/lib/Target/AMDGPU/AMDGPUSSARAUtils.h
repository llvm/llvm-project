//===------- AMDGPUSSARAUtils.h ----------------------------------------*- C++-
//*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SSA_RA_UTILS_H
#define LLVM_LIB_TARGET_AMDGPU_SSA_RA_UTILS_H

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

using namespace llvm;

inline LaneBitmask getOperandLaneMask(const MachineOperand &MO,
                                      const SIRegisterInfo *TRI,
                                      MachineRegisterInfo *MRI) {
  assert(MO.isReg() && MO.getReg().isVirtual() &&
         "Error: Only virtual register allowed!\n");
  if (MO.getSubReg())
    return TRI->getSubRegIndexLaneMask(MO.getSubReg());
  return MRI->getMaxLaneMaskForVReg(MO.getReg());
}

inline unsigned getSubRegIndexForLaneMask(LaneBitmask Mask,
                                          const SIRegisterInfo *TRI) {
  for (unsigned Idx = 1; Idx < TRI->getNumSubRegIndices(); ++Idx) {
    if (TRI->getSubRegIndexLaneMask(Idx) == Mask)
      return Idx;
  }
  return AMDGPU::NoRegister;
}

inline SmallVector<unsigned>
getCoveringSubRegsForLaneMask(LaneBitmask Mask, const TargetRegisterClass *RC,
                              const SIRegisterInfo *TRI) {
  SmallVector<unsigned> Candidates;
  for (unsigned SubIdx = 1; SubIdx < TRI->getNumSubRegIndices(); ++SubIdx) {
    if (!TRI->getSubClassWithSubReg(RC, SubIdx))
      continue;

    LaneBitmask SubMask = TRI->getSubRegIndexLaneMask(SubIdx);
    if ((SubMask & Mask).any()) {
      Candidates.push_back(SubIdx);
    }
  }

  SmallVector<unsigned> OptimalSubIndices;
  llvm::stable_sort(Candidates, [&](unsigned A, unsigned B) {
    return TRI->getSubRegIndexLaneMask(A).getNumLanes() >
           TRI->getSubRegIndexLaneMask(B).getNumLanes();
  });
  for (unsigned SubIdx : Candidates) {
    LaneBitmask SubMask = TRI->getSubRegIndexLaneMask(SubIdx);
    if ((Mask & SubMask) == SubMask) {
      OptimalSubIndices.push_back(SubIdx);
      Mask &= ~SubMask; // remove covered bits
      if (Mask.none())
        break;
    }
  }
  return OptimalSubIndices;
}
#endif // LLVM_LIB_TARGET_AMDGPU_SSA_RA_UTILS_H