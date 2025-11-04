//===- AMDGPURewriteAGPRCopyMFMA.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUREWRITEAGPRCOPYMFMA_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUREWRITEAGPRCOPYMFMA_H

#include "llvm/ADT/SmallVector.h"

namespace llvm {

class MachineFunction;
class MachineDominatorTree;
class MachineInstr;

namespace AMDGPU {

/// Returns true if every reload in \p Loads is jointly dominated by at least
/// one store in \p Stores to the same stack slot \p Slot.
bool checkAGPRCopyMFMAJointDominance(
    const MachineFunction &MF, const MachineDominatorTree &MDT,
    const SmallVectorImpl<MachineInstr *> &Stores,
    const SmallVectorImpl<MachineInstr *> &Loads, int Slot);

} // end namespace AMDGPU
} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUREWRITEAGPRCOPYMFMA_H
