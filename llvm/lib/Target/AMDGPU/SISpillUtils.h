//===- SISpillUtils.h - SI spill helper functions ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SISPILLUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_SISPILLUTILS_H

namespace llvm {

class BitVector;
class MachineBasicBlock;
class MachineFrameInfo;

/// Replace frame index operands with null registers in debug value instructions
/// for the specified spill frame indices.
void clearDebugInfoForSpillFIs(MachineFrameInfo &MFI, MachineBasicBlock &MBB,
                               const BitVector &SpillFIs);

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_SISPILLUTILS_H
