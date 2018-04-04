//===- AMDGPULaneDominator.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPULANEDOMINATOR_H
#define LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPULANEDOMINATOR_H

namespace llvm {

class MachineBasicBlock;

namespace AMDGPU {

bool laneDominates(MachineBasicBlock *MBBA, MachineBasicBlock *MBBB);

} // end namespace AMDGPU
} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPULANEDOMINATOR_H
