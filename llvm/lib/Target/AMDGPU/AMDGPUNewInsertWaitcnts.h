//===--- AMDGPUNewInsertWaitcnts.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUNEWINSERTWAITCNTS_AMDGPUNEWINSERTWAITCNTS_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUNEWINSERTWAITCNTS_AMDGPUNEWINSERTWAITCNTS_H

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

namespace AMDGPU {

/// The main class of the wait insertion pass.
class InsertWaitcnts {
public:
  InsertWaitcnts() = default;
  /// The main entry point of wait insertion into \p MF. Returns true if the IR
  /// is modified.
  bool run(MachineFunction &MF);
};

} // namespace AMDGPU

class AMDGPUNewInsertWaitcntsPass
    : public RequiredPassInfoMixin<AMDGPUNewInsertWaitcntsPass> {
  AMDGPU::InsertWaitcnts IW;

public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUNEWINSERTWAITCNTS_AMDGPUNEWINSERTWAITCNTS_H
