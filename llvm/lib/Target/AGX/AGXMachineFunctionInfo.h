//===-- AGXMachineFunctionInfo.h - AGX machine function info ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AGX_AGXMACHINEFUNCTIONINFO_H
#define LLVM_LIB_TARGET_AGX_AGXMACHINEFUNCTIONINFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

class GlobalVariable;

/// Per kernel state: the uniform register layout of the arguments and the
/// threadgroup memory allocation.
class AGXMachineFunctionInfo : public MachineFunctionInfo {
  /// Half register index of the first implicit argument
  /// (threadgroups_per_grid.xyz), and the end of the uniform range used by
  /// arguments.
  unsigned ImplicitArgHalfIndex = 0;
  unsigned UniformHalfCount = 0;
  /// Virtual registers holding threadgroups_per_grid.xyz.
  Register ImplicitArgVRegs[3];

  /// Threadgroup memory: offset of each __shared__ style variable.
  DenseMap<const GlobalVariable *, unsigned> ThreadgroupOffsets;
  unsigned ThreadgroupSize = 0;

public:
  AGXMachineFunctionInfo(const Function &F, const TargetSubtargetInfo *STI) {}

  MachineFunctionInfo *
  clone(BumpPtrAllocator &Allocator, MachineFunction &DestMF,
        const DenseMap<MachineBasicBlock *, MachineBasicBlock *> &Src2DstMBB)
      const override {
    return DestMF.cloneInfo<AGXMachineFunctionInfo>(*this);
  }

  void setImplicitArgHalfIndex(unsigned I) { ImplicitArgHalfIndex = I; }
  unsigned getImplicitArgHalfIndex() const { return ImplicitArgHalfIndex; }
  void setUniformHalfCount(unsigned N) { UniformHalfCount = N; }
  unsigned getUniformHalfCount() const { return UniformHalfCount; }
  void setImplicitArgVReg(unsigned Dim, Register R) {
    ImplicitArgVRegs[Dim] = R;
  }
  Register getImplicitArgVReg(unsigned Dim) const {
    return ImplicitArgVRegs[Dim];
  }

  /// Offset of \p GV in threadgroup memory, allocating it on first use.
  unsigned allocateThreadgroupVariable(const DataLayout &DL,
                                       const GlobalVariable &GV);
  unsigned getThreadgroupSize() const { return ThreadgroupSize; }
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AGX_AGXMACHINEFUNCTIONINFO_H
