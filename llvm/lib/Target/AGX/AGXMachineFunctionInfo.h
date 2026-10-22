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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Alignment.h"
#include <optional>
#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

class GlobalVariable;
class Module;

namespace AGX {

/// Device and constant global variables live in one buffer, the global data
/// buffer, whose address kernels receive as an implicit argument. The
/// variables are laid out in module order, each aligned to its alignment;
/// the assembly printer emits the initial contents in the .agx.globals
/// section with the same layout.
struct GlobalDataLayout {
  SmallVector<std::pair<const GlobalVariable *, uint64_t>, 8> Offsets;
  uint64_t Size = 0;
  Align MaxAlign;

  bool empty() const { return Offsets.empty(); }
  std::optional<uint64_t> getOffset(const GlobalVariable *GV) const;
};

/// Whether \p GV is stored in the global data buffer.
bool isGlobalDataVariable(const GlobalVariable &GV);

GlobalDataLayout computeGlobalDataLayout(const Module &M);
/// Whether F's instructions use a global that lives in the global data buffer,
/// and so F needs the buffer's address.
bool usesGlobalData(const Function &F);

} // namespace AGX

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
  /// Virtual register holding the global data buffer address, and its first
  /// uniform half register.
  Register GlobalDataVReg;
  unsigned GlobalDataHalfIndex = 0;

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
  /// The first uniform half of each IR argument (~0u if it has none).
  SmallVector<unsigned, 8> ArgHalves;

  void setUniformHalfCount(unsigned N) { UniformHalfCount = N; }
  unsigned getUniformHalfCount() const { return UniformHalfCount; }
  void setImplicitArgVReg(unsigned Dim, Register R) {
    ImplicitArgVRegs[Dim] = R;
  }
  Register getImplicitArgVReg(unsigned Dim) const {
    return ImplicitArgVRegs[Dim];
  }
  void setGlobalData(Register VReg, unsigned HalfIndex) {
    GlobalDataVReg = VReg;
    GlobalDataHalfIndex = HalfIndex;
  }
  Register getGlobalDataVReg() const { return GlobalDataVReg; }
  unsigned getGlobalDataHalfIndex() const { return GlobalDataHalfIndex; }

  /// Offset of \p GV in threadgroup memory. The first call lays out every
  /// threadgroup variable \p F uses, in the order of the module (as PTX
  /// does), the ones without a size (dynamic shared memory) last.
  unsigned allocateThreadgroupVariable(const Function &F,
                                       const GlobalVariable &GV);
  unsigned getThreadgroupSize() const { return ThreadgroupSize; }
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AGX_AGXMACHINEFUNCTIONINFO_H
