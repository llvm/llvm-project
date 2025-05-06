#if LLPC_BUILD_NPI
//===- AMDGPUVGPRIndexingAnalysis.h - VGPR indexing analysis --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Determine properties of VGPR indexing instructions.
/// Currently, only determine VGPR usage intervals.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUVGPRINDEXINGANALYSIS_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUVGPRINDEXINGANALYSIS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include <unordered_map>

namespace llvm {

class GCNSubtarget;
class MachineFunction;
class TargetMachine;
class SIMachineFunctionInfo;
class SIRegisterInfo;
class MachineRegisterInfo;
class SIInstrInfo;

std::pair<unsigned, unsigned>
getAMDGPUPrivateObjectNodeInfo(const MDNode *Node);

struct AMDGPULaneSharedIdxInfo {
  // The implicitly used regs for the MI this Info is associated with, only
  // available if precisely known.
  std::optional<std::pair<unsigned, unsigned>> UsedRegs;
  AMDGPULaneSharedIdxInfo() = delete;
};

struct AMDGPUPrivateObjectIdxInfo {
  // The implicitly used regs for the MI this Info is associated with, only
  // available if precisely known.
  std::optional<std::pair<unsigned, unsigned>> UsedRegs;
  // Pointer to a promoted private object, if applicable
  const MDNode *Obj;
  // Size of the object
  unsigned Size;
  // Offset of the object in the private segment
  unsigned Offset;
  AMDGPUPrivateObjectIdxInfo() = delete;
};

// Track indexing info for all V_LOAD_IDX and V_STORE_IDX instructions
class AMDGPUIndexingInfo {
private:
  std::unordered_map<const MachineInstr *, AMDGPULaneSharedIdxInfo>
      LaneSharedIdxInfos;
  std::unordered_map<const MachineInstr *, AMDGPUPrivateObjectIdxInfo>
      PrivateObjectIdxInfos;
  unsigned LaneSharedSize = 0;

public:
  friend class AMDGPUVGPRIndexingAnalysis;
  friend class AMDGPUIndexingInfoWrapper;

  std::optional<std::reference_wrapper<const AMDGPULaneSharedIdxInfo>>
  getLaneSharedIdxInfo(const MachineInstr *MI) const;

  std::optional<std::reference_wrapper<const AMDGPUPrivateObjectIdxInfo>>
  getPrivateObjectIdxInfo(const MachineInstr *MI) const;

  unsigned getLaneSharedSize() const { return LaneSharedSize; }
  AMDGPUIndexingInfo() {}
  AMDGPUIndexingInfo(unsigned LaneSharedSize)
      : LaneSharedSize(LaneSharedSize) {}
};

// Analysis pass that exposes the \c AMDGPUIndexingInfo for a machine function
class AMDGPUVGPRIndexingAnalysis
    : public AnalysisInfoMixin<AMDGPUVGPRIndexingAnalysis> {
  friend AnalysisInfoMixin<AMDGPUVGPRIndexingAnalysis>;
  static AnalysisKey Key;

public:
  using Result = AMDGPUIndexingInfo;
  Result run(MachineFunction &MF, MachineFunctionAnalysisManager &MFAM);
};

class AMDGPUIndexingInfoWrapper : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUIndexingInfoWrapper() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU VGPR Indexing Analysis";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  const AMDGPUIndexingInfo &getIndexingInfo() const { return IndexingInfo; }

private:
  AMDGPUIndexingInfo IndexingInfo;
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUVGPRINDEXINGANALYSIS_H
#endif /* LLPC_BUILD_NPI */
