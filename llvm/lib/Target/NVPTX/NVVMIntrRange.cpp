//===- NVVMIntrRange.cpp - Set range attributes for NVVM intrinsics -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass adds appropriate range attributes for calls to NVVM
// intrinsics that return a limited range of values.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "NVVMProperties.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/PassManager.h"
#include <cstdint>

using namespace llvm;

#define DEBUG_TYPE "nvvm-intr-range"

namespace {
class NVVMIntrRange : public FunctionPass {
public:
  static char ID;
  NVVMIntrRange() : FunctionPass(ID) {}

  bool runOnFunction(Function &) override;
};
} // namespace

FunctionPass *llvm::createNVVMIntrRangePass() { return new NVVMIntrRange(); }

char NVVMIntrRange::ID = 0;
INITIALIZE_PASS(NVVMIntrRange, "nvvm-intr-range",
                "Add !range metadata to NVVM intrinsics.", false, false)

// Adds the passed-in [Low,High) range information as metadata to the
// passed-in call instruction.
static bool addRangeAttr(uint64_t Low, uint64_t High, IntrinsicInst *II) {
  if (II->getMetadata(LLVMContext::MD_range))
    return false;

  const uint64_t BitWidth = II->getType()->getIntegerBitWidth();
  ConstantRange Range(APInt(BitWidth, Low), APInt(BitWidth, High));

  if (auto CurrentRange = II->getRange())
    Range = Range.intersectWith(CurrentRange.value());

  II->addRangeRetAttr(Range);
  return true;
}

static bool runNVVMIntrRange(Function &F) {
  struct Vector3 {
    unsigned X, Y, Z;
  };

  // All these annotations are only valid for kernel functions.
  if (!isKernelFunction(F))
    return false;

  auto ReqNTID = getReqNTID(F);
  const auto OverallMaxNTID = getOverallMaxNTID(F);
  auto ClusterDim = getClusterDim(F);
  const auto MaxClusterRank = getMaxClusterRank(F);

  // If this function lacks any range information, do nothing.
  if (ReqNTID.empty() && !OverallMaxNTID && ClusterDim.empty() &&
      !MaxClusterRank)
    return false;

  const unsigned MaxNTID =
      OverallMaxNTID.value_or(std::numeric_limits<unsigned>::max());

  // When reqntid is specified, block dimensions are exact compile-time
  // constants. Otherwise, use maxntid (capped at hardware limits) as upper
  // bounds.
  Vector3 MinBlockDim, MaxBlockDim;
  if (!ReqNTID.empty()) {
    ReqNTID.resize(3, 1);
    MinBlockDim = MaxBlockDim = {ReqNTID[0], ReqNTID[1], ReqNTID[2]};
  } else {
    MinBlockDim = {1, 1, 1};
    MaxBlockDim = {std::min(1024u, MaxNTID), std::min(1024u, MaxNTID),
                   std::min(64u, MaxNTID)};
  }

  const bool HasClusterInfo = !ClusterDim.empty() || MaxClusterRank;

  // When cluster_dim is specified, cluster dimensions are exact compile-time
  // constants. Otherwise, use maxclusterrank (capped at hardware limits) as
  // upper bounds.
  Vector3 MinClusterDim, MaxClusterDim;
  uint64_t MinClusterSize, MaxClusterSize;
  if (!ClusterDim.empty()) {
    ClusterDim.resize(3, 1);
    MinClusterDim =
        MaxClusterDim = {ClusterDim[0], ClusterDim[1], ClusterDim[2]};
    MinClusterSize = MaxClusterSize =
        ClusterDim[0] * ClusterDim[1] * ClusterDim[2];
  } else {
    const unsigned MaxNctaPerCluster =
        MaxClusterRank.value_or(std::numeric_limits<unsigned>::max());
    MinClusterDim = {1, 1, 1};
    MaxClusterDim = {std::min(0x7fffffffu, MaxNctaPerCluster),
                     std::min(0xffffu, MaxNctaPerCluster),
                     std::min(0xffffu, MaxNctaPerCluster)};
    MinClusterSize = 1;
    MaxClusterSize = MaxNctaPerCluster;
  }

  const auto ProcessIntrinsic = [&](IntrinsicInst *II) -> bool {
    switch (II->getIntrinsicID()) {
    // Index within block
    case Intrinsic::nvvm_read_ptx_sreg_tid_x:
      return addRangeAttr(0, MaxBlockDim.X, II);
    case Intrinsic::nvvm_read_ptx_sreg_tid_y:
      return addRangeAttr(0, MaxBlockDim.Y, II);
    case Intrinsic::nvvm_read_ptx_sreg_tid_z:
      return addRangeAttr(0, MaxBlockDim.Z, II);

    // Block size: use single-value range when reqntid is specified;
    // InstCombine will fold these to constants later.
    case Intrinsic::nvvm_read_ptx_sreg_ntid_x:
      return addRangeAttr(MinBlockDim.X, MaxBlockDim.X + 1, II);
    case Intrinsic::nvvm_read_ptx_sreg_ntid_y:
      return addRangeAttr(MinBlockDim.Y, MaxBlockDim.Y + 1, II);
    case Intrinsic::nvvm_read_ptx_sreg_ntid_z:
      return addRangeAttr(MinBlockDim.Z, MaxBlockDim.Z + 1, II);

    // Cluster size: use single-value ranges when cluster_dim is specified;
    // InstCombine will fold cluster_nctaid.* / cluster_nctarank to constants
    // later.
    case Intrinsic::nvvm_read_ptx_sreg_cluster_ctaid_x:
      return addRangeAttr(0, MaxClusterDim.X, II);
    case Intrinsic::nvvm_read_ptx_sreg_cluster_ctaid_y:
      return addRangeAttr(0, MaxClusterDim.Y, II);
    case Intrinsic::nvvm_read_ptx_sreg_cluster_ctaid_z:
      return addRangeAttr(0, MaxClusterDim.Z, II);
    case Intrinsic::nvvm_read_ptx_sreg_cluster_nctaid_x:
      return addRangeAttr(MinClusterDim.X, MaxClusterDim.X + 1, II);
    case Intrinsic::nvvm_read_ptx_sreg_cluster_nctaid_y:
      return addRangeAttr(MinClusterDim.Y, MaxClusterDim.Y + 1, II);
    case Intrinsic::nvvm_read_ptx_sreg_cluster_nctaid_z:
      return addRangeAttr(MinClusterDim.Z, MaxClusterDim.Z + 1, II);

    case Intrinsic::nvvm_read_ptx_sreg_cluster_ctarank:
      return HasClusterInfo && addRangeAttr(0, MaxClusterSize, II);
    case Intrinsic::nvvm_read_ptx_sreg_cluster_nctarank:
      return HasClusterInfo &&
             addRangeAttr(MinClusterSize, MaxClusterSize + 1, II);
    default:
      return false;
    }
  };

  // Go through the calls in this function.
  bool Changed = false;
  for (Instruction &I : instructions(F))
    if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I))
      Changed |= ProcessIntrinsic(II);

  return Changed;
}

bool NVVMIntrRange::runOnFunction(Function &F) { return runNVVMIntrRange(F); }

PreservedAnalyses NVVMIntrRangePass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  return runNVVMIntrRange(F) ? PreservedAnalyses::none()
                             : PreservedAnalyses::all();
}
