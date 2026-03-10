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
#include "NVPTXUtilities.h"
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

  const auto OverallReqNTID = getOverallReqNTID(F);
  const auto OverallMaxNTID = getOverallMaxNTID(F);
  const auto OverallClusterRank = getOverallClusterRank(F);

  // If this function lacks any range information, do nothing.
  if (!(OverallReqNTID || OverallMaxNTID || OverallClusterRank))
    return false;

  const unsigned FunctionNTID = OverallReqNTID.value_or(
      OverallMaxNTID.value_or(std::numeric_limits<unsigned>::max()));

  const unsigned FunctionClusterRank =
      OverallClusterRank.value_or(std::numeric_limits<unsigned>::max());

  const Vector3 MaxBlockSize{std::min(1024u, FunctionNTID),
                             std::min(1024u, FunctionNTID),
                             std::min(64u, FunctionNTID)};

  // We conservatively use the maximum grid size as an upper bound for the
  // cluster rank.
  const Vector3 MaxClusterRank{std::min(0x7fffffffu, FunctionClusterRank),
                               std::min(0xffffu, FunctionClusterRank),
                               std::min(0xffffu, FunctionClusterRank)};

  const auto ProccessIntrinsic = [&](IntrinsicInst *II) -> bool {
    switch (II->getIntrinsicID()) {
    // Index within block
    case Intrinsic::nvvm_read_ptx_sreg_tid_x:
      return addRangeAttr(0, MaxBlockSize.X, II);
    case Intrinsic::nvvm_read_ptx_sreg_tid_y:
      return addRangeAttr(0, MaxBlockSize.Y, II);
    case Intrinsic::nvvm_read_ptx_sreg_tid_z:
      return addRangeAttr(0, MaxBlockSize.Z, II);

    // Block size
    case Intrinsic::nvvm_read_ptx_sreg_ntid_x:
      return addRangeAttr(1, MaxBlockSize.X + 1, II);
    case Intrinsic::nvvm_read_ptx_sreg_ntid_y:
      return addRangeAttr(1, MaxBlockSize.Y + 1, II);
    case Intrinsic::nvvm_read_ptx_sreg_ntid_z:
      return addRangeAttr(1, MaxBlockSize.Z + 1, II);

    // Cluster size
    case Intrinsic::nvvm_read_ptx_sreg_cluster_ctaid_x:
      return addRangeAttr(0, MaxClusterRank.X, II);
    case Intrinsic::nvvm_read_ptx_sreg_cluster_ctaid_y:
      return addRangeAttr(0, MaxClusterRank.Y, II);
    case Intrinsic::nvvm_read_ptx_sreg_cluster_ctaid_z:
      return addRangeAttr(0, MaxClusterRank.Z, II);
    case Intrinsic::nvvm_read_ptx_sreg_cluster_nctaid_x:
      return addRangeAttr(1, MaxClusterRank.X + 1, II);
    case Intrinsic::nvvm_read_ptx_sreg_cluster_nctaid_y:
      return addRangeAttr(1, MaxClusterRank.Y + 1, II);
    case Intrinsic::nvvm_read_ptx_sreg_cluster_nctaid_z:
      return addRangeAttr(1, MaxClusterRank.Z + 1, II);

    case Intrinsic::nvvm_read_ptx_sreg_cluster_ctarank:
      if (OverallClusterRank)
        return addRangeAttr(0, FunctionClusterRank, II);
      break;
    case Intrinsic::nvvm_read_ptx_sreg_cluster_nctarank:
      if (OverallClusterRank)
        return addRangeAttr(1, FunctionClusterRank + 1, II);
      break;
    default:
      return false;
    }
    return false;
  };

  // Go through the calls in this function.
  bool Changed = false;
  for (Instruction &I : instructions(F))
    if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I))
      Changed |= ProccessIntrinsic(II);

  return Changed;
}

bool NVVMIntrRange::runOnFunction(Function &F) { return runNVVMIntrRange(F); }

PreservedAnalyses NVVMIntrRangePass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  return runNVVMIntrRange(F) ? PreservedAnalyses::none()
                             : PreservedAnalyses::all();
}
