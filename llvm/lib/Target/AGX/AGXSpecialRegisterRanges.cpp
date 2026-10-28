//===-- AGXSpecialRegisterRanges.cpp - Bounds of special registers --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The thread and threadgroup indices have known bounds: a threadgroup has at
// most 1024 threads, a SIMD-group 32, and grid positions and sizes are below
// 2^31 (CUDA's limits, within which ZLUDA launches). This pass gives the
// special register reads range return attributes just before instruction
// selection, which turns them into AssertZext nodes: signed divisions and
// remainders of indices by powers of two then select as shifts and masks.
//
// The bounds are not given earlier (as NVPTX does in the intrinsic table):
// with them, IndVarSimplify widens induction variables to 64 bits and loop
// strength reduction adds 64-bit byte offsets next to the 32-bit indices,
// which costs registers (llvm/utils/agx-bench EXPERIMENTS.md L14-E1).
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAGX.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "agx-special-register-ranges"

static cl::opt<bool> AGXSpecialRegisterRanges(
    "agx-sr-ranges", cl::init(true), cl::Hidden,
    cl::desc("Bound the thread and threadgroup indices before instruction "
             "selection"));

namespace {
class AGXSpecialRegisterRangesPass : public FunctionPass {
public:
  static char ID;
  AGXSpecialRegisterRangesPass() : FunctionPass(ID) {}
  StringRef getPassName() const override {
    return "AGX Special Register Ranges";
  }
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};
} // namespace

char AGXSpecialRegisterRangesPass::ID = 0;

INITIALIZE_PASS(AGXSpecialRegisterRangesPass, DEBUG_TYPE,
                "AGX Special Register Ranges", false, false)

FunctionPass *llvm::createAGXSpecialRegisterRangesPass() {
  return new AGXSpecialRegisterRangesPass();
}

/// [Lo, Hi) of the special register an intrinsic reads, if bounded.
static std::optional<std::pair<uint64_t, uint64_t>> bounds(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::agx_thread_position_in_threadgroup_x:
  case Intrinsic::agx_thread_position_in_threadgroup_y:
  case Intrinsic::agx_thread_position_in_threadgroup_z:
  case Intrinsic::agx_thread_index_in_threadgroup:
    return std::pair(0, 1024);
  case Intrinsic::agx_threads_per_threadgroup_x:
  case Intrinsic::agx_threads_per_threadgroup_y:
  case Intrinsic::agx_threads_per_threadgroup_z:
    return std::pair(1, 1025);
  case Intrinsic::agx_thread_index_in_simdgroup:
  case Intrinsic::agx_simdgroup_index_in_threadgroup:
    return std::pair(0, 32);
  case Intrinsic::agx_threadgroup_position_in_grid_x:
  case Intrinsic::agx_threadgroup_position_in_grid_y:
  case Intrinsic::agx_threadgroup_position_in_grid_z:
    return std::pair(0, 0x7fffffff);
  case Intrinsic::agx_threadgroups_per_grid_x:
  case Intrinsic::agx_threadgroups_per_grid_y:
  case Intrinsic::agx_threadgroups_per_grid_z:
    return std::pair(1, 0x80000000);
  default:
    return std::nullopt;
  }
}

bool AGXSpecialRegisterRangesPass::runOnFunction(Function &F) {
  if (!AGXSpecialRegisterRanges || skipFunction(F))
    return false;
  bool Changed = false;
  for (Instruction &I : instructions(F)) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II || !II->getType()->isIntegerTy(32))
      continue;
    auto B = bounds(II->getIntrinsicID());
    if (!B || II->getRetAttributes().hasAttribute(Attribute::Range))
      continue;
    II->addRetAttr(Attribute::get(
        F.getContext(), Attribute::Range,
        ConstantRange(APInt(32, B->first), APInt(32, B->second))));
    Changed = true;
  }
  return Changed;
}
