//===--- MisExpect.cpp - Check the use of llvm.expect with PGO data -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit warnings for potentially incorrect usage of the
// llvm.expect intrinsic. This utility extracts the threshold values from
// metadata associated with the instrumented Branch or Switch instruction. The
// threshold values are then used to determine if a warning should be emmited.
//
// MisExpect metadata is generated when llvm.expect intrinsics are lowered see
// LowerExpectIntrinsic.cpp
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/MisExpect.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstdint>
#include <functional>
#include <numeric>

#define DEBUG_TYPE "misexpect"

using namespace llvm;
using namespace misexpect;

namespace llvm {

// Command line option to enable/disable the warning when profile data suggests
// a mismatch with the use of the llvm.expect intrinsic
static cl::opt<bool> PGOWarnMisExpect(
    "pgo-warn-misexpect", cl::init(false), cl::Hidden,
    cl::desc("Use this option to turn on/off "
             "warnings about incorrect usage of llvm.expect intrinsics."));

} // namespace llvm

namespace {

Instruction *getOprndOrInst(Instruction *I) {
  assert(I != nullptr && "MisExpect target Instruction cannot be nullptr");
  Instruction *Ret = nullptr;
  if (auto *B = dyn_cast<BranchInst>(I)) {
    Ret = dyn_cast<Instruction>(B->getCondition());
  }
  // TODO: Find a way to resolve condition location for switches
  // Using the condition of the switch seems to often resolve to an earlier
  // point in the program, i.e. the calculation of the switch condition, rather
  // than the switches location in the source code. Thus, we should use the
  // instruction to get source code locations rather than the condition to
  // improve diagnostic output, such as the caret. If the same problem exists
  // for branch instructions, then we should remove this function and directly
  // use the instruction
  //
  // else if (auto S = dyn_cast<SwitchInst>(I)) {
  // Ret = I;
  //}
  return Ret ? Ret : I;
}

void emitMisexpectDiagnostic(Instruction *I, LLVMContext &Ctx,
                             uint64_t ProfCount, uint64_t TotalCount) {
  double PercentageCorrect = (double)ProfCount / TotalCount;
  auto PerString =
      formatv("{0:P} ({1} / {2})", PercentageCorrect, ProfCount, TotalCount);
  auto RemStr = formatv(
      "Potential performance regression from use of the llvm.expect intrinsic: "
      "Annotation was correct on {0} of profiled executions.",
      PerString);
  Twine Msg(PerString);
  Instruction *Cond = getOprndOrInst(I);
  if (PGOWarnMisExpect)
    Ctx.diagnose(DiagnosticInfoMisExpect(Cond, Msg));
  OptimizationRemarkEmitter ORE(I->getParent()->getParent());
  ORE.emit(OptimizationRemark(DEBUG_TYPE, "misexpect", Cond) << RemStr.str());
}

} // namespace

namespace llvm {
namespace misexpect {

void verifyMisExpect(Instruction *I, const SmallVector<uint32_t, 4> &Weights,
                     LLVMContext &Ctx) {
  if (auto *MisExpectData = I->getMetadata(LLVMContext::MD_misexpect)) {
    auto *MisExpectDataName = dyn_cast<MDString>(MisExpectData->getOperand(0));
    if (MisExpectDataName &&
        MisExpectDataName->getString().equals("misexpect")) {
      LLVM_DEBUG(llvm::dbgs() << "------------------\n");
      LLVM_DEBUG(llvm::dbgs()
                 << "Function: " << I->getFunction()->getName() << "\n");
      LLVM_DEBUG(llvm::dbgs() << "Instruction: " << *I << ":\n");
      LLVM_DEBUG(for (int Idx = 0, Size = Weights.size(); Idx < Size; ++Idx) {
        llvm::dbgs() << "Weights[" << Idx << "] = " << Weights[Idx] << "\n";
      });

      // extract values from misexpect metadata
      const auto *IndexCint =
          mdconst::dyn_extract<ConstantInt>(MisExpectData->getOperand(1));
      const auto *LikelyCInt =
          mdconst::dyn_extract<ConstantInt>(MisExpectData->getOperand(2));
      const auto *UnlikelyCInt =
          mdconst::dyn_extract<ConstantInt>(MisExpectData->getOperand(3));

      if (!IndexCint || !LikelyCInt || !UnlikelyCInt)
        return;

      const uint64_t Index = IndexCint->getZExtValue();
      const uint64_t LikelyBranchWeight = LikelyCInt->getZExtValue();
      const uint64_t UnlikelyBranchWeight = UnlikelyCInt->getZExtValue();
      const uint64_t ProfileCount = Weights[Index];
      const uint64_t CaseTotal = std::accumulate(
          Weights.begin(), Weights.end(), (uint64_t)0, std::plus<uint64_t>());
      const uint64_t NumUnlikelyTargets = Weights.size() - 1;

      const uint64_t TotalBranchWeight =
          LikelyBranchWeight + (UnlikelyBranchWeight * NumUnlikelyTargets);

      const llvm::BranchProbability LikelyThreshold(LikelyBranchWeight,
                                                    TotalBranchWeight);
      uint64_t ScaledThreshold = LikelyThreshold.scale(CaseTotal);

      LLVM_DEBUG(llvm::dbgs()
                 << "Unlikely Targets: " << NumUnlikelyTargets << ":\n");
      LLVM_DEBUG(llvm::dbgs() << "Profile Count: " << ProfileCount << ":\n");
      LLVM_DEBUG(llvm::dbgs()
                 << "Scaled Threshold: " << ScaledThreshold << ":\n");
      LLVM_DEBUG(llvm::dbgs() << "------------------\n");
      if (ProfileCount < ScaledThreshold)
        emitMisexpectDiagnostic(I, Ctx, ProfileCount, CaseTotal);
    }
  }
}

void checkFrontendInstrumentation(Instruction &I) {
  if (auto *MD = I.getMetadata(LLVMContext::MD_prof)) {
    unsigned NOps = MD->getNumOperands();

    // Only emit misexpect diagnostics if at least 2 branch weights are present.
    // Less than 2 branch weights means that the profiling metadata is:
    //    1) incorrect/corrupted
    //    2) not branch weight metadata
    //    3) completely deterministic
    // In these cases we should not emit any diagnostic related to misexpect.
    if (NOps < 3)
      return;

    // Operand 0 is a string tag "branch_weights"
    if (MDString *Tag = cast<MDString>(MD->getOperand(0))) {
      if (Tag->getString().equals("branch_weights")) {
        SmallVector<uint32_t, 4> RealWeights(NOps - 1);
        for (unsigned i = 1; i < NOps; i++) {
          ConstantInt *Value =
              mdconst::dyn_extract<ConstantInt>(MD->getOperand(i));
          RealWeights[i - 1] = Value->getZExtValue();
        }
        verifyMisExpect(&I, RealWeights, I.getContext());
      }
    }
  }
}

} // namespace misexpect
} // namespace llvm
#undef DEBUG_TYPE
