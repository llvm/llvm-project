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
// MisExpect's implementation relies on two assumptions about how branch weights
// are managed in LLVM.
//
// 1) Frontend profiling weights are always in place before llvm.expect is
// lowered in LowerExpectIntrinsic.cpp. Frontend based instrumentation therefore
// needs to extract the branch weights and then compare them to the weights
// being added by the llvm.expect intrinsic lowering.
//
// 2) Sampling and IR based profiles will *only* have branch weight metadata
// before profiling data is consulted if they are from a lowered llvm.expect
// intrinsic. These profiles thus always extract the expected weights and then
// compare them to the weights collected during profiling to determine if a
// diagnostic message is warranted.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/MisExpect.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/ProfDataUtils.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include <algorithm>
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

// Command line option for setting the diagnostic tolerance threshold
static cl::opt<uint32_t> MisExpectTolerance(
    "misexpect-tolerance", cl::init(0),
    cl::desc("Prevents emitting diagnostics when profile counts are "
             "within N% of the threshold.."));

// Command line option to enable/disable the Remarks when profile data suggests
// that the llvm.expect intrinsic may be profitable
static cl::opt<bool>
    PGOMissingAnnotations("pgo-missing-annotations", cl::init(false),
                          cl::Hidden,
                          cl::desc("Use this option to turn on/off suggestions "
                                   "of missing llvm.expect intrinsics."));
} // namespace llvm

namespace {
struct ProfDataSummary {
  uint64_t Likely;
  uint64_t Unlikely;
  uint64_t RealTotal;
  uint64_t NumUnlikely;
};

enum class DiagKind {
  MisExpect,     // Reports when llvm.expect usage is contradicted by PGO data
  MissingExpect, // Reports when llvm.expect would be profitable
};

uint64_t getScaledThreshold(const ProfDataSummary &PDS) {
  uint64_t TotalBranchWeight = PDS.Likely + (PDS.Unlikely * PDS.NumUnlikely);

  LLVM_DEBUG(dbgs() << "Total Branch Weight = " << TotalBranchWeight << "\n"
                    << "Likely Branch Weight = " << PDS.Likely << "\n");

  // Failing this assert means that we have corrupted metadata.
  assert((TotalBranchWeight >= PDS.Likely) && (TotalBranchWeight > 0)
               && "TotalBranchWeight is less than the Likely branch weight");

  // To determine our threshold value we need to obtain the branch probability
  // for the weights added by llvm.expect and use that proportion to calculate
  // our threshold based on the collected profile data.
  BranchProbability LikelyProbablilty =
      BranchProbability::getBranchProbability(PDS.Likely, TotalBranchWeight);

  return LikelyProbablilty.scale(PDS.RealTotal);
}

bool isAnnotationDiagEnabled(LLVMContext &Ctx) {
  LLVM_DEBUG(dbgs() << "PGOMissingAnnotations = " << PGOMissingAnnotations
                    << "\n");
  return PGOMissingAnnotations || Ctx.getAnnotationDiagsRequested();
}

bool isMisExpectDiagEnabled(LLVMContext &Ctx) {
  return PGOWarnMisExpect || Ctx.getMisExpectWarningRequested();
}

uint32_t getMisExpectTolerance(LLVMContext &Ctx) {
  return std::max(static_cast<uint32_t>(MisExpectTolerance),
                  Ctx.getDiagnosticsMisExpectTolerance());
}

Instruction *getInstCondition(Instruction *I) {
  assert(I != nullptr && "MisExpect target Instruction cannot be nullptr");
  Instruction *Ret = nullptr;
  if (auto *B = dyn_cast<BranchInst>(I)) {
    Ret = dyn_cast<Instruction>(B->getCondition());
  }
  // TODO: Find a way to resolve condition location for switches
  // Using the condition of the switch seems to often resolve to an earlier
  // point in the program, i.e. the calculation of the switch condition, rather
  // than the switch's location in the source code. Thus, we should use the
  // instruction to get source code locations rather than the condition to
  // improve diagnostic output, such as the caret. If the same problem exists
  // for branch instructions, then we should remove this function and directly
  // use the instruction
  //
  else if (auto *S = dyn_cast<SwitchInst>(I)) {
    Ret = dyn_cast<Instruction>(S->getCondition());
  }
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
  Instruction *Cond = getInstCondition(I);
  if (isMisExpectDiagEnabled(Ctx))
    Ctx.diagnose(DiagnosticInfoMisExpect(Cond, Msg));
  OptimizationRemarkEmitter ORE(I->getFunction());
  ORE.emit(OptimizationRemark(DEBUG_TYPE, "misexpect", Cond) << RemStr.str());
}


void emitMissingAnnotationDiag(Instruction *I) {
  const auto *RemStr =
      "Extremely hot condition. Consider adding llvm.expect intrinsic";
  Instruction *Cond = getInstCondition(I);
  OptimizationRemarkEmitter ORE(I->getParent()->getParent());
  ORE.emit(
      OptimizationRemark("missing-annotations", "missing-annotations", Cond)
      << RemStr);
}

uint64_t totalWeight(const ArrayRef<uint32_t> Weights) {
  return std::accumulate(Weights.begin(), Weights.end(), (uint64_t)0,
                         std::plus<uint64_t>());
}

void scaleByTollerance(const Instruction &I, uint64_t &ScaledThreshold) {
  // clamp tolerance range to [0, 100)
  uint32_t Tolerance = getMisExpectTolerance(I.getContext());
  Tolerance = std::clamp(Tolerance, 0u, 99u);

  // Allow users to relax checking by N%  i.e., if they use a 5% tolerance,
  // then we check against 0.95*ScaledThreshold
  if (Tolerance > 0)
    ScaledThreshold *= (1.0 - Tolerance / 100.0);

  LLVM_DEBUG(dbgs() << "Scaled Threshold = " << ScaledThreshold << "\n");
}

void reportDiagnostics(Instruction &I, const ProfDataSummary &PDS,
                       uint32_t ProfiledWeight, DiagKind Kind) {
  uint64_t ScaledThreshold =  getScaledThreshold(PDS);
  scaleByTollerance(I, ScaledThreshold);

  LLVM_DEBUG(dbgs() << "Total Branch Weight = " << PDS.RealTotal << "\n"
                    << "Scaled Threshold = " << ScaledThreshold << "\n"
                    << "Profiled Weight = " << ProfiledWeight << "\n"
                    << "Likely Branch Weight = " << PDS.Likely << "\n");
  // When the profile weight is outside the range, we emit the diagnostic
  switch (Kind) {
  case DiagKind::MisExpect:
    if (ProfiledWeight < ScaledThreshold) {
      emitMisexpectDiagnostic(&I, I.getContext(), ProfiledWeight,
                              PDS.RealTotal);
    }
    return;
  case DiagKind::MissingExpect:
    if (ProfiledWeight > ScaledThreshold) {
      emitMissingAnnotationDiag(&I);
    }
    return;
  };
}

} // namespace

namespace llvm {
namespace misexpect {

void verifyMisExpect(Instruction &I, ArrayRef<uint32_t> RealWeights,
                     ArrayRef<uint32_t> ExpectedWeights) {
  // To determine if we emit a diagnostic, we need to compare the branch weights
  // from the profile to those added by the llvm.expect intrinsic.
  // So first, we extract the "likely" and "unlikely" weights from
  // ExpectedWeights and determine the correct weight in the profile to compare
  // against.
  uint64_t LikelyBranchWeight = 0,
           UnlikelyBranchWeight = std::numeric_limits<uint32_t>::max();
  size_t MaxIndex = 0;
  for (size_t Idx = 0, End = ExpectedWeights.size(); Idx < End; Idx++) {
    uint32_t V = ExpectedWeights[Idx];
    if (LikelyBranchWeight < V) {
      LikelyBranchWeight = V;
      MaxIndex = Idx;
    }
    if (UnlikelyBranchWeight > V) {
      UnlikelyBranchWeight = V;
    }
  }

  const uint64_t ProfiledWeight = RealWeights[MaxIndex];
  const ProfDataSummary PDS = {LikelyBranchWeight, UnlikelyBranchWeight,
                               totalWeight(RealWeights),
                               RealWeights.size() - 1};
  reportDiagnostics(I,PDS, ProfiledWeight, DiagKind::MisExpect);
}

void checkBackendInstrumentation(Instruction &I,
                                 const ArrayRef<uint32_t> RealWeights) {
  // Backend checking assumes any existing weight comes from an `llvm.expect`
  // intrinsic. However, SampleProfiling + ThinLTO add branch weights  multiple
  // times, leading to an invalid assumption in our checking. Backend checks
  // should only operate on branch weights that carry the "!expected" field,
  // since they are guaranteed to be added by the LowerExpectIntrinsic pass.
  if (!hasBranchWeightOrigin(I))
    return;
  SmallVector<uint32_t> ExpectedWeights;
  if (!extractBranchWeights(I, ExpectedWeights))
    return;
  verifyMisExpect(I, RealWeights, ExpectedWeights);
}

void checkFrontendInstrumentation(Instruction &I,
                                  const ArrayRef<uint32_t> ExpectedWeights) {
  SmallVector<uint32_t> RealWeights;
  if (!extractBranchWeights(I, RealWeights))
    return;
  verifyMisExpect(I, RealWeights, ExpectedWeights);
}

void checkExpectAnnotations(Instruction &I,
                            const ArrayRef<uint32_t> ExistingWeights,
                            bool IsFrontend) {
  if (IsFrontend) {
    checkFrontendInstrumentation(I, ExistingWeights);
  } else {
    checkBackendInstrumentation(I, ExistingWeights);
  }
}

void verifyMissingAnnotations(Instruction &I, ArrayRef<uint32_t> RealWeights) {
  // To determine if we emit a diagnostic, we need to compare the branch weights
  // from the profile to those that would be added by the llvm.expect intrinsic
  // and compare it to the real profile to see if it would be profitable.
  uint32_t ProfiledWeight =
      *std::max_element(RealWeights.begin(), RealWeights.end());

  const uint64_t LikelyBranchWeight = 2000;
  const uint64_t UnlikelyBranchWeight = 1;
  const ProfDataSummary PDS = {LikelyBranchWeight, UnlikelyBranchWeight,
                               totalWeight(RealWeights),
                               RealWeights.size() - 1};
  reportDiagnostics(I, PDS, ProfiledWeight, DiagKind::MissingExpect);
}

void checkMissingAnnotations(Instruction &I,
                             const ArrayRef<uint32_t> ExistingWeights,
                             bool IsFrontendInstr) {

  //  exit early if these diagnostics weren't requested
  if (LLVM_LIKELY(!isAnnotationDiagEnabled(I.getContext())))
    return;

  if (IsFrontendInstr) {
    // TODO: Frontend checking will have to be thought through, since we need
    // to do the check on branches that don't have expect intrinsics
  } else {
    SmallVector<uint32_t> ExpectedWeights;
    if (extractBranchWeights(I, ExpectedWeights))
      return;
    verifyMissingAnnotations(I, ExistingWeights);
  }
}

} // namespace misexpect
} // namespace llvm
#undef DEBUG_TYPE
