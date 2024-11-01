//===- FunctionSpecialization.h - Function Specialization -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This specialises functions with constant parameters. Constant parameters
// like function pointers and constant globals are propagated to the callee by
// specializing the function. The main benefit of this pass at the moment is
// that indirect calls are transformed into direct calls, which provides inline
// opportunities that the inliner would not have been able to achieve. That's
// why function specialisation is run before the inliner in the optimisation
// pipeline; that is by design. Otherwise, we would only benefit from constant
// passing, which is a valid use-case too, but hasn't been explored much in
// terms of performance uplifts, cost-model and compile-time impact.
//
// Current limitations:
// - It does not yet handle integer ranges. We do support "literal constants",
//   but that's off by default under an option.
// - The cost-model could be further looked into (it mainly focuses on inlining
//   benefits),
//
// Ideas:
// - With a function specialization attribute for arguments, we could have
//   a direct way to steer function specialization, avoiding the cost-model,
//   and thus control compile-times / code-size.
//
// Todos:
// - Specializing recursive functions relies on running the transformation a
//   number of times, which is controlled by option
//   `func-specialization-max-iters`. Thus, increasing this value and the
//   number of iterations, will linearly increase the number of times recursive
//   functions get specialized, see also the discussion in
//   https://reviews.llvm.org/D106426 for details. Perhaps there is a
//   compile-time friendlier way to control/limit the number of specialisations
//   for recursive functions.
// - Don't transform the function if function specialization does not trigger;
//   the SCCPSolver may make IR changes.
//
// References:
// - 2021 LLVM Dev Mtg “Introducing function specialisation, and can we enable
//   it by default?”, https://www.youtube.com/watch?v=zJiCjeXgV5Q
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_FUNCTIONSPECIALIZATION_H
#define LLVM_TRANSFORMS_IPO_FUNCTIONSPECIALIZATION_H

#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Transforms/Scalar/SCCP.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/SCCPSolver.h"
#include "llvm/Transforms/Utils/SizeOpts.h"

using namespace llvm;

namespace llvm {
// Bookkeeping struct to pass data from the analysis and profitability phase
// to the actual transform helper functions.
struct SpecializationInfo {
  SmallVector<ArgInfo, 8> Args; // Stores the {formal,actual} argument pairs.
  InstructionCost Gain;         // Profitability: Gain = Bonus - Cost.
  Function *Clone;              // The definition of the specialized function.
};

using CallSpecBinding = std::pair<CallBase *, SpecializationInfo>;
// We are using MapVector because it guarantees deterministic iteration
// order across executions.
using SpecializationMap = SmallMapVector<CallBase *, SpecializationInfo, 8>;

class FunctionSpecializer {

  /// The IPSCCP Solver.
  SCCPSolver &Solver;

  Module &M;

  /// Analysis manager, needed to invalidate analyses.
  FunctionAnalysisManager *FAM;

  /// Analyses used to help determine if a function should be specialized.
  std::function<const TargetLibraryInfo &(Function &)> GetTLI;
  std::function<TargetTransformInfo &(Function &)> GetTTI;
  std::function<AssumptionCache &(Function &)> GetAC;

  // The number of functions specialised, used for collecting statistics and
  // also in the cost model.
  unsigned NbFunctionsSpecialized = 0;

  SmallPtrSet<Function *, 32> SpecializedFuncs;
  SmallPtrSet<Function *, 32> FullySpecialized;
  DenseMap<Function *, CodeMetrics> FunctionMetrics;

public:
  FunctionSpecializer(
      SCCPSolver &Solver, Module &M, FunctionAnalysisManager *FAM,
      std::function<const TargetLibraryInfo &(Function &)> GetTLI,
      std::function<TargetTransformInfo &(Function &)> GetTTI,
      std::function<AssumptionCache &(Function &)> GetAC)
      : Solver(Solver), M(M), FAM(FAM), GetTLI(GetTLI), GetTTI(GetTTI),
        GetAC(GetAC) {}

  ~FunctionSpecializer() {
    // Eliminate dead code.
    removeDeadFunctions();
    cleanUpSSA();
  }

  bool isClonedFunction(Function *F) { return SpecializedFuncs.count(F); }

  bool run();

private:
  Constant *getPromotableAlloca(AllocaInst *Alloca, CallInst *Call);

  /// A constant stack value is an AllocaInst that has a single constant
  /// value stored to it. Return this constant if such an alloca stack value
  /// is a function argument.
  Constant *getConstantStackValue(CallInst *Call, Value *Val);

  /// Iterate over the argument tracked functions see if there
  /// are any new constant values for the call instruction via
  /// stack variables.
  void promoteConstantStackValues();

  /// Clean up fully specialized functions.
  void removeDeadFunctions();

  /// Remove any ssa_copy intrinsics that may have been introduced.
  void cleanUpSSA();

  // Compute the code metrics for function \p F.
  CodeMetrics &analyzeFunction(Function *F);

  /// This function decides whether it's worthwhile to specialize function
  /// \p F based on the known constant values its arguments can take on. It
  /// only discovers potential specialization opportunities without actually
  /// applying them.
  ///
  /// \returns true if any specializations have been found.
  bool findSpecializations(Function *F, InstructionCost Cost,
                           SmallVectorImpl<CallSpecBinding> &WorkList);

  bool isCandidateFunction(Function *F);

  Function *createSpecialization(Function *F, CallSpecBinding &Specialization);

  /// Compute and return the cost of specializing function \p F.
  InstructionCost getSpecializationCost(Function *F);

  /// Compute a bonus for replacing argument \p A with constant \p C.
  InstructionCost getSpecializationBonus(Argument *A, Constant *C,
                                         const LoopInfo &LI);

  /// Determine if it is possible to specialise the function for constant values
  /// of the formal parameter \p A.
  bool isArgumentInteresting(Argument *A);

  /// Check if the value \p V  (an actual argument) is a constant or can only
  /// have a constant value. Return that constant.
  Constant *getCandidateConstant(Value *V);

  /// Redirects callsites of function \p F to its specialized copies.
  void updateCallSites(Function *F,
                       SmallVectorImpl<CallSpecBinding> &Specializations);
};
} // namespace llvm

#endif // LLVM_TRANSFORMS_IPO_FUNCTIONSPECIALIZATION_H
