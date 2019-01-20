//===- DeadCodeElimination.cpp - Eliminate dead iteration  ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The polyhedral dead code elimination pass analyses a SCoP to eliminate
// statement instances that can be proven dead.
// As a consequence, the code generated for this SCoP may execute a statement
// less often. This means, a statement may be executed only in certain loop
// iterations or it may not even be part of the generated code at all.
//
// This code:
//
//    for (i = 0; i < N; i++)
//        arr[i] = 0;
//    for (i = 0; i < N; i++)
//        arr[i] = 10;
//    for (i = 0; i < N; i++)
//        arr[i] = i;
//
// is e.g. simplified to:
//
//    for (i = 0; i < N; i++)
//        arr[i] = i;
//
// The idea and the algorithm used was first implemented by Sven Verdoolaege in
// the 'ppcg' tool.
//
//===----------------------------------------------------------------------===//

#include "polly/DependenceInfo.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "llvm/Support/CommandLine.h"
#include "isl/flow.h"
#include "isl/isl-noexceptions.h"
#include "isl/map.h"
#include "isl/set.h"
#include "isl/union_map.h"
#include "isl/union_set.h"

using namespace llvm;
using namespace polly;

namespace {

cl::opt<int> DCEPreciseSteps(
    "polly-dce-precise-steps",
    cl::desc("The number of precise steps between two approximating "
             "iterations. (A value of -1 schedules another approximation stage "
             "before the actual dead code elimination."),
    cl::ZeroOrMore, cl::init(-1), cl::cat(PollyCategory));

class DeadCodeElim : public ScopPass {
public:
  static char ID;
  explicit DeadCodeElim() : ScopPass(ID) {}

  /// Remove dead iterations from the schedule of @p S.
  bool runOnScop(Scop &S) override;

  /// Register all analyses and transformation required.
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  /// Return the set of live iterations.
  ///
  /// The set of live iterations are all iterations that write to memory and for
  /// which we can not prove that there will be a later write that _must_
  /// overwrite the same memory location and is consequently the only one that
  /// is visible after the execution of the SCoP.
  ///
  isl::union_set getLiveOut(Scop &S);
  bool eliminateDeadCode(Scop &S, int PreciseSteps);
};
} // namespace

char DeadCodeElim::ID = 0;

// To compute the live outs, we compute for the data-locations that are
// must-written to the last statement that touches these locations. On top of
// this we add all statements that perform may-write accesses.
//
// We could be more precise by removing may-write accesses for which we know
// that they are overwritten by a must-write after. However, at the moment the
// only may-writes we introduce access the full (unbounded) array, such that
// bounded write accesses can not overwrite all of the data-locations. As
// this means may-writes are in the current situation always live, there is
// no point in trying to remove them from the live-out set.
isl::union_set DeadCodeElim::getLiveOut(Scop &S) {
  isl::union_map Schedule = S.getSchedule();
  isl::union_map MustWrites = S.getMustWrites();
  isl::union_map WriteIterations = MustWrites.reverse();
  isl::union_map WriteTimes = WriteIterations.apply_range(Schedule);

  isl::union_map LastWriteTimes = WriteTimes.lexmax();
  isl::union_map LastWriteIterations =
      LastWriteTimes.apply_range(Schedule.reverse());

  isl::union_set Live = LastWriteIterations.range();
  isl::union_map MayWrites = S.getMayWrites();
  Live = Live.unite(MayWrites.domain());
  return Live.coalesce();
}

/// Performs polyhedral dead iteration elimination by:
/// o Assuming that the last write to each location is live.
/// o Following each RAW dependency from a live iteration backwards and adding
///   that iteration to the live set.
///
/// To ensure the set of live iterations does not get too complex we always
/// combine a certain number of precise steps with one approximating step that
/// simplifies the life set with an affine hull.
bool DeadCodeElim::eliminateDeadCode(Scop &S, int PreciseSteps) {
  DependenceInfo &DI = getAnalysis<DependenceInfo>();
  const Dependences &D = DI.getDependences(Dependences::AL_Statement);

  if (!D.hasValidDependences())
    return false;

  isl::union_set Live = getLiveOut(S);
  isl::union_map Dep =
      D.getDependences(Dependences::TYPE_RAW | Dependences::TYPE_RED);
  Dep = Dep.reverse();

  if (PreciseSteps == -1)
    Live = Live.affine_hull();

  isl::union_set OriginalDomain = S.getDomains();
  int Steps = 0;
  while (true) {
    Steps++;

    isl::union_set Extra = Live.apply(Dep);

    if (Extra.is_subset(Live))
      break;

    Live = Live.unite(Extra);

    if (Steps > PreciseSteps) {
      Steps = 0;
      Live = Live.affine_hull();
    }

    Live = Live.intersect(OriginalDomain);
  }

  Live = Live.coalesce();

  bool Changed = S.restrictDomains(Live);

  // FIXME: We can probably avoid the recomputation of all dependences by
  // updating them explicitly.
  if (Changed)
    DI.recomputeDependences(Dependences::AL_Statement);
  return Changed;
}

bool DeadCodeElim::runOnScop(Scop &S) {
  return eliminateDeadCode(S, DCEPreciseSteps);
}

void DeadCodeElim::getAnalysisUsage(AnalysisUsage &AU) const {
  ScopPass::getAnalysisUsage(AU);
  AU.addRequired<DependenceInfo>();
}

Pass *polly::createDeadCodeElimPass() { return new DeadCodeElim(); }

INITIALIZE_PASS_BEGIN(DeadCodeElim, "polly-dce",
                      "Polly - Remove dead iterations", false, false)
INITIALIZE_PASS_DEPENDENCY(DependenceInfo)
INITIALIZE_PASS_DEPENDENCY(ScopInfoRegionPass)
INITIALIZE_PASS_END(DeadCodeElim, "polly-dce", "Polly - Remove dead iterations",
                    false, false)
