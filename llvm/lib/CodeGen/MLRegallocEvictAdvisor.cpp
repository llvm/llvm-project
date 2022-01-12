//===- MLRegAllocEvictAdvisor.cpp - ML eviction advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the ML eviction advisor and reward injection pass
//
//===----------------------------------------------------------------------===//

#include "RegAllocEvictionAdvisor.h"
#include "llvm/Analysis/MLModelRunner.h"
#include "llvm/Analysis/ModelUnderTrainingRunner.h"
#include "llvm/Analysis/NoInferenceModelRunner.h"
#include "llvm/Analysis/Utils/TFUtils.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/Config/config.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetMachine.h"

#include <memory>

using namespace llvm;

#define DEBUG_TYPE "ml-regalloc"

#if defined(LLVM_HAVE_TF_AOT) || defined(LLVM_HAVE_TF_API)
namespace {
// This is the maximum number of interfererring ranges. That's the number of
// distinct AllocationOrder values, which comes from MCRegisterClass::RegsSize.
// For X86, that's 32.
// TODO: find a way to get this, statically, in a programmatic way.
static const int64_t MaxInterferences = 32;

// Logically, we can think of the feature set given to the evaluator as a 2D
// matrix. The rows are the features (see next). The columns correspond to the
// interferences. We treat the candidate virt reg as an 'interference', too, as
// its feature set is the same as that of the interferring ranges. So we'll have
// MaxInterferences + 1 columns and by convention, we will use the last column
// for the virt reg seeking allocation.
static const int64_t CandidateVirtRegPos = MaxInterferences;
static const int64_t NumberOfInterferences = CandidateVirtRegPos + 1;

// Most features are as described above, so we'll reuse this vector in defining
// them.
static const std::vector<int64_t> PerLiveRangeShape{1, NumberOfInterferences};

// --------------
// Features table
// --------------
// For each interfering live range (incl. the candidate) we collect a number of
// features. However, because the features are of different types (and because
// of ML best practices), we organize the tensors per feature, not per
// candidate. Each such tensor has a scalar value corresponding to the
// interferring live range at that position, in the order in AllocationOrder.
// The last position corresponds to the virt reg seeking allocation.
// Exception to all that is the progression feature, which is just a scalar (see
// its documentation for details).
// Note on naming: the "_by_max" are normalized using the largest value of that
// tensor, as observed in the current decision making stage (i.e. for the
// current call to the advisor's tryFindEvictionCandidate)
//
// The feature list format: type, name, shape, documentation.
// Note: we can really just use int64 and float, hence the modeling of some
// bools as int64 values.
#define RA_EVICT_FEATURES_LIST(M)                                              \
  M(int64_t, mask, PerLiveRangeShape,                                          \
    "boolean values, 0 for unavailable candidates (i.e. if a position is 0, "  \
    "it "                                                                      \
    "can't be evicted)")                                                       \
  M(int64_t, is_free, PerLiveRangeShape,                                       \
    "boolean values, 1 if this phys reg is actually free (no interferences)")  \
  M(float, nr_urgent, PerLiveRangeShape,                                       \
    "number of 'urgent' intervals, normalized. Urgent are those that are OK "  \
    "to break cascades")                                                       \
  M(float, nr_broken_hints, PerLiveRangeShape,                                 \
    "if this position were evicted, how many broken hints would there be")     \
  M(int64_t, is_hint, PerLiveRangeShape,                                       \
    "is this a preferred phys reg for the candidate")                          \
  M(int64_t, is_local, PerLiveRangeShape,                                      \
    "is this live range local to a basic block")                               \
  M(float, nr_rematerializable, PerLiveRangeShape,                             \
    "nr rematerializable ranges")                                              \
  M(float, nr_defs_and_uses, PerLiveRangeShape,                                \
    "bb freq - weighed nr defs and uses")                                      \
  M(float, weighed_reads_by_max, PerLiveRangeShape,                            \
    "bb freq - weighed nr of reads, normalized")                               \
  M(float, weighed_writes_by_max, PerLiveRangeShape,                           \
    "bb feq - weighed nr of writes, normalized")                               \
  M(float, weighed_read_writes_by_max, PerLiveRangeShape,                      \
    "bb freq - weighed nr of uses that are both read and writes, normalized")  \
  M(float, weighed_indvars_by_max, PerLiveRangeShape,                          \
    "bb freq - weighed nr of uses that are indvars, normalized")               \
  M(float, hint_weights_by_max, PerLiveRangeShape,                             \
    "bb freq - weighed nr of uses that are hints, normalized")                 \
  M(float, start_bb_freq_by_max, PerLiveRangeShape,                            \
    "the freq in the start block, normalized")                                 \
  M(float, end_bb_freq_by_max, PerLiveRangeShape,                              \
    "freq of end block, normalized")                                           \
  M(float, hottest_bb_freq_by_max, PerLiveRangeShape,                          \
    "hottest BB freq, normalized")                                             \
  M(float, liverange_size, PerLiveRangeShape,                                  \
    "size (instr index diff) of the LR")                                       \
  M(float, use_def_density, PerLiveRangeShape,                                 \
    "the max weight, as computed by the manual heuristic")                     \
  M(int64_t, max_stage, PerLiveRangeShape,                                     \
    "largest stage of an interval in this LR")                                 \
  M(int64_t, min_stage, PerLiveRangeShape,                                     \
    "lowest stage of an interval in this LR")                                  \
  M(float, progress, {1}, "ratio of current queue size to initial size")

// The model learns to pick one of the mask == 1 interferences. This is the name
// of the output tensor.
// The contract with the model is that the output will be guaranteed to be to a
// mask == 1 position.
const char *const DecisionName = "index_to_evict";

// Named features index.
enum FeatureIDs {
#define _FEATURE_IDX(_, name, __, ___) name,
  RA_EVICT_FEATURES_LIST(_FEATURE_IDX)
#undef _FEATURE_IDX
      FeatureCount
};

// The ML advisor will typically have a sparse input to the evaluator, because
// various phys regs won't be available. It's easier (maintenance-wise) to
// bulk-reset the state of the evaluator each time we are about to use it again.
template <typename T> size_t getTotalSize(const std::vector<int64_t> &Shape) {
  size_t Ret = sizeof(T);
  for (const auto V : Shape)
    Ret *= V;
  return Ret;
}

void resetInputs(MLModelRunner &Runner) {
#define _RESET(TYPE, NAME, SHAPE, __)                                          \
  std::memset(Runner.getTensorUntyped(FeatureIDs::NAME), 0,                    \
              getTotalSize<TYPE>(SHAPE));
  RA_EVICT_FEATURES_LIST(_RESET)
#undef _RESET
}

// Development mode-specifics
#ifdef LLVM_HAVE_TF_API
#define _DECL_FEATURES(type, name, shape, _)                                   \
  TensorSpec::createSpec<type>(#name, shape),

static const std::vector<TensorSpec> InputFeatures{
    {RA_EVICT_FEATURES_LIST(_DECL_FEATURES)}};
#undef _DECL_FEATURES
static const TensorSpec Output =
    TensorSpec::createSpec<int64_t>(DecisionName, {1});
static const TensorSpec Reward = TensorSpec::createSpec<float>("reward", {1});

#endif //#ifdef LLVM_HAVE_TF_API
} // namespace
#endif // defined(LLVM_HAVE_TF_AOT) || defined(LLVM_HAVE_TF_API)
