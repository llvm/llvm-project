//===- InlineModelFeatureMaps.h - common model runner defs ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

#ifndef LLVM_ANALYSIS_INLINEMODELFEATUREMAPS_H
#define LLVM_ANALYSIS_INLINEMODELFEATUREMAPS_H

#include "llvm/Analysis/TensorSpec.h"

#include <array>
#include <string>
#include <vector>

namespace llvm {

// List of cost features. A "cost" feature is a summand of the heuristic-based
// inline cost, and we define them separately to preserve the original heuristic
// behavior.
#define INLINE_COST_FEATURE_ITERATOR(M)                                        \
  M(SROASavings, "sroa_savings")                                               \
  M(SROALosses, "sroa_losses")                                                 \
  M(LoadElimination, "load_elimination")                                       \
  M(CallPenalty, "call_penalty")                                               \
  M(CallArgumentSetup, "call_argument_setup")                                  \
  M(LoadRelativeIntrinsic, "load_relative_intrinsic")                          \
  M(LoweredCallArgSetup, "lowered_call_arg_setup")                             \
  M(IndirectCallPenalty, "indirect_call_penalty")                              \
  M(JumpTablePenalty, "jump_table_penalty")                                    \
  M(CaseClusterPenalty, "case_cluster_penalty")                                \
  M(SwitchPenalty, "switch_penalty")                                           \
  M(UnsimplifiedCommonInstructions, "unsimplified_common_instructions")        \
  M(NumLoops, "num_loops")                                                     \
  M(DeadBlocks, "dead_blocks")                                                 \
  M(SimplifiedInstructions, "simplified_instructions")                         \
  M(ConstantArgs, "constant_args")                                             \
  M(ConstantOffsetPtrArgs, "constant_offset_ptr_args")                         \
  M(CallSiteCost, "callsite_cost")                                             \
  M(ColdCcPenalty, "cold_cc_penalty")                                          \
  M(LastCallToStaticBonus, "last_call_to_static_bonus")                        \
  M(IsMultipleBlocks, "is_multiple_blocks")                                    \
  M(NestedInlines, "nested_inlines")                                           \
  M(NestedInlineCostEstimate, "nested_inline_cost_estimate")                   \
  M(Threshold, "threshold")

// clang-format off
enum class InlineCostFeatureIndex : size_t {
#define POPULATE_INDICES(INDEX_NAME, NAME) INDEX_NAME,
  INLINE_COST_FEATURE_ITERATOR(POPULATE_INDICES)
#undef POPULATE_INDICES

  NumberOfFeatures
};
// clang-format on

using InlineCostFeatures =
    std::array<int,
               static_cast<size_t>(InlineCostFeatureIndex::NumberOfFeatures)>;

constexpr bool isHeuristicInlineCostFeature(InlineCostFeatureIndex Feature) {
  return Feature != InlineCostFeatureIndex::SROASavings &&
         Feature != InlineCostFeatureIndex::IsMultipleBlocks &&
         Feature != InlineCostFeatureIndex::DeadBlocks &&
         Feature != InlineCostFeatureIndex::SimplifiedInstructions &&
         Feature != InlineCostFeatureIndex::ConstantArgs &&
         Feature != InlineCostFeatureIndex::ConstantOffsetPtrArgs &&
         Feature != InlineCostFeatureIndex::NestedInlines &&
         Feature != InlineCostFeatureIndex::NestedInlineCostEstimate &&
         Feature != InlineCostFeatureIndex::Threshold;
}

// List of features. Each feature is defined through a triple:
// - the name of an enum member, which will be the feature index
// - a textual name, used for Tensorflow model binding (so it needs to match the
// names used by the Tensorflow model)
// - a documentation description. Currently, that is not used anywhere
// programmatically, and serves as workaround to inability of inserting comments
// in macros.
#define INLINE_FEATURE_ITERATOR(M)                                             \
  M(CalleeBasicBlockCount, "callee_basic_block_count",                         \
    "number of basic blocks of the callee")                                    \
  M(CallSiteHeight, "callsite_height",                                         \
    "position of the call site in the original call graph - measured from "    \
    "the farthest SCC")                                                        \
  M(NodeCount, "node_count",                                                   \
    "total current number of defined functions in the module")                 \
  M(NrCtantParams, "nr_ctant_params",                                          \
    "number of parameters in the call site that are constants")                \
  M(CostEstimate, "cost_estimate", "total cost estimate (threshold - free)")   \
  M(EdgeCount, "edge_count", "total number of calls in the module")            \
  M(CallerUsers, "caller_users",                                               \
    "number of module-internal users of the caller, +1 if the caller is "      \
    "exposed externally")                                                      \
  M(CallerConditionallyExecutedBlocks, "caller_conditionally_executed_blocks", \
    "number of blocks reached from a conditional instruction, in the caller")  \
  M(CallerBasicBlockCount, "caller_basic_block_count",                         \
    "number of basic blocks in the caller")                                    \
  M(CalleeConditionallyExecutedBlocks, "callee_conditionally_executed_blocks", \
    "number of blocks reached from a conditional instruction, in the callee")  \
  M(CalleeUsers, "callee_users",                                               \
    "number of module-internal users of the callee, +1 if the callee is "      \
    "exposed externally")

// clang-format off
enum class FeatureIndex : size_t {
// InlineCost features - these must come first
#define POPULATE_INDICES(INDEX_NAME, NAME) INDEX_NAME,
  INLINE_COST_FEATURE_ITERATOR(POPULATE_INDICES)
#undef POPULATE_INDICES

// Non-cost features
#define POPULATE_INDICES(INDEX_NAME, NAME, COMMENT) INDEX_NAME,
  INLINE_FEATURE_ITERATOR(POPULATE_INDICES)
#undef POPULATE_INDICES

  NumberOfFeatures
};
// clang-format on

constexpr FeatureIndex
inlineCostFeatureToMlFeature(InlineCostFeatureIndex Feature) {
  return static_cast<FeatureIndex>(static_cast<size_t>(Feature));
}

constexpr size_t NumberOfFeatures =
    static_cast<size_t>(FeatureIndex::NumberOfFeatures);

extern const std::array<TensorSpec, NumberOfFeatures> FeatureMap;

extern const char *const DecisionName;
extern const char *const DefaultDecisionName;
extern const char *const RewardName;

using InlineFeatures = std::vector<int64_t>;

} // namespace llvm
#endif // LLVM_ANALYSIS_INLINEMODELFEATUREMAPS_H
