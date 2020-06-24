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

#include <array>
#include <string>
#include <vector>

namespace llvm {

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
  M(EdgeCount, "edge_count",                                                   \
    "number of module-internal users of the caller, +1 if the caller is "      \
    "exposed externally")                                                      \
  M(CallerUsers, "caller_users",                                               \
    "number of blocks reached from a conditional instruction, in the caller")  \
  M(CallerConditionallyExecutedBlocks, "caller_conditionally_executed_blocks", \
    "number of blocks reached from a conditional instruction, in the caller")  \
  M(CallerBasicBlockCount, "caller_basic_block_count",                         \
    "number of basic blocks in the caller")                                    \
  M(CalleeConditionallyExecutedBlocks, "callee_conditionally_executed_blocks", \
    "number of blocks reached from a conditional instruction, in the callee")  \
  M(CalleeUsers, "callee_users",                                               \
    "number of blocks reached from a conditional instruction, in the callee")

enum class FeatureIndex : size_t {
#define POPULATE_INDICES(INDEX_NAME, NAME, COMMENT) INDEX_NAME,
  INLINE_FEATURE_ITERATOR(POPULATE_INDICES)
#undef POPULATE_INDICES
      NumberOfFeatures
};

constexpr size_t NumberOfFeatures =
    static_cast<size_t>(FeatureIndex::NumberOfFeatures);

extern const std::array<std::string, NumberOfFeatures> FeatureNameMap;

extern const char *const DecisionName;
extern const char *const DefaultDecisionName;
extern const char *const RewardName;

using InlineFeatures = std::vector<int64_t>;

} // namespace llvm
#endif // LLVM_ANALYSIS_INLINEMODELFEATUREMAPS_H
