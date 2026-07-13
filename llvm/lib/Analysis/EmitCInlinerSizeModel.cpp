//===- EmitCInlinerSizeModel.cpp - EmitC inliner model wrapper ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// This file implements the wrapper around the EmitC-translated MLGO inliner
/// model.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/EmitCInlinerSizeModel.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-braces"
#endif

namespace llvm::emitc_inliner_model {
#define main action
#include "llvm/Analysis/EmitCInlinerSizeModel.inc"
#undef main
} // namespace llvm::emitc_inliner_model

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

using namespace llvm;

namespace {
template <typename T> inline constexpr bool AlwaysFalse = false;
using I64Ptr = int64_t *;
using I32Ptr = int32_t *;
using F32Ptr = float *;

using InlinerProductionActionTy =
    int64_t (*)(I64Ptr, I64Ptr, I64Ptr, I64Ptr, I64Ptr, I64Ptr, I64Ptr, I64Ptr,
                I64Ptr, I64Ptr, I64Ptr, I64Ptr, I64Ptr, I64Ptr, I64Ptr, I64Ptr,
                I64Ptr, I64Ptr, I64Ptr, I64Ptr, I64Ptr, I64Ptr, I64Ptr, I64Ptr,
                I64Ptr, I64Ptr, I64Ptr, I32Ptr, I64Ptr, I64Ptr, I64Ptr, I64Ptr,
                I64Ptr, I64Ptr, I64Ptr, I64Ptr, F32Ptr, I64Ptr, F32Ptr);
using InlinerMockActionTy = int64_t (*)(I64Ptr, I64Ptr, I64Ptr, I64Ptr, I64Ptr,
                                        I64Ptr, I64Ptr, I64Ptr, I64Ptr, I64Ptr,
                                        I64Ptr, I64Ptr, I64Ptr, I64Ptr, I64Ptr,
                                        I64Ptr, I64Ptr, I64Ptr, I64Ptr, I64Ptr,
                                        I64Ptr, I64Ptr, I64Ptr, I64Ptr, I64Ptr,
                                        I64Ptr, I64Ptr, I32Ptr, I64Ptr, I64Ptr,
                                        I64Ptr, I64Ptr, I64Ptr, I64Ptr, I64Ptr,
                                        I64Ptr, I64Ptr, F32Ptr, I64Ptr, F32Ptr);

struct InlinerRunInputs {
  I64Ptr deadBlocks;
  I64Ptr caseClusterPenalty;
  I64Ptr sroaSavings;
  I64Ptr jumpTablePenalty;
  I64Ptr callsiteHeight;
  I64Ptr calleeBasicBlockCount;
  I64Ptr callArgumentSetup;
  I64Ptr loweredCallArgSetup;
  I64Ptr simplifiedInstructions;
  I64Ptr nrCtantParams;
  I64Ptr isMultipleBlocks;
  I64Ptr loadElimination;
  I64Ptr edgeCount;
  I64Ptr callerUsers;
  I64Ptr callerConditionallyExecutedBlocks;
  I64Ptr constantOffsetPtrArgs;
  I64Ptr callsiteCost;
  I64Ptr callerBasicBlockCount;
  I64Ptr loadRelativeIntrinsic;
  I64Ptr indirectCallPenalty;
  I64Ptr costEstimate;
  I64Ptr threshold;
  I64Ptr nestedInlineCostEstimate;
  I64Ptr unsimplifiedCommonInstructions;
  I64Ptr sroaLosses;
  I64Ptr numLoops;
  I64Ptr switchPenalty;
  I64Ptr calleeUsers;
  I64Ptr nodeCount;
  I64Ptr constantArgs;
  I64Ptr lastCallToStaticBonus;
  I64Ptr coldCCPenalty;
  I64Ptr calleeConditionallyExecutedBlocks;
  I64Ptr callPenalty;
  I64Ptr nestedInlines;
  I32Ptr dummyStepType;
  F32Ptr dummyDiscount;
  F32Ptr dummyReward;
  I64Ptr dummyInliningDefault;
};

template <typename ActionTy>
int64_t runEmitCInlinerAction(const InlinerRunInputs &I) {
  if constexpr (std::is_same_v<ActionTy, InlinerProductionActionTy>) {
    return static_cast<ActionTy>(emitc_inliner_model::action)(
        I.callsiteCost, I.isMultipleBlocks, I.callerConditionallyExecutedBlocks,
        I.dummyInliningDefault, I.coldCCPenalty,
        I.calleeConditionallyExecutedBlocks, I.calleeUsers,
        I.calleeBasicBlockCount, I.nrCtantParams, I.loadRelativeIntrinsic,
        I.jumpTablePenalty, I.unsimplifiedCommonInstructions,
        I.indirectCallPenalty, I.loadElimination, I.callPenalty, I.costEstimate,
        I.caseClusterPenalty, I.nodeCount, I.callArgumentSetup, I.sroaSavings,
        I.loweredCallArgSetup, I.threshold, I.deadBlocks, I.constantArgs,
        I.sroaLosses, I.simplifiedInstructions, I.numLoops, I.dummyStepType,
        I.edgeCount, I.nestedInlines, I.callerBasicBlockCount,
        I.lastCallToStaticBonus, I.nestedInlineCostEstimate, I.callsiteHeight,
        I.constantOffsetPtrArgs, I.switchPenalty, I.dummyDiscount,
        I.callerUsers, I.dummyReward);
  } else if constexpr (std::is_same_v<ActionTy, InlinerMockActionTy>) {
    return static_cast<ActionTy>(emitc_inliner_model::action)(
        I.callerBasicBlockCount, I.callerConditionallyExecutedBlocks,
        I.callerUsers, I.calleeBasicBlockCount,
        I.calleeConditionallyExecutedBlocks, I.calleeUsers, I.nrCtantParams,
        I.nodeCount, I.edgeCount, I.callsiteHeight, I.costEstimate,
        I.sroaSavings, I.sroaLosses, I.loadElimination, I.callPenalty,
        I.callArgumentSetup, I.loadRelativeIntrinsic, I.loweredCallArgSetup,
        I.indirectCallPenalty, I.jumpTablePenalty, I.caseClusterPenalty,
        I.switchPenalty, I.unsimplifiedCommonInstructions, I.numLoops,
        I.deadBlocks, I.simplifiedInstructions, I.constantArgs, I.dummyStepType,
        I.constantOffsetPtrArgs, I.callsiteCost, I.coldCCPenalty,
        I.lastCallToStaticBonus, I.isMultipleBlocks, I.nestedInlines,
        I.nestedInlineCostEstimate, I.threshold, I.dummyInliningDefault,
        I.dummyDiscount, I.callerUsers, I.dummyReward);
  } else {
    static_assert(AlwaysFalse<ActionTy>,
                  "Unsupported EmitC inliner model signature");
  }
}
} // namespace

int EmitCInlinerSizeModel::LookupArgIndex(const std::string &Name) {
  return StringSwitch<int>(Name)
      .Case("feed_dead_blocks", DeadBlocks)
      .Case("feed_case_cluster_penalty", CaseClusterPenalty)
      .Case("feed_sroa_savings", SroaSavings)
      .Case("feed_jump_table_penalty", JumpTablePenalty)
      .Case("feed_callsite_height", CallsiteHeight)
      .Case("feed_callee_basic_block_count", CalleeBasicBlockCount)
      .Case("feed_call_argument_setup", CallArgumentSetup)
      .Case("feed_lowered_call_arg_setup", LoweredCallArgSetup)
      .Case("feed_simplified_instructions", SimplifiedInstructions)
      .Case("feed_nr_ctant_params", NrCtantParams)
      .Case("feed_is_multiple_blocks", IsMultipleBlocks)
      .Case("feed_load_elimination", LoadElimination)
      .Case("feed_edge_count", EdgeCount)
      .Case("feed_caller_users", CallerUsers)
      .Case("feed_caller_conditionally_executed_blocks",
            CallerConditionallyExecutedBlocks)
      .Case("feed_constant_offset_ptr_args", ConstantOffsetPtrArgs)
      .Case("feed_callsite_cost", CallsiteCost)
      .Case("feed_caller_basic_block_count", CallerBasicBlockCount)
      .Case("feed_load_relative_intrinsic", LoadRelativeIntrinsic)
      .Case("feed_indirect_call_penalty", IndirectCallPenalty)
      .Case("feed_cost_estimate", CostEstimate)
      .Case("feed_threshold", Threshold)
      .Case("feed_nested_inline_cost_estimate", NestedInlineCostEstimate)
      .Case("feed_unsimplified_common_instructions",
            UnsimplifiedCommonInstructions)
      .Case("feed_sroa_losses", SroaLosses)
      .Case("feed_num_loops", NumLoops)
      .Case("feed_switch_penalty", SwitchPenalty)
      .Case("feed_callee_users", CalleeUsers)
      .Case("feed_node_count", NodeCount)
      .Case("feed_constant_args", ConstantArgs)
      .Case("feed_last_call_to_static_bonus", LastCallToStaticBonus)
      .Case("feed_cold_cc_penalty", ColdCCPenalty)
      .Case("feed_callee_conditionally_executed_blocks",
            CalleeConditionallyExecutedBlocks)
      .Case("feed_call_penalty", CallPenalty)
      .Case("feed_nested_inlines", NestedInlines)
      .Default(-1);
}

int EmitCInlinerSizeModel::LookupResultIndex(const std::string &Name) {
  return Name == "fetch_inlining_decision" ? 0 : -1;
}

void *EmitCInlinerSizeModel::arg_data(int Index) {
  if (Index < 0 || Index >= NumArgs)
    llvm_unreachable("invalid EmitC inliner input index");
  return Inputs[Index].data();
}

void *EmitCInlinerSizeModel::result_data(int Index) {
  if (Index != 0)
    llvm_unreachable("invalid EmitC inliner result index");
  return Result.data();
}

void EmitCInlinerSizeModel::Run() {
  using ActionTy = decltype(&emitc_inliner_model::action);
  InlinerRunInputs I{};
  I.deadBlocks = Inputs[DeadBlocks].data();
  I.caseClusterPenalty = Inputs[CaseClusterPenalty].data();
  I.sroaSavings = Inputs[SroaSavings].data();
  I.jumpTablePenalty = Inputs[JumpTablePenalty].data();
  I.callsiteHeight = Inputs[CallsiteHeight].data();
  I.calleeBasicBlockCount = Inputs[CalleeBasicBlockCount].data();
  I.callArgumentSetup = Inputs[CallArgumentSetup].data();
  I.loweredCallArgSetup = Inputs[LoweredCallArgSetup].data();
  I.simplifiedInstructions = Inputs[SimplifiedInstructions].data();
  I.nrCtantParams = Inputs[NrCtantParams].data();
  I.isMultipleBlocks = Inputs[IsMultipleBlocks].data();
  I.loadElimination = Inputs[LoadElimination].data();
  I.edgeCount = Inputs[EdgeCount].data();
  I.callerUsers = Inputs[CallerUsers].data();
  I.callerConditionallyExecutedBlocks =
      Inputs[CallerConditionallyExecutedBlocks].data();
  I.constantOffsetPtrArgs = Inputs[ConstantOffsetPtrArgs].data();
  I.callsiteCost = Inputs[CallsiteCost].data();
  I.callerBasicBlockCount = Inputs[CallerBasicBlockCount].data();
  I.loadRelativeIntrinsic = Inputs[LoadRelativeIntrinsic].data();
  I.indirectCallPenalty = Inputs[IndirectCallPenalty].data();
  I.costEstimate = Inputs[CostEstimate].data();
  I.threshold = Inputs[Threshold].data();
  I.nestedInlineCostEstimate = Inputs[NestedInlineCostEstimate].data();
  I.unsimplifiedCommonInstructions =
      Inputs[UnsimplifiedCommonInstructions].data();
  I.sroaLosses = Inputs[SroaLosses].data();
  I.numLoops = Inputs[NumLoops].data();
  I.switchPenalty = Inputs[SwitchPenalty].data();
  I.calleeUsers = Inputs[CalleeUsers].data();
  I.nodeCount = Inputs[NodeCount].data();
  I.constantArgs = Inputs[ConstantArgs].data();
  I.lastCallToStaticBonus = Inputs[LastCallToStaticBonus].data();
  I.coldCCPenalty = Inputs[ColdCCPenalty].data();
  I.calleeConditionallyExecutedBlocks =
      Inputs[CalleeConditionallyExecutedBlocks].data();
  I.callPenalty = Inputs[CallPenalty].data();
  I.nestedInlines = Inputs[NestedInlines].data();
  I.dummyStepType = DummyStepType.data();
  I.dummyDiscount = DummyDiscount.data();
  I.dummyReward = DummyReward.data();
  I.dummyInliningDefault = DummyInliningDefault.data();
  Result[0] = runEmitCInlinerAction<ActionTy>(I);
}
