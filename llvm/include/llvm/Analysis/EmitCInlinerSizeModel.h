//===- EmitCInlinerSizeModel.h - EmitC inliner model wrapper ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// Declares the wrapper around the EmitC-translated MLGO inliner model.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_ANALYSIS_EMITCINLINERSIZEMODEL_H
#define LLVM_LIB_ANALYSIS_EMITCINLINERSIZEMODEL_H

#include <array>
#include <cstdint>
#include <string>

namespace llvm {

class EmitCInlinerSizeModel final {
public:
  int LookupArgIndex(const std::string &Name);
  int LookupResultIndex(const std::string &Name);
  void *arg_data(int Index);
  void *result_data(int Index);
  void Run();

private:
  enum ArgIndex : int {
    DeadBlocks = 0,
    CaseClusterPenalty,
    SroaSavings,
    JumpTablePenalty,
    CallsiteHeight,
    CalleeBasicBlockCount,
    CallArgumentSetup,
    LoweredCallArgSetup,
    SimplifiedInstructions,
    NrCtantParams,
    IsMultipleBlocks,
    LoadElimination,
    EdgeCount,
    CallerUsers,
    CallerConditionallyExecutedBlocks,
    ConstantOffsetPtrArgs,
    CallsiteCost,
    CallerBasicBlockCount,
    LoadRelativeIntrinsic,
    IndirectCallPenalty,
    CostEstimate,
    Threshold,
    NestedInlineCostEstimate,
    UnsimplifiedCommonInstructions,
    SroaLosses,
    NumLoops,
    SwitchPenalty,
    CalleeUsers,
    NodeCount,
    ConstantArgs,
    LastCallToStaticBonus,
    ColdCCPenalty,
    CalleeConditionallyExecutedBlocks,
    CallPenalty,
    NestedInlines,

    NumArgs
  };

  std::array<std::array<int64_t, 1>, NumArgs> Inputs{};
  std::array<int64_t, 1> Result{};

  std::array<int64_t, 1> DummyInliningDefault{};
  std::array<int32_t, 1> DummyStepType{};
  std::array<float, 1> DummyDiscount{};
  std::array<float, 1> DummyReward{};
};

} // namespace llvm

#endif
