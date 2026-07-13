//===- EmitCRegAllocEvictModel.h - EmitC regalloc model wrapper -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Declares the wrapper around the EmitC-translated MLGO regalloc eviction
/// model.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_EMITCREGALLOCEVICTMODEL_H
#define LLVM_CODEGEN_EMITCREGALLOCEVICTMODEL_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

namespace llvm {

class EmitCRegAllocEvictModel final {
public:
  int LookupArgIndex(const std::string &Name);
  int LookupResultIndex(const std::string &Name);
  void *arg_data(int Index);
  void *result_data(int Index);
  void Run();

private:
  static constexpr std::size_t InterferenceCount = 33;

  using F32InterferenceTensor = float[1][InterferenceCount];
  using I64InterferenceTensor = int64_t[1][InterferenceCount];
  using F32Scalar = float[1];
  using I32Scalar = int32_t[1];
  using I64Scalar = int64_t[1];

  enum ArgIndex : int {
    Mask = 0,
    IsFree,
    NrUrgent,
    NrBrokenHints,
    IsHint,
    IsLocal,
    NrRematerializable,
    NrDefsAndUses,
    WeighedReadsByMax,
    WeighedWritesByMax,
    WeighedReadWritesByMax,
    WeighedIndvarsByMax,
    HintWeightsByMax,
    StartBBFreqByMax,
    EndBBFreqByMax,
    HottestBBFreqByMax,
    LiverangeSize,
    UseDefDensity,
    MaxStage,
    MinStage,
    Progress,

    NumArgs
  };

  I64InterferenceTensor MaskInput{};
  I64InterferenceTensor IsFreeInput{};
  F32InterferenceTensor NrUrgentInput{};
  F32InterferenceTensor NrBrokenHintsInput{};
  I64InterferenceTensor IsHintInput{};
  I64InterferenceTensor IsLocalInput{};
  F32InterferenceTensor NrRematerializableInput{};
  F32InterferenceTensor NrDefsAndUsesInput{};
  F32InterferenceTensor WeighedReadsByMaxInput{};
  F32InterferenceTensor WeighedWritesByMaxInput{};
  F32InterferenceTensor WeighedReadWritesByMaxInput{};
  F32InterferenceTensor WeighedIndvarsByMaxInput{};
  F32InterferenceTensor HintWeightsByMaxInput{};
  F32InterferenceTensor StartBBFreqByMaxInput{};
  F32InterferenceTensor EndBBFreqByMaxInput{};
  F32InterferenceTensor HottestBBFreqByMaxInput{};
  F32InterferenceTensor LiverangeSizeInput{};
  F32InterferenceTensor UseDefDensityInput{};
  I64InterferenceTensor MaxStageInput{};
  I64InterferenceTensor MinStageInput{};
  F32Scalar ProgressInput{};

  I32Scalar DummyStepType{};
  F32Scalar DummyDiscount{};
  F32Scalar DummyReward{};
  I64Scalar Result{};
};

} // namespace llvm

#endif
