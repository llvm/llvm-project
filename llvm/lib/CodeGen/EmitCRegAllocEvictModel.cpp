//===- EmitCRegAllocEvictModel.cpp - EmitC regalloc model wrapper ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// This file implements the wrapper around the EmitC-translated MLGO
/// regalloc eviction model.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/EmitCRegAllocEvictModel.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#include <stddef.h>
#include <stdint.h>
#include <type_traits>

namespace llvm::emitc_regalloc_evict_model {
#define main action
#include "llvm/CodeGen/EmitCRegAllocEvictModel.inc"
#undef main
} // namespace llvm::emitc_regalloc_evict_model

using namespace llvm;

namespace {
template <typename T> inline constexpr bool AlwaysFalse = false;
constexpr std::size_t EmitCRegAllocInterferenceCount = 33;
using F32TensorPtr = float (*)[EmitCRegAllocInterferenceCount];
using I64TensorPtr = int64_t (*)[EmitCRegAllocInterferenceCount];
using F32ScalarPtr = float *;
using I32ScalarPtr = int32_t *;
using I64ScalarPtr = int64_t *;

using RegAllocProductionActionTy = int64_t (*)(
    F32TensorPtr, F32TensorPtr, I64TensorPtr, F32TensorPtr, F32TensorPtr,
    F32TensorPtr, F32ScalarPtr, F32TensorPtr, F32TensorPtr, F32TensorPtr,
    I64TensorPtr, I64TensorPtr, F32TensorPtr, F32TensorPtr, I32ScalarPtr,
    F32TensorPtr, I64TensorPtr, F32TensorPtr, I64TensorPtr, F32TensorPtr,
    F32ScalarPtr, F32TensorPtr, I64TensorPtr, F32ScalarPtr);
using RegAllocMaskOnlyActionTy = int64_t (*)(I64ScalarPtr);

struct RegAllocRunInputs {
  F32TensorPtr liverangeSize;
  F32TensorPtr hintWeightsByMax;
  I64TensorPtr isFree;
  F32TensorPtr weighedReadsByMax;
  F32TensorPtr weighedReadWritesByMax;
  F32TensorPtr nrBrokenHints;
  F32ScalarPtr progress;
  F32TensorPtr hottestBBFreqByMax;
  F32TensorPtr useDefDensity;
  F32TensorPtr startBBFreqByMax;
  I64TensorPtr maxStage;
  I64TensorPtr isHint;
  F32TensorPtr nrRematerializable;
  F32TensorPtr weighedWritesByMax;
  I32ScalarPtr dummyStepType;
  F32TensorPtr nrUrgent;
  I64TensorPtr mask;
  F32TensorPtr nrDefsAndUses;
  I64TensorPtr isLocal;
  F32TensorPtr endBBFreqByMax;
  F32ScalarPtr dummyDiscount;
  F32TensorPtr weighedIndvarsByMax;
  I64TensorPtr minStage;
  F32ScalarPtr dummyReward;
  I64ScalarPtr maskFlat;
};

template <typename ActionTy>
int64_t runEmitCRegAllocAction(const RegAllocRunInputs &I) {
  if constexpr (std::is_same_v<ActionTy, RegAllocProductionActionTy>) {
    return static_cast<ActionTy>(emitc_regalloc_evict_model::action)(
        I.liverangeSize, I.hintWeightsByMax, I.isFree, I.weighedReadsByMax,
        I.weighedReadWritesByMax, I.nrBrokenHints, I.progress,
        I.hottestBBFreqByMax, I.useDefDensity, I.startBBFreqByMax, I.maxStage,
        I.isHint, I.nrRematerializable, I.weighedWritesByMax, I.dummyStepType,
        I.nrUrgent, I.mask, I.nrDefsAndUses, I.isLocal, I.endBBFreqByMax,
        I.dummyDiscount, I.weighedIndvarsByMax, I.minStage, I.dummyReward);
  } else if constexpr (std::is_same_v<ActionTy, RegAllocMaskOnlyActionTy>) {
    return static_cast<ActionTy>(emitc_regalloc_evict_model::action)(
        I.maskFlat);
  } else {
    static_assert(AlwaysFalse<ActionTy>,
                  "Unsupported EmitC regalloc eviction model signature");
  }
}
} // namespace

int EmitCRegAllocEvictModel::LookupArgIndex(const std::string &Name) {
  return StringSwitch<int>(Name)
      .Case("feed_mask", Mask)
      .Case("feed_is_free", IsFree)
      .Case("feed_nr_urgent", NrUrgent)
      .Case("feed_nr_broken_hints", NrBrokenHints)
      .Case("feed_is_hint", IsHint)
      .Case("feed_is_local", IsLocal)
      .Case("feed_nr_rematerializable", NrRematerializable)
      .Case("feed_nr_defs_and_uses", NrDefsAndUses)
      .Case("feed_weighed_reads_by_max", WeighedReadsByMax)
      .Case("feed_weighed_writes_by_max", WeighedWritesByMax)
      .Case("feed_weighed_read_writes_by_max", WeighedReadWritesByMax)
      .Case("feed_weighed_indvars_by_max", WeighedIndvarsByMax)
      .Case("feed_hint_weights_by_max", HintWeightsByMax)
      .Case("feed_start_bb_freq_by_max", StartBBFreqByMax)
      .Case("feed_end_bb_freq_by_max", EndBBFreqByMax)
      .Case("feed_hottest_bb_freq_by_max", HottestBBFreqByMax)
      .Case("feed_liverange_size", LiverangeSize)
      .Case("feed_use_def_density", UseDefDensity)
      .Case("feed_max_stage", MaxStage)
      .Case("feed_min_stage", MinStage)
      .Case("feed_progress", Progress)
      .Default(-1);
}

int EmitCRegAllocEvictModel::LookupResultIndex(const std::string &Name) {
  return Name == "fetch_index_to_evict" ? 0 : -1;
}

void *EmitCRegAllocEvictModel::arg_data(int Index) {
  switch (Index) {
  case Mask:
    return MaskInput;
  case IsFree:
    return IsFreeInput;
  case NrUrgent:
    return NrUrgentInput;
  case NrBrokenHints:
    return NrBrokenHintsInput;
  case IsHint:
    return IsHintInput;
  case IsLocal:
    return IsLocalInput;
  case NrRematerializable:
    return NrRematerializableInput;
  case NrDefsAndUses:
    return NrDefsAndUsesInput;
  case WeighedReadsByMax:
    return WeighedReadsByMaxInput;
  case WeighedWritesByMax:
    return WeighedWritesByMaxInput;
  case WeighedReadWritesByMax:
    return WeighedReadWritesByMaxInput;
  case WeighedIndvarsByMax:
    return WeighedIndvarsByMaxInput;
  case HintWeightsByMax:
    return HintWeightsByMaxInput;
  case StartBBFreqByMax:
    return StartBBFreqByMaxInput;
  case EndBBFreqByMax:
    return EndBBFreqByMaxInput;
  case HottestBBFreqByMax:
    return HottestBBFreqByMaxInput;
  case LiverangeSize:
    return LiverangeSizeInput;
  case UseDefDensity:
    return UseDefDensityInput;
  case MaxStage:
    return MaxStageInput;
  case MinStage:
    return MinStageInput;
  case Progress:
    return ProgressInput;
  }
  llvm_unreachable("invalid EmitC regalloc eviction input index");
}

void *EmitCRegAllocEvictModel::result_data(int Index) {
  if (Index != 0)
    llvm_unreachable("invalid EmitC regalloc eviction result index");
  return Result;
}

void EmitCRegAllocEvictModel::Run() {
  using ActionTy = decltype(&emitc_regalloc_evict_model::action);
  RegAllocRunInputs I{};
  I.liverangeSize = LiverangeSizeInput;
  I.hintWeightsByMax = HintWeightsByMaxInput;
  I.isFree = IsFreeInput;
  I.weighedReadsByMax = WeighedReadsByMaxInput;
  I.weighedReadWritesByMax = WeighedReadWritesByMaxInput;
  I.nrBrokenHints = NrBrokenHintsInput;
  I.progress = ProgressInput;
  I.hottestBBFreqByMax = HottestBBFreqByMaxInput;
  I.useDefDensity = UseDefDensityInput;
  I.startBBFreqByMax = StartBBFreqByMaxInput;
  I.maxStage = MaxStageInput;
  I.isHint = IsHintInput;
  I.nrRematerializable = NrRematerializableInput;
  I.weighedWritesByMax = WeighedWritesByMaxInput;
  I.dummyStepType = DummyStepType;
  I.nrUrgent = NrUrgentInput;
  I.mask = MaskInput;
  I.nrDefsAndUses = NrDefsAndUsesInput;
  I.isLocal = IsLocalInput;
  I.endBBFreqByMax = EndBBFreqByMaxInput;
  I.dummyDiscount = DummyDiscount;
  I.weighedIndvarsByMax = WeighedIndvarsByMaxInput;
  I.minStage = MinStageInput;
  I.dummyReward = DummyReward;
  I.maskFlat = MaskInput[0];
  Result[0] = runEmitCRegAllocAction<ActionTy>(I);
}
