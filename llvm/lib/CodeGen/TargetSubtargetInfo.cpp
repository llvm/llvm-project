//===- TargetSubtargetInfo.cpp - General Target Information ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file describes the general parts of a Subtarget.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Intrinsics.h"

using namespace llvm;

TargetSubtargetInfo::TargetSubtargetInfo(
    const Triple &TT, StringRef CPU, StringRef TuneCPU, StringRef FS,
    StringTable PN, ArrayRef<SubtargetFeatureKV> PF,
    ArrayRef<SubtargetSubTypeKV> PD, ArrayRef<SubtargetSubTypeAliasKV> PA,
    const MCSchedModel *PSM, const MCWriteProcResEntry *WPR,
    const MCWriteLatencyEntry *WL, const MCReadAdvanceEntry *RA,
    const InstrStage *IS, const unsigned *OC, const unsigned *FP)
    : MCSubtargetInfo(TT, CPU, TuneCPU, FS, PN, PF, PD, PA, PSM, WPR, WL, RA,
                      IS, OC, FP) {}

TargetSubtargetInfo::~TargetSubtargetInfo() = default;

bool TargetSubtargetInfo::isIntrinsicSupported(unsigned IntrinsicID) const {
  StringRef RequiredFeatures = Intrinsic::getRequiredTargetFeatures(
      static_cast<Intrinsic::ID>(IntrinsicID));

  if (RequiredFeatures.empty())
    return true;

  auto [It, Inserted] = IntrinsicSupportCache.try_emplace(IntrinsicID);
  if (Inserted)
    It->second = !RequiredFeatures.contains(Intrinsic::CustomTargetFeatures) &&
                 checkFeatureExpression(RequiredFeatures);
  return It->second;
}

bool TargetSubtargetInfo::isIntrinsicSupported(
    unsigned IntrinsicID, const CallBase &CB,
    std::optional<StringRef> &RequiredFeatures) const {
  RequiredFeatures.reset();
  if (isIntrinsicSupported(IntrinsicID))
    return true;

  StringRef FeatureExpression = Intrinsic::getRequiredTargetFeatures(
      static_cast<Intrinsic::ID>(IntrinsicID));
  if (!FeatureExpression.contains(Intrinsic::CustomTargetFeatures)) {
    RequiredFeatures = FeatureExpression;
    return false;
  }

  std::optional<StringRef> CustomRequiredFeatures =
      getCustomRequiredTargetFeaturesForIntrinsic(IntrinsicID, CB);
  bool CustomSupported =
      CustomRequiredFeatures && checkFeatureExpression(*CustomRequiredFeatures);
  auto CheckWithCustom = [&](bool Supported) {
    return checkFeatureExpression(FeatureExpression,
                                  [&](StringRef Term) -> std::optional<bool> {
                                    if (Term == Intrinsic::CustomTargetFeatures)
                                      return Supported;
                                    return std::nullopt;
                                  });
  };
  if (CheckWithCustom(CustomSupported))
    return true;

  // Only report the custom check's requirement when satisfying that check
  // would make the complete feature expression true.
  if (!CustomSupported && CheckWithCustom(true))
    RequiredFeatures = CustomRequiredFeatures;
  return false;
}

std::optional<StringRef>
TargetSubtargetInfo::getCustomRequiredTargetFeaturesForIntrinsic(
    unsigned, const CallBase &) const {
  return std::nullopt;
}

bool TargetSubtargetInfo::enableAtomicExpand() const {
  return true;
}

bool TargetSubtargetInfo::enableIndirectBrExpand() const {
  return false;
}

bool TargetSubtargetInfo::enableMachineScheduler() const {
  return false;
}

bool TargetSubtargetInfo::enableJoinGlobalCopies() const {
  return enableMachineScheduler();
}

bool TargetSubtargetInfo::enableRALocalReassignment(
    CodeGenOptLevel OptLevel) const {
  return true;
}

bool TargetSubtargetInfo::enablePostRAScheduler() const {
  return getSchedModel().PostRAScheduler;
}

bool TargetSubtargetInfo::enablePostRAMachineScheduler() const {
  return enableMachineScheduler() && enablePostRAScheduler();
}

bool TargetSubtargetInfo::useAA() const {
  return false;
}

void TargetSubtargetInfo::mirFileLoaded(MachineFunction &MF) const { }
