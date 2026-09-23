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
#include "llvm/Support/ErrorHandling.h"

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

TargetSubtargetInfo::IntrinsicSupport
TargetSubtargetInfo::getIntrinsicSupport(unsigned IntrinsicID) const {
  StringRef RequiredFeatures = Intrinsic::getRequiredTargetFeatures(
      static_cast<Intrinsic::ID>(IntrinsicID));

  if (RequiredFeatures.empty())
    return IntrinsicSupport::Supported;

  auto [It, Inserted] = IntrinsicSupportCache.try_emplace(IntrinsicID);
  if (!Inserted)
    return It->second;

  auto CheckWithCustom = [&](bool CustomSupported) {
    return checkFeatureExpression(RequiredFeatures,
                                  [&](StringRef Term) -> std::optional<bool> {
                                    if (Term == Intrinsic::CustomTargetFeatures)
                                      return CustomSupported;
                                    return std::nullopt;
                                  });
  };

  // Feature expressions only use AND and OR, so the result is monotonic in the
  // custom term. Only defer to the target when the custom term decides it.
  if (CheckWithCustom(false))
    It->second = IntrinsicSupport::Supported;
  else if (RequiredFeatures.contains(Intrinsic::CustomTargetFeatures) &&
           CheckWithCustom(true))
    It->second = IntrinsicSupport::NeedsCustomCheck;
  else
    It->second = IntrinsicSupport::Unsupported;
  return It->second;
}

bool TargetSubtargetInfo::isIntrinsicSupported(unsigned IntrinsicID,
                                               const CallBase &CB) const {
  switch (getIntrinsicSupport(IntrinsicID)) {
  case IntrinsicSupport::Supported:
    return true;
  case IntrinsicSupport::Unsupported:
    return false;
  case IntrinsicSupport::NeedsCustomCheck:
    return isCustomIntrinsicSupported(IntrinsicID, CB);
  }
  llvm_unreachable("unknown intrinsic support kind");
}

bool TargetSubtargetInfo::isCustomIntrinsicSupported(unsigned,
                                                     const CallBase &) const {
  return false;
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
