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

bool TargetSubtargetInfo::isIntrinsicSupported(
    unsigned IntrinsicID, const CallBase &CB,
    StringRef &MissingFeatures) const {
  MissingFeatures = {};
  StringRef RequiredFeatures = Intrinsic::getRequiredTargetFeatures(
      static_cast<Intrinsic::ID>(IntrinsicID));

  if (RequiredFeatures.empty())
    return true;

  bool NeedsCustomCheck =
      RequiredFeatures.consume_back(Intrinsic::CustomTargetFeatures);
  if (NeedsCustomCheck && !RequiredFeatures.empty() &&
      (!RequiredFeatures.consume_back(",") || RequiredFeatures.empty()))
    reportFatalInternalError("$custom must be standalone or follow a comma");

  auto [It, Inserted] = IntrinsicSupportCache.try_emplace(IntrinsicID);
  auto &[Result, CachedMissingFeatures] = It->second;
  if (Inserted) {
    if (RequiredFeatures.contains(Intrinsic::CustomTargetFeatures))
      reportFatalInternalError("$custom must be the final term");
    Result = checkFeatureExpression(RequiredFeatures, CachedMissingFeatures);
  }
  if (!Result) {
    MissingFeatures = CachedMissingFeatures;
    return false;
  }
  return !NeedsCustomCheck || isCustomIntrinsicSupported(IntrinsicID, CB);
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

bool TargetSubtargetInfo::enableSSAMachineScheduler() const { return false; }

bool TargetSubtargetInfo::useAA() const {
  return false;
}

void TargetSubtargetInfo::mirFileLoaded(MachineFunction &MF) const { }
