//===--- OptimizationDelta.h - LLVM Advisor ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once
#include "InsightBase.h"

namespace llvm::advisor {

class OptimizationDeltaInsight final : public Insight {
public:
  InsightKind getKind() const override {
    return InsightKind::OptimizationDelta;
  }
  StringRef getName() const override { return "optimization_delta"; }
  StringRef getDescription() const override {
    return "Compares optimization remark activity against a baseline snapshot";
  }
  StringRef getRequiredCapability() const override {
    return "llvm.remarks.summary";
  }
  bool requiresBaseline() const override { return true; }
  Expected<InsightOutput> analyze(const InsightInput &Input) const override;
};

} // namespace llvm::advisor
