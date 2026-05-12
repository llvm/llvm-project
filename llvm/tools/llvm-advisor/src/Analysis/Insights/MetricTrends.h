//===--- MetricTrends.h - LLVM Advisor -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once
#include "InsightBase.h"

namespace llvm::advisor {

class MetricTrendsInsight final : public Insight {
public:
  InsightKind getKind() const override { return InsightKind::MetricTrends; }
  StringRef getName() const override { return "metric_trends"; }
  StringRef getDescription() const override {
    return "Key IR metrics with interpretation and size-class annotation";
  }
  StringRef getRequiredCapability() const override { return "llvm.ir.summary"; }
  Expected<InsightOutput> analyze(const InsightInput &Input) const override;
};

} // namespace llvm::advisor
