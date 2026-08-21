//===------------------- RegressionClassifier.h - LLVM Advisor -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of RegressionClassifier in Compare
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "Compare/MetricDiff.h"

namespace llvm::advisor {

class RegressionClassifier {
public:
  // Noise threshold: changes within ±threshold_pct% of before value are
  // "unchanged". Default 1% avoids flagging trivial instruction count jitter.
  explicit RegressionClassifier(double ThresholdPct = 1.0)
      : ThresholdPct(ThresholdPct) {}

  StringRef classify(const MetricDelta &Delta) const;

  // Higher-is-better metrics should be inverted: more instructions = worse.
  // Returns severity: "critical" (>10%), "moderate" (>3%), "minor" (>1%),
  // "unchanged", or "improvement".
  StringRef classifyWithSeverity(const MetricDelta &Delta,
                                 bool HigherIsBetter = false) const;

private:
  double ThresholdPct;
};

} // namespace llvm::advisor
