//===--- RegressionClassifier.cpp - LLVM Advisor -------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "Compare/RegressionClassifier.h"

#include <cmath>

namespace llvm::advisor {

StringRef RegressionClassifier::classify(const MetricDelta &Delta) const {
  if (Delta.Before == 0.0 && Delta.Delta == 0.0)
    return "unchanged";
  double PctChange = Delta.Before != 0.0 ? std::abs(Delta.Delta) /
                                               std::abs(Delta.Before) * 100.0
                                         : 0.0;
  if (PctChange < ThresholdPct)
    return "unchanged";
  if (Delta.Delta > 0)
    return "regression";
  return "improvement";
}

StringRef
RegressionClassifier::classifyWithSeverity(const MetricDelta &Delta,
                                           bool HigherIsBetter) const {
  if (Delta.Before == 0.0 && Delta.Delta == 0.0)
    return "unchanged";

  double PctChange =
      Delta.Before != 0.0 ? Delta.Delta / std::abs(Delta.Before) * 100.0 : 0.0;
  double AbsPct = std::abs(PctChange);

  if (AbsPct < ThresholdPct)
    return "unchanged";

  // For higher-is-better (e.g. IPC), an increase is an improvement.
  bool IsRegression = HigherIsBetter ? (PctChange < 0) : (PctChange > 0);

  if (!IsRegression)
    return "improvement";
  if (AbsPct > 10.0)
    return "critical";
  if (AbsPct > 3.0)
    return "moderate";
  return "minor";
}

} // namespace llvm::advisor
