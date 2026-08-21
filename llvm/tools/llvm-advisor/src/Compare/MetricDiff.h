//===------------------- MetricDiff.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of MetricDiff in Compare
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"

namespace llvm::advisor {
struct MetricDelta {
  double Before = 0;
  double After = 0;
  double Delta = 0;
};

inline MetricDelta diffMetric(double Before, double After) {
  return {Before, After, After - Before};
}
} // namespace llvm::advisor
