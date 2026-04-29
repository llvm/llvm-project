//===------------------- Metrics.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Prometheus-compatible metrics counters for observability.
// Tracks performance and operational metrics.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"

namespace llvm::advisor {

/// Simple counter-based metrics tracker.
class Metrics {
public:
  /// Increment a named counter by the given delta.
  void increment(StringRef Name, uint64_t Delta = 1);

  /// Read the current value of a named counter.
  uint64_t get(StringRef Name) const;

  /// Export all counters as simple text lines.
  ///
  /// Each line is "name value".
  std::string toText() const;

private:
  StringMap<uint64_t> Counters;
};

} // namespace llvm::advisor
