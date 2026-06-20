//===--- CallFrequency.h - LLVM Advisor ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once
#include "InsightBase.h"

namespace llvm::advisor {

class CallFrequencyInsight final : public Insight {
public:
  InsightKind getKind() const override { return InsightKind::CallFrequency; }
  StringRef getName() const override { return "call_frequency"; }
  StringRef getDescription() const override {
    return "Most-called functions and call-site fan-out in the call graph";
  }
  StringRef getRequiredCapability() const override { return "llvm.call_graph"; }
  Expected<InsightOutput> analyze(const InsightInput &Input) const override;
};

} // namespace llvm::advisor
